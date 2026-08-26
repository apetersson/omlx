# SPDX-License-Identifier: Apache-2.0
"""Bounded-memory PLE n-gram layout for Qwen4-Exp oQ conversion.

The upstream checkpoint stores its roughly 51B-parameter PLE embedding as
``split_ngram_parts`` independent tensors.  Transformers concatenates all of
them into one runtime embedding, but doing that during an MLX conversion
would transiently allocate the complete BF16 table (about 95 GiB) and the
resulting packed tensor would contain several billion elements.

Expose groups of source shards as virtual, row-addressable tensors instead.
The streaming quantizer reads each group in bounded row chunks and emits a
small collection of independently quantized embedding modules.  Sixteen
groups keeps every Q4 tensor below two GiB for the released checkpoint while
reducing runtime dispatch from 128 shards to 16 groups.
"""

from __future__ import annotations

import logging
import math
import weakref

import mlx.core as mx

logger = logging.getLogger(__name__)

DEFAULT_RUNTIME_GROUPS = 16
_SOURCE_PREFIX = (
    "model.language_model.layers.{layer}.ple.ple_embedding."
    "ngram_embedding.shard_{shard}.weight"
)
_GROUP_PREFIX = (
    "model.language_model.layers.{layer}.ple.ple_embedding."
    "ngram_embedding.groups.{group}.weight"
)


class _GroupedRows:
    """Lazy row concatenation understood by oQ's chunked quantizer."""

    def __init__(self, index, sources: tuple[str, ...], shape, dtype: str):
        self._index_ref = weakref.ref(index)
        self.sources = sources
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self._dtype_name = dtype
        self.dtype = {
            "BF16": mx.bfloat16,
            "F16": mx.float16,
            "F32": mx.float32,
        }.get(dtype, mx.bfloat16)

        offsets = [0]
        for key in sources:
            source_shape = index.source_shape(key)
            offsets.append(offsets[-1] + int(source_shape[0]))
        self._offsets = tuple(offsets)

    @property
    def size(self):
        return math.prod(self.shape)

    @property
    def nbytes(self):
        return self.size * self.dtype.size

    def _index(self):
        index = self._index_ref()
        if index is None:
            raise RuntimeError("Qwen4 n-gram view outlived its tensor index")
        return index

    def _load_rows(self, r0: int, r1: int):
        if r0 < 0 or r1 < r0 or r1 > self.shape[0]:
            raise IndexError(f"invalid row slice [{r0}:{r1}] for {self.shape}")
        if r0 == r1:
            return mx.zeros((0, *self.shape[1:]), dtype=self.dtype)

        # Import lazily to avoid a module cycle: oq imports the registrar while
        # constructing _LazyTensorIndex.
        from omlx.oq import _LazyTensor

        index = self._index()
        pieces = []
        for source_idx, key in enumerate(self.sources):
            source_start = self._offsets[source_idx]
            source_end = self._offsets[source_idx + 1]
            local_start = max(r0, source_start) - source_start
            local_end = min(r1, source_end) - source_start
            if local_start >= local_end:
                continue
            meta = index._index[key]
            lazy = _LazyTensor(meta[0], meta[1], meta[2], meta[3], meta[4], meta[5])
            piece = lazy._load_rows(local_start, local_end)
            if piece.dtype != self.dtype:
                piece = piece.astype(self.dtype)
            pieces.append(piece)

        if not pieces:
            raise RuntimeError(f"no Qwen4 source rows found for [{r0}:{r1}]")
        if len(pieces) == 1:
            return pieces[0]
        result = mx.concatenate(pieces, axis=0)
        mx.eval(result)
        del pieces
        mx.clear_cache()
        return result

    def __getitem__(self, item):
        if isinstance(item, slice):
            start, stop, step = item.indices(self.shape[0])
            if step != 1:
                return self._load_rows(0, self.shape[0])[item]
            return self._load_rows(start, stop)
        if isinstance(item, tuple):
            first, *rest = item
            if isinstance(first, slice):
                value = self[first]
                return value[(slice(None), *rest)]
            value = self._load_rows(int(first), int(first) + 1)
            return value[(0, *rest)]
        return self._load_rows(int(item), int(item) + 1)[0]


def _is_qwen4_exp(config: dict) -> bool:
    return str(config.get("model_type", "")).lower() == "qwen4_exp"


def register(
    index, config: dict, *, runtime_groups: int = DEFAULT_RUNTIME_GROUPS
) -> int:
    """Replace raw Qwen4 PLE shards with grouped lazy tensors.

    Returns the number of runtime groups registered, or zero for checkpoints
    that already carry grouped tensors (including an oQ output).
    """
    if not _is_qwen4_exp(config):
        return 0

    text = config.get("text_config") or {}
    ple_layer_ids = list(text.get("ple_layer_ids") or [])
    split_parts = int(text.get("split_ngram_parts") or 0)
    if not ple_layer_ids or split_parts <= 0:
        return 0
    if runtime_groups <= 0 or split_parts % runtime_groups:
        raise ValueError(
            "Qwen4 PLE split_ngram_parts must be divisible by runtime groups: "
            f"{split_parts} % {runtime_groups}"
        )
    config["omlx_qwen4_ngram_groups"] = runtime_groups

    registered = 0
    parts_per_group = split_parts // runtime_groups
    for one_indexed_layer in ple_layer_ids:
        layer = int(one_indexed_layer) - 1
        sources = tuple(
            _SOURCE_PREFIX.format(layer=layer, shard=shard)
            for shard in range(split_parts)
        )
        present = tuple(key for key in sources if index.source_shape(key) is not None)
        if not present:
            continue
        if len(present) != split_parts:
            missing = [key for key in sources if index.source_shape(key) is None]
            raise ValueError(
                f"incomplete Qwen4 PLE shard set for layer {layer}: "
                f"{len(present)}/{split_parts} present; first missing={missing[0]}"
            )

        shapes = [tuple(index.source_shape(key)) for key in sources]
        dtypes = [index._index[key][5] for key in sources]
        tail = shapes[0][1:]
        if not tail or any(shape[1:] != tail for shape in shapes):
            raise ValueError(f"inconsistent Qwen4 PLE shard shapes: {shapes[:3]}")
        if len(set(dtypes)) != 1:
            raise ValueError(f"mixed Qwen4 PLE shard dtypes: {sorted(set(dtypes))}")

        for group in range(runtime_groups):
            start = group * parts_per_group
            group_sources = sources[start : start + parts_per_group]
            rows = sum(shapes[i][0] for i in range(start, start + parts_per_group))
            shape = (rows, *tail)
            index_ref = weakref.ref(index)

            def materialize(
                group_sources=group_sources,
                shape=shape,
                dtype=dtypes[0],
                index_ref=index_ref,
            ):
                live_index = index_ref()
                if live_index is None:
                    raise RuntimeError("Qwen4 n-gram materializer lost its index")
                return _GroupedRows(live_index, group_sources, shape, dtype)

            index.register_virtual(
                _GROUP_PREFIX.format(layer=layer, group=group),
                shape,
                dtypes[0],
                materialize,
                hides=group_sources,
            )
            registered += 1

    if registered:
        logger.info(
            "Qwen4-Exp PLE: exposed %d bounded-memory runtime groups from %d-way shards",
            registered,
            split_parts,
        )
    return registered
