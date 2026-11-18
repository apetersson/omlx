# SPDX-License-Identifier: Apache-2.0
"""Optional, single-request oQh8 expert paging for local qualification.

Only routed packed weights and their group-affine sidecars are externalized.
Every projection's small BF16 sign vector is still a normal strict-load model
parameter. No Engram file or provider enters this path. There is no change to
the ordinary resident expert implementation when this is not installed.

The budget counts live cached row payloads, *not* total RAM. Compact gathers,
rotations and outputs have a separate workspace guard. Native weights, Metal
allocator/driver overhead, file cache, KV state and locked BF16 tables aren't
part of either number. A dedicated low-memory runner should also disable the
process-wide MLX free-buffer cache (mx.set_cache_limit(0)).
"""

from __future__ import annotations

import fcntl
import os
import sys
from collections import OrderedDict, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from threading import Lock

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from .artifact import Artifact, member
from .convert import ALGORITHM, TABLE_END, stamp
from .experts import rotate_activation
from .quant_io import BITS, GROUP, Spec, View, bank_specs, read_header, require

MAX_PREFILL_TOKENS = 256  # flattened batch * sequence, not the context limit
_MAX_ROUTES = 64
_CHUNK_TOKENS = 8
_MAX_UNIQUE = 32
_WORKSPACE_BYTES = 256 << 20
_IO_CHUNK = 8 << 20
_ROW_FIELDS = ("weight", "oqh_scales", "oqh_biases")  # the order required by gather_qmm
_NUMPY = {"U32": "<u4", "I32": "<i4", "BF16": "<u2", "F32": "<f4"}
_MLX = {"U32": mx.uint32, "I32": mx.int32, "BF16": mx.bfloat16, "F32": mx.float32}


@dataclass(frozen=True)
class _Shard:
    fd: int
    identity: tuple
    views: dict[str, View]

    def check(self):
        require(
            stamp(os.fstat(self.fd)) == self.identity,
            "core shard changed since header audit",
        )


@dataclass(frozen=True)
class _Bank:
    name: str
    shape: tuple[int, int, int]  # logical [expert, out, in]
    shard: _Shard
    parts: tuple[View, ...]  # one expert per view; start is the first expert's offset

    @property
    def row_bytes(self):
        return sum(v.spec.nbytes for v in self.parts)


def _read(fd, view, *, row=0, check=None):
    """Read one validated extent. No mmap, shard-level mx.load or tensor-index guess."""
    spec = view.spec
    size = spec.nbytes
    offset = view.start + row * size

    def pread(pos, n):
        data = os.pread(fd, n, pos)
        require(len(data) == n, "short positional core read")
        return data

    if size <= _IO_CHUNK:
        a = np.frombuffer(pread(offset, size), dtype=_NUMPY[spec.dtype])
    else:
        # A mixed shard can also hold a large *resident* native matrix. Avoid
        # a whole-tensor Python bytes intermediate and Darwin's I/O size limit.
        a = np.empty(spec.shape, dtype=_NUMPY[spec.dtype])
        buf = memoryview(a).cast("B")
        for i in range(0, size, _IO_CHUNK):
            n = min(_IO_CHUNK, size - i)
            buf[i : i + n] = pread(offset + i, n)
    if check == "affine":
        require(
            not np.any((a & 0x7F80) == 0x7F80), "non-finite BF16 expert scale or bias"
        )
    elif check == "sign":
        require(
            np.all((a & 0x7FFF) == 0x3F80),
            "invalid BF16 rotation signs (expected +/-1)",
        )
    out = mx.array(a.reshape(spec.shape))
    return out.view(mx.bfloat16) if spec.dtype == "BF16" else out


class _BoundedProjection(nn.Module):
    def __init__(self, ctx, bank, signs):
        super().__init__()
        self._ctx = ctx
        self._bank = bank
        self.oqh_signs = signs

    def __call__(self, x, indices):
        return self._ctx.project(self._bank, x, indices, self.oqh_signs)


class BoundedExperts:
    """Owns pinned read-only shard descriptors and one global byte-bounded LRU."""

    def __init__(self, max_bytes, top_k, excluded, signs, shards):
        self.excluded_names = frozenset(excluded)
        self.sign_names = frozenset(signs)
        self.mixed_shards = frozenset(shards)
        self._max_bytes = max_bytes
        self._top_k = top_k
        self._shards = {}
        self._files = ExitStack()
        self._lock = Lock()
        self._closed = False
        self._cache = OrderedDict()
        self._used = 0
        self._stats = dict(
            hits=0,
            misses=0,
            evictions=0,
            expert_read_bytes=0,
            resident_read_bytes=0,
            peak_cache_bytes=0,
            gathers=0,
            peak_unique=0,
        )

    @property
    def stats(self):
        with self._lock:
            return dict(
                self._stats,
                max_bytes=self._max_bytes,
                cache_bytes=self._used,
                cache_entries=len(self._cache),
                workspace_limit_bytes=_WORKSPACE_BYTES,
            )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        with self._lock:
            self._closed = True
            self._cache.clear()
            self._used = 0
            self._files.close()
            self._shards.clear()

    def read_resident(self, shard_name, tensor_name):
        """Strict loader's selective read for a retained tensor on a mixed shard."""
        with self._lock:
            require(not self._closed, "bounded expert context is closed")
            require(
                tensor_name not in self.excluded_names,
                "refusing to read a complete expert bank",
            )
            shard = self._shards[shard_name]
            shard.check()
            view = shard.views[tensor_name]
            arr = _read(
                shard.fd, view, check="sign" if tensor_name in self.sign_names else None
            )
            shard.check()
            self._stats["resident_read_bytes"] += view.spec.nbytes
            return arr

    def _row(self, bank, expert):
        key = (bank.name, expert)
        if key in self._cache:
            self._stats["hits"] += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        # Evict before reading a miss. Previous gather results are always
        # evaluated; no old computation graph keeps those evicted rows alive.
        size = bank.row_bytes
        while self._used + size > self._max_bytes:
            _, evicted = self._cache.popitem(last=False)
            self._used -= sum(a.nbytes for a in evicted)
            self._stats["evictions"] += 1
            del evicted
        row = tuple(
            _read(bank.shard.fd, view, row=expert, check="affine" if i else None)
            for i, view in enumerate(bank.parts)
        )
        mx.eval(*row)
        self._cache[key] = row
        self._used += size
        self._stats["misses"] += 1
        self._stats["expert_read_bytes"] += size
        self._stats["peak_cache_bytes"] = max(
            self._stats["peak_cache_bytes"], self._used
        )
        return row

    def _qmm(self, bank, rotated, experts, lhs, rhs, dtype):
        # This function owns the temporary compact RHS. Drop those references
        # before subsequent misses can evict any of these cached rows.
        rows = [self._row(bank, int(i)) for i in experts]
        packed, scales, biases = (mx.stack(part) for part in zip(*rows))
        out = mx.gather_qmm(
            rotated,
            packed,
            scales,
            biases,
            lhs_indices=mx.array(lhs.astype(np.uint32)),
            rhs_indices=mx.array(rhs.astype(np.uint32)),
            transpose=True,
            group_size=GROUP,
            bits=BITS,
            mode="affine",
        ).astype(dtype)
        finite = mx.all(mx.isfinite(out))
        mx.eval(out, finite)  # materialization barrier BEFORE any future LRU eviction
        require(finite.item(), f"non-finite bounded oQh projection: {bank.name}")
        self._stats["gathers"] += 1
        self._stats["peak_unique"] = max(self._stats["peak_unique"], len(experts))
        return out

    def _gather(self, bank, x, ids, signs, limit):
        # Split *unique experts*, not contiguous positions: even when only one
        # row fits, repeated ids within this token chunk incur only one read.
        unique, inverse = np.unique(ids, return_inverse=True)
        inverse = inverse.ravel()
        rotated = rotate_activation(x.astype(mx.bfloat16), signs)
        mx.eval(rotated)
        parts, positions = [], []
        for base in range(0, len(unique), limit):
            experts = unique[base : base + limit]
            pos = np.flatnonzero((inverse >= base) & (inverse < base + len(experts)))
            # MLX gathers matrices from flattened leading dimensions. The gate
            # broadcasts one activation/token; down has a distinct one/route.
            lhs = pos // ids.shape[1] if x.shape[1] == 1 else pos
            parts.append(
                self._qmm(bank, rotated, experts, lhs, inverse[pos] - base, x.dtype)
            )
            positions.append(pos)
        if len(parts) == 1:
            out = parts[0]  # positions already span the complete flattened grid
        else:
            restore = mx.array(np.argsort(np.concatenate(positions)).astype(np.uint32))
            out = mx.take(mx.concatenate(parts), restore, axis=0)
        out = out.reshape(*ids.shape, 1, bank.shape[1])
        mx.eval(out)
        return out

    def project(self, bank, x, indices, signs):
        with self._lock:
            require(not self._closed, "bounded expert context is closed")
            e, o, k = bank.shape
            require(
                indices.ndim == 2 and indices.dtype in (mx.uint32, mx.int32),
                "expected a [tokens, top-k] int32/uint32 route matrix",
            )
            t, r = indices.shape
            require(
                0 < t <= MAX_PREFILL_TOKENS,
                "bounded expert prefill too long; chunk the prompt",
            )
            require(
                r == self._top_k
                and x.ndim == 4
                and x.shape[0] == t
                and x.shape[1] in (1, r)
                and x.shape[2:] == (1, k)
                and x.dtype in (mx.bfloat16, mx.float16, mx.float32),
                "unexpected bounded oQh activation shape, dtype, or route count",
            )
            # Conservative allowance for staging a miss, activation transforms,
            # partial outputs, concatenation, and indices; the remaining bytes
            # bound the compact RHS. This does not model driver allocations.
            row = bank.row_bytes
            reserve = 2 * row + 16 * x.size + 3 * t * r * o * x.itemsize + 64 * t * r
            limit = min(
                _MAX_UNIQUE,
                e - 1,
                self._max_bytes // row,
                (_WORKSPACE_BYTES - reserve) // row,
            )
            require(
                limit >= 1,
                "bounded expert workspace exceeded; use a smaller prefill chunk",
            )
            bank.shard.check()
            ids = np.asarray(
                indices
            )  # bounded and eager; resolves preceding layers/router
            require(
                ids.min() >= 0 and ids.max() < e, "routed expert index out of range"
            )
            outputs = [
                self._gather(
                    bank,
                    x[i : i + _CHUNK_TOKENS],
                    ids[i : i + _CHUNK_TOKENS],
                    signs,
                    limit,
                )
                for i in range(0, t, _CHUNK_TOKENS)
            ]
            bank.shard.check()
            if len(outputs) == 1:
                return outputs[0]
            out = mx.concatenate(outputs)
            mx.eval(out)
            return out


def install_bounded_experts(
    model, model_dir, config, weight_map, max_bytes
) -> BoundedExperts:
    """Install on an *unloaded* model. Caller must close the returned context.

    weight_map is the decoded index's weight_map (tensor name -> core basename).
    Do not mx.load any file in ctx.mixed_shards: call ctx.read_resident for its
    strict-load keys minus ctx.excluded_names. All ctx.sign_names are retained.
    Metadata/header validation is not whole-payload authentication: use an
    already trusted local export, or the loader's optional --verify-core pass.
    """
    require(
        type(max_bytes) is int and max_bytes > 0,
        "cache budget must be a positive integer byte count",
    )
    require(sys.byteorder == "little", "expected little-endian MLX")
    # Also validate standalone callers: authenticate config/index assets, vet
    # all manifest members/sizes, and require the exact supported routing plan.
    artifact = Artifact.open(model_dir)
    require(
        artifact.config == config and artifact.weight_map == weight_map,
        "supplied config or index differs from artifact",
    )
    require(
        artifact.manifest.get("quantization") == ALGORITHM,
        "unsupported routed quantization",
    )
    blocks = config["mlp_layer_types"]
    e, h, m = (
        config[n]
        for n in (
            "n_routed_experts",
            "routed_expert_hidden_size",
            "moe_intermediate_size",
        )
    )
    top_k = config["num_experts_per_tok"]
    require(
        all(type(v) is int and v > 0 for v in (e, h, m, top_k))
        and e > 1
        and top_k <= min(e, _MAX_ROUTES),
        "invalid bounded expert dimensions or top-k",
    )
    require(
        len(blocks) == len(model.model.layers) == config["num_hidden_layers"]
        and set(blocks) <= {"dense", "sparse"},
        "invalid routed layer schedule",
    )

    plans = []
    excluded, signs = set(), set()
    for i, kind in enumerate(blocks):
        if kind == "sparse":
            parent = model.model.layers[i].mlp.experts
            for attr, shape in (
                ("gate_up_proj", (e, 2 * m, h)),
                ("down_proj", (e, h, m)),
            ):
                name = f"model.layers.{i}.mlp.experts.{attr}"
                specs = bank_specs(name, Spec("BF16", shape))
                shard = weight_map[name + ".weight"]
                require(
                    all(weight_map[n] == shard for n in specs),
                    f"split oQh bank: {name}",
                )
                signs.add(name + ".oqh_signs")
                excluded.update(name + "." + field for field in _ROW_FIELDS)
                plans.append((parent, attr, name, shape, shard, specs))
    require(
        not any(n.endswith(TABLE_END) for n in weight_map),
        "external Engram table in core index",
    )
    require(
        {n for n in weight_map if ".mlp.experts." in n} == excluded | signs,
        "unexpected or missing routed expert tensors",
    )
    groups = defaultdict(set)
    for n, shard in weight_map.items():
        groups[shard].add(n)
    ctx = BoundedExperts(
        max_bytes, top_k, excluded, signs, {weight_map[n] for n in excluded}
    )
    params = dict(
        tree_flatten(model.parameters())
    )  # references only; never evaluate placeholders
    try:
        for name in ctx.mixed_shards:
            path = member(artifact.root, name)
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            f = ctx._files.enter_context(os.fdopen(fd, "rb", buffering=0))
            # Do not populate the Darwin file cache with streaming cold misses.
            # This is per descriptor; no global VM settings are modified.
            if sys.platform == "darwin":
                fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
            identity = stamp(os.fstat(fd))
            require(
                identity[2] == artifact.manifest["core_shards"][name]["size"],
                f"changed size: {name}",
            )
            shard = _Shard(fd, identity, read_header(f))
            shard.check()
            require(
                shard.views.keys() == groups[name], f"index/header mismatch: {name}"
            )
            for n, view in shard.views.items():
                require(
                    n in params and params[n].shape == view.spec.shape,
                    f"model/header shape mismatch: {n}",
                )
                if n not in excluded and n not in signs:
                    # Native floating placeholders need not have their serialized
                    # precision. An integer table/index must never be reinterpreted.
                    compatible = (
                        params[n].dtype in (mx.float32, mx.bfloat16)
                        if view.spec.dtype in ("F32", "BF16")
                        else params[n].dtype == _MLX[view.spec.dtype]
                    )
                    require(compatible, f"wrong native core dtype: {n}")
            ctx._shards[name] = shard
        updates = []
        for parent, attr, name, shape, shard_name, specs in plans:
            shard = ctx._shards[shard_name]
            for n, spec in specs.items():
                require(
                    shard.views[n].spec == spec and params[n].dtype == _MLX[spec.dtype],
                    f"wrong packed expert dtype/shape: {n}",
                )
            parts = []
            for field in _ROW_FIELDS:
                v = shard.views[f"{name}.{field}"]
                parts.append(View(v.start, Spec(v.spec.dtype, v.spec.shape[1:])))
            bank = _Bank(name, shape, shard, tuple(parts))
            require(
                bank.row_bytes <= max_bytes,
                f"cache budget below one expert row ({bank.row_bytes} bytes)",
            )
            require(
                3 * bank.row_bytes < _WORKSPACE_BYTES,
                "one expert exceeds bounded workspace",
            )
            updates.append(
                (
                    parent,
                    attr,
                    _BoundedProjection(ctx, bank, params[name + ".oqh_signs"]),
                )
            )
        # No mutation until every bank's source, dtype, shape, and bounds pass.
        for parent, attr, projection in updates:
            setattr(parent, attr, projection)
        return ctx
    except BaseException:
        ctx.close()
        raise
