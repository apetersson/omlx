# SPDX-License-Identifier: Apache-2.0
"""Opt-in fixed expert slots for single-request HY-V4-Flash inference on Metal.

Each gate/up and down projection gets the same number of slots (< #experts),
sized from the *audited physical* rows. The budget is the sum of the anonymous,
page-rounded slot mappings, not a process RSS limit. There is no whole-shard
mapping, anonymous-to-Metal copy, staging array or compact RHS. Read misses
straight into their final slots; MLX sees those very bytes through DLPack.
The inherited bounds still apply to tokens, route count and gather workspace.
As in bounded mode, disable the process-wide free-buffer cache in the runner.

Eager inference only: each QMM completes under the context lock before any
subsequent CPU write. The arrays are private to this context, NOT trainable
model parameters; don't export, transform or run speculative graphs on them.
"""

from __future__ import annotations

import mmap
import os
import platform
import sys
from collections import OrderedDict

import mlx.core as mx
import numpy as np

from . import bounded
from .experts import rotate_activation
from .quant_io import BITS, GROUP, require


def _pages(size):
    return ((size + mmap.PAGESIZE - 1) // mmap.PAGESIZE) * mmap.PAGESIZE


def _plan(banks, budget):
    def cost(slots):
        return sum(_pages(slots * v.spec.nbytes) for b in banks for v in b.parts)

    # Keep even toy banks partial. Unused remainder (< one more slot across
    # all projections, unless capped by expert count) is not allocated.
    hi = min(b.shape[0] for b in banks) - 1
    lo = 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if cost(mid) <= budget:
            lo = mid
        else:
            hi = mid - 1
    require(lo >= 1, f"slot cache budget below one slot/projection ({cost(1)} bytes)")
    return lo, cost(lo)


def _shared_field(view, slots):
    # mmap(-1) is anonymous RAM, never a checkpoint fd. Page-aligned start
    # and padded backing for Metal. Both exporters OWN the backing (DLPack
    # deleter -> ndarray -> mmap); never explicitly unmap a live MTLBuffer.
    spec = view.spec
    arena = mmap.mmap(-1, _pages(slots * spec.nbytes))
    host = np.ndarray(
        (slots, *spec.shape), dtype=bounded._NUMPY[spec.dtype], buffer=arena
    )
    rhs = mx.from_dlpack(host, copy=False)  # fail, do not fall back to a copy
    if spec.dtype == "BF16":
        rhs = rhs.view(mx.bfloat16)  # bit reinterpretation, not a conversion
    # Also check that the final (possibly reinterpreted) native buffer really
    # aliases these writable CPU addresses. memoryview does not cast BF16.
    buf = memoryview(rhs)
    require(
        rhs.shape == host.shape
        and rhs.dtype == bounded._MLX[spec.dtype]
        and buf.c_contiguous
        and buf.nbytes == host.nbytes
        and np.frombuffer(buf, dtype=np.uint8).ctypes.data == host.ctypes.data,
        "Metal slot buffer is not a contiguous zero-copy alias",
    )
    return host, rhs


class _SlotBank:
    """One projection's direct-fill buffers and deterministic batch-pinned LRU."""

    def __init__(self, bank, slots, counters):
        self.bank = bank
        self.live = OrderedDict()  # expert -> slot, oldest first
        self.free = list(range(slots - 1, -1, -1))
        self.stats = counters
        fields = [_shared_field(v, slots) for v in bank.parts]
        self.host, self.rhs = zip(*fields)

    def _fill(self, missing):
        # Sorted unique expert ids from _gather. Adjacent source rows need
        # just one read/field/run, even if their destination slots are not
        # adjacent. Never read through holes or keep Python bytes payloads.
        start = 0
        while start < len(missing):
            stop = start + 1
            while stop < len(missing) and missing[stop][0] == missing[stop - 1][0] + 1:
                stop += 1
            run = missing[start:stop]
            for view, host in zip(self.bank.parts, self.host):
                size = view.spec.nbytes
                rows = [
                    host[slot] for _, slot in run
                ]  # ndarray views, NOT fancy indexing
                self.stats["expert_preadv_calls"] += 1
                n = os.preadv(self.bank.shard.fd, rows, view.start + run[0][0] * size)
                self.stats["expert_read_bytes"] += n
                require(n == len(run) * size, "short positional core read")
                if view.spec.dtype == "BF16":
                    for a in rows:
                        require(
                            not np.any((a & 0x7F80) == 0x7F80),
                            "non-finite BF16 expert scale or bias",
                        )
            start = stop

    def resolve(self, experts):
        # Pin all hits in this gather before choosing victims. A miss early
        # in id order must not evict another id needed by the SAME gather.
        ids = experts.tolist()
        missing = []
        for e in ids:
            if e in self.live:
                self.live.move_to_end(e)
                self.stats["hits"] += 1
            else:
                missing.append(e)
        if missing:
            fills = []
            for e in missing:
                if self.free:
                    slot = self.free.pop()
                else:
                    _, slot = self.live.popitem(last=False)
                    self.stats["evictions"] += 1
                fills.append((e, slot))
            # Invalidate victims BEFORE preadv can write a single byte. An
            # error midway through a run must publish NO partially read row.
            try:
                self._fill(fills)
                self.bank.shard.check()
            except BaseException:
                self.free.extend(slot for _, slot in reversed(fills))
                raise
            self.live.update(fills)
            self.stats["misses"] += len(fills)
        return np.array([self.live[e] for e in ids], dtype=np.uint32)


class SlotExperts(bounded.BoundedExperts):
    """Same strict loader, lifecycle, route checks and stats protocol as bounded."""

    def __init__(self, source, banks):
        super().__init__(
            source._max_bytes,
            source._top_k,
            source.excluded_names,
            source.sign_names,
            source.mixed_shards,
        )
        self._slots, capacity_bytes = _plan(banks, self._max_bytes)
        self._stats["expert_preadv_calls"] = 0
        self._unfenced = False
        self._stores = {}
        self._files.callback(self._stores.clear)  # under the inherited close() lock
        try:
            for bank in banks:
                self._stores[bank.name] = _SlotBank(bank, self._slots, self._stats)
            self._shards.update(source._shards)  # keep audited FDs, not path reopens
            self._files.enter_context(source)
            self._stats["peak_cache_bytes"] = self._used = capacity_bytes
        except BaseException:
            self.close()
            raise

    @property
    def stats(self):
        with self._lock:
            return dict(
                self._stats,
                max_bytes=self._max_bytes,
                cache_bytes=self._used,  # committed capacity including page padding
                cache_entries=sum(len(s.live) for s in self._stores.values()),
                valid_cache_bytes=sum(
                    len(s.live) * s.bank.row_bytes for s in self._stores.values()
                ),
                slots_per_projection=self._slots,
                workspace_limit_bytes=bounded._WORKSPACE_BYTES,
            )

    def _qmm(self, bank, rotated, experts, lhs, rhs, dtype):
        require(
            not self._unfenced,
            "slot context requires close after failed MLX evaluation",
        )
        store = self._stores[bank.name]
        slots = store.resolve(experts)
        # A host-writeable bank may never have an outstanding GPU reader at
        # the next resolve(). Fail closed if evaluation itself is interrupted.
        self._unfenced = True
        out = mx.gather_qmm(
            rotated,
            *store.rhs,
            lhs_indices=None if lhs is None else mx.array(lhs.astype(np.uint32)),
            rhs_indices=mx.array(slots[rhs]),
            transpose=True,
            group_size=GROUP,
            bits=BITS,
            mode="affine",
        ).astype(dtype)
        finite = mx.all(mx.isfinite(out))
        mx.eval(out, finite)
        ok = finite.item()
        self._unfenced = False
        require(ok, f"non-finite bounded oQh projection: {bank.name}")
        self._stats["gathers"] += 1
        self._stats["peak_unique"] = max(self._stats["peak_unique"], len(experts))
        return out

    def _gather(self, bank, x, ids, signs, limit):
        limit = min(self._slots, limit)
        if ids.size <= limit:  # common single-token decode; no separate rotation sync
            experts, inverse = np.unique(ids, return_inverse=True)
            rotated = rotate_activation(x.astype(mx.bfloat16), signs)
            return self._qmm(
                bank, rotated, experts, None, inverse.reshape(ids.shape), x.dtype
            )
        # The bounded implementation chunks by unique expert, supplies explicit
        # LHS indices for gate broadcasting vs down lanes, restores route order.
        return super()._gather(bank, x, ids, signs, limit)


def install_slot_experts(model, root, cfg, weight_map, budget) -> SlotExperts:
    """Install on an unloaded, exclusively owned model. Close the returned context.

    Requires Apple Silicon, Metal, zero-copy CPU DLPack import and preadv. No
    implicit fallback (which would change the selected policy or memory use).
    On failure leave the original placeholders and strict parameter tree intact.
    """
    require(
        sys.platform == "darwin"
        and platform.machine() == "arm64"
        and mx.metal.is_available()
        and mx.default_device() == mx.gpu
        and callable(getattr(mx, "from_dlpack", None))
        and callable(getattr(os, "preadv", None))
        and os.sysconf("SC_IOV_MAX") >= bounded._MAX_UNIQUE,
        "expert slots require Apple Silicon Metal, zero-copy DLPack and positional preadv",
    )
    originals = []
    for layer in model.model.layers:
        parent = getattr(layer.mlp, "experts", None)
        if parent is not None:
            for attr in ("gate_up_proj", "down_proj"):
                originals.append((parent, attr, getattr(parent, attr)))
    # Reuse the *entire* audited installer, retaining its pinned descriptors
    # and projection wrappers; it performs no routed payload reads/allocations.
    source = bounded.install_bounded_experts(model, root, cfg, weight_map, budget)
    ctx = None
    try:
        projections = [getattr(p, attr) for p, attr, _ in originals]
        ctx = SlotExperts(source, [p._bank for p in projections])
        for p in projections:
            p._ctx = ctx
        return ctx
    except BaseException:
        for parent, attr, original in originals:
            setattr(parent, attr, original)
        if ctx is not None:
            ctx.close()
        source.close()
        raise
