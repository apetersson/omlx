# SPDX-License-Identifier: Apache-2.0
"""Direct-fill slot tests. Reproduce the isolated benchmark with

    MLX_ENABLE_TF32=0 python -m pytest --noconftest -q -s tests/test_hy4_flash_slot_cache.py -k benchmark

No checkpoint download. Only the two tiny synthetic projections or one synthetic
32-expert layer with real row dimensions are exercised. "Cold" means cold in
both expert stores (count actual transfers), not a machine-wide disk-cache drop.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import platform
import resource
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import weakref
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from omlx.patches.hy_v4_flash import bounded
from omlx.patches.hy_v4_flash import slot_cache as sc
from omlx.patches.hy_v4_flash.quant_io import (
    Spec,
    View,
    Writer,
    bank_specs,
    read_header,
)
from tests import test_hy4_flash_bounded as ref

bundle = ref.bundle
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin"
    or platform.machine() != "arm64"
    or not mx.metal.is_available()
    or not hasattr(os, "preadv"),
    reason="direct-fill expert slots need Apple Silicon Metal and preadv",
)


def _budget(slots, h=ref.H, m=ref.M):
    page = mmap.PAGESIZE
    # Separate anonymous mappings for packed, scales, biases for gate/up AND down.
    return sum(
        (slots * n + page - 1) // page * page
        for w in (2 * m * h, m * h)
        for n in (w, w // 64, w // 64)
    )


def _install(art, slots, load=True):
    model = ref._model(art.config)
    ctx = sc.install_slot_experts(
        model, art.root, art.config, art.weight_map, _budget(slots)
    )
    try:
        if load:
            ref.load_weights(model, art, ctx)
        return model, ctx
    except BaseException:
        ctx.close()
        raise


@pytest.mark.parametrize("slots", [1, 3, ref.E + 1])
def test_random_order_lanes_rotation_and_retained_results(bundle, slots):
    art, weights = bundle
    resident = ref._model(art.config)
    resident.load_weights(list(weights.items()), strict=True)
    model, ctx = _install(art, slots)
    rng = np.random.default_rng(43)
    keep = []
    hosts = [weakref.ref(a) for s in ctx._stores.values() for a in s.host]
    with ctx:
        assert ctx.stats["slots_per_projection"] == min(slots, ref.E - 1)
        assert ctx.stats["cache_entries"] == 0
        assert ctx.stats["cache_bytes"] == _budget(min(slots, ref.E - 1))
        for step, t in enumerate((1, 13, 3, 17, 1)):
            ids = mx.array(
                rng.integers(0, ref.E, (t, ref.R)),
                dtype=mx.uint32 if step % 2 else mx.int32,
            )
            for attr, k, lanes, dt in (
                ("gate_up_proj", ref.H, 1, mx.bfloat16),
                ("down_proj", ref.M, ref.R, mx.float32 if step % 2 else mx.float16),
            ):
                x = ref._normal((t, lanes, 1, k), 77 + step).astype(dt)
                expected = getattr(ref._banks(resident), attr)(x, ids)
                result = getattr(ref._banks(model), attr)(x, ids)
                assert result.shape == expected.shape and result.dtype == dt
                assert mx.array_equal(expected, result).item()
                keep.append((expected, result))
        assert ctx.stats["evictions"] > 0
        assert ctx.stats["peak_unique"] <= min(slots, ref.E - 1)
        assert (
            ctx.stats["valid_cache_bytes"] <= ctx.stats["cache_bytes"] <= _budget(slots)
        )
    ctx.close()
    assert all(h() is None for h in hosts)  # even with every result still alive
    assert all(mx.array_equal(a, b).item() for a, b in keep)
    assert ctx.stats["cache_bytes"] == ctx.stats["cache_entries"] == 0
    with pytest.raises(ValueError, match="closed"):
        ref._banks(model).gate_up_proj(mx.ones((1, 1, 1, ref.H)), mx.array([[0, 1, 2]]))


def test_exact_direct_reads_fixed_native_rhs_and_pinned_lru(bundle, monkeypatch):
    art, weights = bundle
    with (art.root / ref.SHARD).open("rb", buffering=0) as f:
        header = read_header(f)
    reads, gathers = [], []
    preadv, gather = os.preadv, mx.gather_qmm

    def spy(fd, rows, pos):
        reads.append((pos, [(r.ctypes.data, r.nbytes) for r in rows]))
        return preadv(fd, rows, pos)

    def native(x, w, s, b, **kwargs):
        gathers.append(
            [
                np.frombuffer(memoryview(a), dtype=np.uint8).ctypes.data
                for a in (w, s, b)
            ]
        )
        assert w.shape[0] == s.shape[0] == b.shape[0] == 3  # not compacted or full-bank
        return gather(x, w, s, b, **kwargs)

    monkeypatch.setattr(os, "preadv", spy)
    monkeypatch.setattr(mx, "gather_qmm", native)
    monkeypatch.setattr(
        mx, "stack", lambda *a, **kw: pytest.fail("assembled compact RHS")
    )
    monkeypatch.setattr(
        mx, "load", lambda *a, **kw: pytest.fail("loaded mixed expert shard")
    )
    model, ctx = _install(art, 3, load=False)
    with ctx:
        assert not reads
        assert ctx.excluded_names == {
            f"{p}.{s}" for p in (ref.GATE, ref.DOWN) for s in ref.PARTS
        }
        assert ctx.sign_names == {p + ".oqh_signs" for p in (ref.GATE, ref.DOWN)}
        assert ctx.mixed_shards == {ref.SHARD}
        ref.load_weights(model, art, ctx)
        live = dict(tree_flatten(model.parameters()))
        assert live.keys() == weights.keys() - ctx.excluded_names
        assert all(mx.array_equal(a, weights[n]).item() for n, a in live.items())
        with pytest.raises(ValueError, match="complete expert bank"):
            ctx.read_resident(ref.SHARD, ref.GATE + ".weight")
        # First fill 2 -> slot0 and 3 -> slot1; then hit2 pins slot0,
        # 0 -> free slot2 and 1 -> victim slot1 (noncontiguous iovecs).
        gate = ref._banks(model).gate_up_proj
        x = ref._normal((1, 1, 1, ref.H), 85)
        ids = mx.array([[3, 2, 3]])
        out = gate(x, ids)
        assert mx.array_equal(out, gate(x, ids)).item()
        gate(x, mx.array([[0, 1, 2]]))
        assert (
            len(reads) == 6
            and len(gathers) == 3
            and gathers[0] == gathers[1] == gathers[2]
        )
        for j, suffix in enumerate(ref.PARTS):
            v = header[f"{ref.GATE}.{suffix}"]
            row, base = v.spec.nbytes // ref.E, gathers[0][j]
            assert reads[j] == (v.start + 2 * row, [(base, row), (base + row, row)])
            assert reads[j + 3] == (v.start, [(base + 2 * row, row), (base + row, row)])
        s = ctx.stats
        assert (s["hits"], s["misses"], s["evictions"], s["expert_preadv_calls"]) == (
            3,
            4,
            1,
            6,
        )
        assert s["expert_read_bytes"] == 4 * ref.GATE_ROW
        assert s["cache_entries"] == 3 and s["valid_cache_bytes"] == 3 * ref.GATE_ROW
        assert s["cache_bytes"] == s["peak_cache_bytes"] == _budget(3)


def test_partial_writes_invalidate_victims_not_hits(bundle, monkeypatch):
    art, weights = bundle
    model, ctx = _install(art, 3)
    resident = ref._model(art.config)
    resident.load_weights(list(weights.items()), strict=True)
    gate, gold = ref._banks(model).gate_up_proj, ref._banks(resident).gate_up_proj
    x = ref._normal((1, 1, 1, ref.H), 18)
    with ctx:
        gate(x, mx.array([[0, 1, 2]]))
        preadv = os.preadv
        with monkeypatch.context() as p:
            # Write the first whole victim and part of the second, then report short.
            p.setattr(
                os,
                "preadv",
                lambda fd, rows, pos: preadv(
                    fd, [rows[0], memoryview(rows[1]).cast("B")[:1]], pos
                ),
            )
            with pytest.raises(ValueError, match="short positional core read"):
                gate(x, mx.array([[2, 4, 3]]))
        assert ctx.stats["cache_entries"] == 1
        for route in ([[2, 2, 2]], [[1, 0, 1]], [[4, 3, 2]]):
            ids = mx.array(route)
            assert mx.array_equal(gate(x, ids), gold(x, ids)).item()


def test_slot_validation_and_failed_install_restores_tree(bundle, monkeypatch):
    art, _ = bundle
    model = ref._model(art.config)
    before = dict(tree_flatten(model.parameters()))
    for budget in (False, 0, _budget(1) - 1):
        with pytest.raises(ValueError, match="budget"):
            sc.install_slot_experts(model, art.root, art.config, art.weight_map, budget)
    with monkeypatch.context() as p:
        # Simulate an importer copying despite copy=False. Must NOT continue.
        p.setattr(mx, "from_dlpack", lambda a, *, copy: mx.array(a.copy()))
        with pytest.raises(ValueError, match="zero-copy"):
            sc.install_slot_experts(
                model, art.root, art.config, art.weight_map, _budget(1)
            )
    after = dict(tree_flatten(model.parameters()))
    assert after.keys() == before.keys() and all(
        after[n] is a for n, a in before.items()
    )
    with monkeypatch.context() as p:
        p.delattr(os, "preadv")
        with pytest.raises(ValueError, match="preadv"):
            sc.install_slot_experts(
                model, art.root, art.config, art.weight_map, _budget(1)
            )
    model, ctx = _install(art, 3)
    gate, x = ref._banks(model).gate_up_proj, mx.ones((1, 1, 1, ref.H))
    with ctx:
        for ids in (
            mx.array([1, 2, 3]),
            mx.array([[1, 2]]),
            mx.array([[1.0, 2.0, 3.0]]),
            mx.array([[-1, 0, 0]]),
            mx.array([[ref.E, 2, 3]]),
        ):
            with pytest.raises(ValueError):
                gate(x, ids)
        assert ctx.stats["expert_read_bytes"] == 0
        with (art.root / ref.SHARD).open("ab") as f:
            f.write(b"x")
        with pytest.raises(ValueError, match="changed since header audit"):
            gate(x, mx.array([[0, 1, 2]]))


def test_failed_qmm_fence_forbids_further_slot_writes(bundle, monkeypatch):
    art, _ = bundle
    model, ctx = _install(art, 3)
    gate = ref._banks(model).gate_up_proj
    x = mx.ones((1, 1, 1, ref.H))
    with ctx:

        def interrupted(*args):
            raise RuntimeError("interrupted MLX evaluation")

        with monkeypatch.context() as p:
            p.setattr(mx, "eval", interrupted)
            with pytest.raises(RuntimeError, match="interrupted"):
                gate(x, mx.array([[0, 1, 2]]))
        # Don't overwrite even ONE previous slot after an uncertain GPU fence.
        calls = ctx.stats["expert_preadv_calls"]
        with pytest.raises(ValueError, match="requires close"):
            gate(x, mx.array([[2, 3, 4]]))
        assert ctx.stats["expert_preadv_calls"] == calls


def test_sidecar_signs_and_nonfinite_output_rejected(bundle):
    art, _ = bundle
    with (art.root / ref.SHARD).open("r+b", buffering=0) as f:
        views = read_header(f)
        v = views[ref.GATE + ".oqh_biases"]
        f.seek(v.start + 4 * v.spec.nbytes // ref.E)
        f.write(struct.pack("<H", 0x7FC0))
        f.seek(views[ref.DOWN + ".oqh_signs"].start)
        f.write(b"\x00\x00")
    model, ctx = _install(art, 3, load=False)
    with ctx:
        with pytest.raises(ValueError, match="invalid BF16 rotation signs"):
            ctx.read_resident(ref.SHARD, ref.DOWN + ".oqh_signs")
        x, gate = mx.ones((1, 1, 1, ref.H)), ref._banks(model).gate_up_proj
        gate(x, mx.array([[0, 1, 2]]))
        with pytest.raises(ValueError, match="non-finite BF16"):
            gate(x, mx.array([[2, 3, 4]]))
        assert ctx.stats["cache_entries"] == 1
        with pytest.raises(ValueError, match="non-finite bounded"):
            gate(mx.full(x.shape, mx.inf), mx.array([[2, 2, 2]]))


# Real per-expert transfer and native gather dimensions, NOT a full model.
_BH, _BM, _BE, _BR = 1536, 1536, 32, 16


def test_full_checkpoint_slot_counts_from_shapes_only():
    banks = []
    for name, n, k in ((ref.GATE, 2 * _BM, _BH), (ref.DOWN, _BH, _BM)):
        shape = (512, n, k)
        specs = bank_specs(name, Spec("BF16", shape))
        parts = tuple(
            View(0, Spec(s.dtype, s.shape[1:]))
            for field in ref.PARTS
            for s in (specs[f"{name}.{field}"],)
        )
        banks.append(bounded._Bank(name, shape, None, parts))
    for gib, expected in zip((16, 32, 48, 64, 80), (57, 114, 172, 229, 287)):
        slots, size = sc._plan(banks * 41, gib << 30)
        assert slots == expected
        assert size == 41 * _budget(slots, _BH, _BM) <= gib << 30
        assert 41 * _budget(slots + 1, _BH, _BM) > gib << 30


def _benchmark_checkpoint(root):
    cfg = dict(
        ref._config(),
        n_routed_experts=_BE,
        num_experts_per_tok=_BR,
        routed_expert_hidden_size=_BH,
        moe_intermediate_size=_BM,
    )
    params = dict(tree_flatten(ref._model(cfg).parameters()))  # scalar placeholders
    dtype = {v: k for k, v in bounded._MLX.items()}
    specs = {k: Spec(dtype[a.dtype], a.shape) for k, a in params.items()}
    rng = np.random.default_rng(843)
    with (root / ref.SHARD).open("w+b", buffering=0) as f:
        writer = Writer(f, specs)
        for n, s in specs.items():
            if s.dtype == "U32":  # independent packed codes, one row at a time
                for _ in range(_BE):
                    writer.append(
                        n, rng.integers(2**32, size=s.shape[1:], dtype=np.uint32)
                    )
            else:
                word = (
                    0x3A80
                    if n.endswith(".oqh_scales")
                    else 0xBE00
                    if n.endswith(".oqh_biases")
                    else 0x3F80
                    if n.endswith(".oqh_signs")
                    else 0
                )
                a = np.full(s.shape, word, dtype=bounded._NUMPY[s.dtype])
                if n.endswith(".oqh_signs"):
                    a.ravel()[::3] |= 0x8000
                writer.append(n, a)
        record = writer.finish(4 << 20)
    files = {
        "config.json": json.dumps(cfg),
        ref.INDEX: json.dumps({"weight_map": dict.fromkeys(params, ref.SHARD)}),
        "tokenizer.json": "{}",
        "chat_template.jinja": "unused",
    }
    for n, content in files.items():
        (root / n).write_text(content)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": ref.SCHEMA,
                "source": {"repo": ref.SOURCE, "revision": ref.REVISION},
                "text_only": True,
                "quantization": ref.ALGORITHM,
                "core_shards": {ref.SHARD: record},
                "assets": {n: ref._record(root / n) for n in files},
            }
        )
    )


def _measure(mode, root):
    # Fresh process per policy so ru_maxrss (bytes on Darwin) cannot miss a
    # temporary allocation between Python samples, or inherit the other peak.
    mx.set_cache_limit(0)
    art = ref.Artifact.open(root)
    model = ref._model(art.config)
    budget = _budget(_BR, _BH, _BM)
    rng = np.random.default_rng(81)
    ids = [
        mx.array(a[None, :], dtype=mx.int32)
        for a in np.array_split(rng.permutation(_BE), 2)
    ]
    x = ref._normal((1, _BH), 210).astype(mx.bfloat16)
    mix = mx.full((1, _BR), 1 / _BR, dtype=mx.float32)
    mx.eval(x, mix, *ids)
    mx.clear_cache()

    def rss():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    start_rss, start_mlx = rss(), mx.get_active_memory()
    mx.reset_peak_memory()
    install = (
        sc.install_slot_experts if mode == "slots" else bounded.install_bounded_experts
    )
    result = dict(mode=mode, budget=budget, start_rss=start_rss, start_mlx=start_mlx)
    with install(model, art.root, art.config, art.weight_map, budget) as ctx:
        result["install_peak_rss"] = rss()
        ref.load_weights(model, art, ctx)
        experts = ref._banks(model)

        def run(i):
            start = time.perf_counter_ns()
            y = experts(x, ids[i], mix)
            mx.eval(y)  # time completion, not merely dispatch
            ms = (time.perf_counter_ns() - start) / 1e6
            return ms, hashlib.sha256(memoryview(y).cast("B")).hexdigest()

        run(0)  # warm kernels and fill all slots of gate/up and down
        for phase, route in (("hot", [0] * 15), ("cold", [1, 0, 1])):
            before = ctx.stats
            times, hashes = zip(*(run(i) for i in route))
            after = ctx.stats
            result[phase] = dict(
                median_ms=statistics.median(times),
                hashes=sorted(set(hashes)),
                **{
                    k: after[k] - before[k]
                    for k in (
                        "hits",
                        "misses",
                        "evictions",
                        "expert_read_bytes",
                        "gathers",
                    )
                },
            )
        result.update(
            update_peak_rss=rss(),
            peak_mlx=mx.get_peak_memory(),
            cache_bytes=ctx.stats["cache_bytes"],
        )
    return result


def test_sub_gib_hot_cold_benchmark_and_peak_rss():
    reports = {}
    with tempfile.TemporaryDirectory(prefix="hy4-test-slots-") as directory:
        _benchmark_checkpoint(Path(directory))
        for mode in ("bounded", "slots"):
            run = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tests.test_hy4_flash_slot_cache",
                    mode,
                    directory,
                ],
                env=dict(os.environ, MLX_ENABLE_TF32="0"),
                text=True,
                check=True,
                capture_output=True,
                timeout=120,
            )
            r = reports[mode] = json.loads(run.stdout.strip().splitlines()[-1])
            print(json.dumps(r, sort_keys=True), flush=True)
            assert r["hot"]["expert_read_bytes"] == r["hot"]["misses"] == 0
            assert r["hot"]["hits"] == 2 * _BR * 15
            assert r["cold"]["misses"] == r["cold"]["evictions"] == 2 * _BR * 3
            assert r["cold"]["expert_read_bytes"] == 3 * _BR * (
                3 * _BH * _BM * 33 // 32
            )
            assert r["update_peak_rss"] < 1 << 30
            assert r["cache_bytes"] <= r["budget"]
    for phase in ("hot", "cold"):
        assert reports["bounded"][phase]["hashes"] == reports["slots"][phase]["hashes"]
    s = reports["slots"]
    # 111+ MiB of fixed RHS, >72 MiB gate packed buffer. Both installation
    # and repeated complete replacement must stay within ONE copy + small
    # native/driver/activation allowance. RSS and MLX high-water are separate.
    for key in ("install_peak_rss", "update_peak_rss"):
        assert s[key] <= s["start_rss"] + s["budget"] + (24 << 20)
    assert s["peak_mlx"] <= s["start_mlx"] + s["budget"] + (8 << 20)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in {"slots", "bounded"}:
        raise SystemExit("use the pytest benchmark (see module docstring)")
    print(json.dumps(_measure(sys.argv[1], Path(sys.argv[2]))))
