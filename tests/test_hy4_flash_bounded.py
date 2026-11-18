# SPDX-License-Identifier: Apache-2.0
"""Synthetic physical shards + the actual resident/streamed MLX oQh operators."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from omlx.patches.hy_v4_flash import bounded
from omlx.patches.hy_v4_flash.artifact import INDEX, REVISION, SCHEMA, SOURCE, Artifact
from omlx.patches.hy_v4_flash.convert import ALGORITHM
from omlx.patches.hy_v4_flash.experts import build_mlp
from omlx.patches.hy_v4_flash.quant_io import read_header
from omlx.patches.hy_v4_flash.runner import load_weights

PREFIX = "model.layers.0.mlp.experts."
GATE, DOWN = PREFIX + "gate_up_proj", PREFIX + "down_proj"
SHARD = "model-00001-of-00001.safetensors"
PARTS = ("weight", "oqh_scales", "oqh_biases")
E, H, M, R = 5, 512, 256, 3
GATE_ROW = 2 * M * H + 4 * (2 * M) * (H // 128)


def _config():
    return dict(
        model_type="hy_v4_flash",
        hidden_act="silu",
        hidden_size=16,
        rms_norm_eps=1e-5,
        num_hidden_layers=1,
        mlp_layer_types=["sparse"],
        routed_expert_hidden_size=H,
        moe_intermediate_size=M,
        n_routed_experts=E,
        num_experts_per_tok=R,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        swiglu_limit=10.0,
        omlx_oqh={
            "version": 1,
            "tensors": {
                n: {"bits": 8, "block": 256, "group_size": 128} for n in (GATE, DOWN)
            },
        },
    )


def _model(config):
    root = nn.Module()
    root.model = nn.Module()
    block = nn.Module()
    block.mlp = build_mlp(SimpleNamespace(**config), 0)
    root.model.layers = [block]
    return root


def _banks(model):
    return model.model.layers[0].mlp.experts


def _normal(shape, seed):
    return mx.random.normal(shape, key=mx.random.key(seed))


def _record(path):
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size": path.stat().st_size,
    }


@pytest.fixture
def bundle(tmp_path):
    cfg = _config()
    weights = dict(tree_flatten(_model(cfg).parameters()))
    # Produce a tiny REAL oQh bank independent of the runtime and converter:
    # two signs for two projections, signed normalized WHT -> group-affine RTN.
    for s, (name, o, k) in enumerate(((GATE, 2 * M, H), (DOWN, H, M))):
        raw = _normal((E, o, k), 21 + s).astype(mx.bfloat16).astype(mx.float32) * 0.125
        signs = mx.where(_normal((k // 256, 256), 42 + s) >= 0, 1.0, -1.0)
        rotated = mx.hadamard_transform(raw.reshape(-1, k // 256, 256) * signs).reshape(
            raw.shape
        )
        p, a, b = mx.quantize(rotated, bits=8, group_size=128, mode="affine")
        for suffix, tensor in zip((*PARTS, "oqh_signs"), (p, a, b, signs)):
            weights[name + "." + suffix] = (
                tensor if suffix == "weight" else tensor.astype(mx.bfloat16)
            )
    mx.save_safetensors(str(tmp_path / SHARD), weights, metadata={"format": "mlx"})
    mapping = dict.fromkeys(weights, SHARD)
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / INDEX).write_text(json.dumps({"weight_map": mapping}))
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "chat_template.jinja").write_text("synthetic, not used")
    manifest = {
        "schema": SCHEMA,
        "source": {"repo": SOURCE, "revision": REVISION},
        "text_only": True,
        "quantization": ALGORITHM,
        "core_shards": {SHARD: _record(tmp_path / SHARD)},
        "assets": {
            n: _record(tmp_path / n)
            for n in ("config.json", INDEX, "tokenizer.json", "chat_template.jinja")
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return Artifact.open(tmp_path, verify_core=True), weights


def _install(bundle, budget, *, load=True):
    art, _ = bundle
    model = _model(art.config)
    ctx = bounded.install_bounded_experts(
        model, art.root, art.config, art.weight_map, budget
    )
    try:
        if load:
            load_weights(model, art, ctx)
        return model, ctx
    except BaseException:
        ctx.close()
        raise


def _math_reference(x, ids, name, weights):
    # Explicit Sylvester matrix (no runtime rotate_activation or WHT primitive)
    # and full dequantization, *only* as an independent tiny numeric oracle.
    had = mx.array(
        [
            [(-1 if (i & j).bit_count() % 2 else 1) / 16 for i in range(256)]
            for j in range(256)
        ]
    )
    p, a, b = (weights[f"{name}.{suffix}"] for suffix in PARTS)
    dense = mx.dequantize(p, a, b, group_size=128, bits=8).astype(mx.float32)
    k = x.shape[-1]
    signed = (
        x.astype(mx.bfloat16).astype(mx.float32).reshape(-1, k // 256, 256)
        * weights[name + ".oqh_signs"]
    )
    rotated = (signed @ had).reshape(x.shape).astype(mx.bfloat16).astype(mx.float32)
    return mx.stack(
        [
            mx.stack(
                [
                    rotated[i, 0 if x.shape[1] == 1 else j] @ dense[e].T
                    for j, e in enumerate(route)
                ]
            )
            for i, route in enumerate(ids.tolist())
        ]
    )


@pytest.mark.parametrize(
    "bank,k,lanes,dtype", [(GATE, H, 1, mx.bfloat16), (DOWN, M, R, mx.float32)]
)
def test_chunked_gather_matches_resident_and_dense_oracle(
    bundle, bank, k, lanes, dtype
):
    art, weights = bundle
    resident = _model(art.config)
    resident.load_weights(list(weights.items()), strict=True)
    model, ctx = _install(bundle, GATE_ROW)  # one gate row or two down rows
    attr = bank.split(".")[-1]
    # More tokens than a compact gather, scrambled experts and varying repeats.
    ids = mx.array(([[3, 0, 3], [1, 4, 2], [0, 0, 4]] * 5)[:13], dtype=mx.uint32)
    x = _normal((13, lanes, 1, k), 82).astype(dtype)
    with ctx:
        y = getattr(_banks(model), attr)(x, ids)
        gold = getattr(_banks(resident), attr)(x, ids)
        assert y.shape == gold.shape and y.dtype == dtype
        assert mx.array_equal(y, gold).item()
        ref = _math_reference(x[:3], ids[:3], bank, weights)
        np.testing.assert_allclose(
            np.asarray(y[:3].astype(mx.float32)), np.asarray(ref), atol=0.09, rtol=0.015
        )
        assert 1 <= ctx.stats["peak_unique"] <= (1 if bank == GATE else 2)
        assert ctx.stats["peak_cache_bytes"] <= GATE_ROW
        assert ctx.stats["gathers"] > 1 and ctx.stats["evictions"] > 0


def test_strict_mixed_loader_warm_hits_exact_preads_and_eviction(bundle, monkeypatch):
    art, weights = bundle
    with (art.root / SHARD).open("rb", buffering=0) as f:
        header = read_header(f)
    calls = []
    pread = os.pread

    def spy(fd, n, pos):
        calls.append((n, pos))
        return pread(fd, n, pos)

    def no_shard_load(*args, **kwargs):
        pytest.fail("mx.load would load the mixed expert shard")

    monkeypatch.setattr(os, "pread", spy)
    monkeypatch.setattr(mx, "load", no_shard_load)
    model, ctx = _install(bundle, 2 * GATE_ROW, load=False)
    assert not calls  # validated only packed headers
    with ctx:
        assert ctx.excluded_names == {f"{n}.{s}" for n in (GATE, DOWN) for s in PARTS}
        assert ctx.sign_names == {n + ".oqh_signs" for n in (GATE, DOWN)}
        assert ctx.mixed_shards == {SHARD}
        load_weights(model, art, ctx)
        assert set(calls) == {
            (header[n].spec.nbytes, header[n].start)
            for n in weights.keys() - ctx.excluded_names
        }
        params = dict(tree_flatten(model.parameters()))
        assert params.keys() == weights.keys() - ctx.excluded_names
        assert all(mx.array_equal(params[n], weights[n]).item() for n in params)
        with pytest.raises(ValueError, match="complete expert bank"):
            ctx.read_resident(SHARD, GATE + ".weight")
        gate = _banks(model).gate_up_proj
        x, ids = _normal((1, 1, 1, H), 82), mx.array([[3, 1, 3]], dtype=mx.uint32)
        calls.clear()
        first = gate(x, ids)
        expected = []
        for i in (1, 3):
            for suffix in PARTS:
                v = header[f"{GATE}.{suffix}"]
                size = v.spec.nbytes // E
                expected.append((size, v.start + i * size))
        assert calls == expected
        assert (
            ctx.stats["expert_read_bytes"] == 2 * GATE_ROW and ctx.stats["misses"] == 2
        )
        assert mx.array_equal(first, gate(x, ids)).item()
        assert calls == expected and ctx.stats["hits"] == 2
        gate(x, mx.array([[4, 2, 4]], dtype=mx.uint32))
        assert ctx.stats["evictions"] == 2 and ctx.stats["cache_bytes"] == 2 * GATE_ROW
    assert ctx.stats["cache_bytes"] == 0
    with pytest.raises(ValueError, match="closed"):
        gate(x, ids)
    assert mx.all(mx.isfinite(first)).item()  # output survives eviction + close
    ctx.close()  # idempotent


def test_minimum_budget_deduplicates_before_splitting_routes(bundle):
    model, ctx = _install(bundle, GATE_ROW)
    with ctx:
        ids = mx.array([[3, 0, 3], [0, 3, 0], [3, 3, 0]])
        _banks(model).gate_up_proj(mx.ones((3, 1, 1, H)), ids)
        assert (
            ctx.stats["misses"] == 2 and ctx.stats["expert_read_bytes"] == 2 * GATE_ROW
        )
        assert ctx.stats["gathers"] == 2 and ctx.stats["peak_unique"] == 1


def test_single_lru_shared_by_different_size_projections(bundle):
    model, ctx = _install(bundle, 2 * GATE_ROW)
    with ctx:
        gate, down = _banks(model).gate_up_proj, _banks(model).down_proj
        x = mx.ones((1, 1, 1, H))
        gate(x, mx.array([[0, 1, 0]]))
        gate(x, mx.array([[0, 0, 0]]))  # most recently used
        down(mx.ones((1, R, 1, M)), mx.array([[1, 2, 1]]))
        stats = ctx.stats
        assert stats["evictions"] == 1 and stats["cache_entries"] == 3
        assert stats["cache_bytes"] == stats["peak_cache_bytes"] == 2 * GATE_ROW
        gate(x, mx.array([[0, 0, 0]]))
        assert (
            ctx.stats["expert_read_bytes"] == stats["expert_read_bytes"]
        )  # gate 0 survived
        gate(x, mx.array([[1, 1, 1]]))
        assert ctx.stats["evictions"] == 3  # two half-size rows evicted for a gate row


def test_retained_outputs_do_not_pin_old_expert_payloads(bundle):
    model, ctx = _install(bundle, GATE_ROW)
    with ctx:
        gate = _banks(model).gate_up_proj
        x = mx.ones((1, 1, 1, H))
        outputs = [gate(x, mx.zeros((1, R), dtype=mx.uint32))]
        start = mx.get_active_memory()
        for i in range(1, 21):
            outputs.append(gate(x, mx.full((1, R), i % E, dtype=mx.uint32)))
        # Keep *all* results alive, never mx.eval them from this test. Even
        # with generous allocator alignment slack, old row payloads (~5MiB)
        # must not survive via those outputs' computation graphs.
        assert mx.get_active_memory() - start < 1 << 20
        assert ctx.stats["evictions"] == 20 and len(outputs) == 21


def test_short_pread_does_not_cache_partial_row(bundle, monkeypatch):
    model, ctx = _install(bundle, GATE_ROW)
    with ctx:
        original = os.pread
        monkeypatch.setattr(os, "pread", lambda fd, n, pos: original(fd, n - 1, pos))
        with pytest.raises(ValueError, match="short positional core read"):
            _banks(model).gate_up_proj(mx.ones((1, 1, 1, H)), mx.array([[4, 0, 4]]))
        assert ctx.stats["cache_bytes"] == ctx.stats["cache_entries"] == 0


def test_never_assemble_even_a_tiny_complete_bank(bundle, monkeypatch):
    orig = mx.gather_qmm
    seen = []

    def gather(x, w, *args, **kwargs):
        seen.append((x.shape[0], w.shape[0]))
        return orig(x, w, *args, **kwargs)

    monkeypatch.setattr(mx, "gather_qmm", gather)
    model, ctx = _install(bundle, 32 * GATE_ROW)
    ids = mx.array([[0, 1, 2], [2, 3, 4]] * 6, dtype=mx.int32)
    with ctx:
        _banks(model).gate_up_proj(mx.ones((12, 1, 1, H)), ids)
    assert len(seen) > 1 and all(t <= 8 and 0 < u < E for t, u in seen)


def test_nonfinite_payload_only_checked_on_selected_row(bundle):
    art, _ = bundle
    with (art.root / SHARD).open("r+b", buffering=0) as f:
        v = read_header(f)[GATE + ".oqh_scales"]
        f.seek(v.start + 4 * v.spec.nbytes // E)
        f.write(struct.pack("<H", 0x7FC0))
    model, ctx = _install(bundle, 2 * GATE_ROW)
    with ctx:
        gate = _banks(model).gate_up_proj
        gate(mx.ones((1, 1, 1, H)), mx.array([[2, 0, 2]]))
        with pytest.raises(ValueError, match="non-finite BF16"):
            gate(mx.ones((1, 1, 1, H)), mx.array([[4, 4, 4]]))
        with pytest.raises(ValueError, match="non-finite bounded"):
            gate(mx.full((1, 1, 1, H), mx.inf), mx.array([[0, 0, 0]]))


def test_invalid_signs_and_mutation_rejected(bundle):
    art, _ = bundle
    with (art.root / SHARD).open("r+b", buffering=0) as f:
        v = read_header(f)[GATE + ".oqh_signs"]
        f.seek(v.start)
        f.write(b"\0\0")
    model, ctx = _install(bundle, 2 * GATE_ROW, load=False)
    with ctx:
        with pytest.raises(ValueError, match="invalid BF16 rotation signs"):
            ctx.read_resident(SHARD, GATE + ".oqh_signs")
        with (art.root / SHARD).open("ab") as f:
            f.write(b"x")
        with pytest.raises(ValueError, match="changed since header audit"):
            _banks(model).gate_up_proj(mx.ones((1, 1, 1, H)), mx.array([[0, 1, 0]]))


def test_cap_invalid_budget_routes_tokens_and_workspace_before_io(bundle, monkeypatch):
    art, _ = bundle
    model = _model(art.config)
    for bad in (False, -1, 0, 2.5, float("inf"), GATE_ROW - 1):
        with pytest.raises(ValueError, match="budget"):
            bounded.install_bounded_experts(
                model, art.root, art.config, art.weight_map, bad
            )
    assert GATE + ".weight" in dict(tree_flatten(model.parameters()))
    model, ctx = _install(bundle, 2 * GATE_ROW)
    with ctx:
        gate = _banks(model).gate_up_proj
        x = mx.ones((1, 1, 1, H))
        for ids in (
            mx.array([0, 1, 2]),
            mx.array([[0, 0]]),
            mx.array([[-1, 1, 2]]),
            mx.array([[E, 1, 2]]),
            mx.array([[0.3, 1.0, 2.0]]),
        ):
            with pytest.raises(ValueError):
                gate(x, ids)
        t = bounded.MAX_PREFILL_TOKENS + 1
        with pytest.raises(ValueError, match="prefill too long"):
            gate(mx.broadcast_to(x, (t, 1, 1, H)), mx.zeros((t, R), dtype=mx.uint32))
        monkeypatch.setattr(bounded, "_WORKSPACE_BYTES", 128)
        with pytest.raises(ValueError, match="workspace"):
            gate(x, mx.array([[0, 1, 2]]))
        assert ctx.stats["expert_read_bytes"] == 0


@pytest.mark.parametrize("field", ["dtype", "shape", "data_offsets"])
def test_bad_physical_bank_header_fails_before_model_mutation(bundle, field):
    art, _ = bundle
    with (art.root / SHARD).open("r+b", buffering=0) as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        entry = header[GATE + ".weight"]
        if field == "dtype":
            entry[field] = "I32"  # same element size and extent but wrong packed format
        elif field == "shape":
            entry[field] = list(reversed(entry[field]))  # same number of bytes
        else:
            entry[field] = [entry[field][0], entry[field][1] - 1]
        raw = json.dumps(header, separators=(",", ":")).encode()
        assert len(raw) <= n
        f.seek(8)
        f.write(raw.ljust(n, b" "))
    model = _model(art.config)
    with pytest.raises(ValueError, match="dtype/shape|shape mismatch|extent"):
        bounded.install_bounded_experts(
            model, art.root, art.config, art.weight_map, GATE_ROW
        )
    assert set(dict(tree_flatten(model.parameters()))) == set(art.weight_map)
