import mlx.core as mx

from omlx.oq import (
    _build_model_sanitizer,
    _build_quant_plan,
    _proxy_quant_bits,
    universal_quant_predicate,
)

CONFIG = {
    "model_type": "qwen4_exp",
    "architectures": ["Qwen4ExpForConditionalGeneration"],
    "vision_config": {"hidden_size": 32},
    "omlx_qwen4_ngram_groups": 2,
    "text_config": {
        "num_hidden_layers": 1,
        "num_experts": 512,
        "hidden_size": 32,
        "ple_layer_ids": [1],
        "split_ngram_parts": 4,
    },
}


def test_qwen4_sensitive_projection_floors():
    cases = {
        "language_model.model.layers.0.self_attn.indexer.index_qk_proj": 8,
        "language_model.model.layers.0.attn_hyper_connection.block_inject_weight": 8,
        "language_model.model.layers.0.attn_hyper_connection.input_mix_weight_down": 6,
        "language_model.model.layers.0.ple.key_proj": 8,
    }
    for path, expected_bits in cases.items():
        got = universal_quant_predicate(path, None, CONFIG, 4)
        assert got["bits"] == expected_bits, path
    assert (
        universal_quant_predicate(
            "language_model.model.layers.0.ple.conv1d", None, CONFIG, 4
        )
        is False
    )
    ngram = universal_quant_predicate(
        "language_model.model.layers.0.ple.ple_embedding." "ngram_embedding.groups.0",
        None,
        CONFIG,
        4,
    )
    assert ngram == {"bits": 4, "group_size": 32, "mode": "affine"}


def test_qwen4_proxy_uses_q3_but_final_plan_keeps_q4_ngram_g32():
    assert _proxy_quant_bits(CONFIG) == 3
    ngram_path = "model.layers.0.ple.ple_embedding.ngram_embedding.groups.0.weight"
    shapes = {
        ngram_path: (1_000, 160),
        "model.layers.0.mlp.switch_mlp.gate_proj.weight": (4, 32, 32),
        "model.layers.0.self_attn.indexer.index_qk_proj.weight": (40, 32),
    }
    plan = _build_quant_plan(
        shapes,
        {**CONFIG, "_oq_sensitivity_map": {"0": 1.0}},
        4,
        target_bpw=5.0,
        hard_cap_bpw=5.2,
    )

    assert plan.boost_map[ngram_path] == {
        "bits": 4,
        "group_size": 32,
        "mode": "affine",
    }


def test_qwen4_streaming_sanitizer_maps_vlm_and_experts():
    weights = {
        "model.language_model.layers.0.mlp.experts.gate_up_proj": mx.zeros((2, 8, 32)),
        "model.language_model.layers.0.mlp.experts.down_proj": mx.zeros((2, 32, 4)),
        "model.language_model.layers.0.attn_hyper_connection.hc_norm.weight": mx.zeros(
            (128,)
        ),
        "model.visual.patch_embed.proj.weight": mx.zeros((32, 32)),
        "lm_head.weight": mx.zeros((64, 32)),
    }
    sanitizer = _build_model_sanitizer(dict(CONFIG))
    got = sanitizer(weights)

    prefix = "language_model.model.layers.0.mlp.switch_mlp"
    assert got[f"{prefix}.gate_proj.weight"].shape == (2, 4, 32)
    assert got[f"{prefix}.up_proj.weight"].shape == (2, 4, 32)
    assert got[f"{prefix}.down_proj.weight"].shape == (2, 32, 4)
    assert "vision_tower.patch_embed.proj.weight" in got
    assert "language_model.lm_head.weight" in got
    norm = got["language_model.model.layers.0.attn_hyper_connection.hc_norm.weight"]
    assert mx.all(norm == 1.0).item()


def test_qwen4_gated_delta_norm_is_not_zero_centered():
    sanitizer = _build_model_sanitizer(dict(CONFIG), text_only=True)
    key = "model.language_model.layers.0.linear_attn.norm.weight"
    got = sanitizer({key: mx.ones((8,))})

    assert mx.all(got["model.layers.0.linear_attn.norm.weight"] == 1.0).item()


def test_qwen4_already_sanitized_norm_is_not_shifted_again():
    from omlx.patches.qwen4_exp.sanitize import sanitize_weights

    key = "model.layers.0.attn_hyper_connection.hc_norm.weight"
    got = sanitize_weights(
        {key: mx.ones((128,))},
        dict(CONFIG),
        text_only=True,
        already_sanitized=True,
    )

    assert mx.all(got[key] == 1.0).item()


def test_qwen4_raw_ngram_shards_group_during_sanitize():
    weights = {}
    for shard in range(4):
        key = (
            "model.language_model.layers.0.ple.ple_embedding.ngram_embedding."
            f"shard_{shard}.weight"
        )
        weights[key] = mx.full((2, 32), shard, dtype=mx.bfloat16)

    sanitizer = _build_model_sanitizer(dict(CONFIG))
    got = sanitizer(weights)
    groups = sorted(key for key in got if ".ngram_embedding.groups." in key)
    assert len(groups) == 2
    assert got[groups[0]].shape == (4, 32)
    assert not any("shard_" in key for key in got)
