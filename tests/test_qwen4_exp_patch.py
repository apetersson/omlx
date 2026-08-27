import importlib
import json
import sys
from unittest.mock import MagicMock

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten


def _tiny_config():
    return {
        "model_type": "qwen4_exp",
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "omlx_qwen4_ngram_groups": 2,
        "text_config": {
            "model_type": "qwen4_exp_text",
            "vocab_size": 128,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "rms_norm_eps": 1e-6,
            "linear_num_value_heads": 2,
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 32,
            "linear_conv_kernel_dim": 4,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 16,
            "hc_count": 2,
            "hc_lowrank": 8,
            "ple_layer_ids": [1],
            "ple_embed_dim": 32,
            "ple_conv_kernel_size": 3,
            "ngram_size": 3,
            "heads_per_ngram": 2,
            "ngram_vocab_size_base": 11,
            "make_ngram_vocab_size_divisible_by": 2,
            "split_ngram_parts": 2,
            "indexer_n_heads": 2,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 8,
            "indexer_budget": 4,
            "indexer_compress_ratio": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "eos_token_id": 3,
            "partial_rotary_factor": 0.5,
            "rope_theta": 10_000.0,
        },
    }


def _load_module():
    from omlx.patches.qwen4_exp import apply_qwen4_exp_patch

    apply_qwen4_exp_patch()
    return importlib.import_module("mlx_lm.models.qwen4_exp")


def test_patch_registers_and_resolves_qwen4_exp():
    module = _load_module()
    from mlx_lm.utils import _get_classes

    model_cls, args_cls = _get_classes(_tiny_config())

    assert sys.modules["mlx_lm.models.qwen4_exp"] is module
    assert model_cls is module.Model
    assert args_cls is module.ModelArgs


def test_patch_is_idempotent():
    from omlx.patches.qwen4_exp import apply_qwen4_exp_patch, is_applied

    first = apply_qwen4_exp_patch()
    second = apply_qwen4_exp_patch()

    assert is_applied() is True
    assert first in (True, False)
    assert second is False


def test_tokenizer_patch_avoids_unknown_model_config_warning(tmp_path, monkeypatch):
    import mlx_lm.tokenizer_utils as tokenizer_utils
    from transformers import PreTrainedConfig

    import omlx.patches.qwen4_exp as qwen4_patch

    (tmp_path / "config.json").write_text(json.dumps(_tiny_config()))
    original = MagicMock()
    original.from_pretrained.return_value = object()
    monkeypatch.setattr(tokenizer_utils, "AutoTokenizer", original)
    monkeypatch.setattr(qwen4_patch, "_TOKENIZER_PATCHED", False)

    qwen4_patch._patch_tokenizer()
    tokenizer_utils.AutoTokenizer.from_pretrained(tmp_path)

    call = original.from_pretrained.call_args
    assert isinstance(call.kwargs["config"], PreTrainedConfig)
    assert call.kwargs["config"].model_type == ""


def test_runtime_parameter_layout_matches_sanitized_checkpoint():
    module = _load_module()
    model = module.Model(module.ModelArgs.from_dict(_tiny_config()))
    keys = {key for key, _ in tree_flatten(model.parameters())}

    expected = {
        "model.layers.0.ple.ple_embedding.ngram_heads_vocab_sizes",
        "model.layers.0.ple.ple_embedding.ngram_heads_offsets",
        "model.layers.0.ple.ple_embedding.layer_multipliers",
        "model.layers.0.ple.ple_embedding.ngram_embedding.groups.0.weight",
        "model.layers.0.ple.key_proj.weight",
        "model.layers.1.self_attn.indexer.index_qk_proj.weight",
        "model.layers.1.mlp.switch_mlp.gate_proj.weight",
        "model.hyper_connection_mixer.hc_norm.weight",
        "lm_head.weight",
    }
    assert expected <= keys


def test_runtime_does_not_resanitize_converted_checkpoint_norms():
    module = _load_module()
    config = _tiny_config()
    config["omlx_qwen4_weights_sanitized"] = True
    model = module.Model(module.ModelArgs.from_dict(config))
    key = "model.layers.0.attn_hyper_connection.hc_norm.weight"

    sanitized = model.sanitize({key: mx.ones((64,))})

    assert mx.all(sanitized[key] == 1.0).item()


def test_cached_decode_matches_full_forward():
    module = _load_module()
    mx.random.seed(7)
    model = module.Model(module.ModelArgs.from_dict(_tiny_config()))
    model.eval()
    tokens = mx.array([[5, 9, 3, 11, 7, 4]], dtype=mx.int32)

    full = model(tokens)
    cache = model.make_cache()
    pieces = [
        model(tokens[:, index : index + 1], cache=cache)
        for index in range(tokens.shape[1])
    ]
    cached = mx.concatenate(pieces, axis=1)
    mx.eval(full, cached)

    np.testing.assert_allclose(
        np.asarray(cached), np.asarray(full), rtol=2e-4, atol=2e-4
    )


def test_calibration_layer_walk_matches_native_text_model():
    from omlx.oq import _forward_layer_result, _prepare_layer_inputs

    module = _load_module()
    mx.random.seed(17)
    model = module.Model(module.ModelArgs.from_dict(_tiny_config()))
    model.eval()
    tokens = mx.array([[5, 9, 3, 11]], dtype=mx.int32)

    expected = model.model(tokens)
    embedded = model.model.embed_tokens(tokens)
    hidden, masks, state = _prepare_layer_inputs(
        model, model.model.layers, tokens, embedded
    )

    assert state["kind"] == "qwen4_exp"
    assert hidden.shape[-1] == embedded.shape[-1] * model.text_args.hc_count
    for layer_idx, layer in enumerate(model.model.layers):
        hidden, _ = _forward_layer_result(
            layer,
            hidden,
            masks[layer_idx],
            state,
            layer_idx=layer_idx,
        )
        assert hidden is not None
    actual = model.model.hyper_connection_mixer(hidden)
    mx.eval(expected, actual)

    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=2e-4, atol=2e-4
    )


def test_runtime_supports_continuous_batch_generation():
    module = _load_module()
    from mlx_lm.generate import BatchGenerator

    mx.random.seed(11)
    model = module.Model(module.ModelArgs.from_dict(_tiny_config()))
    model.eval()
    generator = BatchGenerator(
        model,
        max_tokens=2,
        prefill_batch_size=2,
        completion_batch_size=2,
        sampler=lambda logits: mx.argmax(logits, axis=-1),
    )
    uids = generator.insert([[5, 9, 3], [7, 4]], max_tokens=[2, 2])
    finished = []
    for _ in range(8):
        _, responses = generator.next()
        finished.extend(
            response for response in responses if response.finish_reason is not None
        )
        if len(finished) == 2:
            break

    assert uids == [0, 1]
    assert {response.uid for response in finished} == {0, 1}
    assert all(response.finish_reason == "length" for response in finished)


def test_grouped_rms_norm_normalizes_each_hyper_connection_stream():
    module = _load_module()
    norm = module.Qwen4RMSNorm(4, 1e-6, group_size=2)
    norm.weight = mx.array([1.0, 2.0, 3.0, 4.0])
    values = mx.array([[[3.0, 4.0, 5.0, 12.0]]])

    got = norm(values)
    rms = np.array([np.sqrt(12.5 + 1e-6), np.sqrt(84.5 + 1e-6)])
    expected = np.array([[[3 / rms[0], 8 / rms[0], 15 / rms[1], 48 / rms[1]]]])
    np.testing.assert_allclose(np.asarray(got), expected, rtol=2e-6, atol=2e-6)


def test_model_discovery_routes_qwen4_checkpoint_to_text_engine(tmp_path):
    from omlx.model_discovery import detect_model_type

    (tmp_path / "config.json").write_text(json.dumps(_tiny_config()))
    assert detect_model_type(tmp_path) == "llm"
