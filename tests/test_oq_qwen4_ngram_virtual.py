import json

import mlx.core as mx
import numpy as np

from omlx.oq import (
    _build_streaming_proxy_for_sensitivity,
    _estimate_streaming_proxy_bytes,
    _LazyTensorIndex,
    _quantize_chunked,
)
from omlx.patches.qwen4_exp.virtual_ngram import _GroupedRows


def _config():
    return {
        "model_type": "qwen4_exp",
        "text_config": {
            "ple_layer_ids": [2],
            "split_ngram_parts": 4,
        },
    }


def _checkpoint(tmp_path):
    tensors = {}
    expected = []
    for shard in range(4):
        value = np.arange(96, dtype=np.float32).reshape(3, 32) + shard * 100
        expected.append(value)
        tensors[
            "model.language_model.layers.1.ple.ple_embedding."
            f"ngram_embedding.shard_{shard}.weight"
        ] = mx.array(value, dtype=mx.bfloat16)
    path = tmp_path / "model.safetensors"
    mx.save_safetensors(str(path), tensors)
    return path, expected


def test_qwen4_ngram_shards_become_bounded_runtime_groups(tmp_path, monkeypatch):
    from omlx.patches.qwen4_exp import virtual_ngram

    monkeypatch.setattr(virtual_ngram, "DEFAULT_RUNTIME_GROUPS", 2)
    path, expected = _checkpoint(tmp_path)
    # register_virtual_tensors uses its default argument captured at definition,
    # so call the model-specific registrar explicitly for this miniature layout.
    idx = _LazyTensorIndex([path], config={})
    assert virtual_ngram.register(idx, _config(), runtime_groups=2) == 2

    keys = set(idx.logical_metadata())
    assert not any("shard_" in key for key in keys)
    group_keys = sorted(key for key in keys if ".groups." in key)
    assert len(group_keys) == 2
    assert idx.logical_metadata()[group_keys[0]][0] == (6, 32)

    group = idx[group_keys[0]]
    assert isinstance(group, _GroupedRows)
    assert np.array_equal(
        np.array(group._load_rows(2, 5).astype(mx.float32)),
        np.concatenate(expected[:2], axis=0)[2:5],
    )


def test_qwen4_grouped_rows_quantize_without_full_materialization(tmp_path):
    from omlx.patches.qwen4_exp import virtual_ngram

    path, _ = _checkpoint(tmp_path)
    idx = _LazyTensorIndex([path], config={})
    virtual_ngram.register(idx, _config(), runtime_groups=2)
    key = next(key for key in idx.logical_metadata() if ".groups.0.weight" in key)
    grouped = idx[key]

    qw, scales, biases = _quantize_chunked(
        grouped, group_size=32, bits=4, mode="affine"
    )
    mx.eval(qw, scales, biases)
    assert qw.shape[0] == grouped.shape[0]
    assert scales.shape == biases.shape


def test_qwen4_ngram_incomplete_shard_set_fails_closed(tmp_path):
    path, _ = _checkpoint(tmp_path)
    arrays = mx.load(str(path))
    arrays.pop(
        "model.language_model.layers.1.ple.ple_embedding."
        "ngram_embedding.shard_3.weight"
    )
    broken = tmp_path / "broken.safetensors"
    mx.save_safetensors(str(broken), arrays)

    idx = _LazyTensorIndex([broken], config={})
    from omlx.patches.qwen4_exp import virtual_ngram

    try:
        virtual_ngram.register(idx, _config(), runtime_groups=2)
    except ValueError as exc:
        assert "incomplete Qwen4 PLE shard set" in str(exc)
    else:
        raise AssertionError("incomplete checkpoint was accepted")


def test_qwen4_proxy_is_text_only_q3_with_g32_ngram_groups(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = {
        "model_type": "qwen4_exp",
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "vision_config": {"hidden_size": 32},
        "text_config": {
            "num_hidden_layers": 1,
            "num_experts": 512,
            "hidden_size": 32,
            "ple_layer_ids": [1],
            "split_ngram_parts": 16,
        },
    }
    (source / "config.json").write_text(json.dumps(config))
    tensors = {
        "model.language_model.layers.0.self_attn.q_proj.weight": mx.ones(
            (32, 32), dtype=mx.bfloat16
        ),
        "model.visual.patch_embed.proj.weight": mx.ones((32, 32), dtype=mx.bfloat16),
    }
    for shard in range(16):
        tensors[
            "model.language_model.layers.0.ple.ple_embedding."
            f"ngram_embedding.shard_{shard}.weight"
        ] = mx.full((2, 160), shard, dtype=mx.bfloat16)
    mx.save_safetensors(str(source / "model.safetensors"), tensors)
    # External macOS volumes commonly create these resource-fork sidecars.
    # They match ``*.safetensors`` but are not safetensors files.
    (source / "._model.safetensors").write_bytes(b"not a tensor header")

    index = _LazyTensorIndex([source / "model.safetensors"], config=config)
    q3_bytes = _estimate_streaming_proxy_bytes(
        index,
        config,
        base_bits=3,
        text_only=True,
        preserve_mtp=False,
    )
    q2_bytes = _estimate_streaming_proxy_bytes(
        index,
        config,
        base_bits=2,
        text_only=True,
        preserve_mtp=False,
    )
    assert q2_bytes < q3_bytes

    output = tmp_path / "proxy"
    _build_streaming_proxy_for_sensitivity(str(source), output, dtype="bfloat16")

    output_config = json.loads((output / "config.json").read_text())
    quant = output_config["quantization"]
    assert quant["bits"] == 3
    assert "vision_config" not in output_config
    group_specs = {
        key: value for key, value in quant.items() if ".ngram_embedding.groups." in key
    }
    assert len(group_specs) == 16
    assert all(
        spec == {"bits": 3, "group_size": 32, "mode": "affine"}
        for spec in group_specs.values()
    )

    output_weights = mx.load(str(output / "model.safetensors"))
    assert not any("vision" in key for key in output_weights)
    assert (
        len(
            [
                key
                for key in output_weights
                if ".groups." in key and key.endswith(".weight")
            ]
        )
        == 16
    )
