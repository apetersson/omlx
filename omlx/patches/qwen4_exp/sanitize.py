# SPDX-License-Identifier: Apache-2.0
"""Weight-name conversion for the upstream Qwen4-Exp checkpoint."""

from __future__ import annotations

import mlx.core as mx

from .virtual_ngram import DEFAULT_RUNTIME_GROUPS


def _is_text_norm(key: str) -> bool:
    if "language_model" not in key and not key.startswith(
        ("model.layers.", "model.hyper_connection_mixer")
    ):
        return False
    return key.endswith(".weight") and any(
        token in key
        for token in (
            ".hc_norm.",
            ".norm_key.",
            ".norm_query.",
            ".norm_conv.",
            ".q_norm.",
            ".k_norm.",
            ".q_layernorm.",
            ".k_layernorm.",
        )
    )


def _group_raw_ngram_shards(weights: dict, config: dict) -> dict:
    """Fallback grouping when virtual registration was unavailable.

    The normal streaming path has already replaced raw shards with virtual
    group keys, so this branch is primarily useful for small unit fixtures.
    It deliberately groups only one runtime tensor at a time.
    """
    text = config.get("text_config") or {}
    split_parts = int(text.get("split_ngram_parts") or 0)
    runtime_groups = int(
        config.get("omlx_qwen4_ngram_groups") or DEFAULT_RUNTIME_GROUPS
    )
    if split_parts <= 0 or split_parts % runtime_groups:
        return weights
    per_group = split_parts // runtime_groups

    for one_indexed_layer in text.get("ple_layer_ids") or []:
        layer = int(one_indexed_layer) - 1
        for root in ("model.language_model", "model"):
            prefix = f"{root}.layers.{layer}.ple.ple_embedding.ngram_embedding"
            sources = [f"{prefix}.shard_{idx}.weight" for idx in range(split_parts)]
            present = [key in weights for key in sources]
            if not any(present):
                continue
            if not all(present):
                raise ValueError(
                    "incomplete Qwen4 PLE shard set during sanitize: "
                    f"{sum(present)}/{split_parts}"
                )
            for group in range(runtime_groups):
                start = group * per_group
                tensors = [
                    weights.pop(key) for key in sources[start : start + per_group]
                ]
                weights[f"{prefix}.groups.{group}.weight"] = mx.concatenate(
                    tensors, axis=0
                )
    return weights


def sanitize_weights(weights: dict, config: dict, *, text_only: bool = False) -> dict:
    """Map Hugging Face Qwen4-Exp weights onto oMLX module names."""
    weights = _group_raw_ngram_shards(dict(weights), config)
    text = config.get("text_config") or {}
    num_layers = int(text.get("num_hidden_layers") or 0)

    # HF packs every expert's gate and up projections together. MLX's
    # SwitchGLU keeps them as two packed expert matrices.
    for layer in range(num_layers):
        for root in ("model.language_model", "model"):
            prefix = f"{root}.layers.{layer}.mlp"
            gate_up_key = f"{prefix}.experts.gate_up_proj"
            down_key = f"{prefix}.experts.down_proj"
            if gate_up_key in weights:
                gate_up = weights.pop(gate_up_key)
                mid = gate_up.shape[-2] // 2
                weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_up[..., :mid, :]
                weights[f"{prefix}.switch_mlp.up_proj.weight"] = gate_up[..., mid:, :]
            if down_key in weights:
                weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(down_key)

    sanitized = {}
    for source_key, value in weights.items():
        key = source_key
        if text_only:
            if key.startswith("model.visual"):
                continue
            if key.startswith("model.language_model"):
                key = key.replace("model.language_model", "model", 1)
        else:
            if key.startswith("model.visual"):
                key = key.replace("model.visual", "vision_tower", 1)
            elif key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model", 1)
            elif key.startswith("lm_head"):
                key = key.replace("lm_head", "language_model.lm_head", 1)

        if "conv1d.weight" in key and value.shape[-1] != 1:
            value = value.moveaxis(2, 1)
        if _is_text_norm(key) and getattr(value, "ndim", 0) == 1:
            value = value + 1.0
        sanitized[key] = value

    return sanitized


def cast_predicate(key: str) -> bool:
    """Keep recurrent decay parameters in their checkpoint precision."""
    return not key.endswith(("A_log", "dt_bias"))
