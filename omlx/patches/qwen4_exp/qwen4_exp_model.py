# SPDX-License-Identifier: Apache-2.0
"""Text runtime for Qwen4-Exp, adapted from the upstream Transformers model.

This module intentionally lives behind an oMLX registration patch until
mlx-lm grows native ``qwen4_exp`` support.  It implements the released
checkpoint's Gated DeltaNet / QSA / hyper-connection / PLE stack and keeps
the enormous PLE embedding in bounded runtime groups.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import sum_gradients
from mlx_lm.models.base import (
    BaseModelArgs,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import ArraysCache, CacheList, KVCache
from mlx_lm.models.gated_delta import gated_delta_update
from mlx_lm.models.qwen3_5 import GatedDeltaNet
from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

from omlx.patches.qwen4_exp.sanitize import cast_predicate, sanitize_weights
from omlx.patches.qwen4_exp.virtual_ngram import DEFAULT_RUNTIME_GROUPS


@dataclass
class TextModelArgs(BaseModelArgs):
    model_type: str = "qwen4_exp_text"
    hidden_size: int = 2560
    num_hidden_layers: int = 48
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    vocab_size: int = 248320
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"
    output_gate_type: str = "sigmoid"
    linear_num_value_heads: int = 48
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    norm_topk_prob: bool = True
    hc_count: int = 4
    hc_lowrank: int = 320
    ple_layer_ids: list[int] = field(default_factory=list)
    ple_embed_dim: int = 2560
    ple_conv_kernel_size: int = 4
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    seed: int = 1234
    split_ngram_parts: int = 128
    omlx_qwen4_ngram_groups: int = DEFAULT_RUNTIME_GROUPS
    indexer_n_heads: int = 4
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 128
    indexer_budget: int = 2048
    indexer_compress_ratio: int = 4
    full_attention_interval: int = 4
    layer_types: list[str] | None = None
    eos_token_id: int | list[int] | None = None
    partial_rotary_factor: float = 0.25
    rope_theta: float = 10_000_000.0
    rope_parameters: dict[str, Any] | None = None

    def __post_init__(self):
        if self.rope_parameters:
            self.partial_rotary_factor = float(
                self.rope_parameters.get(
                    "partial_rotary_factor", self.partial_rotary_factor
                )
            )
            self.rope_theta = float(
                self.rope_parameters.get("rope_theta", self.rope_theta)
            )
        if self.layer_types is None:
            self.layer_types = [
                (
                    "linear_attention"
                    if (idx + 1) % self.full_attention_interval
                    else "qwen_sparse_attention"
                )
                for idx in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types = [
                "qwen_sparse_attention" if value == "full_attention" else value
                for value in self.layer_types
            ]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict
    omlx_qwen4_ngram_groups: int = DEFAULT_RUNTIME_GROUPS

    @classmethod
    def from_dict(cls, params):
        if "text_config" not in params:
            return cls(
                model_type=params.get("model_type", "qwen4_exp"),
                text_config=params,
                omlx_qwen4_ngram_groups=int(
                    params.get("omlx_qwen4_ngram_groups", DEFAULT_RUNTIME_GROUPS)
                ),
            )
        return super().from_dict(params)


def _apply_rope(x: mx.array, positions: mx.array, dims: int, base: float) -> mx.array:
    """Apply non-traditional (rotate-half) RoPE at arbitrary positions."""
    if dims <= 0:
        return x
    inv = base ** (-mx.arange(0, dims, 2, dtype=mx.float32) / dims)
    angles = positions.astype(mx.float32)[..., None] * inv
    cos, sin = mx.cos(angles), mx.sin(angles)
    if positions.ndim == 1:
        cos, sin = cos[None, None, :, :], sin[None, None, :, :]
    elif positions.ndim == 2:
        cos, sin = cos[:, None, :, :], sin[:, None, :, :]
    first, second = x[..., : dims // 2], x[..., dims // 2 : dims]
    rotated = mx.concatenate(
        [first * cos - second * sin, second * cos + first * sin], axis=-1
    )
    return mx.concatenate([rotated, x[..., dims:]], axis=-1)


def _cache_positions(offset, length: int, batch: int) -> mx.array:
    steps = mx.arange(length, dtype=mx.int32)
    if isinstance(offset, mx.array) and offset.ndim > 0:
        return offset.astype(mx.int32)[:, None] + steps[None, :]
    return mx.broadcast_to(
        steps[None, :] + int(offset),
        (batch, length),
    )


class Qwen4RMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float, activation: str):
        super().__init__()
        self.weight = mx.ones(hidden_size)
        self.eps = eps
        self.activation = activation

    def __call__(self, x, gate):
        normalized = mx.fast.rms_norm(x, self.weight, self.eps)
        gate = gate.astype(mx.float32)
        gate = mx.sigmoid(gate) if self.activation == "sigmoid" else nn.silu(gate)
        return (normalized.astype(mx.float32) * gate).astype(x.dtype)


class Qwen4RMSNorm(nn.Module):
    """Qwen4 zero-centered RMSNorm, optionally normalized per HC stream."""

    def __init__(self, hidden_size: int, eps: float, group_size: int | None = None):
        super().__init__()
        if group_size is not None and hidden_size % group_size:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"group_size ({group_size})"
            )
        # The sanitizer converts HF's zero-centered weight to a conventional
        # multiplicative weight by adding one.
        self.weight = mx.ones(hidden_size)
        self.eps = eps
        self.group_size = group_size

    def __call__(self, x):
        original_shape = x.shape
        if self.group_size is not None:
            x = x.reshape(*x.shape[:-1], -1, self.group_size)
        x = mx.fast.rms_norm(x, None, self.eps)
        if self.group_size is not None:
            x = x.reshape(original_shape)
        return (x.astype(mx.float32) * self.weight.astype(mx.float32)).astype(x.dtype)


class Qwen4GatedDeltaNet(GatedDeltaNet):
    def __init__(self, args: TextModelArgs):
        super().__init__(args)
        self.norm = Qwen4RMSNormGated(
            self.head_v_dim,
            args.rms_norm_eps,
            args.output_gate_type or args.hidden_act,
        )

    def __call__(self, inputs, mask=None, cache=None):
        """Run GDN with Qwen4's exact L2 normalization convention."""
        batch, length, _ = inputs.shape
        if self.sharding_group is not None:
            inputs = sum_gradients(self.sharding_group)(inputs)

        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(
            batch, length, self.num_v_heads, self.head_v_dim
        )
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        conv_state = cache[0] if cache is not None else None
        if conv_state is None:
            conv_state = mx.zeros(
                (batch, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )
        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, length)
                positions = (ends[:, None] + mx.arange(keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -keep:, :])
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            value.reshape(batch, length, heads, dims)
            for value, heads, dims in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        # Transformers/FLA uses true L2 normalization with epsilon after the
        # reduction. Qwen3.5's algebraically convenient RMS formulation puts
        # a head-dimension-scaled epsilon inside the reduction; that small
        # difference is amplified by Qwen4's sigmoid output gate.
        q = q * mx.rsqrt(mx.sum(q * q, axis=-1, keepdims=True) + 1e-6)
        k = k * mx.rsqrt(mx.sum(k * k, axis=-1, keepdims=True) + 1e-6)
        q = q * (self.head_k_dim**-0.5)

        state = cache[1] if cache is not None else None
        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
        )
        if cache is not None:
            cache[1] = state
            cache.advance(length)

        out = self.norm(out, z)
        out = self.out_proj(out.reshape(batch, length, -1))
        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)
        return out


class Qwen4GatedResidual(nn.Module):
    def __init__(self, args: TextModelArgs, use_combine: bool = True):
        super().__init__()
        self.hc_count = args.hc_count
        self.hidden_size = args.hidden_size
        hc_hidden = self.hc_count * self.hidden_size
        self.hc_norm = Qwen4RMSNorm(
            hc_hidden, args.rms_norm_eps, group_size=args.hidden_size
        )
        self.input_mix_weight_down = nn.Linear(hc_hidden, args.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(args.hc_lowrank, hc_hidden, bias=False)
        self.block_inject_weight = (
            nn.Linear(hc_hidden, self.hc_count, bias=False) if use_combine else None
        )

    def __call__(self, hyper_input):
        normalized = self.hc_norm(hyper_input)
        mix = nn.silu(self.input_mix_weight_down(normalized) / self.hc_count)
        mix = mx.sigmoid(self.input_mix_weight_up(mix))
        mix = mix.reshape(*mix.shape[:-1], self.hc_count, self.hidden_size)
        streams = normalized.reshape(
            *normalized.shape[:-1], self.hc_count, self.hidden_size
        )
        mixed = (mix * streams).mean(axis=-2)
        if self.block_inject_weight is None:
            return mixed
        injection = 2 * mx.sigmoid(self.block_inject_weight(normalized) / self.hc_count)
        return mixed, hyper_input, injection


_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    return all(value % divisor != 0 for divisor in range(3, math.isqrt(value) + 1, 2))


def _find_nth_prime_after(start: int, count: int) -> int:
    value = start
    for _ in range(count):
        value += 1
        while not _is_prime(value):
            value += 1
    return value


class Qwen4GroupedNGramEmbedding(nn.Module):
    def __init__(self, args: TextModelArgs, layer_idx: int, ple_layer_index: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.context_len = args.ngram_size - 1
        self.ngram_size = args.ngram_size
        self.heads_per_ngram = args.heads_per_ngram
        self.ngram_heads = self.context_len * self.heads_per_ngram
        self.eos_token_id = (
            args.eos_token_id[0]
            if isinstance(args.eos_token_id, list)
            else int(args.eos_token_id or 0)
        )
        head_dim = args.ple_embed_dim // self.ngram_heads

        sizes, offsets, total = [], [], 0
        for head in range(self.ngram_heads):
            global_head = ple_layer_index * self.ngram_heads + head
            size = _find_nth_prime_after(
                args.ngram_vocab_size_base - 1, global_head + 1
            )
            sizes.append(size)
            offsets.append(total)
            total += size
        divisor = args.make_ngram_vocab_size_divisible_by
        padded_total = math.ceil(total / divisor) * divisor
        self.ngram_heads_vocab_sizes = mx.array(sizes, dtype=mx.int64)
        self.ngram_heads_offsets = mx.array(offsets, dtype=mx.int64)

        max_long = (1 << 63) - 1
        half_bound = max(1, (max_long // max(args.vocab_size, 1)) // 2)
        base_seed = args.seed + 10007 * ple_layer_index
        multipliers = []
        for idx in range(args.ngram_size):
            value = (base_seed + _SPLITMIX_GAMMA * (idx + 1)) & _MASK64
            multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
        self.layer_multipliers = mx.array(multipliers, dtype=mx.int64)

        group_count = int(args.omlx_qwen4_ngram_groups)
        if padded_total % group_count:
            raise ValueError(
                f"Qwen4 padded n-gram vocab {padded_total} is not divisible by {group_count}"
            )
        self.ngram_embedding = Qwen4GroupedEmbeddingTables(
            padded_total, group_count, head_dim
        )

    def _shift_right_ignore_eos(self, token_ids, shift):
        if shift == 0:
            return token_ids
        batch, length = token_ids.shape
        positions = mx.arange(length, dtype=mx.int64)
        eos_positions = mx.where(token_ids == self.eos_token_id, positions, -1)
        inclusive = mx.cummax(eos_positions, axis=1)
        previous = mx.concatenate(
            [mx.full((batch, 1), -1, dtype=mx.int64), inclusive[:, :-1]], axis=1
        )
        position_in_segment = positions[None, :] - (previous + 1)
        source = positions - shift
        gathered = mx.take_along_axis(
            token_ids, mx.broadcast_to(mx.maximum(source, 0), token_ids.shape), axis=1
        )
        valid = (position_in_segment >= shift) & (source[None, :] >= 0)
        return mx.where(valid, gathered, self.eos_token_id)

    def __call__(self, input_ids, cache: ArraysCache | None):
        input_ids = input_ids.astype(mx.int64)
        if cache is not None and cache[3] is not None:
            previous = cache[3]
        else:
            previous = mx.full(
                (input_ids.shape[0], self.context_len),
                self.eos_token_id,
                dtype=mx.int64,
            )
        history = mx.concatenate([previous, input_ids], axis=1)
        if cache is not None:
            cache[3] = mx.contiguous(history[:, -self.context_len :])

        shifted = [
            self._shift_right_ignore_eos(history, shift)
            for shift in range(self.ngram_size)
        ]
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for position in range(1, ngram):
                mixed = mx.bitwise_xor(
                    mixed, shifted[position] * self.layer_multipliers[position]
                )
            sizes = self.ngram_heads_vocab_sizes[start:end]
            offsets = self.ngram_heads_offsets[start:end]
            blocks.append((mixed[..., None] % sizes) + offsets)
        ids = mx.concatenate(blocks, axis=-1)[:, -input_ids.shape[1] :]
        return self.ngram_embedding(ids).reshape(*ids.shape[:2], -1)


class Qwen4GroupedEmbeddingTables(nn.Module):
    def __init__(self, total_rows: int, group_count: int, head_dim: int):
        super().__init__()
        self.group_rows = total_rows // group_count
        self.groups = [
            nn.Embedding(self.group_rows, head_dim) for _ in range(group_count)
        ]

    def __call__(self, ids):
        output = None
        for group_idx, embedding in enumerate(self.groups):
            start = group_idx * self.group_rows
            local = mx.clip(ids - start, 0, self.group_rows - 1)
            value = embedding(local)
            mask = ((ids >= start) & (ids < start + self.group_rows))[..., None]
            value = mx.where(mask, value, 0)
            output = value if output is None else output + value
        return output


class Qwen4PLELayer(nn.Module):
    def __init__(self, args: TextModelArgs, layer_idx: int, ple_layer_index: int):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.hc_count = args.hc_count
        self.ple_embedding = Qwen4GroupedNGramEmbedding(
            args, layer_idx, ple_layer_index
        )
        self.key_proj = nn.Linear(
            args.ple_embed_dim, args.hc_count * args.hidden_size, bias=False
        )
        self.value_proj = nn.Linear(args.ple_embed_dim, args.hidden_size, bias=False)
        hc_hidden = args.hc_count * args.hidden_size
        self.norm_key = Qwen4RMSNorm(
            hc_hidden, args.rms_norm_eps, group_size=args.hidden_size
        )
        self.norm_query = Qwen4RMSNorm(
            hc_hidden, args.rms_norm_eps, group_size=args.hidden_size
        )
        self.norm_conv = Qwen4RMSNorm(
            hc_hidden, args.rms_norm_eps, group_size=args.hidden_size
        )
        self.kernel_size = args.ple_conv_kernel_size
        self.dilation = args.ngram_size
        self.state_length = (self.kernel_size - 1) * self.dilation
        self.conv1d = nn.Conv1d(
            hc_hidden, hc_hidden, self.kernel_size, groups=hc_hidden, bias=False
        )

    def _short_conv(self, x, cache):
        previous = cache[2] if cache is not None else None
        if previous is None:
            previous = mx.zeros(
                (x.shape[0], self.state_length, x.shape[-1]), dtype=x.dtype
            )
        full = mx.concatenate([previous, x], axis=1)
        if cache is not None:
            cache[2] = mx.contiguous(full[:, -self.state_length :])
        weight = self.conv1d.weight[..., 0]
        pieces = [
            full[:, idx * self.dilation : idx * self.dilation + x.shape[1], :]
            * weight[:, idx]
            for idx in range(self.kernel_size)
        ]
        return nn.silu(sum(pieces))

    def __call__(self, hidden, input_ids, cache):
        embedded = self.ple_embedding(input_ids, cache)
        keys = self.norm_key(self.key_proj(embedded)).reshape(
            *hidden.shape[:-1], self.hc_count, self.hidden_size
        )
        queries = self.norm_query(hidden).reshape(
            *hidden.shape[:-1], self.hc_count, self.hidden_size
        )
        value = self.value_proj(embedded)
        gate = (keys * queries).sum(axis=-1, keepdims=True) / math.sqrt(
            self.hidden_size
        )
        gate = mx.sign(gate) * mx.sqrt(mx.maximum(mx.abs(gate), 1e-6))
        gated = (mx.sigmoid(gate) * value[..., None, :]).reshape(*hidden.shape)
        return gated + self._short_conv(self.norm_conv(gated), cache)


class Qwen4QSAIndexer(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.n_heads = args.indexer_n_heads
        self.head_dim = args.indexer_head_dim
        self.compress_ratio = args.indexer_compress_ratio
        self.block_topk = args.indexer_budget // args.indexer_compress_ratio
        self.rotary_dim = int(args.head_dim * args.partial_rotary_factor)
        self.rope_theta = args.rope_theta
        self.index_qk_proj = nn.Linear(
            args.hidden_size,
            (args.indexer_n_heads + 1) * args.indexer_head_dim,
            bias=False,
        )
        self.q_layernorm = Qwen4RMSNorm(self.head_dim, args.rms_norm_eps)
        self.k_layernorm = Qwen4RMSNorm(self.head_dim, args.rms_norm_eps)

    def __call__(self, hidden, cache: KVCache | None, offset: int):
        batch, length, _ = hidden.shape
        qk = self.index_qk_proj(hidden).reshape(
            batch, length, self.n_heads + 1, self.head_dim
        )
        queries = self.q_layernorm(qk[..., : self.n_heads, :]).transpose(0, 2, 1, 3)
        raw = qk[..., self.n_heads, :]
        if cache is not None:
            raw, _ = cache.update_and_fetch(raw[:, None, :, :], raw[:, None, :, :])
            raw = raw[:, 0]
        positions = _cache_positions(offset, length, batch)
        queries = _apply_rope(queries, positions, self.rotary_dim, self.rope_theta)
        return queries, raw

    def sparse_mask(self, queries, raw_keys, offset: int):
        batch, _, length, _ = queries.shape
        key_length = raw_keys.shape[1]
        complete_blocks = key_length // self.compress_ratio
        q_positions = _cache_positions(offset, length, batch)
        slots = mx.arange(key_length, dtype=mx.int32)
        causal = slots[None, None, :] <= q_positions[:, :, None]
        if complete_blocks == 0:
            return causal[:, None, :, :]

        usable = raw_keys[:, : complete_blocks * self.compress_ratio]
        pooled = (
            usable.reshape(batch, complete_blocks, self.compress_ratio, self.head_dim)
            .astype(mx.float32)
            .mean(axis=2)
            .astype(raw_keys.dtype)
        )
        pooled = self.k_layernorm(pooled)
        starts = mx.arange(complete_blocks, dtype=mx.int32) * self.compress_ratio
        pooled = _apply_rope(
            pooled[:, None, :, :], starts, self.rotary_dim, self.rope_theta
        )[:, 0]
        scores = (
            queries.astype(mx.float32).transpose(0, 2, 1, 3)
            @ pooled.astype(mx.float32).transpose(0, 2, 1)[:, None, :, :]
        )
        scores = mx.maximum(scores, 0).sum(axis=2) / math.sqrt(self.head_dim)
        eligible = (starts + self.compress_ratio - 1)[None, None, :] <= q_positions[
            :, :, None
        ]
        scores = mx.where(eligible, scores, -mx.inf)
        k = min(self.block_topk, complete_blocks)
        selected_blocks = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        selected_tokens = (
            selected_blocks[..., None] * self.compress_ratio
            + mx.arange(self.compress_ratio, dtype=selected_blocks.dtype)
        ).reshape(batch, length, -1)
        sparse = mx.zeros((batch, length, key_length), dtype=mx.bool_)
        sparse = mx.put_along_axis(sparse, selected_tokens, mx.array(True), axis=-1)
        tail_start = ((q_positions + 1) // self.compress_ratio) * self.compress_ratio
        tail = (slots[None, None, :] >= tail_start[:, :, None]) & causal
        return (sparse | tail)[:, None, :, :] & causal[:, None, :, :]


class Qwen4Attention(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.num_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = args.head_dim**-0.5
        self.rotary_dim = int(args.head_dim * args.partial_rotary_factor)
        self.rope_theta = args.rope_theta
        self.q_proj = nn.Linear(
            args.hidden_size, self.num_heads * self.head_dim * 2, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, args.hidden_size, bias=False
        )
        self.q_norm = Qwen4RMSNorm(self.head_dim, args.rms_norm_eps)
        self.k_norm = Qwen4RMSNorm(self.head_dim, args.rms_norm_eps)
        self.indexer = Qwen4QSAIndexer(args)

    def __call__(self, x, cache: CacheList | None):
        batch, length, _ = x.shape
        kv_cache = cache[0] if cache is not None else None
        index_cache = cache[1] if cache is not None else None
        offset = kv_cache.offset if kv_cache is not None else 0
        index_queries, raw_keys = self.indexer(x, index_cache, offset)

        q, gate = mx.split(
            self.q_proj(x).reshape(batch, length, self.num_heads, -1), 2, axis=-1
        )
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(
            self.k_proj(x).reshape(batch, length, self.num_kv_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        v = (
            self.v_proj(x)
            .reshape(batch, length, self.num_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        positions = _cache_positions(offset, length, batch)
        q = _apply_rope(q, positions, self.rotary_dim, self.rope_theta)
        k = _apply_rope(k, positions, self.rotary_dim, self.rope_theta)
        if kv_cache is not None:
            k, v = kv_cache.update_and_fetch(k, v)
        mask = self.indexer.sparse_mask(index_queries, raw_keys, offset)
        out = scaled_dot_product_attention(
            q, k, v, cache=kv_cache, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(out * mx.sigmoid(gate.reshape(batch, length, -1)))


class Qwen4DecoderLayer(nn.Module):
    def __init__(self, args: TextModelArgs, layer_idx: int):
        super().__init__()
        self.is_linear = args.layer_types[layer_idx] == "linear_attention"
        if self.is_linear:
            self.linear_attn = Qwen4GatedDeltaNet(args)
        else:
            self.self_attn = Qwen4Attention(args)
        self.mlp = Qwen3NextSparseMoeBlock(args)
        ple_index = (
            args.ple_layer_ids.index(layer_idx + 1)
            if layer_idx + 1 in args.ple_layer_ids
            else None
        )
        self.ple = (
            Qwen4PLELayer(args, layer_idx, ple_index) if ple_index is not None else None
        )
        self.attn_hyper_connection = Qwen4GatedResidual(args)
        self.mlp_hyper_connection = Qwen4GatedResidual(args)

    def __call__(self, hidden, input_ids, mask=None, cache=None):
        if self.ple is not None:
            hidden = hidden + self.ple(hidden, input_ids, cache)
        mixed, residual, injection = self.attn_hyper_connection(hidden)
        out = (
            self.linear_attn(mixed, mask, cache)
            if self.is_linear
            else self.self_attn(mixed, cache)
        )
        hidden = residual + (out[..., None, :] * injection[..., :, None]).reshape(
            *residual.shape
        )
        mixed, residual, injection = self.mlp_hyper_connection(hidden)
        out = self.mlp(mixed)
        return residual + (out[..., None, :] * injection[..., :, None]).reshape(
            *residual.shape
        )


class Qwen4TextModel(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Qwen4DecoderLayer(args, idx) for idx in range(args.num_hidden_layers)
        ]
        self.hyper_connection_mixer = Qwen4GatedResidual(args, use_combine=False)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        hidden = (
            self.embed_tokens(inputs) if input_embeddings is None else input_embeddings
        )
        hidden = mx.tile(hidden, (1, 1, self.args.hc_count))
        if cache is None:
            cache = [None] * len(self.layers)
        first_linear = next(
            (idx for idx, layer in enumerate(self.layers) if layer.is_linear), None
        )
        ssm_mask = (
            create_ssm_mask(hidden, cache[first_linear])
            if first_linear is not None
            else None
        )
        for layer, layer_cache in zip(self.layers, cache):
            hidden = layer(
                hidden,
                inputs,
                mask=ssm_mask if layer.is_linear else None,
                cache=layer_cache,
            )
        return self.hyper_connection_mixer(hidden)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        text_dict = dict(args.text_config)
        text_dict["omlx_qwen4_ngram_groups"] = args.omlx_qwen4_ngram_groups
        self.text_args = TextModelArgs.from_dict(text_dict)
        self.model_type = args.model_type
        self.model = Qwen4TextModel(self.text_args)
        if not self.text_args.tie_word_embeddings:
            self.lm_head = nn.Linear(
                self.text_args.hidden_size, self.text_args.vocab_size, bias=False
            )

    def __call__(self, inputs, cache=None, input_embeddings=None):
        hidden = self.model(inputs, cache=cache, input_embeddings=input_embeddings)
        return (
            self.model.embed_tokens.as_linear(hidden)
            if self.text_args.tie_word_embeddings
            else self.lm_head(hidden)
        )

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            ArraysCache(size=4) if layer.is_linear else CacheList(KVCache(), KVCache())
            for layer in self.layers
        ]

    def sanitize(self, weights):
        return sanitize_weights(
            weights,
            {
                "model_type": "qwen4_exp",
                "text_config": self.args.text_config,
                "omlx_qwen4_ngram_groups": self.args.omlx_qwen4_ngram_groups,
            },
            text_only=True,
        )

    @property
    def cast_predicate(self):
        return cast_predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            return not path.endswith("mlp.gate")

        return predicate
