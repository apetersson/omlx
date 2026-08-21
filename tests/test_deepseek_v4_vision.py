# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 DeepEncoderV2 sidecar bridge tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from PIL import Image

from omlx.models.deepseek_v4_vision import (
    DS4V_HEADER,
    DS4V_MAGIC,
    DS4V_VERSION,
    HIDDEN_SIZE,
    IMAGE_TOKEN_ID,
    DeepEncoderV2Sidecar,
    deepseek_v4_vision_prefill_step_size,
)


def _sidecar(tmp_path):
    for name in ("tower.safetensors", "projector.safetensors", "encode.py"):
        (tmp_path / name).write_bytes(b"x")
    config = {
        "model_type": "deepseek_v4",
        "vocab_size": 129280,
        "vision_config": {
            "model_type": "deepencoder_v2",
            "tower_path": "tower.safetensors",
            "projector_path": "projector.safetensors",
            "encoder_path": "encode.py",
            "image_token_id": IMAGE_TOKEN_ID,
            "hidden_size": HIDDEN_SIZE,
            "tiles": 2,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return DeepEncoderV2Sidecar.from_model_path(tmp_path)


def test_ds4v_parser_preserves_routes_and_float32_rows(tmp_path):
    sidecar = _sidecar(tmp_path)
    routes = np.full((257,), IMAGE_TOKEN_ID, dtype="<i4")
    embeddings = np.arange(257 * HIDDEN_SIZE, dtype="<f4").reshape(257, HIDDEN_SIZE)
    digest = bytes(range(32))
    header = DS4V_HEADER.pack(
        DS4V_MAGIC,
        DS4V_VERSION,
        DS4V_HEADER.size,
        257,
        HIDDEN_SIZE,
        0,
        -1,
        -1,
        0,
        digest,
    )
    payload = tmp_path / "image.ds4v"
    payload.write_bytes(header + routes.tobytes() + embeddings.tobytes())

    parsed_routes, parsed_embeddings, parsed_digest = sidecar.read_ds4v(payload)

    assert np.array_equal(parsed_routes, routes)
    assert np.array_equal(parsed_embeddings, embeddings)
    assert parsed_digest == digest.hex()


def test_prepare_injects_rows_but_retains_hash_route_ids():
    from omlx.engine.vlm import VLMBatchedEngine

    class Tokenizer:
        def encode(self, prompt, add_special_tokens=False):
            assert prompt == "rendered"
            assert add_special_tokens is False
            return [10, IMAGE_TOKEN_ID, 11]

    class Receiver:
        def __init__(self):
            self.model = self

        def embed_tokens(self, token_ids):
            values = token_ids.astype(mx.float32)[..., None]
            return mx.broadcast_to(values, (*token_ids.shape, HIDDEN_SIZE))

    class Wrapped:
        def __init__(self):
            self.config = SimpleNamespace(model_type="deepseek_v4_vision")
            self.language_model = Receiver()

        def encode_image(self, image, control="real"):
            assert image.size == (8, 8)
            assert control == "real"
            return (
                np.full((257,), IMAGE_TOKEN_ID, dtype=np.int32),
                np.ones((257, HIDDEN_SIZE), dtype=np.float32),
                "real:" + "ab" * 32,
            )

    engine = VLMBatchedEngine("test")
    engine._vlm_model = Wrapped()
    engine._tokenizer = Tokenizer()
    engine._apply_chat_template = lambda *args, **kwargs: "rendered"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "unused"}},
                {"type": "text", "text": "describe"},
            ],
        }
    ]

    token_ids, merged, extra, image_hash, start, ranges = (
        engine._prepare_deepseek_v4_vision_inputs(
            messages,
            [Image.new("RGB", (8, 8), "red")],
        )
    )
    mx.eval(merged)

    assert token_ids == [10] + [IMAGE_TOKEN_ID] * 257 + [11]
    assert merged.shape == (1, 259, HIDDEN_SIZE)
    assert float(merged[0, 0, 0].item()) == 10.0
    assert float(merged[0, 1, 0].item()) == 1.0
    assert float(merged[0, -1, 0].item()) == 11.0
    assert extra is None
    assert len(image_hash) == 64
    assert start == 1
    assert ranges == [(1, image_hash)]


def test_message_normalizer_preserves_image_text_order():
    from omlx.engine.vlm import VLMBatchedEngine
    from omlx.models.deepseek_v4_vision import IMAGE_TOKEN

    normalized, count = VLMBatchedEngine._deepseek_v4_vision_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {"type": "image_url", "image_url": {"url": "unused"}},
                    {"type": "text", "text": "after"},
                ],
            }
        ]
    )

    assert normalized == [{"role": "user", "content": f"before\n{IMAGE_TOKEN}\nafter"}]
    assert count == 1


def test_artifact_prefill_step_size_is_optional_and_scoped(tmp_path):
    _sidecar(tmp_path)
    assert deepseek_v4_vision_prefill_step_size(tmp_path) is None

    path = tmp_path / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["vision_config"]["prefill_step_size"] = 32
    path.write_text(json.dumps(config), encoding="utf-8")
    assert deepseek_v4_vision_prefill_step_size(tmp_path) == 32

    config["vision_config"]["model_type"] = "other"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert deepseek_v4_vision_prefill_step_size(tmp_path) is None


def test_artifact_prefill_step_size_rejects_invalid_values(tmp_path):
    _sidecar(tmp_path)
    path = tmp_path / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["vision_config"]["prefill_step_size"] = 0
    path.write_text(json.dumps(config), encoding="utf-8")

    try:
        deepseek_v4_vision_prefill_step_size(tmp_path)
    except ValueError as exc:
        assert "positive integer" in str(exc)
    else:
        raise AssertionError("invalid prefill marker was accepted")
