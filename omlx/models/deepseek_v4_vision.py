# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 + FlyCockpit DeepEncoderV2 sidecar support.

The receiver remains an ordinary mlx-lm DeepSeek-V4 model.  Vision is kept
as a reference-precision BF16 sidecar and executed out of process by the
catalog's pinned encoder script.  This avoids teaching mlx-vlm about the
PyTorch-only DeepEncoderV2 tower while preserving the raw route-token IDs
needed by DeepSeek-V4's first hash-routed MoE layers.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import mlx.nn as nn
import numpy as np
from PIL import Image

VISION_CONFIG_TYPE = "deepencoder_v2"
VISION_MODEL_TYPE = "deepseek_v4_vision"
IMAGE_TOKEN = "<｜image｜>"
IMAGE_TOKEN_ID = 129279
HIDDEN_SIZE = 4096
VOCAB_SIZE = 129280
DS4V_MAGIC = b"DS4VEMB1"
DS4V_VERSION = 1
DS4V_HEADER = struct.Struct("<8sIIIIiiiI32s")
VisionControl = Literal["real", "zero", "shuffle"]


def _read_model_config(model_path: str | Path) -> dict[str, Any]:
    path = Path(model_path) / "config.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def is_deepseek_v4_vision_config(config: dict[str, Any]) -> bool:
    """Return whether *config* declares oMLX's DeepEncoderV2 bridge."""
    vision = config.get("vision_config")
    return (
        isinstance(vision, dict)
        and vision.get("model_type") == VISION_CONFIG_TYPE
        and str(config.get("model_type", "")).replace("-", "_").lower() == "deepseek_v4"
    )


def is_deepseek_v4_vision_path(model_path: str | Path) -> bool:
    return is_deepseek_v4_vision_config(_read_model_config(model_path))


@dataclass(frozen=True)
class DeepEncoderV2Sidecar:
    model_dir: Path
    tower: Path
    projector: Path
    encoder: Path
    image_token_id: int
    hidden_size: int
    vocab_size: int
    tiles: int
    timeout_seconds: int

    @classmethod
    def from_model_path(cls, model_path: str | Path) -> DeepEncoderV2Sidecar:
        model_dir = Path(model_path).resolve()
        config = _read_model_config(model_dir)
        if not is_deepseek_v4_vision_config(config):
            raise ValueError(
                f"{model_dir} does not declare vision_config.model_type="
                f"{VISION_CONFIG_TYPE!r}"
            )
        vision = config["vision_config"]

        def resolve(name: str) -> Path:
            raw = vision.get(name)
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f"vision_config.{name} must be a non-empty path")
            candidate = (model_dir / raw).resolve()
            if not candidate.is_file():
                raise FileNotFoundError(
                    f"vision_config.{name} does not exist: {candidate}"
                )
            return candidate

        image_token_id = int(vision.get("image_token_id", IMAGE_TOKEN_ID))
        hidden_size = int(vision.get("hidden_size", HIDDEN_SIZE))
        vocab_size = int(config.get("vocab_size", VOCAB_SIZE))
        tiles = int(vision.get("tiles", 2))
        timeout_seconds = int(vision.get("encoder_timeout_seconds", 600))
        if image_token_id != IMAGE_TOKEN_ID:
            raise ValueError(
                f"DeepEncoderV2 bridge requires image token {IMAGE_TOKEN_ID}, "
                f"got {image_token_id}"
            )
        if hidden_size != HIDDEN_SIZE:
            raise ValueError(
                f"DeepEncoderV2 bridge requires hidden size {HIDDEN_SIZE}, "
                f"got {hidden_size}"
            )
        if tiles < 0 or tiles > 4:
            raise ValueError("vision_config.tiles must be between 0 and 4")
        if timeout_seconds <= 0:
            raise ValueError("vision_config.encoder_timeout_seconds must be positive")
        return cls(
            model_dir=model_dir,
            tower=resolve("tower_path"),
            projector=resolve("projector_path"),
            encoder=resolve("encoder_path"),
            image_token_id=image_token_id,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            tiles=tiles,
            timeout_seconds=timeout_seconds,
        )

    def _python(self) -> str:
        override = os.environ.get("OMLX_DEEPSEEK_V4_VISION_PYTHON", "").strip()
        candidates = [
            override,
            "/private/tmp/dsv4-vision-venv/bin/python",
            sys.executable,
        ]
        for candidate in candidates:
            if candidate and Path(candidate).is_file():
                return candidate
        raise FileNotFoundError(
            "No DeepSeek-V4 vision encoder Python was found; set "
            "OMLX_DEEPSEEK_V4_VISION_PYTHON"
        )

    @staticmethod
    def _pixel_shuffle(image: Image.Image) -> Image.Image:
        """Deterministically destroy spatial structure for a causal control."""
        rgb = image.convert("RGB")
        pixels = np.asarray(rgb, dtype=np.uint8).copy()
        digest = hashlib.sha256(
            pixels.tobytes(order="C")
            + struct.pack("<II", int(rgb.width), int(rgb.height))
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        flat = pixels.reshape(-1, 3)
        shuffled = flat[rng.permutation(flat.shape[0])].reshape(pixels.shape)
        return Image.fromarray(shuffled, mode="RGB")

    def encode(
        self, image: Image.Image, control: VisionControl = "real"
    ) -> tuple[np.ndarray, np.ndarray, str]:
        if control not in ("real", "zero", "shuffle"):
            raise ValueError(f"unsupported DeepSeek-V4 vision control: {control!r}")
        encoded_image = self._pixel_shuffle(image) if control == "shuffle" else image

        with tempfile.TemporaryDirectory(prefix="omlx-dsv4-vision-") as tmp:
            temp_dir = Path(tmp)
            image_path = temp_dir / "input.png"
            output_path = temp_dir / "embeddings.ds4v"
            encoded_image.convert("RGB").save(image_path, format="PNG")
            command = [
                self._python(),
                str(self.encoder),
                "--image",
                str(image_path),
                "--output",
                str(output_path),
                "--tower",
                str(self.tower),
                "--adapter",
                str(self.projector),
                "--tiles",
                str(self.tiles),
            ]
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout).strip()
                raise RuntimeError(
                    "DeepEncoderV2 subprocess failed "
                    f"(exit {completed.returncode}): {detail[-2000:]}"
                )
            routes, embeddings, digest = self.read_ds4v(output_path)

        if control == "zero":
            embeddings.fill(0)
        return routes, embeddings, f"{control}:{digest}"

    def read_ds4v(self, path: str | Path) -> tuple[np.ndarray, np.ndarray, str]:
        payload = Path(path).read_bytes()
        if len(payload) < DS4V_HEADER.size:
            raise ValueError("truncated DS4VEMB1 payload")
        (
            magic,
            version,
            header_size,
            token_count,
            hidden_size,
            _flags,
            _span_start,
            _span_end,
            _reserved,
            image_digest,
        ) = DS4V_HEADER.unpack_from(payload)
        if magic != DS4V_MAGIC or version != DS4V_VERSION:
            raise ValueError("unsupported DS4VEMB payload")
        if header_size != DS4V_HEADER.size:
            raise ValueError(f"unexpected DS4VEMB header size: {header_size}")
        if hidden_size != self.hidden_size or token_count <= 0:
            raise ValueError(
                f"unexpected DS4VEMB shape: ({token_count}, {hidden_size})"
            )
        route_bytes = token_count * np.dtype("<i4").itemsize
        embed_count = token_count * hidden_size
        expected_size = (
            header_size + route_bytes + embed_count * np.dtype("<f4").itemsize
        )
        if len(payload) != expected_size:
            raise ValueError(
                f"DS4VEMB payload size mismatch: {len(payload)} != {expected_size}"
            )
        routes = np.frombuffer(
            payload, dtype="<i4", count=token_count, offset=header_size
        ).copy()
        embeddings = (
            np.frombuffer(
                payload,
                dtype="<f4",
                count=embed_count,
                offset=header_size + route_bytes,
            )
            .reshape(token_count, hidden_size)
            .copy()
        )
        if routes.min() < 0 or routes.max() >= self.vocab_size:
            raise ValueError("DS4VEMB route IDs fall outside the receiver vocabulary")
        if not np.isfinite(embeddings).all():
            raise ValueError("DS4VEMB contains non-finite embeddings")
        return routes, embeddings, image_digest.hex()


class DeepseekV4VisionModel(nn.Module):
    """A lightweight VLM-shaped wrapper around the loaded text receiver."""

    def __init__(self, language_model: nn.Module, sidecar: DeepEncoderV2Sidecar):
        super().__init__()
        self.language_model = language_model
        self.sidecar = sidecar
        args = getattr(language_model, "args", None)
        self.config = SimpleNamespace(
            model_type=VISION_MODEL_TYPE,
            text_config=args,
            eos_token_id=getattr(args, "eos_token_id", None),
        )

    def encode_image(
        self, image: Image.Image, control: VisionControl = "real"
    ) -> tuple[np.ndarray, np.ndarray, str]:
        return self.sidecar.encode(image, control)


def load_deepseek_v4_vision(
    model_name: str,
    *,
    model_settings: Any | None = None,
) -> tuple[DeepseekV4VisionModel, Any]:
    """Load the receiver with mlx-lm and return a VLM-shaped wrapper/tokenizer."""
    from ..utils.model_loading import load_text_model

    sidecar = DeepEncoderV2Sidecar.from_model_path(model_name)
    language_model, tokenizer = load_text_model(
        model_name,
        model_settings=model_settings,
    )
    return DeepseekV4VisionModel(language_model, sidecar), tokenizer
