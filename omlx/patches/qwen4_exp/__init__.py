# SPDX-License-Identifier: Apache-2.0
"""Qwen4-Exp compatibility helpers vendored ahead of upstream MLX support."""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

TRANSFORMERS_PR = "https://github.com/huggingface/transformers/pull/48337"
TRANSFORMERS_PR_HEAD = "b61b98bea4cd99ff97da2ca0aa4fa34e8800d10e"

_APPLIED = False
_TOKENIZER_PATCHED = False


def _register_model() -> None:
    qualname = "mlx_lm.models.qwen4_exp"
    if qualname in sys.modules:
        return
    source = Path(__file__).with_name("qwen4_exp_model.py")
    spec = importlib.util.spec_from_file_location(qualname, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot register {qualname} from {source}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "mlx_lm.models"
    sys.modules[qualname] = module
    spec.loader.exec_module(module)


def _patch_tokenizer() -> None:
    global _TOKENIZER_PATCHED
    if _TOKENIZER_PATCHED:
        return
    import mlx_lm.tokenizer_utils as tokenizer_utils
    from transformers import PreTrainedConfig

    original = tokenizer_utils.AutoTokenizer

    class Qwen4AwareAutoTokenizer:
        @staticmethod
        def from_pretrained(model_path, *args, **kwargs):
            try:
                return original.from_pretrained(model_path, *args, **kwargs)
            except (AttributeError, ValueError) as exc:
                config_path = Path(model_path) / "config.json"
                try:
                    is_qwen4 = (
                        json.loads(config_path.read_text()).get("model_type")
                        == "qwen4_exp"
                    )
                except Exception:
                    is_qwen4 = False
                if not is_qwen4 or "config" in kwargs:
                    raise
                logger.warning(
                    "Transformers does not yet recognize qwen4_exp; using its "
                    "published tokenizer files with a generic config (%s)",
                    exc,
                )
                return original.from_pretrained(
                    model_path, *args, config=PreTrainedConfig(), **kwargs
                )

    tokenizer_utils.AutoTokenizer = Qwen4AwareAutoTokenizer
    _TOKENIZER_PATCHED = True


def apply_qwen4_exp_patch() -> bool:
    """Register the vendored text runtime before mlx-lm resolves classes."""
    global _APPLIED
    if _APPLIED:
        return False
    _register_model()
    _patch_tokenizer()
    _APPLIED = True
    logger.info(
        "Qwen4-Exp MLX patch applied (Transformers PR head %s)",
        TRANSFORMERS_PR_HEAD[:8],
    )
    return True


def is_applied() -> bool:
    return _APPLIED


__all__ = [
    "TRANSFORMERS_PR",
    "TRANSFORMERS_PR_HEAD",
    "apply_qwen4_exp_patch",
    "is_applied",
]
