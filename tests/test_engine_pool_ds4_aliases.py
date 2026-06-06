# SPDX-License-Identifier: Apache-2.0
"""EnginePool resolution tests for DS4 per-model aliases."""

from unittest.mock import MagicMock

from omlx.engine_pool import EnginePool
from omlx.model_settings import ModelSettings


def _pool_with_gguf(tmp_path, filename: str = "Foo.gguf") -> EnginePool:
    (tmp_path / filename).write_bytes(b"0" * 1000)
    pool = EnginePool()
    pool.discover_models(str(tmp_path))
    return pool


def test_ds4_suffix_alias_resolves_to_base_model(tmp_path):
    """Built-in DS4 suffix aliases resolve to the discovered GGUF entry."""
    pool = _pool_with_gguf(tmp_path, "DeepSeek V4 Flash Q2_K.gguf")

    assert pool.resolve_model_id("deepseek-v4-flash-q2-k-chat", None) == (
        "deepseek-v4-flash-q2-k"
    )
    assert pool.resolve_model_id("omlx/DEEPSEEK-V4-FLASH-Q2-K-REASONER", None) == (
        "deepseek-v4-flash-q2-k"
    )
    assert pool.resolve_model_id("deepseek-v4-flash-q2-k-think-max", None) == (
        "deepseek-v4-flash-q2-k"
    )


def test_ds4_suffix_alias_uses_existing_model_alias(tmp_path):
    """User-defined model_alias values also receive DS4 suffix aliases."""
    pool = _pool_with_gguf(tmp_path)

    settings_manager = MagicMock()
    settings_manager.get_all_settings.return_value = {
        "foo": ModelSettings(model_alias="gpt-4o"),
    }

    assert pool.resolve_model_id("gpt-4o", settings_manager) == "foo"
    assert pool.resolve_model_id("gpt-4o-chat", settings_manager) == "foo"
    assert pool.resolve_model_id("omlx/gpt-4o-reasoner", settings_manager) == "foo"


def test_global_ds4_native_aliases_are_not_auto_created(tmp_path):
    """deepseek-chat/reasoner do not route globally without a DS4 base model."""
    pool = _pool_with_gguf(tmp_path)

    assert pool.resolve_model_id("deepseek-chat", None) == "deepseek-chat"
    assert pool.resolve_model_id("deepseek-reasoner", None) == "deepseek-reasoner"
