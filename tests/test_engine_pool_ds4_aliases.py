# SPDX-License-Identifier: Apache-2.0
"""EnginePool resolution tests for DS4 per-model aliases."""

import json
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
    settings_manager.get_exposed_profile_source_model_id.return_value = None
    settings_manager.get_all_settings.return_value = {
        "foo": ModelSettings(model_alias="gpt-4o"),
    }

    assert pool.resolve_model_id("gpt-4o", settings_manager) == "foo"
    assert pool.resolve_model_id("gpt-4o-chat", settings_manager) == "foo"
    assert pool.resolve_model_id("omlx/gpt-4o-reasoner", settings_manager) == "foo"


def test_model_alias_ignores_undiscovered_settings_entry(tmp_path):
    """Stale per-model settings do not resolve aliases to missing models."""
    pool = _pool_with_gguf(tmp_path)

    settings_manager = MagicMock()
    settings_manager.get_exposed_profile_source_model_id.return_value = None
    settings_manager.get_all_settings.return_value = {
        "missing": ModelSettings(model_alias="gpt-4o"),
    }

    assert pool.resolve_model_id("gpt-4o", settings_manager) == "gpt-4o"


def test_ds4_gguf_file_path_is_not_dropped_as_missing(tmp_path):
    """DS4 GGUF entries are files, not directories with config.json."""
    pool = _pool_with_gguf(tmp_path)
    entry = pool.get_entry("foo")

    assert entry is not None
    pool._raise_if_model_path_missing_locked("foo", entry)
    assert pool.get_entry("foo") is entry


def test_global_ds4_native_aliases_are_not_auto_created(tmp_path):
    """deepseek-chat/reasoner do not route globally without a DS4 base model."""
    pool = _pool_with_gguf(tmp_path)

    assert pool.resolve_model_id("deepseek-chat", None) == "deepseek-chat"
    assert pool.resolve_model_id("deepseek-reasoner", None) == "deepseek-reasoner"


def test_ds4_source_filename_and_path_resolve_to_normalized_id(tmp_path):
    """Original GGUF filenames/paths resolve to the discovered DS4 entry."""
    filename = "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf"
    gguf = tmp_path / filename
    pool = _pool_with_gguf(tmp_path, filename)
    model_id = "deepseek-v4-flash-iq2xxs-w2q2k-aprojq8-sexpq8-outq8-chat-v2"

    assert pool.resolve_model_id(filename, None) == model_id
    assert pool.resolve_model_id(filename.removesuffix(".gguf"), None) == model_id
    assert pool.resolve_model_id(str(gguf), None) == model_id
    assert pool.resolve_model_id(f"{filename}-chat", None) == model_id


def test_ds4_source_filename_disambiguates_mlx_collision(tmp_path):
    """A .gguf source name can select DS4 when the stem collides with MLX."""
    (tmp_path / "Foo.gguf").write_bytes(b"0" * 1000)
    mlx_model = tmp_path / "foo"
    mlx_model.mkdir()
    (mlx_model / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (mlx_model / "model.safetensors").write_bytes(b"0" * 1000)
    pool = EnginePool()
    pool.discover_models(str(tmp_path))

    assert pool.resolve_model_id("Foo", None) == "foo"
    assert pool.resolve_model_id("Foo.gguf", None) == "foo:ds4"
    assert pool.resolve_model_id("Foo.gguf-chat", None) == "foo:ds4"
    assert pool.resolve_model_id("foo-chat", None) == "foo-chat"


def test_ds4_full_path_source_alias_matches_exact_path_before_basename(tmp_path):
    """Full GGUF paths do not resolve to another entry with the same basename."""
    first = tmp_path / "a" / "model.gguf"
    second = tmp_path / "b" / "model.gguf"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"0" * 1000)
    second.write_bytes(b"1" * 1000)
    pool = EnginePool()
    pool.discover_models(str(tmp_path))

    assert pool.resolve_model_id(str(first), None) == "a"
    assert pool.resolve_model_id(str(second), None) == "b"
    assert pool.resolve_model_id(f"{second}-chat", None) == "b"
