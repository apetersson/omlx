# SPDX-License-Identifier: Apache-2.0
"""Admin settings safeguards for DS4-discovered GGUF models."""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from omlx.admin import routes as admin_routes
from omlx.engine_pool import EngineEntry, EnginePool
from omlx.model_settings import ModelSettings


def _ds4_pool(model_path: str = "/models/Foo.gguf") -> EnginePool:
    pool = EnginePool()
    pool._entries["foo"] = EngineEntry(
        model_id="foo",
        model_path=model_path,
        model_type="llm",
        engine_type="ds4",
        estimated_size=1000,
    )
    return pool


@pytest.mark.asyncio
async def test_update_model_settings_rejects_ds4_model_type_override(monkeypatch):
    """Admin settings must not reroute DS4 GGUF entries to MLX engines."""
    pool = _ds4_pool()
    manager = MagicMock()
    manager.get_settings.return_value = ModelSettings()
    monkeypatch.setattr(admin_routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(admin_routes, "_get_settings_manager", lambda: manager)
    monkeypatch.setattr(admin_routes, "_get_server_state", lambda: None)

    with pytest.raises(HTTPException) as exc_info:
        await admin_routes.update_model_settings(
            "foo",
            admin_routes.ModelSettingsRequest(model_type_override="vlm"),
            is_admin=True,
        )

    assert exc_info.value.status_code == 400
    assert "DS4 GGUF" in exc_info.value.detail
    entry = pool.get_entry("foo")
    assert entry.model_type == "llm"
    assert entry.engine_type == "ds4"
    manager.set_settings.assert_not_called()


@pytest.mark.asyncio
async def test_update_model_settings_can_clear_stale_ds4_model_type_override(
    monkeypatch,
):
    """Clearing a stale override keeps DS4 entries pinned to their DS4 engine."""
    pool = _ds4_pool()
    manager = MagicMock()
    manager.get_settings.return_value = ModelSettings(model_type_override="vlm")
    monkeypatch.setattr(admin_routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(admin_routes, "_get_settings_manager", lambda: manager)
    monkeypatch.setattr(admin_routes, "_get_server_state", lambda: None)

    result = await admin_routes.update_model_settings(
        "foo",
        admin_routes.ModelSettingsRequest(model_type_override=None),
        is_admin=True,
    )

    assert result["model_type"] == "llm"
    assert result["engine_type"] == "ds4"
    saved = manager.set_settings.call_args.args[1]
    assert saved.model_type_override is None
    entry = pool.get_entry("foo")
    assert entry.model_type == "llm"
    assert entry.engine_type == "ds4"


@pytest.mark.asyncio
async def test_list_models_exposes_ds4_display_name(monkeypatch, tmp_path):
    """Admin model list keeps source-cased DS4 display names."""
    gguf = tmp_path / "Foo.gguf"
    gguf.write_bytes(b"0" * 1000)
    pool = _ds4_pool(str(gguf))
    pool.get_entry("foo").display_name = "Foo"
    manager = MagicMock()
    manager.get_all_settings.return_value = {}
    monkeypatch.setattr(admin_routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(admin_routes, "_get_settings_manager", lambda: manager)
    monkeypatch.setattr(admin_routes, "_get_server_state", lambda: None)
    monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: None)

    result = await admin_routes.list_models(is_admin=True)

    assert result["models"][0]["id"] == "foo"
    assert result["models"][0]["engine_type"] == "ds4"
    assert result["models"][0]["display_name"] == "Foo"
