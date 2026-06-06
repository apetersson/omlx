# SPDX-License-Identifier: Apache-2.0
"""Admin settings safeguards for DS4-discovered GGUF models."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from omlx.admin import routes as admin_routes
from omlx.engine.ds4 import DS4ProxyError
from omlx.engine_pool import EngineEntry, EnginePool
from omlx.model_settings import ModelSettings
from omlx.settings import DS4_MAX_CONTEXT_TOKENS

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _FakeStatusDS4Engine:
    def get_stats(self):
        return {
            "backend": "ds4",
            "host": "127.0.0.1",
            "port": 49152,
            "pid": 12345,
            "running": True,
            "crashed": False,
            "rss_bytes": 2 * 1024**3,
            "context_tokens": 100_000,
            "log_path": "/tmp/ds4/foo/ds4.log",
            "recent_logs": "stdout: ready\nstderr: warning",
        }


class _FakeLoadedDS4Engine:
    def __init__(self, *, active: bool = False, race_active: bool = False):
        self.active = active
        self.race_active = race_active
        self.restarted_context_tokens = None

    def has_active_requests(self) -> bool:
        return self.active

    async def restart_with_context(self, context_tokens: int | None) -> bool:
        if self.race_active:
            raise DS4ProxyError(
                "DS4 context change requires a backend restart, but the backend "
                "is currently serving another request; retry when idle"
            )
        self.restarted_context_tokens = context_tokens
        return True


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
async def test_update_model_settings_sets_ds4_context_and_restarts_loaded(
    monkeypatch,
):
    """DS4 per-model context overrides restart loaded DS4 engines."""
    pool = _ds4_pool()
    fake_engine = _FakeLoadedDS4Engine()
    pool.get_entry("foo").engine = fake_engine
    manager = MagicMock()
    manager.get_settings.return_value = ModelSettings()
    monkeypatch.setattr(admin_routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(admin_routes, "_get_settings_manager", lambda: manager)
    monkeypatch.setattr(admin_routes, "_get_server_state", lambda: None)

    result = await admin_routes.update_model_settings(
        "foo",
        admin_routes.ModelSettingsRequest(ds4_context_tokens=100_000),
        is_admin=True,
    )

    saved = manager.set_settings.call_args.args[1]
    assert saved.ds4_context_tokens == 100_000
    assert fake_engine.restarted_context_tokens == 100_000
    assert result["requires_reload"] is True
    assert result["auto_reloaded"] is True
    assert result["ds4_context_restarted"] is True


@pytest.mark.asyncio
async def test_update_model_settings_clears_ds4_context_to_auto(monkeypatch):
    """Null or non-positive DS4 context values clear back to auto."""
    pool = _ds4_pool()
    manager = MagicMock()
    manager.get_settings.return_value = ModelSettings(ds4_context_tokens=100_000)
    monkeypatch.setattr(admin_routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(admin_routes, "_get_settings_manager", lambda: manager)
    monkeypatch.setattr(admin_routes, "_get_server_state", lambda: None)

    result = await admin_routes.update_model_settings(
        "foo",
        admin_routes.ModelSettingsRequest(ds4_context_tokens=0),
        is_admin=True,
    )

    saved = manager.set_settings.call_args.args[1]
    assert saved.ds4_context_tokens is None
    assert result["settings"].get("ds4_context_tokens") is None


@pytest.mark.asyncio
async def test_update_model_settings_rejects_ds4_context_above_limit(monkeypatch):
    """DS4 context overrides are capped to the DS4-supported UI maximum."""
    pool = _ds4_pool()
    manager = MagicMock()
    manager.get_settings.return_value = ModelSettings()
    monkeypatch.setattr(admin_routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(admin_routes, "_get_settings_manager", lambda: manager)
    monkeypatch.setattr(admin_routes, "_get_server_state", lambda: None)

    with pytest.raises(HTTPException) as exc_info:
        await admin_routes.update_model_settings(
            "foo",
            admin_routes.ModelSettingsRequest(
                ds4_context_tokens=DS4_MAX_CONTEXT_TOKENS + 1
            ),
            is_admin=True,
        )

    assert exc_info.value.status_code == 400
    assert "ds4_context_tokens" in exc_info.value.detail
    manager.set_settings.assert_not_called()


@pytest.mark.asyncio
async def test_update_model_settings_rejects_ds4_context_when_active(monkeypatch):
    """Context restart avoids interrupting active DS4 proxy requests."""
    pool = _ds4_pool()
    pool.get_entry("foo").engine = _FakeLoadedDS4Engine(active=True)
    manager = MagicMock()
    manager.get_settings.return_value = ModelSettings()
    monkeypatch.setattr(admin_routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(admin_routes, "_get_settings_manager", lambda: manager)
    monkeypatch.setattr(admin_routes, "_get_server_state", lambda: None)

    with pytest.raises(HTTPException) as exc_info:
        await admin_routes.update_model_settings(
            "foo",
            admin_routes.ModelSettingsRequest(ds4_context_tokens=100_000),
            is_admin=True,
        )

    assert exc_info.value.status_code == 409
    assert "retry when idle" in exc_info.value.detail
    manager.set_settings.assert_not_called()


@pytest.mark.asyncio
async def test_update_model_settings_allows_noop_ds4_context_while_active(
    monkeypatch,
):
    """Full-form saves do not reject active DS4 when context is unchanged."""
    pool = _ds4_pool()
    pool.get_entry("foo").engine = _FakeLoadedDS4Engine(active=True)
    manager = MagicMock()
    manager.get_settings.return_value = ModelSettings(ds4_context_tokens=100_000)
    monkeypatch.setattr(admin_routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(admin_routes, "_get_settings_manager", lambda: manager)
    monkeypatch.setattr(admin_routes, "_get_server_state", lambda: None)

    result = await admin_routes.update_model_settings(
        "foo",
        admin_routes.ModelSettingsRequest(ds4_context_tokens=100_000),
        is_admin=True,
    )

    saved = manager.set_settings.call_args.args[1]
    assert saved.ds4_context_tokens == 100_000
    assert result["requires_reload"] is False
    assert result["ds4_context_restarted"] is False


@pytest.mark.asyncio
async def test_update_model_settings_does_not_persist_if_restart_races_active(
    monkeypatch,
):
    """A request becoming active during restart still returns 409 without saving."""
    pool = _ds4_pool()
    pool.get_entry("foo").engine = _FakeLoadedDS4Engine(race_active=True)
    manager = MagicMock()
    manager.get_settings.return_value = ModelSettings()
    monkeypatch.setattr(admin_routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(admin_routes, "_get_settings_manager", lambda: manager)
    monkeypatch.setattr(admin_routes, "_get_server_state", lambda: None)

    with pytest.raises(HTTPException) as exc_info:
        await admin_routes.update_model_settings(
            "foo",
            admin_routes.ModelSettingsRequest(ds4_context_tokens=100_000),
            is_admin=True,
        )

    assert exc_info.value.status_code == 409
    assert "retry when idle" in exc_info.value.detail
    manager.set_settings.assert_not_called()


@pytest.mark.asyncio
async def test_update_model_settings_rejects_ds4_context_for_non_ds4(monkeypatch):
    """DS4 context overrides are not accepted for MLX model entries."""
    pool = EnginePool()
    pool._entries["foo"] = EngineEntry(
        model_id="foo",
        model_path="/models/Foo",
        model_type="llm",
        engine_type="batched",
        estimated_size=1000,
    )
    manager = MagicMock()
    manager.get_settings.return_value = ModelSettings()
    monkeypatch.setattr(admin_routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(admin_routes, "_get_settings_manager", lambda: manager)
    monkeypatch.setattr(admin_routes, "_get_server_state", lambda: None)

    with pytest.raises(HTTPException) as exc_info:
        await admin_routes.update_model_settings(
            "foo",
            admin_routes.ModelSettingsRequest(ds4_context_tokens=100_000),
            is_admin=True,
        )

    assert exc_info.value.status_code == 400
    assert "DS4 GGUF" in exc_info.value.detail
    manager.set_settings.assert_not_called()


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


@pytest.mark.asyncio
async def test_list_models_exposes_ds4_admin_status(monkeypatch, tmp_path):
    """Admin model list includes DS4 lifecycle/log/context details."""
    gguf = tmp_path / "Foo.gguf"
    gguf.write_bytes(b"0" * 1000)
    pool = _ds4_pool(str(gguf))
    entry = pool.get_entry("foo")
    entry.engine = _FakeStatusDS4Engine()
    entry.actual_size = 2 * 1024**3
    manager = MagicMock()
    manager.get_all_settings.return_value = {
        "foo": ModelSettings(ds4_context_tokens=100_000)
    }
    monkeypatch.setattr(admin_routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(admin_routes, "_get_settings_manager", lambda: manager)
    monkeypatch.setattr(admin_routes, "_get_server_state", lambda: None)
    monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: None)

    result = await admin_routes.list_models(is_admin=True)

    model = result["models"][0]
    assert model["engine_type"] == "ds4"
    assert model["actual_size_formatted"] == "2.00 GB"
    assert model["settings"]["ds4_context_tokens"] == 100_000
    assert model["ds4"]["status"] == "running"
    assert model["ds4"]["running"] is True
    assert model["ds4"]["port"] == 49152
    assert model["ds4"]["rss_formatted"] == "2.00 GB"
    assert model["ds4"]["context_tokens"] == 100_000
    assert model["ds4"]["context_tokens_formatted"] == "100,000"
    assert model["ds4"]["log_path"] == "/tmp/ds4/foo/ds4.log"
    assert model["ds4"]["recent_log_lines"] == [
        "stdout: ready",
        "stderr: warning",
    ]


def test_model_settings_modal_exposes_ds4_context_controls():
    """Admin UI exposes DS4 context controls only for DS4-backed models."""
    template = (
        _PROJECT_ROOT
        / "omlx/admin/templates/dashboard/_modal_model_settings.html"
    ).read_text(encoding="utf-8")

    assert "selectedModel?.engine_type === 'ds4'" in template
    assert 'x-model.number="modelSettings.ds4_context_tokens"' in template
    assert 'max="1000000"' in template
    assert ':disabled="selectedModel?.engine_type === \'ds4\'"' in template
    assert "modal.model_settings.ds4_context_tokens" in template


def test_dashboard_saves_ds4_context_only_for_ds4_models():
    """Frontend payload avoids sending DS4-only fields for non-DS4 models."""
    js = (_PROJECT_ROOT / "omlx/admin/static/js/dashboard.js").read_text(
        encoding="utf-8"
    )

    assert "ds4_context_tokens: settings.ds4_context_tokens ?? null" in js
    assert "model_type_override: model.engine_type === 'ds4' ? ''" in js
    assert "model_type_override: this.selectedModel?.engine_type === 'ds4'" in js
    assert "this.selectedModel?.engine_type === 'ds4'" in js
    assert "{ ds4_context_tokens: this.modelSettings.ds4_context_tokens || null }" in js


def test_ds4_context_ui_strings_are_localized():
    """All admin locales include the DS4 context-control strings."""
    required_keys = {
        "modal.model_settings.ds4_model_type_hint",
        "modal.model_settings.ds4_section_label",
        "modal.model_settings.ds4_context_hint",
        "modal.model_settings.ds4_context_tokens",
        "modal.model_settings.ds4_context_auto",
        "modal.model_settings.ds4_status_context",
        "modal.model_settings.ds4_status_rss",
        "modal.model_settings.ds4_status_log",
    }
    for path in (_PROJECT_ROOT / "omlx/admin/i18n").glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert required_keys <= set(data), path.name
