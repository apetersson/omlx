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
from omlx.model_settings import ModelSettings, ModelSettingsManager
from omlx.settings import DS4_MAX_CONTEXT_TOKENS, GlobalSettings

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


def _memory_info(total_bytes: int) -> dict[str, int | str]:
    return {
        "total_bytes": total_bytes,
        "total_formatted": "128.00 GB",
        "auto_limit_formatted": "102.40 GB",
        "available_bytes": 0,
        "omlx_phys_footprint_bytes": 0,
        "free_memory_bytes": 0,
        "inactive_memory_bytes": 0,
        "active_memory_bytes": 0,
        "iogpu_wired_limit_bytes": 0,
        "omlx_wired_limit_request_bytes": 0,
    }


@pytest.mark.asyncio
async def test_get_global_settings_exposes_ds4_backend_controls(monkeypatch, tmp_path):
    """Global settings API includes DS4 defaults, paths, and lifecycle status."""
    settings = GlobalSettings(base_path=tmp_path)
    settings.cache.ssd_cache_max_size = "1GB"
    settings.ds4.kv_disk_space_mb = 1234
    settings.ds4.kv_cache_continued_interval_tokens = 4096
    settings.ds4.ssd_streaming = "on"
    settings.ds4.power = 77
    pool = _ds4_pool()
    pool.get_entry("foo").engine = _FakeStatusDS4Engine()
    monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: settings)
    monkeypatch.setattr(admin_routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(
        admin_routes,
        "get_system_memory_info",
        lambda: _memory_info(128 * 1024**3),
    )
    monkeypatch.setattr(
        admin_routes,
        "get_ssd_disk_info",
        lambda _path: {"total_bytes": 10**12, "total_formatted": "1.00 TB"},
    )

    result = await admin_routes.get_global_settings(is_admin=True)

    ds4 = result["ds4"]
    assert ds4["enabled"] is True
    assert ds4["support_dir"] == str(tmp_path / "support" / "ds4")
    assert ds4["context_default_tokens"] is None
    assert ds4["auto_context_tokens"] == 100_000
    assert ds4["auto_context_tokens_formatted"] == "100,000"
    assert ds4["kv_cache_enabled"] is True
    assert ds4["kv_root"] == str(tmp_path / "ds4-kv")
    assert ds4["kv_disk_space_mb"] == 1234
    assert ds4["kv_cache_continued_interval_tokens"] == 4096
    assert ds4["ssd_streaming"] == "on"
    assert ds4["power"] == 77
    assert ds4["status"] == "running"
    assert ds4["available_models"] == 1
    assert ds4["loaded_count"] == 1
    assert ds4["running_count"] == 1
    assert ds4["loaded_models"][0]["id"] == "foo"


@pytest.mark.asyncio
async def test_update_global_settings_saves_ds4_backend_controls(monkeypatch, tmp_path):
    """Global settings save path persists DS4 launch controls."""
    settings = GlobalSettings(base_path=tmp_path)
    save = MagicMock()
    monkeypatch.setattr(settings, "save", save)
    monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: settings)

    result = await admin_routes.update_global_settings(
        admin_routes.GlobalSettingsRequest(
            ds4_enabled=False,
            ds4_support_dir="  /tmp/ds4-support  ",
            ds4_context_default_tokens=100_000,
            ds4_ready_timeout_ms=42,
            ds4_kv_cache_enabled=False,
            ds4_kv_root="  /tmp/ds4-kv  ",
            ds4_kv_disk_space_mb=1234,
            ds4_kv_cache_continued_interval_tokens=4096,
            ds4_ssd_streaming="on",
            ds4_power=77,
        ),
        is_admin=True,
    )

    assert result["success"] is True
    assert "ds4" in result["runtime_applied"]
    assert settings.ds4.enabled is False
    assert settings.ds4.support_dir == "/tmp/ds4-support"
    assert settings.ds4.context_default_tokens == 100_000
    assert settings.ds4.ready_timeout_ms == 42
    assert settings.ds4.kv_cache_enabled is False
    assert settings.ds4.kv_root == "/tmp/ds4-kv"
    assert settings.ds4.kv_disk_space_mb == 1234
    assert settings.ds4.kv_cache_continued_interval_tokens == 4096
    assert settings.ds4.ssd_streaming == "on"
    assert settings.ds4.power == 77
    save.assert_called_once()


@pytest.mark.asyncio
async def test_update_global_settings_clears_ds4_context_to_auto(
    monkeypatch,
    tmp_path,
):
    """Global DS4 context default supports returning to adaptive auto mode."""
    settings = GlobalSettings(base_path=tmp_path)
    settings.ds4.context_default_tokens = 100_000
    save = MagicMock()
    monkeypatch.setattr(settings, "save", save)
    monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: settings)

    await admin_routes.update_global_settings(
        admin_routes.GlobalSettingsRequest(ds4_context_default_tokens=0),
        is_admin=True,
    )

    assert settings.ds4.context_default_tokens is None
    save.assert_called_once()


@pytest.mark.asyncio
async def test_update_global_settings_rejects_invalid_ds4_controls(
    monkeypatch,
    tmp_path,
):
    """DS4 global settings still flow through central validation."""
    settings = GlobalSettings(base_path=tmp_path)
    settings.ds4.power = 77
    save = MagicMock()
    monkeypatch.setattr(settings, "save", save)
    monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: settings)

    with pytest.raises(HTTPException) as exc_info:
        await admin_routes.update_global_settings(
            admin_routes.GlobalSettingsRequest(ds4_power=101),
            is_admin=True,
        )

    assert exc_info.value.status_code == 400
    assert "ds4.power" in str(exc_info.value.detail)
    assert settings.ds4.power == 77
    save.assert_not_called()


@pytest.mark.asyncio
async def test_update_global_settings_does_not_mutate_ds4_on_other_rejections(
    monkeypatch,
    tmp_path,
):
    """Rejected non-DS4 fields do not leak paired DS4 changes into memory."""
    settings = GlobalSettings(base_path=tmp_path)
    settings.ds4.power = 77
    save = MagicMock()
    monkeypatch.setattr(settings, "save", save)
    monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: settings)

    with pytest.raises(HTTPException) as exc_info:
        await admin_routes.update_global_settings(
            admin_routes.GlobalSettingsRequest(
                ds4_power=66,
                markitdown_max_file_size_mb=0,
            ),
            is_admin=True,
        )

    assert exc_info.value.status_code == 400
    assert "markitdown_max_file_size_mb" in exc_info.value.detail
    assert settings.ds4.power == 77
    save.assert_not_called()


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
async def test_update_model_settings_sets_ds4_max_context_and_restarts_loaded(
    monkeypatch,
):
    """DS4 context windows restart loaded DS4 engines."""
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
        admin_routes.ModelSettingsRequest(max_context_window=100_000),
        is_admin=True,
    )

    saved = manager.set_settings.call_args.args[1]
    assert saved.max_context_window == 100_000
    assert fake_engine.restarted_context_tokens == 100_000
    assert result["requires_reload"] is True
    assert result["auto_reloaded"] is True
    assert result["ds4_context_restarted"] is True


@pytest.mark.asyncio
async def test_update_model_settings_clears_ds4_max_context_to_auto(monkeypatch):
    """Null or non-positive DS4 context values clear back to auto."""
    pool = _ds4_pool()
    manager = MagicMock()
    manager.get_settings.return_value = ModelSettings(max_context_window=100_000)
    monkeypatch.setattr(admin_routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(admin_routes, "_get_settings_manager", lambda: manager)
    monkeypatch.setattr(admin_routes, "_get_server_state", lambda: None)

    result = await admin_routes.update_model_settings(
        "foo",
        admin_routes.ModelSettingsRequest(max_context_window=0),
        is_admin=True,
    )

    saved = manager.set_settings.call_args.args[1]
    assert saved.max_context_window is None
    assert result["settings"].get("max_context_window") is None


@pytest.mark.asyncio
async def test_update_model_settings_rejects_ds4_max_context_above_limit(monkeypatch):
    """DS4 context windows are capped to the DS4-supported UI maximum."""
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
                max_context_window=DS4_MAX_CONTEXT_TOKENS + 1
            ),
            is_admin=True,
        )

    assert exc_info.value.status_code == 400
    assert "max_context_window" in exc_info.value.detail
    manager.set_settings.assert_not_called()


@pytest.mark.asyncio
async def test_update_model_settings_rejects_ds4_max_context_when_active(monkeypatch):
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
            admin_routes.ModelSettingsRequest(max_context_window=100_000),
            is_admin=True,
        )

    assert exc_info.value.status_code == 409
    assert "retry when idle" in exc_info.value.detail
    manager.set_settings.assert_not_called()


@pytest.mark.asyncio
async def test_update_model_settings_allows_noop_ds4_max_context_while_active(
    monkeypatch,
):
    """Full-form saves do not reject active DS4 when context is unchanged."""
    pool = _ds4_pool()
    pool.get_entry("foo").engine = _FakeLoadedDS4Engine(active=True)
    manager = MagicMock()
    manager.get_settings.return_value = ModelSettings(max_context_window=100_000)
    monkeypatch.setattr(admin_routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(admin_routes, "_get_settings_manager", lambda: manager)
    monkeypatch.setattr(admin_routes, "_get_server_state", lambda: None)

    result = await admin_routes.update_model_settings(
        "foo",
        admin_routes.ModelSettingsRequest(max_context_window=100_000),
        is_admin=True,
    )

    saved = manager.set_settings.call_args.args[1]
    assert saved.max_context_window == 100_000
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
            admin_routes.ModelSettingsRequest(max_context_window=100_000),
            is_admin=True,
        )

    assert exc_info.value.status_code == 409
    assert "retry when idle" in exc_info.value.detail
    manager.set_settings.assert_not_called()


@pytest.mark.asyncio
async def test_apply_profile_restarts_loaded_ds4_on_max_context_window(
    monkeypatch,
    tmp_path,
):
    """Profile-applied max_context_window also restarts loaded DS4 engines."""
    pool = _ds4_pool()
    fake_engine = _FakeLoadedDS4Engine()
    pool.get_entry("foo").engine = fake_engine
    manager = ModelSettingsManager(tmp_path)
    manager.set_settings("foo", ModelSettings(max_context_window=32_768))
    manager.save_profile(
        "foo",
        "long-context",
        "Long Context",
        None,
        {"max_context_window": 100_000},
    )
    monkeypatch.setattr(admin_routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(admin_routes, "_get_settings_manager", lambda: manager)

    result = await admin_routes.apply_model_profile(
        "foo",
        "long-context",
        is_admin=True,
    )

    assert result["settings"]["max_context_window"] == 100_000
    assert result["ds4_context_restarted"] is True
    assert fake_engine.restarted_context_tokens == 100_000
    assert manager.get_settings("foo").max_context_window == 100_000


@pytest.mark.asyncio
async def test_update_model_settings_accepts_max_context_for_non_ds4(monkeypatch):
    """max_context_window remains the shared context setting for MLX models."""
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

    result = await admin_routes.update_model_settings(
        "foo",
        admin_routes.ModelSettingsRequest(max_context_window=100_000),
        is_admin=True,
    )

    saved = manager.set_settings.call_args.args[1]
    assert saved.max_context_window == 100_000
    assert result["settings"]["max_context_window"] == 100_000


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
async def test_local_models_manager_lists_ds4_gguf_files(monkeypatch, tmp_path):
    """Models Manager's local list includes discovered DS4 GGUF files."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    gguf = model_dir / "DeepSeek-V4-Flash-IQ2XXS-w2Q2K.gguf"
    gguf.write_bytes(b"g" * 2048)
    mlx_dir = model_dir / "Qwen-MLX"
    mlx_dir.mkdir()
    (mlx_dir / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (mlx_dir / "model.safetensors").write_bytes(b"m" * 1024)

    pool = EnginePool()
    pool.discover_models(str(model_dir))
    settings = MagicMock()
    settings.base_path = tmp_path
    settings.model.get_model_dirs.return_value = [model_dir]
    monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: settings)
    monkeypatch.setattr(admin_routes, "_get_engine_pool", lambda: pool)

    result = await admin_routes.list_hf_models(is_admin=True)
    by_name = {model["name"]: model for model in result["models"]}

    assert "Qwen-MLX" in by_name
    ds4_id = "deepseek-v4-flash-iq2xxs-w2q2k"
    assert ds4_id in by_name
    assert by_name[ds4_id]["display_name"] == "DeepSeek-V4-Flash-IQ2XXS-w2Q2K"
    assert by_name[ds4_id]["engine_type"] == "ds4"
    assert by_name[ds4_id]["backend_label"] == "DS4-GGUF"
    assert by_name[ds4_id]["path"] == str(gguf)
    assert by_name[ds4_id]["size"] == 2048


@pytest.mark.asyncio
async def test_local_models_manager_deletes_ds4_gguf_file(monkeypatch, tmp_path):
    """Deleting a listed DS4 GGUF removes the file and refreshes discovery."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    gguf = model_dir / "DeepSeek-V4-Flash-IQ2XXS-w2Q2K.gguf"
    gguf.write_bytes(b"g" * 2048)
    pool = EnginePool()
    pool.discover_models(str(model_dir))
    ds4_id = "deepseek-v4-flash-iq2xxs-w2q2k"

    settings = MagicMock()
    settings.base_path = tmp_path
    settings.model.get_model_dirs.return_value = [model_dir]
    settings.get_effective_model_dirs.return_value = [model_dir]
    settings_manager = MagicMock()
    settings_manager.get_pinned_model_ids.return_value = []
    monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: settings)
    monkeypatch.setattr(admin_routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(admin_routes, "_get_settings_manager", lambda: settings_manager)

    result = await admin_routes.delete_hf_model(model_name=ds4_id, is_admin=True)

    assert result["success"] is True
    assert not gguf.exists()
    assert ds4_id not in pool.get_model_ids()
    settings_manager.delete_settings.assert_called_once_with(ds4_id)


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
        "foo": ModelSettings(max_context_window=100_000)
    }
    monkeypatch.setattr(admin_routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(admin_routes, "_get_settings_manager", lambda: manager)
    monkeypatch.setattr(admin_routes, "_get_server_state", lambda: None)
    monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: None)

    result = await admin_routes.list_models(is_admin=True)

    model = result["models"][0]
    assert model["engine_type"] == "ds4"
    assert model["actual_size_formatted"] == "2.00 GB"
    assert model["settings"]["max_context_window"] == 100_000
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


def test_models_manager_can_render_ds4_gguf_entries():
    """Models Manager local list can show discovered DS4 GGUF files."""
    template = (_PROJECT_ROOT / "omlx/admin/templates/dashboard/_models.html").read_text(
        encoding="utf-8"
    )

    assert "model.display_name || model.name" in template
    assert "model.backend_label" in template
    assert "DS4-GGUF" in (_PROJECT_ROOT / "omlx/admin/routes.py").read_text(
        encoding="utf-8"
    )


def test_model_settings_modal_exposes_ds4_supported_controls_only():
    """Admin UI hides unsupported MLX-only controls for DS4-backed models."""
    template = (
        _PROJECT_ROOT
        / "omlx/admin/templates/dashboard/_modal_model_settings.html"
    ).read_text(encoding="utf-8")

    assert "selectedModel?.engine_type === 'ds4'" in template
    assert 'x-model.number="modelSettings.max_context_window"' in template
    assert ':max="selectedModel?.engine_type === \'ds4\' ? 1000000 : null"' in template
    assert "modal.model_settings.max_context_window" in template
    assert "modelSettings." + "ds4_" + "context_tokens" not in template
    assert "selectedModel?.engine_type !== 'ds4'" in template
    assert "selectedModel?.engine_type !== 'ds4' && reasoningParsers.length > 0" in template
    assert "selectedModel?.engine_type !== 'ds4' && (!selectedModel?.model_type" in template


def test_dashboard_saves_ds4_supported_settings_only():
    """Frontend payload sends only DS4-supported settings for DS4 models."""
    js = (_PROJECT_ROOT / "omlx/admin/static/js/dashboard.js").read_text(
        encoding="utf-8"
    )

    assert "max_context_window: settings.max_context_window || null" in js
    assert "model_type_override: model.engine_type === 'ds4' ? ''" in js
    assert "const isDs4 = this.selectedModel?.engine_type === 'ds4'" in js
    assert "if (isDs4)" in js
    assert "max_context_window: this.modelSettings.max_context_window || null" in js
    assert "ds4_" + "context_tokens" not in js
    assert "repetition_penalty: null" in js
    assert "presence_penalty: null" in js
    assert "force_sampling: false" in js
    assert "guided_grammar_enabled: false" in js


def test_dashboard_formats_ds4_activity_phase_tps_metadata():
    """Active Models activity rows can display DS4 phase/token-rate metadata."""
    js = (_PROJECT_ROOT / "omlx/admin/static/js/dashboard.js").read_text(
        encoding="utf-8"
    )

    assert "activity.current_tokens" in js
    assert "activity.total_tokens" in js
    assert "activity.tokens_per_second" in js
    assert "activity.chunk_tokens_per_second" in js


def test_global_settings_ui_exposes_ds4_backend_controls():
    """Admin global settings includes DS4 backend status and launch controls."""
    template = (_PROJECT_ROOT / "omlx/admin/templates/dashboard/_settings.html").read_text(
        encoding="utf-8"
    )

    assert "settings.ds4.section_label" in template
    assert "globalSettings.ds4.status" in template
    assert "globalSettings.ds4.enabled" in template
    assert "globalSettings.ds4.support_dir" in template
    assert "globalSettings.ds4.context_default_tokens" in template
    assert "globalSettings.ds4.kv_cache_enabled" in template
    assert "globalSettings.ds4.kv_root" in template
    assert "globalSettings.ds4.kv_disk_space_mb" in template
    assert "globalSettings.ds4.ssd_streaming" in template
    assert "globalSettings.ds4.power" in template


def test_dashboard_saves_ds4_global_settings():
    """Frontend payload persists DS4 global backend settings."""
    js = (_PROJECT_ROOT / "omlx/admin/static/js/dashboard.js").read_text(
        encoding="utf-8"
    )

    assert "ds4: { ...this.globalSettings.ds4, ...data.ds4 }" in js
    assert "ds4_enabled: this.globalSettings.ds4.enabled" in js
    assert "ds4_support_dir: this.globalSettings.ds4.support_dir" in js
    assert "ds4_context_default_tokens: this.globalSettings.ds4.context_default_tokens || null" in js
    assert "ds4_kv_cache_enabled: this.globalSettings.ds4.kv_cache_enabled" in js
    assert "ds4_kv_root: this.globalSettings.ds4.kv_root" in js
    assert "ds4_kv_disk_space_mb: this.globalSettings.ds4.kv_disk_space_mb" in js
    assert "ds4_ssd_streaming: this.globalSettings.ds4.ssd_streaming" in js
    assert "ds4_power: this.globalSettings.ds4.power" in js


def test_ds4_context_ui_strings_are_localized():
    """All admin locales include the DS4 context-control strings."""
    required_keys = {
        "modal.model_settings.ds4_model_type_hint",
        "modal.model_settings.ds4_section_label",
        "modal.model_settings.ds4_context_hint",
        "modal.model_settings.max_context_window",
        "modal.model_settings.ds4_status_context",
        "modal.model_settings.ds4_status_rss",
        "modal.model_settings.ds4_status_log",
        "settings.ds4.section_label",
        "settings.ds4.status_label",
        "settings.ds4.enabled",
        "settings.ds4.support_dir",
        "settings.ds4.context_default",
        "settings.ds4.context_auto",
        "settings.ds4.kv_cache_enabled",
        "settings.ds4.kv_root",
        "settings.ds4.kv_disk_space",
        "settings.ds4.ssd_streaming",
        "settings.ds4.power",
        "settings.ds4.next_launch_badge",
    }
    for path in (_PROJECT_ROOT / "omlx/admin/i18n").glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert required_keys <= set(data), path.name
