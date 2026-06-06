# SPDX-License-Identifier: Apache-2.0
"""Tests for DS4 EnginePool lifecycle integration."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from omlx.engine.ds4 import DS4ProcessEngine, DS4ProxyError, DS4ProxyResponse
from omlx.engine_pool import EnginePool
from omlx.server import ServerState, app
from omlx.settings import DS4_THINK_MAX_CONTEXT_TOKENS, DS4Settings


class FakeManagedProcess:
    """Small stand-in for DS4ManagedProcess that avoids spawning subprocesses."""

    instances: list[FakeManagedProcess] = []

    def __init__(self, config, *, max_log_lines: int = 500):
        self.config = config
        self.max_log_lines = max_log_lines
        self.process = None
        self.port = None
        self.command = None
        self.log_path = None
        self.logs = []
        self.started = False
        self.stopped = False
        FakeManagedProcess.instances.append(self)

    @property
    def is_running(self) -> bool:
        return (
            self.started
            and self.process is not None
            and self.process.returncode is None
        )

    async def start(self) -> None:
        self.started = True
        self.port = self.config.port or 49152
        self.command = self.config.build_command(self.port)
        self.log_path = self.config.log_path
        self.process = SimpleNamespace(pid=12345, returncode=None)

    async def stop(self) -> None:
        self.stopped = True
        if self.process is not None:
            self.process.returncode = 0

    def recent_log_text(self) -> str:
        return "fake ds4 logs"

    def crash(self, returncode: int = 9) -> None:
        if self.process is not None:
            self.process.returncode = returncode


def _patch_fake_process(monkeypatch):
    FakeManagedProcess.instances = []
    monkeypatch.setattr("omlx.engine.ds4.DS4ManagedProcess", FakeManagedProcess)


def _pool_with_ds4(tmp_path, *, ds4_enabled: bool = True) -> EnginePool:
    (tmp_path / "Foo.gguf").write_bytes(b"0" * 1000)
    pool = EnginePool(
        base_path=tmp_path,
        ds4_settings=DS4Settings(
            support_dir=str(tmp_path / "support" / "ds4"),
            kv_root=str(tmp_path / "kv"),
            debug_dir=str(tmp_path / "debug"),
            enabled=ds4_enabled,
        ),
    )
    pool._get_final_ceiling = lambda: 0
    pool.discover_models(str(tmp_path))
    return pool


@pytest.mark.asyncio
async def test_ds4_process_engine_starts_and_stops_fake_process(monkeypatch, tmp_path):
    _patch_fake_process(monkeypatch)
    gguf = tmp_path / "Foo.gguf"
    gguf.write_bytes(b"0" * 1000)
    engine = DS4ProcessEngine(
        model_id="foo",
        model_path=gguf,
        settings=DS4Settings(
            support_dir=str(tmp_path / "support" / "ds4"),
            kv_root=str(tmp_path / "kv"),
        ),
        base_path=tmp_path,
    )

    await engine.start()
    try:
        assert engine.is_running is True
        assert engine.port == 49152
        assert engine.pid == 12345
        stats = engine.get_stats()
        assert stats["backend"] == "ds4"
        assert stats["host"] == "127.0.0.1"
        assert stats["port"] == 49152
        assert stats["running"] is True
        assert stats["log_path"] == str(
            tmp_path / "logs" / "ds4-debug" / "foo" / "ds4.log"
        )
        assert stats["recent_logs"] == "fake ds4 logs"
    finally:
        await engine.stop()

    assert engine.is_running is False
    assert FakeManagedProcess.instances[0].stopped is True


@pytest.mark.asyncio
async def test_ds4_process_engine_raises_context_for_think_max(
    monkeypatch, tmp_path
):
    _patch_fake_process(monkeypatch)
    gguf = tmp_path / "Foo.gguf"
    gguf.write_bytes(b"0" * 1000)
    settings = DS4Settings(
        context_default_tokens=32_768,
        support_dir=str(tmp_path / "support" / "ds4"),
        kv_root=str(tmp_path / "kv"),
    )
    engine = DS4ProcessEngine(
        model_id="foo",
        model_path=gguf,
        settings=settings,
        base_path=tmp_path,
    )

    await engine.start()
    raised = await engine.ensure_min_context(DS4_THINK_MAX_CONTEXT_TOKENS)

    try:
        assert raised is True
        assert engine.context_tokens == DS4_THINK_MAX_CONTEXT_TOKENS
        assert settings.context_default_tokens == 32_768
        assert len(FakeManagedProcess.instances) == 2
        assert FakeManagedProcess.instances[0].stopped is True
        assert FakeManagedProcess.instances[1].started is True
        assert FakeManagedProcess.instances[1].config.context_tokens == (
            DS4_THINK_MAX_CONTEXT_TOKENS
        )
        assert "--ctx" in FakeManagedProcess.instances[1].command
        assert str(DS4_THINK_MAX_CONTEXT_TOKENS) in FakeManagedProcess.instances[1].command
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_ds4_process_engine_serializes_concurrent_context_raises(
    monkeypatch, tmp_path
):
    _patch_fake_process(monkeypatch)

    async def slow_stop(self):
        await asyncio.sleep(0.01)
        self.stopped = True
        self.running = False

    monkeypatch.setattr(FakeManagedProcess, "stop", slow_stop)
    gguf = tmp_path / "Foo.gguf"
    gguf.write_bytes(b"0" * 1000)
    engine = DS4ProcessEngine(
        model_id="foo",
        model_path=gguf,
        settings=DS4Settings(
            context_default_tokens=32_768,
            support_dir=str(tmp_path / "support" / "ds4"),
            kv_root=str(tmp_path / "kv"),
        ),
        base_path=tmp_path,
    )

    await engine.start()
    try:
        results = await asyncio.gather(
            engine.ensure_min_context(DS4_THINK_MAX_CONTEXT_TOKENS),
            engine.ensure_min_context(DS4_THINK_MAX_CONTEXT_TOKENS),
        )
        assert results == [True, False]
        assert len(FakeManagedProcess.instances) == 2
        assert FakeManagedProcess.instances[0].stopped is True
        assert FakeManagedProcess.instances[1].started is True
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_ds4_process_engine_think_max_context_noops_when_already_high(
    monkeypatch, tmp_path
):
    _patch_fake_process(monkeypatch)
    gguf = tmp_path / "Foo.gguf"
    gguf.write_bytes(b"0" * 1000)
    engine = DS4ProcessEngine(
        model_id="foo",
        model_path=gguf,
        settings=DS4Settings(
            context_default_tokens=DS4_THINK_MAX_CONTEXT_TOKENS,
            support_dir=str(tmp_path / "support" / "ds4"),
            kv_root=str(tmp_path / "kv"),
        ),
        base_path=tmp_path,
    )

    await engine.start()
    try:
        raised = await engine.ensure_min_context(DS4_THINK_MAX_CONTEXT_TOKENS)
        assert raised is False
        assert len(FakeManagedProcess.instances) == 1
        assert FakeManagedProcess.instances[0].stopped is False
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_ds4_process_engine_rejects_context_raise_while_active(
    monkeypatch, tmp_path
):
    _patch_fake_process(monkeypatch)
    gguf = tmp_path / "Foo.gguf"
    gguf.write_bytes(b"0" * 1000)
    engine = DS4ProcessEngine(
        model_id="foo",
        model_path=gguf,
        settings=DS4Settings(
            context_default_tokens=32_768,
            support_dir=str(tmp_path / "support" / "ds4"),
            kv_root=str(tmp_path / "kv"),
        ),
        base_path=tmp_path,
    )

    await engine.start()
    engine._increment_active_requests()
    try:
        with pytest.raises(DS4ProxyError, match="retry when idle"):
            await engine.ensure_min_context(DS4_THINK_MAX_CONTEXT_TOKENS)
        assert len(FakeManagedProcess.instances) == 1
        assert FakeManagedProcess.instances[0].stopped is False
    finally:
        engine._decrement_active_requests()
        await engine.stop()


@pytest.mark.asyncio
async def test_ds4_process_engine_restarts_crashed_backend_before_request(
    monkeypatch, tmp_path
):
    _patch_fake_process(monkeypatch)
    gguf = tmp_path / "Foo.gguf"
    gguf.write_bytes(b"0" * 1000)
    engine = DS4ProcessEngine(
        model_id="foo",
        model_path=gguf,
        settings=DS4Settings(
            support_dir=str(tmp_path / "support" / "ds4"),
            kv_root=str(tmp_path / "kv"),
        ),
        base_path=tmp_path,
    )

    def fake_proxy_response(self, path, body):
        try:
            assert self.is_running is True
            assert path == "/v1/chat/completions"
            assert body == {"model": "foo"}
            return DS4ProxyResponse(
                status_code=200,
                headers={"content-type": "application/json"},
                body=b'{"ok":true}',
            )
        finally:
            self._decrement_active_requests()

    monkeypatch.setattr(
        DS4ProcessEngine,
        "_proxy_json_response_blocking",
        fake_proxy_response,
    )

    await engine.start()
    FakeManagedProcess.instances[0].crash(returncode=9)

    response = await engine.proxy_chat_completion({"model": "foo"})

    try:
        assert response.body == b'{"ok":true}'
        assert len(FakeManagedProcess.instances) == 2
        assert FakeManagedProcess.instances[0].stopped is True
        assert FakeManagedProcess.instances[1].started is True
        stats = engine.get_stats()
        assert stats["running"] is True
        assert stats["crashed"] is False
        assert stats["crash_count"] == 1
        assert stats["restart_count"] == 1
        assert stats["last_crash_exit_code"] == 9
        assert stats["last_crash_logs"] == "fake ds4 logs"
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_ds4_process_engine_rejects_crash_restart_while_active(
    monkeypatch, tmp_path
):
    _patch_fake_process(monkeypatch)
    gguf = tmp_path / "Foo.gguf"
    gguf.write_bytes(b"0" * 1000)
    engine = DS4ProcessEngine(
        model_id="foo",
        model_path=gguf,
        settings=DS4Settings(
            support_dir=str(tmp_path / "support" / "ds4"),
            kv_root=str(tmp_path / "kv"),
        ),
        base_path=tmp_path,
    )

    await engine.start()
    FakeManagedProcess.instances[0].crash(returncode=9)
    engine._increment_active_requests()
    try:
        with pytest.raises(DS4ProxyError, match="retry when idle"):
            await engine.restart_if_crashed()
        assert len(FakeManagedProcess.instances) == 1
        assert FakeManagedProcess.instances[0].stopped is False
    finally:
        engine._decrement_active_requests()
        await engine.stop()


@pytest.mark.asyncio
async def test_ds4_process_engine_protocol_methods_are_explicitly_deferred(tmp_path):
    gguf = tmp_path / "Foo.gguf"
    gguf.write_bytes(b"0" * 1000)
    engine = DS4ProcessEngine(model_id="foo", model_path=gguf, base_path=tmp_path)

    with pytest.raises(RuntimeError, match="protocol forwarding"):
        await engine.chat([])
    with pytest.raises(RuntimeError, match="protocol forwarding"):
        await engine.generate("hello")


@pytest.mark.asyncio
async def test_engine_pool_loads_and_unloads_ds4_entries(monkeypatch, tmp_path):
    _patch_fake_process(monkeypatch)
    pool = _pool_with_ds4(tmp_path)

    engine = await pool.get_engine("foo")

    assert isinstance(engine, DS4ProcessEngine)
    assert engine.is_running is True
    assert pool.get_loaded_model_ids() == ["foo"]
    entry = pool.get_entry("foo")
    assert entry is not None
    assert entry.engine is engine
    assert entry.actual_size is not None
    status = pool.get_status()["models"][0]
    assert status["id"] == "foo"
    assert status["loaded"] is True
    assert status["engine_type"] == "ds4"
    assert status["ds4"]["running"] is True
    assert status["ds4"]["port"] == 49152

    await pool._unload_engine("foo")

    assert entry.engine is None
    assert pool.get_loaded_model_ids() == []
    assert FakeManagedProcess.instances[0].stopped is True


@pytest.mark.asyncio
async def test_engine_pool_preloads_pinned_ds4_models(monkeypatch, tmp_path):
    _patch_fake_process(monkeypatch)
    (tmp_path / "Foo.gguf").write_bytes(b"0" * 1000)
    pool = EnginePool(base_path=tmp_path, ds4_settings=DS4Settings())
    pool._get_final_ceiling = lambda: 0
    pool.discover_models(str(tmp_path), pinned_models=["foo"])

    await pool.preload_pinned_models()

    assert pool.get_entry("foo").engine is not None
    assert FakeManagedProcess.instances[0].started is True


@pytest.mark.asyncio
async def test_engine_pool_restarts_crashed_pinned_ds4_models(
    monkeypatch, tmp_path
):
    _patch_fake_process(monkeypatch)
    (tmp_path / "Foo.gguf").write_bytes(b"0" * 1000)
    pool = EnginePool(base_path=tmp_path, ds4_settings=DS4Settings())
    pool._get_final_ceiling = lambda: 0
    pool.discover_models(str(tmp_path), pinned_models=["foo"])
    await pool.preload_pinned_models()

    FakeManagedProcess.instances[0].crash(returncode=7)
    restarted = await pool.restart_crashed_pinned_ds4()

    assert restarted == ["foo"]
    assert len(FakeManagedProcess.instances) == 2
    assert FakeManagedProcess.instances[0].stopped is True
    assert FakeManagedProcess.instances[1].started is True
    status = pool.get_status()["models"][0]
    assert status["ds4"]["running"] is True
    assert status["ds4"]["restart_count"] == 1
    assert status["ds4"]["last_crash_exit_code"] == 7


@pytest.mark.asyncio
async def test_engine_pool_leaves_unpinned_crashed_ds4_stopped_until_request(
    monkeypatch, tmp_path
):
    _patch_fake_process(monkeypatch)
    pool = _pool_with_ds4(tmp_path)
    await pool.get_engine("foo")

    FakeManagedProcess.instances[0].crash(returncode=8)
    restarted = await pool.restart_crashed_pinned_ds4()

    assert restarted == []
    assert len(FakeManagedProcess.instances) == 1
    status = pool.get_status()["models"][0]
    assert status["ds4"]["running"] is False
    assert status["ds4"]["crashed"] is True
    assert status["ds4"]["exit_code"] == 8
    assert status["ds4"]["crash_count"] == 1
    assert status["ds4"]["last_crash_exit_code"] == 8
    assert status["ds4"]["last_crash_logs"] == "fake ds4 logs"

    status_again = pool.get_status()["models"][0]
    assert status_again["ds4"]["crash_count"] == 1


@pytest.mark.asyncio
async def test_ttl_check_restarts_crashed_pinned_ds4_models(
    monkeypatch, tmp_path
):
    _patch_fake_process(monkeypatch)
    (tmp_path / "Foo.gguf").write_bytes(b"0" * 1000)
    pool = EnginePool(base_path=tmp_path, ds4_settings=DS4Settings())
    pool._get_final_ceiling = lambda: 0
    pool.discover_models(str(tmp_path), pinned_models=["foo"])
    await pool.preload_pinned_models()

    class SettingsManager:
        def get_settings(self, model_id):
            from omlx.model_settings import ModelSettings

            return ModelSettings(ttl_seconds=0)

    FakeManagedProcess.instances[0].crash(returncode=7)
    expired = await pool.check_ttl_expirations(SettingsManager())

    assert expired == []
    assert len(FakeManagedProcess.instances) == 2
    assert FakeManagedProcess.instances[1].started is True


@pytest.mark.asyncio
async def test_unpinned_ds4_process_can_be_evicted_under_memory_ceiling(
    monkeypatch, tmp_path
):
    """DS4 estimated memory participates in pre-load LRU admission."""
    _patch_fake_process(monkeypatch)
    monkeypatch.setattr("omlx.engine_pool.get_phys_footprint", lambda: 1_000)
    monkeypatch.setattr("omlx.engine_pool.mx.get_active_memory", lambda: 0)
    (tmp_path / "Foo.gguf").write_bytes(b"0" * 1000)
    (tmp_path / "Bar.gguf").write_bytes(b"0" * 1000)
    pool = EnginePool(base_path=tmp_path, ds4_settings=DS4Settings())
    pool._get_final_ceiling = lambda: 2_500
    pool.discover_models(str(tmp_path))

    await pool.get_engine("foo")
    await pool.get_engine("bar")

    assert pool.get_entry("foo").engine is None
    assert pool.get_entry("bar").engine is not None
    assert FakeManagedProcess.instances[0].stopped is True
    assert FakeManagedProcess.instances[1].started is True
    assert pool.current_model_memory + 1_000 <= 2_500


@pytest.mark.asyncio
async def test_ttl_expiration_unloads_idle_unpinned_ds4_process(monkeypatch, tmp_path):
    _patch_fake_process(monkeypatch)
    pool = _pool_with_ds4(tmp_path)
    await pool.get_engine("foo")

    class SettingsManager:
        def get_settings(self, model_id):
            from omlx.model_settings import ModelSettings

            return ModelSettings(ttl_seconds=0)

    expired = await pool.check_ttl_expirations(SettingsManager())

    assert expired == ["foo"]
    assert pool.get_entry("foo").engine is None
    assert FakeManagedProcess.instances[0].stopped is True


@pytest.mark.asyncio
async def test_engine_pool_rejects_ds4_load_when_backend_disabled(monkeypatch, tmp_path):
    _patch_fake_process(monkeypatch)
    pool = _pool_with_ds4(tmp_path, ds4_enabled=False)

    with pytest.raises(RuntimeError, match="DS4 backend is disabled"):
        await pool.get_engine("foo")

    assert FakeManagedProcess.instances == []


@pytest.mark.asyncio
async def test_disabled_ds4_load_does_not_evict_existing_victim(monkeypatch, tmp_path):
    """Disabled backend errors before admission evicts loaded models."""
    _patch_fake_process(monkeypatch)
    monkeypatch.setattr("omlx.engine_pool.get_phys_footprint", lambda: 0)
    monkeypatch.setattr("omlx.engine_pool.mx.get_active_memory", lambda: 0)
    (tmp_path / "Foo.gguf").write_bytes(b"0" * 1000)
    (tmp_path / "Victim.gguf").write_bytes(b"0" * 1000)
    ds4_settings = DS4Settings(enabled=True)
    pool = EnginePool(base_path=tmp_path, ds4_settings=ds4_settings)
    pool._get_final_ceiling = lambda: 1_500
    pool.discover_models(str(tmp_path))
    await pool.get_engine("victim")

    ds4_settings.enabled = False
    with pytest.raises(RuntimeError, match="DS4 backend is disabled"):
        await pool.get_engine("foo")

    assert pool.get_entry("victim").engine is not None
    assert pool.get_entry("foo").engine is None
    assert FakeManagedProcess.instances[0].stopped is False


def test_public_load_unload_endpoints_manage_ds4_process(monkeypatch, tmp_path):
    """Existing manual model lifecycle endpoints work for DS4 entries."""
    _patch_fake_process(monkeypatch)
    pool = _pool_with_ds4(tmp_path)
    state = ServerState()
    state.engine_pool = pool
    state.api_key = None

    with patch("omlx.server._server_state", state):
        with TestClient(app, raise_server_exceptions=False) as client:
            load_response = client.post("/v1/models/foo/load")
            unload_response = client.post("/v1/models/foo/unload")

    assert load_response.status_code == 200
    assert load_response.json()["status"] == "ok"
    assert unload_response.status_code == 200
    assert unload_response.json()["status"] == "ok"
    assert FakeManagedProcess.instances[0].started is True
    assert FakeManagedProcess.instances[0].stopped is True
