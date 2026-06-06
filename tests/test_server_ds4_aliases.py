# SPDX-License-Identifier: Apache-2.0
"""Tests for DS4 aliases exposed through server model listing."""

import json
from contextlib import contextmanager
from unittest.mock import patch

from fastapi.testclient import TestClient

from omlx.engine_pool import EnginePool
from omlx.model_settings import ModelSettings
from omlx.server import ServerState, app


class _SettingsManager:
    def __init__(self, settings: dict[str, ModelSettings] | None = None):
        self._settings = settings or {}

    def get_settings(self, model_id: str) -> ModelSettings:
        return self._settings.get(model_id, ModelSettings())

    def get_all_settings(self) -> dict[str, ModelSettings]:
        return self._settings


def _ds4_pool(tmp_path, filename: str = "DeepSeek V4 Flash Q2_K.gguf") -> EnginePool:
    (tmp_path / filename).write_bytes(b"0" * 1000)
    pool = EnginePool()
    pool.discover_models(str(tmp_path))
    return pool


@contextmanager
def _client_for_pool(pool: EnginePool, settings_manager: _SettingsManager | None = None):
    state = ServerState()
    state.engine_pool = pool
    state.settings_manager = settings_manager or _SettingsManager()
    state.api_key = None
    with patch("omlx.server._server_state", state):
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client


def test_models_list_includes_ds4_base_and_per_model_aliases(tmp_path):
    pool = _ds4_pool(tmp_path)
    model_id = pool.get_model_ids()[0]

    with _client_for_pool(pool) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    ids = [model["id"] for model in response.json()["data"]]
    assert model_id in ids
    assert f"{model_id}-chat" in ids
    assert f"{model_id}-reasoner" in ids
    assert f"{model_id}-think-max" in ids


def test_models_list_uses_user_alias_as_ds4_alias_base(tmp_path):
    pool = _ds4_pool(tmp_path, filename="Foo.gguf")
    settings_manager = _SettingsManager({"foo": ModelSettings(model_alias="gpt-4o")})

    with _client_for_pool(pool, settings_manager) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    ids = [model["id"] for model in response.json()["data"]]
    assert "gpt-4o" in ids
    assert "gpt-4o-chat" in ids
    assert "gpt-4o-reasoner" in ids
    assert "gpt-4o-think-max" in ids
    assert "deepseek-chat" not in ids
    assert "deepseek-reasoner" not in ids


def test_models_list_deduplicates_ds4_aliases_that_collide_with_real_models(tmp_path):
    (tmp_path / "Foo.gguf").write_bytes(b"0" * 1000)
    mlx_model = tmp_path / "foo-chat"
    mlx_model.mkdir()
    (mlx_model / "config.json").write_text(json.dumps({"model_type": "llama"}))
    pool = EnginePool()
    pool.discover_models(str(tmp_path))

    with _client_for_pool(pool) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    ids = [model["id"] for model in response.json()["data"]]
    assert "foo" in ids
    assert "foo-chat" in ids
    assert ids.count("foo-chat") == 1


def test_models_status_includes_ds4_aliases_for_ui(tmp_path):
    pool = _ds4_pool(tmp_path, filename="Foo.gguf")
    settings_manager = _SettingsManager({"foo": ModelSettings(model_alias="gpt-4o")})

    with _client_for_pool(pool, settings_manager) as client:
        response = client.get("/v1/models/status")

    assert response.status_code == 200
    model = response.json()["models"][0]
    assert model["id"] == "foo"
    assert model["model_alias"] == "gpt-4o"
    assert model["ds4_aliases"] == [
        "gpt-4o-chat",
        "gpt-4o-reasoner",
        "gpt-4o-think-max",
    ]
