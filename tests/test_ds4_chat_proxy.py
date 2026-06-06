# SPDX-License-Identifier: Apache-2.0
"""Tests for DS4 OpenAI chat-completions proxying."""

from __future__ import annotations

import asyncio
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from omlx.api.openai_models import ChatCompletionRequest, Message
from omlx.engine.ds4 import DS4ProcessEngine, DS4ProxyResponse
from omlx.model_settings import ModelSettings
from omlx.server import ServerState, app


class _SettingsManager:
    def __init__(self, settings: ModelSettings | None = None):
        self.settings = settings or ModelSettings()

    def get_settings(self, model_id: str) -> ModelSettings:
        return self.settings

    def get_all_settings(self) -> dict[str, ModelSettings]:
        return {"foo": self.settings}


class _Pool:
    def __init__(self, engine: DS4ProcessEngine):
        self.engine = engine
        self.requested_model_ids: list[str] = []

    def resolve_model_id(self, model_id, settings_manager):
        if model_id in {
            "foo",
            "foo-chat",
            "foo-reasoner",
            "foo-think-max",
            "gpt-4o",
            "gpt-4o-chat",
            "gpt-4o-reasoner",
            "gpt-4o-think-max",
        }:
            return "foo"
        return model_id

    async def get_engine(self, model_id, **kwargs):
        self.requested_model_ids.append(model_id)
        return self.engine

    def get_entry(self, model_id):
        return None

    async def preload_pinned_models(self):
        return None

    async def check_ttl_expirations(self, *args, **kwargs):
        return []

    async def shutdown(self):
        return None


@dataclass
class _StreamingProxy:
    chunks: list[bytes]
    status_code: int = 200
    headers: dict[str, str] | None = None
    closed: bool = False

    def iter_bytes(self):
        yield from self.chunks

    def close(self):
        self.closed = True


class _RawResponseBody:
    def __init__(self, *, content: bytes = b"", chunks: list[bytes] | None = None):
        self.content = content
        self.chunks = chunks or []

    def read(self, decode_content=True):
        return self.content

    def stream(self, amt=None, decode_content=True):
        yield from self.chunks


class _RequestsResponse:
    def __init__(self, *, content: bytes = b"", chunks: list[bytes] | None = None):
        self.status_code = 200
        self.headers = {"Content-Type": "text/event-stream" if chunks else "application/json"}
        self.content = b"decoded-content-should-not-be-used"
        self.chunks = chunks or []
        self.raw = _RawResponseBody(content=content, chunks=chunks)
        self.closed = False

    def iter_content(self, chunk_size=None):
        yield from self.chunks

    def close(self):
        self.closed = True


class _FakeRequestsSession:
    instances: list[_FakeRequestsSession] = []
    next_response: _RequestsResponse | None = None
    post_started: threading.Event | None = None
    allow_post: threading.Event | None = None

    def __init__(self):
        self.trust_env = True
        self.closed = False
        self.calls: list[dict] = []
        self.trust_env_at_post: bool | None = None
        _FakeRequestsSession.instances.append(self)

    def post(self, url, *, json, stream, headers):
        self.trust_env_at_post = self.trust_env
        self.calls.append({"url": url, "json": json, "stream": stream, "headers": headers})
        if self.post_started is not None:
            self.post_started.set()
        if self.allow_post is not None:
            self.allow_post.wait(timeout=5.0)
        return self.next_response or _RequestsResponse(content=b"{}")

    def close(self):
        self.closed = True


class _FakeDS4Engine(DS4ProcessEngine):
    def __init__(self, tmp_path):
        gguf = tmp_path / "Foo.gguf"
        gguf.write_bytes(b"0" * 1000)
        super().__init__(model_id="foo", model_path=gguf, base_path=tmp_path)
        self.proxy_bodies: list[dict] = []
        self.stream_bodies: list[dict] = []

    async def proxy_chat_completion(self, body: dict):
        self.proxy_bodies.append(body)
        return DS4ProxyResponse(
            status_code=201,
            headers={"Content-Type": "application/json; charset=utf-8"},
            body=b'{"ds4":true,"choices":[]}',
        )

    async def open_chat_completion_stream(self, body: dict):
        self.stream_bodies.append(body)
        return _StreamingProxy(
            chunks=[b"data: one\n\n", b"data: [DONE]\n\n"],
            headers={"Content-Type": "text/event-stream"},
        )


@contextmanager
def _client_with_engine(engine: _FakeDS4Engine, settings: ModelSettings | None = None):
    state = ServerState()
    state.engine_pool = _Pool(engine)
    state.settings_manager = _SettingsManager(settings)
    state.api_key = None
    with patch("omlx.server._server_state", state):
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, state.engine_pool


@pytest.mark.asyncio
async def test_ds4_process_engine_proxies_non_streaming_chat(monkeypatch, tmp_path):
    engine = DS4ProcessEngine(
        model_id="foo",
        model_path=tmp_path / "Foo.gguf",
        base_path=tmp_path,
    )
    engine.process = SimpleNamespace(
        is_running=True,
        port=49152,
        process=SimpleNamespace(pid=123),
        command=[],
        recent_log_text=lambda: "",
    )
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:8080")
    _FakeRequestsSession.instances = []
    _FakeRequestsSession.next_response = _RequestsResponse(content=b'{"ok":true}')
    monkeypatch.setattr("omlx.engine.ds4.requests.Session", _FakeRequestsSession)

    response = await engine.proxy_chat_completion({"model": "foo"})

    session = _FakeRequestsSession.instances[0]
    captured = session.calls[0]
    assert session.trust_env_at_post is False
    assert session.closed is True
    assert captured["url"] == "http://127.0.0.1:49152/v1/chat/completions"
    assert captured["json"] == {"model": "foo"}
    assert captured["headers"]["Accept-Encoding"] == "identity"
    assert captured["stream"] is True
    assert response.body == b'{"ok":true}'
    assert engine.has_active_requests() is False


@pytest.mark.asyncio
async def test_ds4_process_engine_stream_tracks_active_until_consumed(
    monkeypatch, tmp_path
):
    engine = DS4ProcessEngine(
        model_id="foo",
        model_path=tmp_path / "Foo.gguf",
        base_path=tmp_path,
    )
    engine.process = SimpleNamespace(
        is_running=True,
        port=49152,
        process=SimpleNamespace(pid=123),
        command=[],
        recent_log_text=lambda: "",
    )
    backend_response = _RequestsResponse(chunks=[b"data: one\n\n", b"data: [DONE]\n\n"])
    _FakeRequestsSession.instances = []
    _FakeRequestsSession.next_response = backend_response
    monkeypatch.setattr("omlx.engine.ds4.requests.Session", _FakeRequestsSession)

    response = await engine.open_chat_completion_stream({"model": "foo"})
    assert engine.has_active_requests() is True
    assert list(response.iter_bytes()) == [b"data: one\n\n", b"data: [DONE]\n\n"]
    session = _FakeRequestsSession.instances[0]
    assert session.trust_env_at_post is False
    assert session.calls[0]["stream"] is True
    assert backend_response.closed is True
    assert session.closed is True
    assert engine.has_active_requests() is False


@pytest.mark.asyncio
async def test_ds4_multiple_streams_remain_active_until_all_thread_closes(
    monkeypatch, tmp_path
):
    engine = DS4ProcessEngine(
        model_id="foo",
        model_path=tmp_path / "Foo.gguf",
        base_path=tmp_path,
    )
    engine.process = SimpleNamespace(
        is_running=True,
        port=49152,
        process=SimpleNamespace(pid=123),
        command=[],
        recent_log_text=lambda: "",
    )
    _FakeRequestsSession.instances = []
    monkeypatch.setattr("omlx.engine.ds4.requests.Session", _FakeRequestsSession)

    first_backend_response = _RequestsResponse(chunks=[b"data: first\n\n"])
    _FakeRequestsSession.next_response = first_backend_response
    first = await engine.open_chat_completion_stream({"model": "foo"})
    second_backend_response = _RequestsResponse(chunks=[b"data: second\n\n"])
    _FakeRequestsSession.next_response = second_backend_response
    second = await engine.open_chat_completion_stream({"model": "foo"})

    assert engine.has_active_requests() is True
    await asyncio.to_thread(first.close)
    assert engine.has_active_requests() is True
    await asyncio.to_thread(second.close)
    assert engine.has_active_requests() is False


@pytest.mark.asyncio
async def test_ds4_stream_response_close_releases_active_without_iteration(
    monkeypatch, tmp_path
):
    engine = DS4ProcessEngine(
        model_id="foo",
        model_path=tmp_path / "Foo.gguf",
        base_path=tmp_path,
    )
    engine.process = SimpleNamespace(
        is_running=True,
        port=49152,
        process=SimpleNamespace(pid=123),
        command=[],
        recent_log_text=lambda: "",
    )
    backend_response = _RequestsResponse(chunks=[b"data: never-read\n\n"])
    _FakeRequestsSession.instances = []
    _FakeRequestsSession.next_response = backend_response
    monkeypatch.setattr("omlx.engine.ds4.requests.Session", _FakeRequestsSession)

    response = await engine.open_chat_completion_stream({"model": "foo"})
    assert engine.has_active_requests() is True
    response.close()

    assert list(response.iter_bytes()) == []
    assert backend_response.closed is True
    assert _FakeRequestsSession.instances[0].closed is True
    assert engine.has_active_requests() is False


@pytest.mark.asyncio
async def test_ds4_non_streaming_cancellation_keeps_active_until_thread_finishes(
    monkeypatch, tmp_path
):
    engine = DS4ProcessEngine(
        model_id="foo",
        model_path=tmp_path / "Foo.gguf",
        base_path=tmp_path,
    )
    engine.process = SimpleNamespace(
        is_running=True,
        port=49152,
        process=SimpleNamespace(pid=123),
        command=[],
        recent_log_text=lambda: "",
    )
    read_started = threading.Event()
    allow_read = threading.Event()
    backend_response = _RequestsResponse(content=b'{"ok":true}')

    def blocking_read(decode_content=True):
        read_started.set()
        allow_read.wait(timeout=5.0)
        return b'{"ok":true}'

    backend_response.raw.read = blocking_read
    _FakeRequestsSession.instances = []
    _FakeRequestsSession.next_response = backend_response
    _FakeRequestsSession.post_started = None
    _FakeRequestsSession.allow_post = None
    monkeypatch.setattr("omlx.engine.ds4.requests.Session", _FakeRequestsSession)

    task = asyncio.create_task(engine.proxy_chat_completion({"model": "foo"}))
    assert await asyncio.to_thread(read_started.wait, 2.0)
    assert engine.has_active_requests() is True
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert engine.has_active_requests() is True

    allow_read.set()
    for _ in range(50):
        if not engine.has_active_requests():
            break
        await asyncio.sleep(0.02)

    assert engine.has_active_requests() is False
    assert backend_response.closed is True
    assert _FakeRequestsSession.instances[0].closed is True


@pytest.mark.asyncio
async def test_ds4_stream_open_cancellation_cleans_up_unclaimed_response(
    monkeypatch, tmp_path
):
    engine = DS4ProcessEngine(
        model_id="foo",
        model_path=tmp_path / "Foo.gguf",
        base_path=tmp_path,
    )
    engine.process = SimpleNamespace(
        is_running=True,
        port=49152,
        process=SimpleNamespace(pid=123),
        command=[],
        recent_log_text=lambda: "",
    )
    backend_response = _RequestsResponse(chunks=[b"data: later\n\n"])
    post_started = threading.Event()
    allow_post = threading.Event()
    _FakeRequestsSession.instances = []
    _FakeRequestsSession.next_response = backend_response
    _FakeRequestsSession.post_started = post_started
    _FakeRequestsSession.allow_post = allow_post
    monkeypatch.setattr("omlx.engine.ds4.requests.Session", _FakeRequestsSession)

    task = asyncio.create_task(engine.open_chat_completion_stream({"model": "foo"}))
    assert await asyncio.to_thread(post_started.wait, 2.0)
    assert engine.has_active_requests() is True
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert engine.has_active_requests() is True

    allow_post.set()
    for _ in range(50):
        if not engine.has_active_requests():
            break
        await asyncio.sleep(0.02)

    assert engine.has_active_requests() is False
    assert backend_response.closed is True
    assert _FakeRequestsSession.instances[0].closed is True
    _FakeRequestsSession.post_started = None
    _FakeRequestsSession.allow_post = None


def test_ds4_chat_non_streaming_proxies_raw_response_and_applies_defaults(tmp_path):
    engine = _FakeDS4Engine(tmp_path)
    settings = ModelSettings(
        temperature=0.25,
        top_p=0.5,
        top_k=7,
        repetition_penalty=1.2,
        presence_penalty=0.4,
        max_tokens=123,
        force_sampling=True,
    )

    with _client_with_engine(engine, settings) as (client, pool):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "foo-chat",
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0.9,
                "top_p": 0.8,
            },
        )

    assert response.status_code == 201
    assert response.content == b'{"ds4":true,"choices":[]}'
    assert pool.requested_model_ids == ["foo"]
    body = engine.proxy_bodies[0]
    assert body["model"] == "deepseek-chat"
    assert body["messages"] == [{"role": "user", "content": "hello", "partial": False}]
    assert body["temperature"] == 0.9
    assert body["top_p"] == 0.8
    assert body["top_k"] == 7
    assert body["repetition_penalty"] == 1.2
    assert body["presence_penalty"] == 0.4
    assert body["frequency_penalty"] == 0.0
    assert body["xtc_probability"] == 0.0
    assert body["xtc_threshold"] == 0.1
    assert body["max_tokens"] == 123


@pytest.mark.asyncio
async def test_ds4_streaming_response_closes_proxy_when_send_start_fails():
    from omlx.server import _DS4StreamingResponse

    proxy = _StreamingProxy(
        chunks=[b"data: never-started\n\n"],
        headers={"Content-Type": "text/event-stream"},
    )
    response = _DS4StreamingResponse(proxy, media_type="text/event-stream")

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            raise RuntimeError("client disconnected before stream body")

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [],
    }
    with pytest.raises(RuntimeError, match="client disconnected"):
        await response(scope, receive, send)

    assert proxy.closed is True


def test_ds4_chat_streaming_preserves_backend_sse_bytes(tmp_path):
    engine = _FakeDS4Engine(tmp_path)

    with _client_with_engine(engine) as (client, _pool):
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "foo-reasoner",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        ) as response:
            body = b"".join(response.iter_bytes())

    assert response.status_code == 200
    assert body == b"data: one\n\ndata: [DONE]\n\n"
    assert engine.stream_bodies[0]["model"] == "deepseek-reasoner"
    assert engine.stream_bodies[0]["stream"] is True


def test_ds4_proxy_body_preserves_openai_schema_alias(tmp_path):
    from omlx.server import _build_ds4_chat_proxy_body

    engine = _FakeDS4Engine(tmp_path)
    state = ServerState()
    state.engine_pool = _Pool(engine)
    state.settings_manager = _SettingsManager()
    request = ChatCompletionRequest(
        model="foo",
        messages=[Message(role="user", content="json please")],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "result",
                "schema": {"type": "object"},
            },
        },
    )

    with patch("omlx.server._server_state", state):
        body = _build_ds4_chat_proxy_body(request, "foo")

    assert body["response_format"]["json_schema"]["schema"] == {"type": "object"}
    assert "schema_" not in body["response_format"]["json_schema"]


def test_ds4_base_model_id_ending_with_alias_suffix_is_preserved(tmp_path):
    from omlx.server import _build_ds4_chat_proxy_body

    engine = _FakeDS4Engine(tmp_path)
    state = ServerState()
    state.engine_pool = _Pool(engine)
    state.settings_manager = _SettingsManager()
    request = ChatCompletionRequest(
        model="foo-chat",
        messages=[Message(role="user", content="hello")],
    )

    with patch("omlx.server._server_state", state):
        body = _build_ds4_chat_proxy_body(request, "foo-chat")

    assert body["model"] == "foo-chat"

    request = ChatCompletionRequest(
        model="FOO-CHAT",
        messages=[Message(role="user", content="hello")],
    )
    with patch("omlx.server._server_state", state):
        body = _build_ds4_chat_proxy_body(request, "foo-chat")

    assert body["model"] == "FOO-CHAT"


def test_ds4_think_max_alias_injects_reasoning_effort_with_user_alias(tmp_path):
    from omlx.server import _build_ds4_chat_proxy_body

    engine = _FakeDS4Engine(tmp_path)
    state = ServerState()
    state.engine_pool = _Pool(engine)
    state.settings_manager = _SettingsManager(ModelSettings(model_alias="gpt-4o"))
    request = ChatCompletionRequest(
        model="gpt-4o-think-max",
        messages=[Message(role="user", content="hello")],
        reasoning_effort="low",
    )

    with patch("omlx.server._server_state", state):
        body = _build_ds4_chat_proxy_body(request, "foo")

    assert body["model"] == "gpt-4o"
    assert body["reasoning_effort"] == "max"

    request = ChatCompletionRequest(
        model="omlx/gpt-4o-think-max",
        messages=[Message(role="user", content="hello")],
    )
    with patch("omlx.server._server_state", state):
        body = _build_ds4_chat_proxy_body(request, "foo")

    assert body["model"] == "gpt-4o"
    assert body["reasoning_effort"] == "max"
