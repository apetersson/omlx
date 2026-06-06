# SPDX-License-Identifier: Apache-2.0
"""EnginePool adapter for OMLX-managed DS4 subprocesses.

This adapter owns DS4 lifecycle management and byte-preserving proxy helpers for
protocol endpoints that have been wired through OMLX.  BaseEngine generation
methods still raise clear errors for endpoints that are not proxied yet.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from ..ds4_process import DS4_HOST, DS4LaunchConfig, DS4ManagedProcess
from ..settings import DEFAULT_BASE_PATH, DS4Settings
from .base import BaseEngine, GenerationOutput


class DS4ProxyError(RuntimeError):
    """Raised when OMLX cannot contact the managed DS4 backend."""


@dataclass(frozen=True)
class DS4ProxyResponse:
    """Raw non-streaming DS4 HTTP response."""

    status_code: int
    headers: dict[str, str]
    body: bytes


@dataclass
class DS4StreamingProxyResponse:
    """Raw streaming DS4 HTTP response with close-time cleanup."""

    status_code: int
    headers: dict[str, str]
    response: requests.Response
    on_close: Callable[[], None]
    session: requests.Session | None = None
    closed: bool = False

    def close(self) -> None:
        """Close the upstream response and run cleanup exactly once."""
        if self.closed:
            return
        self.closed = True
        self.response.close()
        if self.session is not None:
            self.session.close()
        self.on_close()

    def iter_bytes(self) -> Iterator[bytes]:
        """Yield DS4 response bytes without parsing or reformatting them."""
        if self.closed:
            return
        try:
            raw = getattr(self.response, "raw", None)
            if raw is not None and hasattr(raw, "stream"):
                for chunk in raw.stream(65536, decode_content=False):
                    if chunk:
                        yield chunk
            else:
                for chunk in self.response.iter_content(chunk_size=None):
                    if chunk:
                        yield chunk
        finally:
            self.close()


class DS4ProcessEngine(BaseEngine):
    """Minimal BaseEngine wrapper around one managed ds4-server process."""

    def __init__(
        self,
        *,
        model_id: str,
        model_path: str | Path,
        settings: DS4Settings | None = None,
        base_path: str | Path | None = None,
        context_tokens: int | None = None,
        auto_enable_ssd_streaming: bool = False,
    ):
        self.model_id = model_id
        self._model_path = Path(model_path)
        self.settings = settings or DS4Settings()
        self.base_path = Path(base_path) if base_path is not None else DEFAULT_BASE_PATH
        self.context_tokens = context_tokens
        self.auto_enable_ssd_streaming = auto_enable_ssd_streaming
        self.process: DS4ManagedProcess | None = None
        self._active_requests = 0
        self._active_requests_lock = threading.Lock()

    @property
    def model_name(self) -> str:
        """Return the GGUF model path used to launch DS4."""
        return str(self._model_path)

    @property
    def tokenizer(self) -> Any:
        """DS4 tokenization stays inside the subprocess."""
        return None

    @property
    def model_type(self) -> str:
        """Expose a stable model type for server-side type checks."""
        return "ds4"

    @property
    def port(self) -> int | None:
        """Localhost port selected for the running DS4 process."""
        return self.process.port if self.process is not None else None

    @property
    def pid(self) -> int | None:
        """Subprocess PID, if DS4 has been started."""
        process = self.process.process if self.process is not None else None
        return process.pid if process is not None else None

    @property
    def is_running(self) -> bool:
        """Return True while the managed DS4 subprocess is alive."""
        return self.process is not None and self.process.is_running

    async def start(self) -> None:
        """Start DS4 and wait for readiness."""
        if self.is_running:
            return
        config = DS4LaunchConfig(
            model_id=self.model_id,
            gguf_path=self._model_path,
            settings=self.settings,
            base_path=self.base_path,
            context_tokens=self.context_tokens,
            auto_enable_ssd_streaming=self.auto_enable_ssd_streaming,
        )
        self.process = DS4ManagedProcess(config)
        try:
            await self.process.start()
        except Exception:
            self.process = None
            raise

    async def stop(self) -> None:
        """Stop the managed DS4 subprocess."""
        process = self.process
        if process is not None:
            await process.stop()
        self.process = None

    def _increment_active_requests(self) -> None:
        with self._active_requests_lock:
            self._active_requests += 1

    def _decrement_active_requests(self) -> None:
        with self._active_requests_lock:
            if self._active_requests > 0:
                self._active_requests -= 1

    def has_active_requests(self) -> bool:
        """Return True while OMLX is proxying requests to DS4."""
        with self._active_requests_lock:
            return self._active_requests > 0

    def _backend_url(self, path: str) -> str:
        if self.port is None or not self.is_running:
            raise DS4ProxyError("DS4 backend process is not running")
        return f"http://{DS4_HOST}:{self.port}{path}"

    @staticmethod
    def _response_headers(response: requests.Response) -> dict[str, str]:
        """Return response headers that are safe to reflect through OMLX."""
        excluded = {"connection", "transfer-encoding"}
        return {
            key: value
            for key, value in response.headers.items()
            if key.lower() not in excluded
        }

    def _proxy_json_request_blocking(
        self,
        path: str,
        body: dict[str, Any],
        *,
        stream: bool,
    ) -> tuple[requests.Session, requests.Response]:
        session = requests.Session()
        session.trust_env = False
        try:
            response = session.post(
                self._backend_url(path),
                json=body,
                stream=stream,
                headers={
                    "Content-Type": "application/json",
                    "Accept-Encoding": "identity",
                },
            )
        except requests.RequestException as exc:
            session.close()
            raise DS4ProxyError(f"DS4 backend request failed: {exc}") from exc
        return session, response

    @staticmethod
    def _raw_body(response: requests.Response) -> bytes:
        raw = getattr(response, "raw", None)
        if raw is not None and hasattr(raw, "read"):
            return raw.read(decode_content=False)
        return response.content

    def _proxy_json_response_blocking(
        self,
        path: str,
        body: dict[str, Any],
    ) -> DS4ProxyResponse:
        try:
            session, response = self._proxy_json_request_blocking(
                path,
                body,
                stream=True,
            )
            try:
                try:
                    raw_body = self._raw_body(response)
                except Exception as exc:  # noqa: BLE001 - normalize backend I/O failures
                    raise DS4ProxyError(
                        f"DS4 backend response read failed: {exc}"
                    ) from exc
                return DS4ProxyResponse(
                    status_code=response.status_code,
                    headers=self._response_headers(response),
                    body=raw_body,
                )
            finally:
                response.close()
                session.close()
        finally:
            self._decrement_active_requests()

    async def _proxy_json_endpoint(
        self,
        path: str,
        body: dict[str, Any],
    ) -> DS4ProxyResponse:
        self._increment_active_requests()
        task = asyncio.create_task(
            asyncio.to_thread(
                self._proxy_json_response_blocking,
                path,
                body,
            )
        )
        return await asyncio.shield(task)

    async def proxy_chat_completion(self, body: dict[str, Any]) -> DS4ProxyResponse:
        """Forward one non-streaming OpenAI chat completion request to DS4."""
        return await self._proxy_json_endpoint("/v1/chat/completions", body)

    async def proxy_completion(self, body: dict[str, Any]) -> DS4ProxyResponse:
        """Forward one non-streaming OpenAI text completion request to DS4."""
        return await self._proxy_json_endpoint("/v1/completions", body)

    async def _open_json_stream(
        self,
        path: str,
        body: dict[str, Any],
    ) -> DS4StreamingProxyResponse:
        self._increment_active_requests()
        task = asyncio.create_task(
            asyncio.to_thread(
                self._proxy_json_request_blocking,
                path,
                body,
                stream=True,
            )
        )
        claimed = False

        def _cleanup_unclaimed(done: asyncio.Task) -> None:
            nonlocal claimed
            if claimed:
                return
            try:
                session, response = done.result()
            except Exception:
                self._decrement_active_requests()
                return
            response.close()
            session.close()
            self._decrement_active_requests()

        try:
            session, response = await asyncio.shield(task)
            claimed = True
        except asyncio.CancelledError:
            task.add_done_callback(_cleanup_unclaimed)
            raise
        except Exception:
            claimed = True
            self._decrement_active_requests()
            raise

        def _decrement_active() -> None:
            self._decrement_active_requests()

        return DS4StreamingProxyResponse(
            status_code=response.status_code,
            headers=self._response_headers(response),
            response=response,
            on_close=_decrement_active,
            session=session,
        )

    async def open_chat_completion_stream(
        self, body: dict[str, Any]
    ) -> DS4StreamingProxyResponse:
        """Open a streaming OpenAI chat completion request to DS4."""
        return await self._open_json_stream("/v1/chat/completions", body)

    async def open_completion_stream(
        self, body: dict[str, Any]
    ) -> DS4StreamingProxyResponse:
        """Open a streaming OpenAI text completion request to DS4."""
        return await self._open_json_stream("/v1/completions", body)

    def get_process_rss_bytes(self) -> int | None:
        """Return the DS4 subprocess RSS when psutil can observe it."""
        pid = self.pid
        if pid is None:
            return None
        try:
            import psutil

            return int(psutil.Process(pid).memory_info().rss)
        except Exception:  # noqa: BLE001 - status should be best-effort only
            return None

    def get_stats(self) -> dict[str, Any]:
        """Return DS4 lifecycle/status fields for admin/status endpoints."""
        command = self.process.command if self.process is not None else None
        logs = self.process.recent_log_text() if self.process is not None else ""
        return {
            "backend": "ds4",
            "host": DS4_HOST,
            "port": self.port,
            "pid": self.pid,
            "running": self.is_running,
            "rss_bytes": self.get_process_rss_bytes(),
            "command": command,
            "recent_logs": logs,
        }

    def get_cache_stats(self) -> dict[str, Any] | None:
        """DS4 cache metrics are not available until protocol metrics land."""
        return None

    def _protocol_not_implemented(self) -> RuntimeError:
        return RuntimeError(
            "DS4 backend lifecycle is available, but protocol forwarding has not "
            "been implemented yet"
        )

    async def generate(self, *args, **kwargs) -> GenerationOutput:
        """Text completions are proxied through the server route."""
        raise self._protocol_not_implemented()

    async def stream_generate(self, *args, **kwargs) -> AsyncIterator[GenerationOutput]:
        """Streaming completions are proxied through the server route."""
        raise self._protocol_not_implemented()
        yield  # pragma: no cover - keeps this method an async iterator

    async def chat(self, *args, **kwargs) -> GenerationOutput:
        """Chat completions are forwarded in a later DS4 protocol slice."""
        raise self._protocol_not_implemented()

    async def stream_chat(self, *args, **kwargs) -> AsyncIterator[GenerationOutput]:
        """Streaming chat is forwarded in a later DS4 protocol slice."""
        raise self._protocol_not_implemented()
        yield  # pragma: no cover - keeps this method an async iterator
