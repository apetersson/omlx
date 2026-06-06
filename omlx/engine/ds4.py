# SPDX-License-Identifier: Apache-2.0
"""EnginePool adapter for OMLX-managed DS4 subprocesses.

This adapter intentionally stops at lifecycle management.  Protocol-specific
request forwarding is implemented in a later slice, so generation methods raise
clear errors while load/unload/status can already exercise the managed backend.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from ..ds4_process import DS4_HOST, DS4LaunchConfig, DS4ManagedProcess
from ..settings import DEFAULT_BASE_PATH, DS4Settings
from .base import BaseEngine, GenerationOutput


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

    def has_active_requests(self) -> bool:
        """Protocol forwarding is not wired yet, so no requests run here."""
        return False

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
        """Text completions are forwarded in a later DS4 protocol slice."""
        raise self._protocol_not_implemented()

    async def stream_generate(self, *args, **kwargs) -> AsyncIterator[GenerationOutput]:
        """Streaming completions are forwarded in a later DS4 protocol slice."""
        raise self._protocol_not_implemented()
        yield  # pragma: no cover - keeps this method an async iterator

    async def chat(self, *args, **kwargs) -> GenerationOutput:
        """Chat completions are forwarded in a later DS4 protocol slice."""
        raise self._protocol_not_implemented()

    async def stream_chat(self, *args, **kwargs) -> AsyncIterator[GenerationOutput]:
        """Streaming chat is forwarded in a later DS4 protocol slice."""
        raise self._protocol_not_implemented()
        yield  # pragma: no cover - keeps this method an async iterator
