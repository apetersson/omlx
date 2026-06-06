# SPDX-License-Identifier: Apache-2.0
"""Tests for managed DS4 subprocess scaffolding."""

from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from omlx.ds4_process import (
    DS4_HOST,
    DS4LaunchConfig,
    DS4ManagedProcess,
    DS4ProcessError,
    safe_ds4_fs_name,
)
from omlx.ds4_support import DS4_METAL_FILES, DS4_SERVER_BINARY, DS4SupportError
from omlx.settings import DS4Settings


def _write_support_tree(root: Path, script: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    binary = root / DS4_SERVER_BINARY
    binary.write_text(script)
    binary.chmod(0o755)
    (root / "LICENSE").write_text("MIT\n")
    (root / "README.md").write_text("DS4\n")
    metal_dir = root / "metal"
    metal_dir.mkdir()
    for name in DS4_METAL_FILES:
        (metal_dir / name).write_text("// metal\n")
    return binary


def _ready_server_script() -> str:
    return f"""#!{sys.executable}
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

parser = argparse.ArgumentParser()
parser.add_argument('--chdir')
parser.add_argument('--model')
parser.add_argument('--host')
parser.add_argument('--port', type=int)
parser.add_argument('--power')
parser.add_argument('--ctx')
parser.add_argument('--kv-disk-dir')
parser.add_argument('--kv-disk-space-mb')
parser.add_argument('--kv-cache-continued-interval-tokens')
parser.add_argument('--ssd-streaming', action='store_true')
parser.add_argument('--trace')
args, _ = parser.parse_known_args()
print('fake ds4 argv model=' + str(args.model), flush=True)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/v1/models':
            body = json.dumps({{'object': 'list', 'data': [{{'id': 'fake'}}]}}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()
    def log_message(self, *args):
        return

HTTPServer((args.host, args.port), Handler).serve_forever()
"""


def _never_ready_script() -> str:
    return f"""#!{sys.executable}
import time
print('fake ds4 never ready', flush=True)
time.sleep(60)
"""


class TestDS4LaunchConfig:
    """Tests for DS4 launch command construction."""

    def test_safe_ds4_fs_name(self):
        """Model ids become filesystem-safe per-model artifact names."""
        assert safe_ds4_fs_name("DeepSeek/V4:Flash") == "deepseek-v4-flash"
        assert safe_ds4_fs_name(" ... ") == "ds4-model"

    def test_build_command_includes_ds4_flags(self, tmp_path):
        """Command construction preserves the DS4 performance-path flags."""
        support = tmp_path / "support" / "ds4"
        _write_support_tree(support, _ready_server_script())
        gguf = tmp_path / "models" / "DeepSeek.gguf"
        gguf.parent.mkdir()
        gguf.write_bytes(b"gguf")
        settings = DS4Settings(
            support_dir=str(support),
            context_default_tokens=100_000,
            kv_root=str(tmp_path / "kv"),
            kv_disk_space_mb=1234,
            kv_cache_continued_interval_tokens=4096,
            ssd_streaming="auto",
            power=77,
            trace_enabled=True,
            trace_dir=str(tmp_path / "traces"),
        )
        config = DS4LaunchConfig(
            model_id="DeepSeek/V4:Flash",
            gguf_path=gguf,
            settings=settings,
            base_path=tmp_path,
            port=12345,
            auto_enable_ssd_streaming=True,
            trace_timestamp="20260101-010203",
            platform_system="Darwin",
            platform_machine="arm64",
        )

        command = config.build_command(12345)

        assert command[0] == str(support / DS4_SERVER_BINARY)
        assert command[command.index("--chdir") + 1] == str(support.resolve())
        assert command[command.index("--model") + 1] == str(gguf)
        assert command[command.index("--host") + 1] == DS4_HOST
        assert command[command.index("--port") + 1] == "12345"
        assert command[command.index("--ctx") + 1] == "100000"
        assert command[command.index("--kv-disk-space-mb") + 1] == "1234"
        assert (
            command[command.index("--kv-cache-continued-interval-tokens") + 1]
            == "4096"
        )
        assert command[command.index("--power") + 1] == "77"
        assert "--ssd-streaming" in command
        trace_arg = command[command.index("--trace") + 1]
        assert trace_arg.endswith("deepseek-v4-flash-20260101-010203.trace")
        assert "deepseek-v4-flash" in command[command.index("--kv-disk-dir") + 1]

    def test_launch_config_rejects_non_localhost_host(self, tmp_path):
        """Managed DS4 is never allowed to bind to LAN interfaces."""
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"gguf")

        with pytest.raises(ValueError, match="127.0.0.1"):
            DS4LaunchConfig(
                model_id="model",
                gguf_path=gguf,
                base_path=tmp_path,
                host="0.0.0.0",
            )

    def test_launch_config_is_frozen_after_localhost_validation(self, tmp_path):
        """The localhost-only invariant cannot be bypassed by mutation."""
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"gguf")
        config = DS4LaunchConfig(
            model_id="model",
            gguf_path=gguf,
            base_path=tmp_path,
        )

        with pytest.raises(FrozenInstanceError):
            config.host = "0.0.0.0"  # type: ignore[misc]

    def test_build_command_omits_kv_trace_and_ssd_when_disabled(self, tmp_path):
        """Disabled optional DS4 features are not passed to ds4-server."""
        support = tmp_path / "support" / "ds4"
        _write_support_tree(support, _ready_server_script())
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"gguf")
        settings = DS4Settings(
            support_dir=str(support),
            kv_cache_enabled=False,
            ssd_streaming="off",
            trace_enabled=False,
        )
        config = DS4LaunchConfig(
            model_id="model",
            gguf_path=gguf,
            settings=settings,
            base_path=tmp_path,
            port=12345,
            platform_system="Darwin",
            platform_machine="arm64",
        )

        command = config.build_command(12345)

        assert "--kv-disk-dir" not in command
        assert "--kv-disk-space-mb" not in command
        assert "--ssd-streaming" not in command
        assert "--trace" not in command


class TestDS4ManagedProcess:
    """Tests for managed DS4 subprocess lifecycle."""

    @pytest.mark.asyncio
    async def test_start_waits_for_models_readiness_and_captures_logs(self, tmp_path):
        """A fake ds4-server is started, probed, and terminated cleanly."""
        support = tmp_path / "support" / "ds4"
        _write_support_tree(support, _ready_server_script())
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"gguf")
        settings = DS4Settings(
            support_dir=str(support),
            kv_root=str(tmp_path / "kv"),
            debug_dir=str(tmp_path / "debug"),
            ready_timeout_ms=2_000,
        )
        managed = DS4ManagedProcess(
            DS4LaunchConfig(
                model_id="model",
                gguf_path=gguf,
                settings=settings,
                base_path=tmp_path,
                platform_system="Darwin",
                platform_machine="arm64",
            )
        )

        await managed.start()
        try:
            assert managed.is_running is True
            assert managed.port is not None
            assert managed.command is not None
            assert "--host" in managed.command
            assert (tmp_path / "kv" / "model").is_dir()
            assert (tmp_path / "debug" / "model").is_dir()
            assert any("fake ds4 argv" in line.text for line in managed.logs)
        finally:
            await managed.stop()

        assert managed.is_running is False

    @pytest.mark.asyncio
    async def test_start_timeout_stops_process_and_reports_logs(self, tmp_path):
        """Readiness timeout terminates the subprocess and includes logs."""
        support = tmp_path / "support" / "ds4"
        _write_support_tree(support, _never_ready_script())
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"gguf")
        settings = DS4Settings(
            support_dir=str(support),
            ready_timeout_ms=1_500,
        )
        managed = DS4ManagedProcess(
            DS4LaunchConfig(
                model_id="model",
                gguf_path=gguf,
                settings=settings,
                base_path=tmp_path,
                platform_system="Darwin",
                platform_machine="arm64",
            )
        )

        with pytest.raises(DS4ProcessError) as exc_info:
            await managed.start()

        assert managed.is_running is False
        assert "timed out" in str(exc_info.value)
        assert "fake ds4 never ready" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_missing_support_files_block_start(self, tmp_path):
        """Start fails before spawning when support validation fails."""
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"gguf")
        managed = DS4ManagedProcess(
            DS4LaunchConfig(
                model_id="model",
                gguf_path=gguf,
                settings=DS4Settings(support_dir=str(tmp_path / "missing")),
                base_path=tmp_path,
                platform_system="Darwin",
                platform_machine="arm64",
            )
        )

        with pytest.raises(DS4SupportError):
            await managed.start()

        assert managed.process is None
