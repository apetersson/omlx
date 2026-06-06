# SPDX-License-Identifier: Apache-2.0
"""Tests for DS4 support-file validation and copy helpers."""

import os
from pathlib import Path

import pytest

from omlx.ds4_support import (
    BUNDLED_DS4_SUPPORT_DIR_NAME,
    BUNDLED_DS4_SUPPORT_ENV,
    DS4_METAL_FILES,
    DS4_SERVER_BINARY,
    DS4SupportError,
    copy_ds4_support_files,
    find_bundled_ds4_support_dir,
    inspect_ds4_support,
    install_bundled_ds4_support_files,
    is_ds4_supported_platform,
    require_ds4_support,
    required_ds4_support_relative_paths,
)
from omlx.settings import DS4Settings


def _write_complete_support_tree(root: Path, *, executable: bool = True) -> None:
    binary = root / DS4_SERVER_BINARY
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"#!/bin/sh\n")
    binary.chmod(0o755 if executable else 0o644)
    (root / "LICENSE").write_text("MIT\n")
    (root / "README.md").write_text("DS4\n")
    metal_dir = root / "metal"
    metal_dir.mkdir()
    for name in DS4_METAL_FILES:
        (metal_dir / name).write_text("// metal\n")


class TestDS4SupportInspection:
    """Tests for DS4 support status inspection."""

    def test_required_paths_include_binary_license_readme_and_metal(self):
        """Required relative paths match unpatched DS4 runtime needs."""
        paths = required_ds4_support_relative_paths()

        assert DS4_SERVER_BINARY in paths
        assert "LICENSE" in paths
        assert "README.md" in paths
        assert "metal/flash_attn.metal" in paths
        assert "metal/set_rows.metal" in paths

    def test_supported_platform_detection(self):
        """V1 DS4 backend is macOS Apple Silicon only."""
        assert is_ds4_supported_platform("Darwin", "arm64") is True
        assert is_ds4_supported_platform("Darwin", "aarch64") is True
        assert is_ds4_supported_platform("Linux", "aarch64") is False
        assert is_ds4_supported_platform("Darwin", "x86_64") is False

    def test_complete_support_tree_is_ready(self, tmp_path):
        """A complete support dir is ready for later process launch."""
        support = tmp_path / "support" / "ds4"
        _write_complete_support_tree(support)
        settings = DS4Settings(support_dir=str(support))

        status = inspect_ds4_support(
            settings,
            base_path=tmp_path,
            system="Darwin",
            machine="arm64",
        )

        assert status.ready is True
        assert status.error_message() is None
        assert status.support_dir == support.resolve()
        assert status.binary_path == (support / DS4_SERVER_BINARY).resolve()

    def test_missing_support_files_report_clear_error(self, tmp_path):
        """Missing support files produce a user-facing reinstall error message."""
        support = tmp_path / "support" / "ds4"
        support.mkdir(parents=True)
        settings = DS4Settings(support_dir=str(support))

        status = inspect_ds4_support(
            settings,
            base_path=tmp_path,
            system="Darwin",
            machine="arm64",
        )

        assert status.ready is False
        assert status.binary_missing is True
        assert "LICENSE" in status.missing_files
        assert "metal/flash_attn.metal" in status.missing_files
        message = status.error_message()
        assert message is not None
        assert "missing DS4 binary" in message
        assert "missing DS4 support files" in message
        assert str(support.resolve()) in message

    def test_binary_override_does_not_require_support_dir_binary(self, tmp_path):
        """Advanced binary override still validates Metal/support files."""
        support = tmp_path / "support" / "ds4"
        _write_complete_support_tree(support)
        (support / DS4_SERVER_BINARY).unlink()
        override = tmp_path / "custom" / "ds4-server"
        override.parent.mkdir()
        override.write_bytes(b"#!/bin/sh\n")
        override.chmod(0o755)
        settings = DS4Settings(
            support_dir=str(support),
            binary_path=str(override),
        )

        status = inspect_ds4_support(
            settings,
            base_path=tmp_path,
            system="Darwin",
            machine="arm64",
        )

        assert status.ready is True
        assert status.binary_path == override.resolve()
        assert DS4_SERVER_BINARY not in status.missing_files

    def test_non_executable_binary_is_not_ready(self, tmp_path):
        """Existing but non-executable ds4-server is reported distinctly."""
        support = tmp_path / "support" / "ds4"
        _write_complete_support_tree(support, executable=False)
        settings = DS4Settings(support_dir=str(support))

        status = inspect_ds4_support(
            settings,
            base_path=tmp_path,
            system="Darwin",
            machine="arm64",
        )

        assert status.ready is False
        assert status.binary_not_executable is True
        assert "not executable" in (status.error_message() or "")

    def test_unsupported_platform_is_reported(self, tmp_path):
        """Support inspection enforces macOS Apple Silicon v1 target."""
        support = tmp_path / "support" / "ds4"
        _write_complete_support_tree(support)
        settings = DS4Settings(support_dir=str(support))

        status = inspect_ds4_support(
            settings,
            base_path=tmp_path,
            system="Linux",
            machine="arm64",
        )

        assert status.ready is False
        assert status.unsupported_platform is True
        assert "macOS Apple Silicon" in (status.error_message() or "")

    def test_require_ds4_support_raises_clear_error(self, tmp_path):
        """Callers can raise a clear error instead of probing status manually."""
        settings = DS4Settings(support_dir=str(tmp_path / "missing"))

        with pytest.raises(DS4SupportError) as exc_info:
            require_ds4_support(
                settings,
                base_path=tmp_path,
                system="Darwin",
                machine="arm64",
            )

        assert "missing DS4 binary" in str(exc_info.value)


class TestDS4BundledSupport:
    """Tests for bundled app-resource DS4 support discovery/install."""

    def test_find_bundled_support_dir_from_env(self, tmp_path):
        source = tmp_path / "bundle" / BUNDLED_DS4_SUPPORT_DIR_NAME
        source.mkdir(parents=True)

        found = find_bundled_ds4_support_dir(
            env={BUNDLED_DS4_SUPPORT_ENV: str(source)},
            module_file=tmp_path / "Resources" / "omlx" / "ds4_support.py",
        )

        assert found == source.resolve()

    def test_find_bundled_support_dir_next_to_app_resources(self, tmp_path):
        resources = tmp_path / "oMLX.app" / "Contents" / "Resources"
        source = resources / BUNDLED_DS4_SUPPORT_DIR_NAME
        source.mkdir(parents=True)
        module_file = resources / "omlx" / "ds4_support.py"
        module_file.parent.mkdir()
        module_file.write_text("# module\n")

        found = find_bundled_ds4_support_dir(env={}, module_file=module_file)

        assert found == source.resolve()

    def test_install_bundled_support_files_to_default_dir(self, tmp_path):
        source = tmp_path / "Resources" / BUNDLED_DS4_SUPPORT_DIR_NAME
        _write_complete_support_tree(source)
        base_path = tmp_path / "base"

        result = install_bundled_ds4_support_files(
            DS4Settings(),
            base_path=base_path,
            source_dir=source,
        )

        assert result is not None
        assert result.destination_dir == (base_path / "support" / "ds4").resolve()
        assert (base_path / "support" / "ds4" / DS4_SERVER_BINARY).is_file()
        assert (base_path / "support" / "ds4" / "metal" / "dense.metal").is_file()

    def test_install_bundled_support_skips_custom_support_dir(self, tmp_path):
        source = tmp_path / "Resources" / BUNDLED_DS4_SUPPORT_DIR_NAME
        _write_complete_support_tree(source)
        custom = tmp_path / "custom-support"

        result = install_bundled_ds4_support_files(
            DS4Settings(support_dir=str(custom)),
            base_path=tmp_path / "base",
            source_dir=source,
        )

        assert result is None
        assert not custom.exists()


class TestDS4SupportCopy:
    """Tests for copying bundled DS4 support files."""

    def test_copy_required_support_files(self, tmp_path):
        """Only required support files are copied into the destination."""
        source = tmp_path / "resources" / "ds4"
        destination = tmp_path / "support" / "ds4"
        _write_complete_support_tree(source)
        (source / "ignored.txt").write_text("ignore me")

        result = copy_ds4_support_files(source, destination)

        assert result.source_dir == source.resolve()
        assert result.destination_dir == destination.resolve()
        assert len(result.copied_files) == len(required_ds4_support_relative_paths())
        assert (destination / DS4_SERVER_BINARY).is_file()
        assert os.access(destination / DS4_SERVER_BINARY, os.X_OK)
        assert (destination / "metal" / "flash_attn.metal").is_file()
        assert not (destination / "ignored.txt").exists()

    def test_copy_skips_existing_files_without_overwrite(self, tmp_path):
        """Existing files are preserved unless overwrite is requested."""
        source = tmp_path / "resources" / "ds4"
        destination = tmp_path / "support" / "ds4"
        _write_complete_support_tree(source)
        _write_complete_support_tree(destination)
        (destination / "README.md").write_text("keep\n")

        result = copy_ds4_support_files(source, destination)

        assert result.copied_files == ()
        assert (destination / "README.md").read_text() == "keep\n"

    def test_copy_overwrites_when_requested(self, tmp_path):
        """Explicit overwrite refreshes already-copied support files."""
        source = tmp_path / "resources" / "ds4"
        destination = tmp_path / "support" / "ds4"
        _write_complete_support_tree(source)
        _write_complete_support_tree(destination)
        (source / "README.md").write_text("new\n")
        (destination / "README.md").write_text("old\n")

        result = copy_ds4_support_files(source, destination, overwrite=True)

        assert result.copied_files
        assert (destination / "README.md").read_text() == "new\n"

    def test_copy_rejects_incomplete_source_tree(self, tmp_path):
        """Missing bundled files fail clearly; no fetch/build is attempted."""
        source = tmp_path / "resources" / "ds4"
        source.mkdir(parents=True)
        destination = tmp_path / "support" / "ds4"

        with pytest.raises(DS4SupportError) as exc_info:
            copy_ds4_support_files(source, destination)

        assert "Bundled DS4 support files are incomplete" in str(exc_info.value)
        assert not destination.exists()
