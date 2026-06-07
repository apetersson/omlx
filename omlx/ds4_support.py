# SPDX-License-Identifier: Apache-2.0
"""Support-file checks for the managed DS4/GGUF backend.

The DS4 process backend is launched from a user support directory rather than
built or fetched at runtime.  This module owns the small, testable pieces that
validate/copy those support files before later process-launch code consumes
those paths.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .settings import DEFAULT_BASE_PATH, DS4Settings

DS4_SERVER_BINARY = "ds4-server"
BUNDLED_DS4_SUPPORT_ENV = "OMLX_BUNDLED_DS4_SUPPORT_DIR"
BUNDLED_DS4_SUPPORT_DIR_NAME = "DS4Support"
DS4_SUPPORT_FILES: tuple[str, ...] = (
    "LICENSE",
    "README.md",
)
DS4_REQUIRED_CLI_FLAGS: tuple[str, ...] = ("--ssd-streaming",)
DS4_METAL_FILES: tuple[str, ...] = (
    "flash_attn.metal",
    "dense.metal",
    "moe.metal",
    "dsv4_hc.metal",
    "unary.metal",
    "dsv4_kv.metal",
    "dsv4_rope.metal",
    "dsv4_misc.metal",
    "argsort.metal",
    "cpy.metal",
    "concat.metal",
    "get_rows.metal",
    "sum_rows.metal",
    "softmax.metal",
    "repeat.metal",
    "glu.metal",
    "norm.metal",
    "bin.metal",
    "set_rows.metal",
)


class DS4SupportError(RuntimeError):
    """Raised when DS4 support files are unavailable or incomplete."""


@dataclass(frozen=True)
class DS4SupportStatus:
    """Result of inspecting the configured DS4 support directory."""

    support_dir: Path
    binary_path: Path
    missing_files: tuple[str, ...]
    binary_missing: bool
    binary_not_executable: bool
    unsupported_platform: bool
    platform_name: str
    binary_capability_error: str | None = None

    @property
    def ready(self) -> bool:
        """True when all required support files are present and launchable."""
        return not (
            self.missing_files
            or self.binary_missing
            or self.binary_not_executable
            or self.unsupported_platform
            or self.binary_capability_error
        )

    def error_message(self) -> str | None:
        """Return a clear user-facing error, or ``None`` when ready."""
        problems: list[str] = []
        if self.unsupported_platform:
            problems.append(
                "DS4 backend is supported only on macOS Apple Silicon "
                f"(detected {self.platform_name})"
            )
        if self.binary_missing:
            problems.append(f"missing DS4 binary: {self.binary_path}")
        elif self.binary_not_executable:
            problems.append(f"DS4 binary is not executable: {self.binary_path}")
        elif self.binary_capability_error:
            problems.append(
                f"DS4 binary is incompatible: {self.binary_path}: "
                f"{self.binary_capability_error}"
            )
        if self.missing_files:
            missing = ", ".join(self.missing_files)
            problems.append(f"missing DS4 support files under {self.support_dir}: {missing}")
        if not problems:
            return None
        return "; ".join(problems)


@dataclass(frozen=True)
class DS4SupportCopyResult:
    """Files copied into the DS4 support directory."""

    source_dir: Path
    destination_dir: Path
    copied_files: tuple[Path, ...]


def _platform_name(system: str | None = None, machine: str | None = None) -> str:
    system = system or platform.system()
    machine = machine or platform.machine()
    return f"{system} {machine}".strip()


def is_ds4_supported_platform(
    system: str | None = None, machine: str | None = None
) -> bool:
    """Return True for the v1 DS4 target platform: macOS Apple Silicon."""
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower()
    return system == "darwin" and machine in {"arm64", "aarch64"}


def required_ds4_support_relative_paths(*, include_binary: bool = True) -> tuple[str, ...]:
    """Return required support paths relative to the DS4 support directory."""
    paths: list[str] = []
    if include_binary:
        paths.append(DS4_SERVER_BINARY)
    paths.extend(DS4_SUPPORT_FILES)
    paths.extend(f"metal/{name}" for name in DS4_METAL_FILES)
    return tuple(paths)


def _missing_relative_paths(root: Path, relative_paths: Iterable[str]) -> tuple[str, ...]:
    return tuple(rel for rel in relative_paths if not (root / rel).is_file())


def _inspect_ds4_binary_capabilities(binary_path: Path) -> str | None:
    """Return a compatibility error when the DS4 binary lacks required flags."""
    try:
        completed = subprocess.run(
            [str(binary_path), "--help"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "timed out while probing --help"
    except OSError as exc:
        return f"failed to run --help: {exc}"

    output = f"{completed.stdout}\n{completed.stderr}"
    if completed.returncode != 0:
        return f"--help exited with status {completed.returncode}"
    missing = tuple(flag for flag in DS4_REQUIRED_CLI_FLAGS if flag not in output)
    if missing:
        return "missing required CLI option(s): " + ", ".join(missing)
    return None


def inspect_ds4_support(
    settings: DS4Settings | None = None,
    *,
    base_path: str | Path | None = None,
    system: str | None = None,
    machine: str | None = None,
) -> DS4SupportStatus:
    """Inspect configured DS4 support files without mutating the filesystem."""
    settings = settings or DS4Settings()
    base = Path(base_path).expanduser().resolve() if base_path else DEFAULT_BASE_PATH
    support_dir = settings.get_support_dir(base)
    binary_path = settings.get_binary_path(base)
    binary_override = settings.binary_path is not None
    missing_files = _missing_relative_paths(
        support_dir,
        required_ds4_support_relative_paths(include_binary=not binary_override),
    )
    binary_missing = not binary_path.is_file()
    binary_not_executable = binary_path.is_file() and not os.access(binary_path, os.X_OK)
    platform_name = _platform_name(system, machine)
    unsupported_platform = not is_ds4_supported_platform(system, machine)
    binary_capability_error = None
    if not (binary_missing or binary_not_executable or unsupported_platform):
        binary_capability_error = _inspect_ds4_binary_capabilities(binary_path)

    return DS4SupportStatus(
        support_dir=support_dir,
        binary_path=binary_path,
        missing_files=missing_files,
        binary_missing=binary_missing,
        binary_not_executable=binary_not_executable,
        unsupported_platform=unsupported_platform,
        platform_name=platform_name,
        binary_capability_error=binary_capability_error,
    )


def require_ds4_support(
    settings: DS4Settings | None = None,
    *,
    base_path: str | Path | None = None,
    system: str | None = None,
    machine: str | None = None,
) -> DS4SupportStatus:
    """Return support status or raise ``DS4SupportError`` with a clear message."""
    status = inspect_ds4_support(
        settings,
        base_path=base_path,
        system=system,
        machine=machine,
    )
    if not status.ready:
        raise DS4SupportError(status.error_message() or "DS4 support is unavailable")
    return status


def find_bundled_ds4_support_dir(
    *,
    env: Mapping[str, str] | None = None,
    module_file: str | Path | None = None,
) -> Path | None:
    """Locate bundled DS4 support files shipped next to the app resources.

    The Swift app bundle copies Python sources into ``Contents/Resources/omlx``
    and DS4 runtime files into ``Contents/Resources/DS4Support``.  A hidden env
    override keeps tests and alternate packagers deterministic.
    """
    environment = os.environ if env is None else env
    if bundled_override := environment.get(BUNDLED_DS4_SUPPORT_ENV):
        return Path(bundled_override).expanduser().resolve()

    module_path = Path(module_file or __file__).expanduser().resolve()
    try:
        resources_dir = module_path.parents[1]
    except IndexError:
        return None

    for name in (BUNDLED_DS4_SUPPORT_DIR_NAME, "ds4"):
        candidate = resources_dir / name
        if candidate.is_dir():
            return candidate
    return None


def install_bundled_ds4_support_files(
    settings: DS4Settings | None = None,
    *,
    base_path: str | Path | None = None,
    source_dir: str | Path | None = None,
    overwrite: bool = False,
) -> DS4SupportCopyResult | None:
    """Copy bundled DS4 support files into the default user support dir.

    Custom ``ds4.support_dir`` values are treated as an explicit user choice and
    are left untouched.  Returning ``None`` means no bundled source was present
    or the user configured a custom support directory.
    """
    settings = settings or DS4Settings()
    if settings.support_dir is not None:
        return None

    base = Path(base_path).expanduser().resolve() if base_path else DEFAULT_BASE_PATH
    source = (
        Path(source_dir).expanduser().resolve()
        if source_dir is not None
        else find_bundled_ds4_support_dir()
    )
    if source is None:
        return None
    return copy_ds4_support_files(
        source,
        settings.get_support_dir(base),
        overwrite=overwrite,
    )


def copy_ds4_support_files(
    source_dir: str | Path,
    destination_dir: str | Path,
    *,
    overwrite: bool = False,
) -> DS4SupportCopyResult:
    """Copy required DS4 support files from a bundled resource directory.

    The copy is intentionally deterministic: only the required binary, license,
    README, and Metal source files are copied.  Missing bundled inputs raise a
    clear error instead of attempting to rebuild or fetch DS4.
    """
    source = Path(source_dir).expanduser().resolve()
    destination = Path(destination_dir).expanduser().resolve()
    required = required_ds4_support_relative_paths(include_binary=True)
    missing = _missing_relative_paths(source, required)
    if missing:
        raise DS4SupportError(
            "Bundled DS4 support files are incomplete under "
            f"{source}: {', '.join(missing)}"
        )
    capability_error = _inspect_ds4_binary_capabilities(source / DS4_SERVER_BINARY)
    if capability_error:
        raise DS4SupportError(
            f"Bundled DS4 binary is incompatible under {source}: {capability_error}"
        )

    copied: list[Path] = []
    for rel in required:
        src = source / rel
        dst = destination / rel
        if dst.exists() and not overwrite:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(dst)

    return DS4SupportCopyResult(
        source_dir=source,
        destination_dir=destination,
        copied_files=tuple(copied),
    )
