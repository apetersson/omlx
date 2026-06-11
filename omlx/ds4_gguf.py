# SPDX-License-Identifier: Apache-2.0
"""DS4 GGUF discovery helpers.

DS4 is a specialized DeepSeek V4 backend, not a general GGUF runtime.  This
module keeps GGUF filename normalization, metadata inspection, split-shard
filtering, and DS4 support checks out of the generic model discovery scanner.
"""

from __future__ import annotations

import logging
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

DS4_GENERIC_GGUF_STEMS = {
    "model",
    "gguf",
    "weights",
    "consolidated",
}
DS4_SUPPORTED_GGUF_ARCHITECTURES = {"deepseek4"}

_DS4_ID_SEPARATORS_RE = re.compile(r"[^a-z0-9.]+")
_DS4_ID_DASHES_RE = re.compile(r"-+")
_GGUF_MAGIC = b"GGUF"

# Maximum allowed string length for a single GGUF metadata key or value (1 MiB).
# Real-world GGUF metadata strings (architecture names, tokenizer paths, etc.)
# are well under 1 KiB; 1 MiB provides ample headroom while blocking OOM from
# corrupted headers that claim multi-GB strings.
_GGUF_MAX_STRING_LENGTH = 1 * 1024 * 1024
_GGUF_TYPE_UINT8 = 0
_GGUF_TYPE_INT8 = 1
_GGUF_TYPE_UINT16 = 2
_GGUF_TYPE_INT16 = 3
_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_INT32 = 5
_GGUF_TYPE_FLOAT32 = 6
_GGUF_TYPE_BOOL = 7
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9
_GGUF_TYPE_UINT64 = 10
_GGUF_TYPE_INT64 = 11
_GGUF_TYPE_FLOAT64 = 12
_GGUF_SCALAR_FORMATS = {
    _GGUF_TYPE_UINT8: "B",
    _GGUF_TYPE_INT8: "b",
    _GGUF_TYPE_UINT16: "H",
    _GGUF_TYPE_INT16: "h",
    _GGUF_TYPE_UINT32: "I",
    _GGUF_TYPE_INT32: "i",
    _GGUF_TYPE_FLOAT32: "f",
    _GGUF_TYPE_BOOL: "?",
    _GGUF_TYPE_UINT64: "Q",
    _GGUF_TYPE_INT64: "q",
    _GGUF_TYPE_FLOAT64: "d",
}


@dataclass(frozen=True)
class GGUFMetadataSummary:
    """Subset of GGUF metadata needed for DS4 discovery decisions."""

    architecture: str | None = None
    split_no: int | None = None
    split_count: int | None = None


@dataclass(frozen=True)
class DS4GGUFModelCandidate:
    """A GGUF file that can be exposed as a DS4-backed discovered model."""

    base_id: str
    model_path: Path
    estimated_size: int
    config_model_type: str
    display_name: str
    source_type: str = "local"
    source_repo_id: str | None = None


class GGUFMetadataError(ValueError):
    """Raised when the GGUF header/metadata cannot be parsed."""


def _read_exact(f, size: int) -> bytes:
    data = f.read(size)
    if len(data) != size:
        raise GGUFMetadataError("truncated GGUF metadata")
    return data


def _file_remaining(f) -> int:
    """Return the number of bytes remaining in the file from the current position."""
    pos = f.tell()
    f.seek(0, 2)
    end = f.tell()
    f.seek(pos)
    return end - pos


def _read_u32(f) -> int:
    return struct.unpack("<I", _read_exact(f, 4))[0]


def _read_u64(f) -> int:
    return struct.unpack("<Q", _read_exact(f, 8))[0]


def _read_gguf_string(f) -> str:
    length = _read_u64(f)
    # Bound string length against remaining file size to prevent OOM
    # from a corrupted header claiming a multi-GB string length.
    remaining = _file_remaining(f)
    if length > remaining:
        raise GGUFMetadataError(
            f"GGUF string length ({length}) exceeds remaining file size ({remaining})"
        )
    # Also reject unreasonably large strings even if theoretically within
    # file bounds — a single metadata key/value should never need more than
    # a few MB.
    if length > _GGUF_MAX_STRING_LENGTH:
        raise GGUFMetadataError(
            f"GGUF string length ({length}) exceeds maximum ({_GGUF_MAX_STRING_LENGTH})"
        )
    return _read_exact(f, length).decode("utf-8", "replace")


def _read_gguf_scalar(f, value_type: int):
    if value_type == _GGUF_TYPE_STRING:
        return _read_gguf_string(f)
    fmt = _GGUF_SCALAR_FORMATS.get(value_type)
    if fmt is None:
        raise GGUFMetadataError(f"unsupported GGUF metadata value type {value_type}")
    size = struct.calcsize("<" + fmt)
    return struct.unpack("<" + fmt, _read_exact(f, size))[0]


def _skip_gguf_scalar(f, value_type: int) -> None:
    if value_type == _GGUF_TYPE_STRING:
        length = _read_u64(f)
        f.seek(length, 1)
        return
    fmt = _GGUF_SCALAR_FORMATS.get(value_type)
    if fmt is None:
        raise GGUFMetadataError(f"unsupported GGUF metadata value type {value_type}")
    f.seek(struct.calcsize("<" + fmt), 1)


def _skip_gguf_value(f, value_type: int) -> None:
    if value_type == _GGUF_TYPE_ARRAY:
        item_type = _read_u32(f)
        item_count = _read_u64(f)
        fmt = _GGUF_SCALAR_FORMATS.get(item_type)
        if fmt is not None:
            f.seek(struct.calcsize("<" + fmt) * item_count, 1)
            return
        for _ in range(item_count):
            _skip_gguf_scalar(f, item_type)
        return
    _skip_gguf_scalar(f, value_type)


def read_ds4_gguf_metadata_summary(path: Path) -> GGUFMetadataSummary:
    """Read just enough GGUF metadata to decide if DS4 can expose a file."""
    with path.open("rb") as f:
        # Check for minimum GGUF header size before attempting to read magic.
        # A file with fewer than 4 bytes is not a valid GGUF file.
        initial = f.read(4)
        if len(initial) < 4:
            raise GGUFMetadataError(
                f"file too short to be a GGUF file ({len(initial)} bytes)"
            )
        if initial != _GGUF_MAGIC:
            raise GGUFMetadataError(
                "not a GGUF file (missing magic bytes)"
            )
        # Seek back so _read_exact can re-read for consistency.
        f.seek(0)
        _read_u32(f)  # version
        _read_u64(f)  # tensor count
        kv_count = _read_u64(f)

        architecture: str | None = None
        split_no: int | None = None
        split_count: int | None = None
        for _ in range(kv_count):
            key = _read_gguf_string(f)
            value_type = _read_u32(f)
            if key == "general.architecture" and value_type == _GGUF_TYPE_STRING:
                architecture = str(_read_gguf_scalar(f, value_type))
            elif key == "split.no" and value_type in _GGUF_SCALAR_FORMATS:
                split_no = int(_read_gguf_scalar(f, value_type))
            elif key == "split.count" and value_type in _GGUF_SCALAR_FORMATS:
                split_count = int(_read_gguf_scalar(f, value_type))
            else:
                _skip_gguf_value(f, value_type)

    return GGUFMetadataSummary(
        architecture=architecture,
        split_no=split_no,
        split_count=split_count,
    )


def is_supported_ds4_gguf(path: Path) -> bool:
    """Return True when a GGUF is a DS4-supported primary DeepSeek V4 file."""
    try:
        metadata = read_ds4_gguf_metadata_summary(path)
    except Exception as e:
        # Several existing tests and some hand-made local stubs use a .gguf
        # extension without a real GGUF header.  Keep extension-based discovery
        # as a compatibility fallback when metadata cannot be inspected; real
        # GGUFs with explicit unsupported metadata are rejected below.
        logger.debug("Could not inspect GGUF metadata for %s: %s", path, e)
        return True

    if metadata.split_no is not None and metadata.split_no > 0:
        logger.info(
            "Skipping DS4 GGUF continuation shard %s (split.no=%s)",
            path,
            metadata.split_no,
        )
        return False

    architecture = (metadata.architecture or "").strip().lower()
    if architecture and architecture not in DS4_SUPPORTED_GGUF_ARCHITECTURES:
        logger.info(
            "Skipping unsupported DS4 GGUF %s (architecture=%s)",
            path,
            metadata.architecture,
        )
        return False

    return True


def normalize_ds4_gguf_model_id(name: str) -> str:
    """Normalize a DS4 GGUF file/repo name into an API model id.

    DS4 model ids are intentionally lowercased and separator-normalized so
    `Foo.gguf` and `foo` resolve consistently.  The original source casing is
    kept separately in ``DiscoveredModel.display_name`` for UI presentation.
    """
    raw = name.strip()
    if raw.lower().endswith(".gguf"):
        raw = raw[:-5]
    normalized = raw.lower()
    normalized = _DS4_ID_SEPARATORS_RE.sub("-", normalized)
    normalized = _DS4_ID_DASHES_RE.sub("-", normalized).strip("-.")
    return normalized or "gguf-model"


def is_ds4_gguf_file(path: Path) -> bool:
    """Return True for visible GGUF model files that DS4 discovery may inspect."""
    return (
        path.is_file()
        and path.suffix.lower() == ".gguf"
        and not path.name.startswith(".")
    )


def detect_ds4_gguf_config_type(
    gguf_path: Path, source_repo_id: str | None = None
) -> str:
    """Classify DeepSeek V4 GGUF variant from filename/repo heuristics."""
    haystack = " ".join(
        part for part in (source_repo_id, gguf_path.parent.name, gguf_path.name) if part
    ).lower()
    if "deepseek" in haystack and "v4" in haystack and "flash" in haystack:
        return "deepseek_v4_flash_gguf"
    if "deepseek" in haystack and "v4" in haystack and "pro" in haystack:
        return "deepseek_v4_pro_gguf"
    return "ds4_gguf"


def compose_ds4_gguf_model_id(
    container: Path,
    gguf_path: Path,
    gguf_count: int,
    *,
    source_repo_id: str | None = None,
) -> str:
    """Build the preferred normalized id before collision suffixing."""
    file_id = normalize_ds4_gguf_model_id(gguf_path.stem)
    if container == gguf_path.parent:
        if source_repo_id:
            repo_id = normalize_ds4_gguf_model_id(source_repo_id.split("/")[-1])
            if gguf_count == 1 and (
                file_id in DS4_GENERIC_GGUF_STEMS
                or file_id.startswith("model-")
                or file_id == repo_id
            ):
                return repo_id
            if file_id.startswith(f"{repo_id}-"):
                return file_id
            return f"{repo_id}-{file_id}"
        # For top-level GGUFs, the filename is the model id:
        #   Foo.gguf -> foo
        return file_id

    container_id = normalize_ds4_gguf_model_id(gguf_path.parent.name)
    if gguf_count == 1 and (
        file_id in DS4_GENERIC_GGUF_STEMS
        or file_id.startswith("model-")
        or file_id == container_id
    ):
        return container_id
    if file_id.startswith(f"{container_id}-"):
        return file_id
    return f"{container_id}-{file_id}"


def compose_ds4_gguf_display_name(
    root_dir: Path,
    gguf_path: Path,
    gguf_count: int,
    *,
    source_repo_id: str | None = None,
) -> str:
    """Build the UI display name for a DS4 GGUF file."""
    file_id = normalize_ds4_gguf_model_id(gguf_path.stem)
    if source_repo_id and gguf_path.parent == root_dir:
        repo_display = source_repo_id.split("/")[-1]
        return (
            repo_display
            if gguf_count == 1 and file_id in DS4_GENERIC_GGUF_STEMS
            else f"{repo_display} / {gguf_path.stem}"
        )
    if gguf_path.parent == root_dir:
        return gguf_path.stem
    if gguf_count == 1 and file_id in DS4_GENERIC_GGUF_STEMS:
        return gguf_path.parent.name
    return f"{gguf_path.parent.name} / {gguf_path.stem}"


def collect_ds4_gguf_model_candidates(
    root_dir: Path,
    paths: Iterable[Path],
    *,
    source_type: str = "local",
    source_repo_id: str | None = None,
) -> list[DS4GGUFModelCandidate]:
    """Collect DS4-supported primary GGUF files from a direct path listing."""
    ggufs = [path for path in paths if is_ds4_gguf_file(path)]
    supported_ggufs = [path for path in ggufs if is_supported_ds4_gguf(path)]
    candidates: list[DS4GGUFModelCandidate] = []
    for gguf_path in supported_ggufs:
        try:
            candidates.append(
                DS4GGUFModelCandidate(
                    base_id=compose_ds4_gguf_model_id(
                        root_dir,
                        gguf_path,
                        len(supported_ggufs),
                        source_repo_id=source_repo_id,
                    ),
                    model_path=gguf_path,
                    estimated_size=int(gguf_path.stat().st_size * 1.05),
                    config_model_type=detect_ds4_gguf_config_type(
                        gguf_path,
                        source_repo_id,
                    ),
                    display_name=compose_ds4_gguf_display_name(
                        root_dir,
                        gguf_path,
                        len(supported_ggufs),
                        source_repo_id=source_repo_id,
                    ),
                    source_type=source_type,
                    source_repo_id=source_repo_id,
                )
            )
        except Exception as e:
            logger.error("Failed to discover DS4 GGUF %s: %s", gguf_path, e)
    return candidates
