#!/usr/bin/env python3
"""Build a 128 GiB-oriented text-only oQ4e Qwen3.8-Flash-Next checkpoint."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

from safetensors import safe_open

from omlx.oq import quantize_oq_streaming

DEFAULT_SOURCE = Path("/Volumes/Samsung_4TB/models/Qwen3.8-Flash-Next")
DEFAULT_OUTPUT = Path("/Volumes/Samsung_4TB/models/Qwen3.8-Flash-Next-oQ4e-128GB")
EXPECTED_SHARDS = 131
MINIMUM_FREE_GIB = 190


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--target-bpw", type=float, default=4.6)
    parser.add_argument("--hard-cap-bpw", type=float, default=4.7)
    return parser.parse_args()


def _preflight(source: Path, output: Path) -> list[Path]:
    if not (source / "config.json").is_file():
        raise SystemExit(f"missing source config: {source / 'config.json'}")
    shards = sorted(source.glob("model-*.safetensors"))
    if len(shards) != EXPECTED_SHARDS:
        raise SystemExit(
            f"source download is incomplete: {len(shards)}/{EXPECTED_SHARDS} shards"
        )
    if output.exists():
        raise SystemExit(f"output already exists: {output}")

    # Opening each header catches interrupted files without reading tensor data.
    for shard in shards:
        try:
            with safe_open(str(shard), framework="numpy") as handle:
                next(iter(handle.keys()), None)
        except Exception as exc:
            raise SystemExit(
                f"invalid or incomplete shard {shard.name}: {exc}"
            ) from exc

    free = shutil.disk_usage(output.parent).free
    if free < MINIMUM_FREE_GIB * 1024**3:
        raise SystemExit(
            f"only {free / 1024**3:.1f} GiB free; at least "
            f"{MINIMUM_FREE_GIB} GiB is required for the Q3 calibration proxy "
            "and final oQ4e output"
        )
    return shards


def _wired_limit_mb() -> int:
    try:
        result = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "iogpu.wired_limit_mb"],
            check=True,
            capture_output=True,
            text=True,
        )
        return int(result.stdout.strip() or 0)
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0


def main() -> None:
    args = _arguments()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    _preflight(source, output)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    wired_mb = _wired_limit_mb()
    if wired_mb:
        logging.info("kernel Metal wired limit: %.2f GiB", wired_mb / 1024)
    else:
        logging.warning(
            "iogpu.wired_limit_mb is unset; conversion is safe, but serving "
            "the finished model benefits from raising it to your usual 118 GiB"
        )

    imatrix_cache = (
        output.parent
        / ".oqe_imatrix"
        / f"{source.name}-text-oQ4e-s{args.samples}-l{args.sequence_length}.npz"
    )

    def progress(phase, percent, detail="", _meta=None):
        logging.info("[%s %5.1f%%] %s", phase, percent, detail)

    quantize_oq_streaming(
        str(source),
        str(output),
        oq_level=4,
        group_size=64,
        progress_callback=progress,
        text_only=True,
        target_bpw=args.target_bpw,
        hard_cap_bpw=args.hard_cap_bpw,
        dtype="bfloat16",
        preserve_mtp=False,
        auto_proxy_sensitivity=True,
        trust_remote_code=False,
        enhanced=True,
        imatrix_cache_path=str(imatrix_cache),
        imatrix_reuse_cache=True,
        imatrix_strict=False,
        imatrix_num_samples=args.samples,
        imatrix_seq_length=args.sequence_length,
    )

    size = sum(path.stat().st_size for path in output.glob("*.safetensors"))
    logging.info("finished %s (%.2f GiB)", output, size / 1024**3)


if __name__ == "__main__":
    main()
