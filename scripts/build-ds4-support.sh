#!/usr/bin/env bash
# Build/stage the prebuilt DS4 support tree consumed by the macOS app bundle.
#
# The runtime app never builds or fetches DS4. Release builders run this script
# ahead of apps/omlx-mac/Scripts/build.sh so the bundle can copy the validated
# packaging/DS4Support tree into Contents/Resources/DS4Support.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/build-ds4-support.sh [options]

Build ds4-server from a local upstream ds4 checkout and stage the validated
runtime support tree for the macOS app bundle.

Options:
  --source DIR     ds4 source checkout (default: $OMLX_DS4_SOURCE_DIR,
                   ../ds4, then ../ds4-apetersson)
  --out DIR        Destination support tree (default: $OMLX_DS4_SUPPORT_OUT or
                   packaging/DS4Support)
  --skip-build     Do not run make; validate/copy an already-built ds4-server
                   from the source tree (also OMLX_DS4_SKIP_BUILD=1)
  -h, --help       Show this help

Environment:
  PYTHON_BIN       Python used to run omlx.ds4_support validation/copy helper
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

log() {
    printf '==> %s\n' "$*"
}

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"

SOURCE_DIR="${OMLX_DS4_SOURCE_DIR:-}"
OUT_DIR="${OMLX_DS4_SUPPORT_OUT:-$REPO_ROOT/packaging/DS4Support}"
SKIP_BUILD="${OMLX_DS4_SKIP_BUILD:-0}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --source)
            [ "$#" -ge 2 ] || die "--source requires a directory"
            SOURCE_DIR="$2"
            shift 2
            ;;
        --out)
            [ "$#" -ge 2 ] || die "--out requires a directory"
            OUT_DIR="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

if [ -z "$SOURCE_DIR" ]; then
    if [ -d "$REPO_ROOT/../ds4" ]; then
        SOURCE_DIR="$REPO_ROOT/../ds4"
    elif [ -d "$REPO_ROOT/../ds4-apetersson" ]; then
        SOURCE_DIR="$REPO_ROOT/../ds4-apetersson"
    else
        die "no DS4 source found; pass --source or set OMLX_DS4_SOURCE_DIR"
    fi
fi

SOURCE_DIR="$(CDPATH= cd -- "$SOURCE_DIR" && pwd)" || die "DS4 source not found: $SOURCE_DIR"
mkdir -p "$(dirname -- "$OUT_DIR")"
OUT_DIR="$(CDPATH= cd -- "$(dirname -- "$OUT_DIR")" && pwd)/$(basename -- "$OUT_DIR")"

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
        PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
    else
        PYTHON_BIN="$(command -v python3)"
    fi
fi
[ -x "$PYTHON_BIN" ] || die "PYTHON_BIN is not executable: $PYTHON_BIN"

if [ "$SKIP_BUILD" = "1" ]; then
    log "Skipping DS4 build; using existing $SOURCE_DIR/ds4-server"
else
    [ -f "$SOURCE_DIR/Makefile" ] || die "DS4 source has no Makefile: $SOURCE_DIR"
    log "Building ds4-server in $SOURCE_DIR"
    make -C "$SOURCE_DIR" ds4-server
fi

[ -x "$SOURCE_DIR/ds4-server" ] || die "missing executable ds4-server in $SOURCE_DIR"

log "Staging DS4 support files into $OUT_DIR"
rm -rf "$OUT_DIR"
PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" - "$SOURCE_DIR" "$OUT_DIR" <<'PY'
import sys
from omlx.ds4_support import copy_ds4_support_files

source, destination = sys.argv[1], sys.argv[2]
result = copy_ds4_support_files(source, destination, overwrite=True)
print(f"copied {len(result.copied_files)} DS4 support files")
PY

log "DS4 support tree ready: $OUT_DIR"
