# oMLX macOS App Packaging

Produces the venvstacks Python layers that the Swift macOS bundle
embeds. Building the user-facing `.app` itself is owned by
[`apps/omlx-mac/Scripts/build.sh`](../apps/omlx-mac/Scripts/build.sh);
this directory only hands it a `_export/` tree of Python layers.

> **PyObjC menubar retired.** The earlier Python / PyObjC menubar
> (`packaging/omlx_app/`) and the `packaging/build.py` `.app` + DMG
> pipeline that wrapped it have been removed. The Swift app under
> [`apps/omlx-mac/`](../apps/omlx-mac/) is now the only macOS bundle.

## Requirements

- macOS 15.0+ (Sequoia) — required by MLX ≥ 0.29.2
- Apple Silicon (M1/M2/M3/M4)
- Python 3.11+ on the host
- venvstacks (installed via `pip install -e ".[dev]"` from the repo
  root, or any of `uv`, `pipx run`)

## Build

```bash
# Re-export the venvstacks layers (cold ~10-20 min, warm ~4 min)
python packaging/build.py --venvstacks-only

# Stable fingerprint of the inputs that drive the export shape — used
# by build.sh to decide whether to re-export
python packaging/build.py --print-fingerprint
```

Then the Swift bundle:

```bash
apps/omlx-mac/Scripts/build.sh release             # full bundle
apps/omlx-mac/Scripts/build.sh release --no-rebuild-donor   # reuse _export/
```

DS4/GGUF backend releases should provide a prebuilt support tree containing
`ds4-server`, `LICENSE`, `README.md`, and `metal/*.metal` files. To build and
stage that tree from a local `ds4-apetersson` checkout before the app bundle
step:

```bash
scripts/build-ds4-support.sh --source ../ds4-apetersson
OMLX_REQUIRE_DS4_BUNDLE=1 apps/omlx-mac/Scripts/build.sh release
```

The helper writes `packaging/DS4Support/`, which the bundle step picks up
automatically. Alternatively, point the bundle step at another validated tree
with `OMLX_DS4_BUNDLE_SOURCE=/path/to/ds4-support` (and set
`OMLX_REQUIRE_DS4_BUNDLE=1` in release jobs); the build copies only the
validated runtime files into `Contents/Resources/DS4Support`. Validation also
probes `ds4-server --help` and rejects stale binaries that lack the current OMLX
launch flags such as `--ssd-streaming`, so release artifacts must be built from
a DS4 checkout new enough for the managed backend. On first server start from
the app bundle, oMLX seeds the user support directory
(`~/Library/Application Support/oMLX` / `~/.omlx` base path `support/ds4`) from
that bundled resource. No DS4 build or network fetch happens at runtime.

## Output

```
packaging/
├── _build/         # venvstacks intermediate layers
├── _export/        # venvstacks export — embedded into the .app
└── _wheels/        # cached local wheels (e.g. mlx + mlx-metal pins)
```

## Layer Configuration

| Layer | Contents |
|-------|----------|
| Runtime (`cpython-3.11`) | Python 3.11 |
| Framework (`mlx-base`) | MLX, mlx-lm, mlx-vlm, FastAPI, transformers, mlx-audio, paroquant, spaCy |

No application layer — the Swift app is the application surface.

## Installation

1. Open the DMG produced by the Swift build.
2. Drag `oMLX.app` to Applications.
3. Launch the app (appears in the menubar).
4. Walk through the first-run wizard (Storage + API key), then Start
   Server.
