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

DS4/GGUF backend releases include a vendored macOS arm64 support tree under
`omlx/vendor/ds4/darwin-arm64` containing `ds4-server`, `LICENSE`, `README.md`,
`manifest.json`, and `metal/*.metal` files. The app bundle step uses that tree
by default and copies only the validated runtime files into
`Contents/Resources/DS4Support`. To override it with a freshly staged local
build, run:

```bash
scripts/build-ds4-support.sh --source ../ds4
OMLX_REQUIRE_DS4_BUNDLE=1 apps/omlx-mac/Scripts/build.sh release
```

The helper writes `packaging/DS4Support/`, which the bundle step picks up before
the vendored fallback. Alternatively, point the bundle step at another validated
tree with `OMLX_DS4_BUNDLE_SOURCE=/path/to/ds4-support` (and set
`OMLX_REQUIRE_DS4_BUNDLE=1` in release jobs). Validation probes
`ds4-server --help` and rejects stale binaries that lack the current OMLX launch
flags such as `--ssd-streaming`. On first server start from the app bundle, oMLX
seeds the user support directory (`~/Library/Application Support/oMLX` /
`~/.omlx` base path `support/ds4`) from that bundled resource. No DS4 build or
network fetch happens at runtime.

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

The Swift build (`build.sh release`) produces
`apps/omlx-mac/build/Stage/oMLX.app` directly — no DMG step. To install:

1. Drag `apps/omlx-mac/build/Stage/oMLX.app` to `/Applications`, or
   `open` it in-place to launch from `apps/omlx-mac/build/Stage/`.
2. Launch the app (appears in the menubar).
3. Walk through the first-run wizard (Storage + API key), then Start
   Server.

> The DMGs on the [Releases](https://github.com/jundot/omlx/releases)
> page are produced by an off-tree maintainer pipeline, not by anything
> in this repo. End users follow the Releases install path; this
> section is for developers building from source.
