# Hy4 Flash: local oQh8 text inference on Apple Silicon

This branch adds a single-request text runner, not an upstream model registration or a GUI server preset. Use the fork CLI below, not the stock v0.8.0 app, mlx-vlm's Hy4 Preview loader, or a generic oQ converter. The [original instruct checkpoint](https://huggingface.co/tencent/Hy4-flash/tree/d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959) and its license remain separate. Vision, DFly speculative decoding, continuous batching, cluster and shared prefix caches are **not** supported here.

## Convert and run

Use this branch in an isolated oMLX development environment with its pinned mlx-lm and MLX 0.32.4. If another oMLX is already installed, invoke `python -m omlx.cli` from this checkout using that environment. No inference command downloads Python model code. The native oQh extension is optional: signed MLX Hadamard plus `mx.gather_qmm` is the equivalent GPU path when the extension is absent.

~~~sh
# Set SRC, OUT and SSD to absolute directories. OUT must not already exist.
REV=d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959
hf download tencent/Hy4-flash --revision "$REV" --local-dir "$SRC"
hf cache verify tencent/Hy4-flash --revision "$REV" --local-dir "$SRC" --fail-on-missing-files
python -m omlx.cli hy4-flash convert --source "$SRC" --dry-run
python -m omlx.cli hy4-flash convert --source "$SRC" --output "$OUT" --chunk-mib 32
python -m omlx.cli hy4-flash inspect --model "$OUT" --verify-core
~~~

The converted directory is **not standalone**. Keep the original first Engram shard in `$SRC`, plus the three original late shards (#3, #2, #1) in `$SSD` on the internal Data SSD. Do not stage the first shard or a combined 118 GiB copy. Existing byte-identical GGUF-stage shards can be reused. Authentication receipts are small, private, machine-specific, and must live outside both source directories. Create a **new**, nonexisting receipt pathname under a private directory; keep all weight files immutable after verification.

~~~sh
umask 077
# The parent of $SSD must exist. --check preflights without copying.
python -m omlx.cli hy4-flash stage-tables --model "$OUT" --source "$SRC" --ssd-dir "$SSD"
python -m omlx.cli hy4-flash verify-tables --model "$OUT" --source "$SRC" --ssd-dir "$SSD" --receipt "$RECEIPT"
# Set MLX_ENABLE_TF32 before importing MLX for strict FP32 routing/indexer reproducibility.
MLX_ENABLE_TF32=0 python -m omlx.cli hy4-flash generate \
  --model "$OUT" --source "$SRC" --ssd-dir "$SSD" --receipt "$RECEIPT" \
  --prompt 'What is the capital of France? Answer in one word.' \
  --prefill-chunk 8 --max-tokens 8 --expert-cache-gib 4 --json
~~~

A positive `--expert-cache-gib` reads only selected packed experts and bounds its explicit LRU. It prioritizes coexistence with other programs, not speed: cold expert reads can approach **4.46 GiB per decoded token**. The core's location determines the speed of those reads. For the fast resident path, omit the flag or pass 0, **only when enough unified memory is available**. It keeps all experts in memory, but still reads only selected late Engram rows. Use one model at a time; a 220 GB budget is not an OS reservation. Neither command raises global wired-memory limits or adjusts swap. Context limit defaults to 4096 as a request guard; long-context performance is not qualified. JSON output distinguishes prefill time, actual decode iterations, terminal token IDs, MLX peak allocation, and per-table lookup/residency counters.

## Precision and placement

- All 82 fused routed banks (41 layers × gate/up and down) use normalized signed 256-point Walsh–Hadamard rotation, group-128 affine **8-bit** weights in the rotated basis, and activation-side rotation before the gathered matrix multiplication. The converter uses deterministic RTN, **not calibrated GPTQ**; this is not perplexity parity or a claim to match a calibrated oQh quant. Other 1,350 text tensors are copied byte-for-byte, including routers, attention/indexers and iHC. Source tensor types are preserved, not blanket-cast to BF16.
- Four trained BF16 Engram tables are external, unchanged: layers 1, 11, 21 and 31 (zero-based). The first complete shard is memory-mapped, `mlock`ed, counted with `mincore` and retained for the lifetime of the model. If locking fails, generation fails; there is no quiet pageable fallback. The last three open only by FD after authentication; unique requested 192-byte rows are read with positional I/O. The trained FP8 round-trip applies to *gathered activations*, not stored weights. Positional reads need not be physical SSD misses if macOS already cached data.
- Original snapshot: ~410.29 GiB tensor payload. Output core: **155.754 GiB**, 141 safetensors shards, 1,678 tensors. Resident core plus first BF16 table is **185.26 GiB** before KV, scratch, the OS and other users. Internal late stage is **88.52 GiB**, shared rather than duplicated. Conversion checks all original full-shard checksums (including omitted modalities) and rereads all outputs before atomic publication.

[Reference architecture](https://github.com/huggingface/transformers/blob/c2ec850510a37978ea83df226589623d5f2be776/src/transformers/models/hy_v4_flash/modeling_hy_v4_flash.py). Scoped regression suite: `python -m pytest -q tests/test_hy4_flash_*.py`. Includes complete public schema geometry without loading weights, independently calculated iHC/MoE/attention/table cases, short and sparse-context full-vs-incremental outputs, corrupted artifact failures, actual tiny mmap wiring and bounded/resident expert parity.
