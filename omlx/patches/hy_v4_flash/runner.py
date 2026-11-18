# SPDX-License-Identifier: Apache-2.0
"""Offline, single-request Hy4 Flash text inference. All paths must be local."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

from .artifact import Artifact, member


def render_user(artifact, text, raw, effort):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(artifact.root / "tokenizer.json"))
    if not raw:
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        env = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
        )
        template = env.from_string((artifact.root / "chat_template.jinja").read_text())
        text = template.render(
            messages=[{"role": "user", "content": text}],
            tools=[],
            add_generation_prompt=True,
            reasoning_effort=effort,
        )
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    if not ids:
        raise ValueError("prompt produced no tokens")
    return tokenizer, ids


def weight_shapes(model):
    from mlx.utils import tree_flatten

    return {n: tuple(w.shape) for n, w in tree_flatten(model.parameters())}


def load_weights(model, artifact, experts=None):
    """Check the complete index; evaluate one shard at a time, never all source tensors."""
    import mlx.core as mx

    from .quant_io import read_header

    skip = set() if experts is None else experts.excluded_names
    shapes = weight_shapes(model)
    required = artifact.weight_map.keys() - skip
    if shapes.keys() != required:
        raise ValueError(
            f"model/checkpoint mismatch: missing {sorted(shapes.keys() - required)[:8]}; "
            f"unexpected {sorted(required - shapes.keys())[:8]}"
        )
    by_file = defaultdict(set)
    for key, shard in artifact.weight_map.items():
        by_file[shard].add(key)
    dtypes = {"U32": mx.uint32, "I32": mx.int32, "F32": mx.float32, "BF16": mx.bfloat16}
    for shard, keys in sorted(by_file.items()):
        path = member(artifact.root, shard)
        with path.open("rb", buffering=0) as f:
            header = read_header(f)
        if header.keys() != keys:
            raise ValueError(f"{shard}: index and actual header disagree")
        wanted = sorted(keys - skip)
        if not wanted:
            continue
        if experts is not None and shard in experts.mixed_shards:
            weights = {n: experts.read_resident(shard, n) for n in wanted}
        else:
            if keys & skip:
                raise ValueError(
                    "refusing to read excluded expert banks into unified memory"
                )
            weights = mx.load(str(path))
        for n, array in weights.items():
            if (
                tuple(array.shape) != shapes[n]
                or array.dtype != dtypes[header[n].spec.dtype]
            ):
                raise ValueError(f"wrong tensor type or shape: {n}")
        model.load_weights(list(weights.items()), strict=False)
        mx.eval(list(weights.values()))
        del weights


def greedy(model, ids, *, eos, count, chunk):
    """Greedy generation with a fresh request cache and distinct prefill/decode timers."""
    import mlx.core as mx

    cache = model.make_cache()

    def advance(batch):
        last = model(mx.array([batch], dtype=mx.int32), cache=cache, logits_to_keep=1)[
            :, -1
        ]
        mx.eval(last)
        if not mx.all(mx.isfinite(last)).item():
            raise ValueError("non-finite model logits")
        return last

    start = time.monotonic()
    for i in range(0, len(ids), chunk):
        last = advance(ids[i : i + chunk])
    first = time.monotonic()
    out = []
    steps = 0
    for i in range(count):
        token = int(mx.argmax(last, axis=-1).item())
        out.append(token)
        if token in eos or i + 1 == count:
            break
        last = advance([token])
        steps += 1
    finished = time.monotonic()
    return out, {
        "prompt_tokens": len(ids),
        "new_tokens": len(out),
        "stopped_on_eos": out[-1] in eos,
        "prefill_seconds": first - start,
        "decode_steps": steps,
        "decode_seconds": finished - first,
        "decode_tokens_per_second": steps / (finished - first)
        if steps and finished > first
        else None,
    }


def generate(args, artifact):
    import mlx.core as mx

    from .model import Model, ModelArgs
    from .residency import open_tables

    if (
        args.max_tokens < 1
        or not 1 <= args.prefill_chunk <= 256
        or args.expert_cache_gib < 0
    ):
        raise ValueError("invalid token, prefill chunk, or expert cache bound")
    tok, ids = render_user(
        artifact, args.prompt, args.raw_prompt, args.reasoning_effort
    )
    if len(ids) + args.max_tokens > min(
        artifact.config["max_position_embeddings"], args.context_limit
    ):
        raise ValueError("prompt and requested generation exceed the context limit")
    eos = artifact.config["eos_token_id"]
    eos = {eos} if isinstance(eos, int) else set(eos)
    setup = time.monotonic()
    with ExitStack() as stack:
        model = Model(ModelArgs.from_dict(artifact.config))
        model.eval()
        experts = None
        if args.expert_cache_gib:
            # Limit allocator recycling as well as the explicit per-expert LRU.
            previous_cache_limit = mx.set_cache_limit(0)
            stack.callback(mx.set_cache_limit, previous_cache_limit)
            mx.clear_cache()
            from .bounded import install_bounded_experts

            experts = install_bounded_experts(
                model,
                artifact.root,
                artifact.config,
                artifact.weight_map,
                args.expert_cache_gib << 30,
            )
            stack.callback(experts.close)
        load_weights(model, artifact, experts)
        with open_tables(
            artifact.root / "manifest.json",
            first_root=args.source,
            late_root=args.ssd_dir,
            receipt=args.receipt,
        ) as provider:
            model.set_engram_tables(provider)
            setup_seconds = time.monotonic() - setup
            tokens, metrics = greedy(
                model, ids, eos=eos, count=args.max_tokens, chunk=args.prefill_chunk
            )
            text = tok.decode(tokens, skip_special_tokens=True)
            result = {
                "text": text,
                "token_ids": tokens,
                "load_and_wire_seconds": setup_seconds,
                **metrics,
                "engram": provider.stats(),
                "mlx_peak_bytes": mx.get_peak_memory(),
                "expert_mode": "bounded" if experts else "resident",
            }
            if experts:
                result["expert_cache"] = experts.stats
            if args.json:
                print(json.dumps(result, ensure_ascii=False, allow_nan=False))
            else:
                print(text)
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in result.items()
                            if k not in ("text", "token_ids")
                        }
                    ),
                    file=sys.stderr,
                )


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["convert"]:
        from .convert import main as convert

        return convert(argv[1:])
    p = argparse.ArgumentParser(prog="omlx hy4-flash", description=__doc__)
    subs = p.add_subparsers(dest="command", required=True)
    subs.add_parser("convert", help="stream and verify an oQh8 text export")
    for cmd, helptext in [
        ("inspect", "check the local oQh export"),
        ("verify-tables", "authenticate the four BF16 source files"),
        ("generate", "run a single text generation"),
    ]:
        a = subs.add_parser(cmd, help=helptext)
        a.add_argument(
            "--model", required=True, type=Path, help="converted native oQh8 directory"
        )
        a.add_argument(
            "--verify-core",
            action="store_true",
            help="also hash every core shard in full",
        )
        if cmd in ("verify-tables", "generate"):
            a.add_argument(
                "--source",
                required=True,
                type=Path,
                help="original verified BF16 snapshot (first table)",
            )
            a.add_argument(
                "--ssd-dir",
                required=True,
                type=Path,
                help="internal SSD directory containing the three verified late shards",
            )
            a.add_argument(
                "--receipt",
                required=True,
                type=Path,
                help="private local table verification receipt",
            )
        if cmd == "generate":
            a.add_argument("--prompt", required=True)
            a.add_argument("--raw-prompt", action="store_true")
            a.add_argument(
                "--reasoning-effort",
                choices=["no_think", "high", "max"],
                default="no_think",
            )
            a.add_argument("--max-tokens", type=int, default=32)
            a.add_argument("--prefill-chunk", type=int, default=16)
            a.add_argument(
                "--context-limit",
                type=int,
                default=4096,
                help="request guard, not architecture context size",
            )
            a.add_argument(
                "--expert-cache-gib",
                type=int,
                default=0,
                help="0: resident experts (fast); positive: bounded expert reads",
            )
            a.add_argument(
                "--json",
                action="store_true",
                help="include token IDs, performance and live residency counters",
            )
    args = p.parse_args(argv)
    try:
        a = Artifact.open(args.model, verify_core=args.verify_core)
        if args.command == "inspect":
            print(json.dumps(a.summary(), indent=2))
        elif args.command == "verify-tables":
            from .residency import prepare_table_receipt

            prepare_table_receipt(
                a.root / "manifest.json",
                first_root=args.source,
                late_root=args.ssd_dir,
                receipt=args.receipt,
            )
        elif args.command == "generate":
            generate(args, a)
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        p.exit(1, f"hy4-flash: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
