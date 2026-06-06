# SPDX-License-Identifier: Apache-2.0
"""Helpers for OMLX-managed DS4 per-model aliases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DS4AliasKind = Literal["chat", "reasoner", "think-max"]

DS4_ALIAS_SUFFIXES: dict[str, DS4AliasKind] = {
    "-chat": "chat",
    "-reasoner": "reasoner",
    "-think-max": "think-max",
}
DS4_ALIAS_ORDER: tuple[DS4AliasKind, ...] = ("chat", "reasoner", "think-max")
DS4_ALIAS_SUFFIX_BY_KIND: dict[DS4AliasKind, str] = {
    kind: suffix for suffix, kind in DS4_ALIAS_SUFFIXES.items()
}


@dataclass(frozen=True)
class DS4Alias:
    """Description of an OMLX-visible DS4 alias for one model entry."""

    alias_id: str
    base_model_id: str
    kind: DS4AliasKind
    ds4_model: str | None = None
    reasoning_effort: str | None = None


def ds4_aliases_for_model(model_id: str) -> tuple[DS4Alias, ...]:
    """Return the standard OMLX-visible DS4 aliases for *model_id*."""
    aliases: list[DS4Alias] = []
    for kind in DS4_ALIAS_ORDER:
        suffix = DS4_ALIAS_SUFFIX_BY_KIND[kind]
        aliases.append(
            DS4Alias(
                alias_id=f"{model_id}{suffix}",
                base_model_id=model_id,
                kind=kind,
                ds4_model=ds4_model_for_alias_kind(kind),
                reasoning_effort=ds4_reasoning_effort_for_alias_kind(kind),
            )
        )
    return tuple(aliases)


def parse_ds4_alias_id(alias_id: str) -> tuple[str, DS4AliasKind] | None:
    """Split a possible DS4 alias into ``(base_model_id, alias_kind)``.

    Matching is suffix-case-insensitive so OpenAI-compatible clients that vary
    casing still resolve normalized DS4 model ids, while the returned base keeps
    the caller-provided spelling for the engine pool's usual exact/case-insensitive
    lookup order.
    """
    lowered = alias_id.lower()
    for suffix, kind in DS4_ALIAS_SUFFIXES.items():
        if lowered.endswith(suffix) and len(alias_id) > len(suffix):
            return alias_id[: -len(suffix)], kind
    return None


def ds4_model_for_alias_kind(kind: DS4AliasKind) -> str | None:
    """Return the DS4-native model name implied by an alias kind, if any."""
    if kind == "chat":
        return "deepseek-chat"
    if kind == "reasoner":
        return "deepseek-reasoner"
    return None


def ds4_reasoning_effort_for_alias_kind(kind: DS4AliasKind) -> str | None:
    """Return the request-level reasoning_effort implied by an alias kind."""
    if kind == "think-max":
        return "max"
    return None
