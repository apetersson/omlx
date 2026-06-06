# SPDX-License-Identifier: Apache-2.0
"""Tests for DS4 per-model alias helpers."""

from omlx.ds4_aliases import (
    ds4_aliases_for_model,
    ds4_model_for_alias_kind,
    ds4_reasoning_effort_for_alias_kind,
    parse_ds4_alias_id,
)


def test_ds4_aliases_for_model_include_standard_suffixes():
    aliases = ds4_aliases_for_model("deepseek-v4-flash-q2")

    assert [alias.alias_id for alias in aliases] == [
        "deepseek-v4-flash-q2-chat",
        "deepseek-v4-flash-q2-reasoner",
        "deepseek-v4-flash-q2-think-max",
    ]
    assert [alias.kind for alias in aliases] == ["chat", "reasoner", "think-max"]


def test_ds4_alias_metadata_preserves_ds4_native_alias_meaning():
    aliases = {alias.kind: alias for alias in ds4_aliases_for_model("foo")}

    assert aliases["chat"].ds4_model == "deepseek-chat"
    assert aliases["reasoner"].ds4_model == "deepseek-reasoner"
    assert aliases["think-max"].ds4_model is None
    assert aliases["think-max"].reasoning_effort == "max"


def test_parse_ds4_alias_id_keeps_base_spelling_and_matches_suffix_case_insensitive():
    assert parse_ds4_alias_id("Foo-CHAT") == ("Foo", "chat")
    assert parse_ds4_alias_id("Foo-reasoner") == ("Foo", "reasoner")
    assert parse_ds4_alias_id("Foo-Think-Max") == ("Foo", "think-max")


def test_parse_ds4_alias_id_is_syntactic_and_rejects_empty_bases():
    assert parse_ds4_alias_id("deepseek-chat") == ("deepseek", "chat")
    assert parse_ds4_alias_id("-chat") is None
    assert parse_ds4_alias_id("deepseek-reasoner") == ("deepseek", "reasoner")


def test_ds4_alias_kind_mappings_are_forwarding_ready():
    assert ds4_model_for_alias_kind("chat") == "deepseek-chat"
    assert ds4_model_for_alias_kind("reasoner") == "deepseek-reasoner"
    assert ds4_model_for_alias_kind("think-max") is None
    assert ds4_reasoning_effort_for_alias_kind("think-max") == "max"
    assert ds4_reasoning_effort_for_alias_kind("chat") is None
