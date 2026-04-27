"""NoiseConfig: dataclass contract + TOML loader + walk-up discovery.

The user-facing surface is `.aelfrice.toml` at the project root or any
ancestor. The dataclass is the implementation; the TOML schema is what
power users edit. Tests cover the schema, the word-boundary safety
(initials don't strip filename extensions), case-insensitivity, and
the resilience contract (missing / malformed / wrong-typed configs
degrade silently to defaults).
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import pytest

from aelfrice.noise_filter import (
    DEFAULT_MIN_WORDS,
    NoiseConfig,
    is_noise,
)


# --- Defaults -----------------------------------------------------------


def test_default_config_drops_all_four_categories() -> None:
    cfg = NoiseConfig.default()
    assert cfg.drop_headings is True
    assert cfg.drop_checklists is True
    assert cfg.drop_fragments is True
    assert cfg.drop_license is True


def test_default_config_min_words_is_four() -> None:
    assert NoiseConfig.default().min_words == DEFAULT_MIN_WORDS


def test_default_config_excludes_are_empty() -> None:
    cfg = NoiseConfig.default()
    assert cfg.exclude_words == ()
    assert cfg.exclude_phrases == ()


# --- exclude_words: word-boundary safety --------------------------------


def test_exclude_words_drops_standalone_word() -> None:
    """The exact case the user flagged: filtering 'jso' should drop a
    paragraph mentioning the standalone token 'jso'."""
    cfg = NoiseConfig(exclude_words=("jso",))
    text = "This belief is owned by jso and concerns the publish path."
    assert is_noise(text, cfg) is True


def test_exclude_words_does_not_strip_substring_matches() -> None:
    """The exact pitfall the user flagged: filtering 'jso' must NOT
    drop a paragraph mentioning 'json'. Word-boundary regex is the
    fix."""
    cfg = NoiseConfig(exclude_words=("jso",))
    text = "We parse json files using the stdlib json module."
    assert is_noise(text, cfg) is False


def test_exclude_words_does_match_when_token_is_followed_by_punct() -> None:
    """Word-boundary semantics: hyphens, periods, and commas are
    non-word characters, so 'jso-files' / 'jso.' / 'jso,' all contain
    the standalone token 'jso' and match. This is the desired
    behaviour — the user adding 'jso' as a stop-word does mean drop
    paragraphs that name them, including in compound directory names."""
    cfg = NoiseConfig(exclude_words=("jso",))
    assert is_noise("Owned by jso-files directory.", cfg) is True
    assert is_noise("Authored by jso, reviewed elsewhere.", cfg) is True
    assert is_noise("Final touch by jso. Shipping today.", cfg) is True


def test_exclude_words_does_not_match_inside_alphanumeric_run() -> None:
    """The pitfall the user flagged: 'json', 'jsonify', 'jsodb' must
    not match because 'jso' is part of a longer word."""
    cfg = NoiseConfig(exclude_words=("jso",))
    assert is_noise("We parse json files using stdlib json module.", cfg) is False
    assert is_noise("The jsonify helper exports configured payloads.", cfg) is False
    assert is_noise("The jsodb backend works fine for our use case.", cfg) is False


def test_exclude_words_is_case_insensitive() -> None:
    cfg = NoiseConfig(exclude_words=("draft",))
    assert is_noise("This is a DRAFT paragraph for review", cfg) is True
    assert is_noise("This is a Draft paragraph for review", cfg) is True


def test_exclude_words_multiple_entries() -> None:
    cfg = NoiseConfig(exclude_words=("draft", "wip"))
    assert is_noise("Marked WIP and not yet ready for landing", cfg) is True
    assert is_noise("Tagged DRAFT pending review", cfg) is True
    assert is_noise("This paragraph is fully ready and reviewed", cfg) is False


def test_exclude_words_empty_strings_are_ignored() -> None:
    """Empty strings would compile to a regex that matches everything
    — treat them as no-op."""
    cfg = NoiseConfig(exclude_words=("",))
    assert is_noise("Any text passes through fine", cfg) is False


# --- exclude_phrases: literal substring match ---------------------------


def test_exclude_phrases_matches_anywhere_in_text() -> None:
    cfg = NoiseConfig(exclude_phrases=("Last updated:",))
    text = "Last updated: 2026-04-27. This file is auto-generated."
    assert is_noise(text, cfg) is True


def test_exclude_phrases_match_in_middle_of_paragraph() -> None:
    cfg = NoiseConfig(exclude_phrases=("TODO:",))
    text = "Some real prose here. TODO: refactor this. More prose."
    assert is_noise(text, cfg) is True


def test_exclude_phrases_is_case_insensitive() -> None:
    cfg = NoiseConfig(exclude_phrases=("Last updated:",))
    assert is_noise("LAST UPDATED: 2026-04-27", cfg) is True
    assert is_noise("last updated: 2026-04-27", cfg) is True


def test_exclude_phrases_matches_substring_inside_word() -> None:
    """Phrase match is literal — no word boundaries. Caller chooses
    whether to space-pad. This is the fundamental difference from
    exclude_words."""
    cfg = NoiseConfig(exclude_phrases=("foo",))
    text = "The foobar function works as expected for inputs."
    assert is_noise(text, cfg) is True


def test_exclude_phrases_empty_strings_are_ignored() -> None:
    cfg = NoiseConfig(exclude_phrases=("",))
    assert is_noise("Any text passes through fine", cfg) is False


# --- Toggles ------------------------------------------------------------


def test_disable_fragments_lets_short_text_through() -> None:
    cfg = NoiseConfig(drop_fragments=False)
    assert is_noise("two words", cfg) is False


def test_disable_headings_lets_heading_through() -> None:
    cfg = NoiseConfig(drop_headings=False)
    assert is_noise("## Architecture overview reference", cfg) is False


def test_disable_checklists_lets_checklist_through() -> None:
    cfg = NoiseConfig(drop_checklists=False)
    text = "- [ ] task one goes here\n- [x] task two goes here"
    assert is_noise(text, cfg) is False


def test_disable_license_lets_copyright_through() -> None:
    cfg = NoiseConfig(drop_license=False)
    text = "Copyright (c) 2026 ExampleCorp. All rights reserved."
    assert is_noise(text, cfg) is False


def test_min_words_override_changes_threshold() -> None:
    """min_words=2 means only 0- or 1-word paragraphs are fragments."""
    cfg = NoiseConfig(min_words=2)
    assert is_noise("two words", cfg) is False
    assert is_noise("oneword", cfg) is True


def test_min_words_zero_disables_fragment_check() -> None:
    cfg = NoiseConfig(min_words=0)
    assert is_noise("anything", cfg) is False


# --- TOML loader: from_mapping ------------------------------------------


def test_from_mapping_empty_section_returns_default() -> None:
    cfg = NoiseConfig.from_mapping({})
    assert cfg == NoiseConfig.default()


def test_from_mapping_disable_fragments_only() -> None:
    cfg = NoiseConfig.from_mapping({"disable": ["fragments"]})
    assert cfg.drop_fragments is False
    assert cfg.drop_headings is True
    assert cfg.drop_checklists is True
    assert cfg.drop_license is True


def test_from_mapping_disable_multiple_categories() -> None:
    cfg = NoiseConfig.from_mapping(
        {"disable": ["fragments", "license", "headings"]}
    )
    assert cfg.drop_fragments is False
    assert cfg.drop_license is False
    assert cfg.drop_headings is False
    assert cfg.drop_checklists is True


def test_from_mapping_disable_is_case_insensitive() -> None:
    cfg = NoiseConfig.from_mapping({"disable": ["Fragments", "LICENSE"]})
    assert cfg.drop_fragments is False
    assert cfg.drop_license is False


def test_from_mapping_unknown_disable_token_silently_ignored() -> None:
    """Typo in the disable list does not turn the whole filter off."""
    cfg = NoiseConfig.from_mapping(
        {"disable": ["fragmints", "lisence"]}
    )
    assert cfg == NoiseConfig.default()


def test_from_mapping_min_words_override() -> None:
    cfg = NoiseConfig.from_mapping({"min_words": 2})
    assert cfg.min_words == 2


def test_from_mapping_negative_min_words_clamped_to_zero() -> None:
    cfg = NoiseConfig.from_mapping({"min_words": -5})
    assert cfg.min_words == 0


def test_from_mapping_non_int_min_words_logs_and_defaults() -> None:
    err = io.StringIO()
    cfg = NoiseConfig.from_mapping(
        {"min_words": "three"}, stderr=err,  # type: ignore[arg-type]
    )
    assert cfg.min_words == DEFAULT_MIN_WORDS
    assert "min_words" in err.getvalue()


def test_from_mapping_bool_min_words_logs_and_defaults() -> None:
    """Python bool is int — but a bool in TOML is almost certainly a
    typo. Reject."""
    err = io.StringIO()
    cfg = NoiseConfig.from_mapping(
        {"min_words": True}, stderr=err,
    )
    assert cfg.min_words == DEFAULT_MIN_WORDS


def test_from_mapping_exclude_words_loaded() -> None:
    cfg = NoiseConfig.from_mapping(
        {"exclude_words": ["jso", "DRAFT"]}
    )
    assert cfg.exclude_words == ("jso", "DRAFT")
    assert is_noise("owned by jso for review", cfg) is True
    assert is_noise("we parse json files daily", cfg) is False


def test_from_mapping_exclude_phrases_loaded() -> None:
    cfg = NoiseConfig.from_mapping(
        {"exclude_phrases": ["Last updated:", "TODO:"]}
    )
    assert cfg.exclude_phrases == ("Last updated:", "TODO:")
    assert is_noise("Last updated: 2026-04-27", cfg) is True


def test_from_mapping_non_list_exclude_words_logs_and_skips() -> None:
    err = io.StringIO()
    cfg = NoiseConfig.from_mapping(
        {"exclude_words": "jso"}, stderr=err,  # type: ignore[arg-type]
    )
    assert cfg.exclude_words == ()
    assert "exclude_words" in err.getvalue()


def test_from_mapping_non_string_entries_skipped() -> None:
    err = io.StringIO()
    cfg = NoiseConfig.from_mapping(
        {"exclude_words": ["jso", 42, "wip"]},  # type: ignore[list-item]
        stderr=err,
    )
    assert cfg.exclude_words == ("jso", "wip")
    assert "non-string entry" in err.getvalue()


def test_from_mapping_unknown_keys_ignored() -> None:
    """Forward-compat: future versions may add fields. Unknown keys
    do not break loading."""
    cfg = NoiseConfig.from_mapping(
        {"min_words": 3, "future_setting": "value"}
    )
    assert cfg.min_words == 3


# --- TOML loader: from_toml_path ----------------------------------------


def test_from_toml_path_reads_file(tmp_path: Path) -> None:
    cfg_path = tmp_path / ".aelfrice.toml"
    cfg_path.write_text(
        '[noise]\n'
        'min_words = 3\n'
        'exclude_words = ["jso"]\n'
        'exclude_phrases = ["TODO:"]\n',
        encoding="utf-8",
    )
    cfg = NoiseConfig.from_toml_path(cfg_path)
    assert cfg.min_words == 3
    assert cfg.exclude_words == ("jso",)
    assert cfg.exclude_phrases == ("TODO:",)


def test_from_toml_path_no_noise_section_returns_default(tmp_path: Path) -> None:
    cfg_path = tmp_path / ".aelfrice.toml"
    cfg_path.write_text("[other]\nfield = 1\n", encoding="utf-8")
    cfg = NoiseConfig.from_toml_path(cfg_path)
    assert cfg == NoiseConfig.default()


def test_from_toml_path_malformed_toml_logs_and_defaults(tmp_path: Path) -> None:
    cfg_path = tmp_path / ".aelfrice.toml"
    cfg_path.write_text("this is = not valid [[ toml", encoding="utf-8")
    err = io.StringIO()
    cfg = NoiseConfig.from_toml_path(cfg_path, stderr=err)
    assert cfg == NoiseConfig.default()
    assert "malformed TOML" in err.getvalue()


def test_from_toml_path_nonexistent_file_logs_and_defaults(tmp_path: Path) -> None:
    err = io.StringIO()
    cfg = NoiseConfig.from_toml_path(
        tmp_path / "missing.toml", stderr=err,
    )
    assert cfg == NoiseConfig.default()
    assert "cannot read" in err.getvalue()


# --- discover: walk-up --------------------------------------------------


def test_discover_finds_config_in_start_dir(tmp_path: Path) -> None:
    (tmp_path / ".aelfrice.toml").write_text(
        '[noise]\nmin_words = 7\n', encoding="utf-8",
    )
    cfg = NoiseConfig.discover(tmp_path)
    assert cfg.min_words == 7


def test_discover_walks_up_to_ancestor(tmp_path: Path) -> None:
    (tmp_path / ".aelfrice.toml").write_text(
        '[noise]\nmin_words = 6\n', encoding="utf-8",
    )
    deep = tmp_path / "src" / "module" / "sub"
    deep.mkdir(parents=True)
    cfg = NoiseConfig.discover(deep)
    assert cfg.min_words == 6


def test_discover_first_file_wins(tmp_path: Path) -> None:
    """Closer config beats farther one."""
    (tmp_path / ".aelfrice.toml").write_text(
        '[noise]\nmin_words = 6\n', encoding="utf-8",
    )
    inner = tmp_path / "subproject"
    inner.mkdir()
    (inner / ".aelfrice.toml").write_text(
        '[noise]\nmin_words = 9\n', encoding="utf-8",
    )
    cfg = NoiseConfig.discover(inner)
    assert cfg.min_words == 9


def test_discover_no_file_returns_default(tmp_path: Path) -> None:
    """Walk to filesystem root finds nothing — return default."""
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    cfg = NoiseConfig.discover(deep)
    # We can't assert exact equality because a real .aelfrice.toml
    # might exist on the user's home tree above tmp_path on some
    # systems. Assert the dataclass shape instead.
    assert isinstance(cfg, NoiseConfig)


# --- Frozen invariant ---------------------------------------------------


def test_noise_config_is_frozen() -> None:
    """Catch accidental mutation in is_noise or callers."""
    cfg = NoiseConfig.default()
    with pytest.raises(Exception):  # noqa: BLE001
        cfg.min_words = 99  # type: ignore[misc]
