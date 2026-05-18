"""Unit tests for session_exclusions (#856)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aelfrice.session_exclusions import (
    EXCLUSIONS_FILENAME,
    SESSION_STATE_FILENAME,
    add_exclusion,
    clear_exclusions,
    exclusions_path,
    is_excluded,
    load_exclusions,
    read_current_session_id,
    save_exclusions,
    session_state_path,
)


def test_exclusions_path_format(tmp_path: Path) -> None:
    assert exclusions_path(tmp_path) == tmp_path / EXCLUSIONS_FILENAME


def test_load_returns_empty_when_file_missing(tmp_path: Path) -> None:
    assert load_exclusions(exclusions_path(tmp_path), "sid-1") == []


def test_load_returns_empty_when_session_id_none(tmp_path: Path) -> None:
    save_exclusions(exclusions_path(tmp_path), "sid-1", ["foo"])
    assert load_exclusions(exclusions_path(tmp_path), None) == []
    assert load_exclusions(exclusions_path(tmp_path), "") == []
    assert load_exclusions(exclusions_path(tmp_path), "   ") == []


def test_load_returns_empty_on_session_id_mismatch(tmp_path: Path) -> None:
    """Auto-clear semantics: previous session's exclusions do not leak."""
    save_exclusions(exclusions_path(tmp_path), "sid-old", ["foo", "bar"])
    assert load_exclusions(exclusions_path(tmp_path), "sid-new") == []


def test_load_returns_patterns_on_session_id_match(tmp_path: Path) -> None:
    save_exclusions(exclusions_path(tmp_path), "sid-1", ["foo", "bar"])
    assert load_exclusions(exclusions_path(tmp_path), "sid-1") == ["foo", "bar"]


def test_load_returns_empty_on_malformed_json(tmp_path: Path) -> None:
    path = exclusions_path(tmp_path)
    path.write_text("not json", encoding="utf-8")
    assert load_exclusions(path, "sid-1") == []


def test_load_returns_empty_on_non_dict_root(tmp_path: Path) -> None:
    path = exclusions_path(tmp_path)
    path.write_text("[]", encoding="utf-8")
    assert load_exclusions(path, "sid-1") == []


def test_load_skips_non_string_pattern_entries(tmp_path: Path) -> None:
    path = exclusions_path(tmp_path)
    path.write_text(
        json.dumps({"session_id": "sid-1", "patterns": ["ok", 42, None, ""]}),
        encoding="utf-8",
    )
    assert load_exclusions(path, "sid-1") == ["ok"]


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    path = exclusions_path(tmp_path)
    save_exclusions(path, "sid-1", ["a", "b"])
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw == {"session_id": "sid-1", "patterns": ["a", "b"]}


def test_save_deduplicates_and_preserves_order(tmp_path: Path) -> None:
    path = exclusions_path(tmp_path)
    save_exclusions(path, "sid-1", ["a", "b", "a", "c", "b"])
    assert load_exclusions(path, "sid-1") == ["a", "b", "c"]


def test_save_strips_empty_strings(tmp_path: Path) -> None:
    path = exclusions_path(tmp_path)
    save_exclusions(path, "sid-1", ["", "a", ""])
    assert load_exclusions(path, "sid-1") == ["a"]


def test_save_creates_parent_directory(tmp_path: Path) -> None:
    nested = tmp_path / "git" / "common" / "aelfrice"
    save_exclusions(exclusions_path(nested), "sid-1", ["a"])
    assert (nested / EXCLUSIONS_FILENAME).exists()


def test_add_exclusion_appends(tmp_path: Path) -> None:
    path = exclusions_path(tmp_path)
    assert add_exclusion(path, "sid-1", "foo") == ["foo"]
    assert add_exclusion(path, "sid-1", "bar") == ["foo", "bar"]


def test_add_exclusion_idempotent(tmp_path: Path) -> None:
    path = exclusions_path(tmp_path)
    add_exclusion(path, "sid-1", "foo")
    assert add_exclusion(path, "sid-1", "foo") == ["foo"]
    # Single entry on disk.
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["patterns"] == ["foo"]


def test_add_exclusion_overwrites_stale_session(tmp_path: Path) -> None:
    """Adding from a new session replaces the stale list."""
    path = exclusions_path(tmp_path)
    save_exclusions(path, "sid-old", ["x", "y"])
    assert add_exclusion(path, "sid-new", "z") == ["z"]
    assert load_exclusions(path, "sid-new") == ["z"]


def test_clear_exclusions_empties_list(tmp_path: Path) -> None:
    path = exclusions_path(tmp_path)
    add_exclusion(path, "sid-1", "foo")
    add_exclusion(path, "sid-1", "bar")
    clear_exclusions(path, "sid-1")
    assert load_exclusions(path, "sid-1") == []


def test_is_excluded_empty_patterns(tmp_path: Path) -> None:
    assert is_excluded("anything", []) is False


def test_is_excluded_substring_match(tmp_path: Path) -> None:
    assert is_excluded("foo bar baz", ["bar"]) is True
    assert is_excluded("foo bar baz", ["qux"]) is False


def test_is_excluded_case_insensitive(tmp_path: Path) -> None:
    assert is_excluded("Hello World", ["hello"]) is True
    assert is_excluded("Hello World", ["WORLD"]) is True


def test_is_excluded_skips_empty_patterns(tmp_path: Path) -> None:
    """An empty-string pattern would otherwise match everything."""
    assert is_excluded("anything", [""]) is False
    assert is_excluded("anything", ["", "x"]) is False


def test_is_excluded_multiple_patterns_any_match(tmp_path: Path) -> None:
    assert is_excluded("aelfrice is great", ["junk", "aelf", "other"]) is True


def test_session_state_path_format(tmp_path: Path) -> None:
    assert session_state_path(tmp_path) == tmp_path / SESSION_STATE_FILENAME


def test_read_current_session_id_missing_file(tmp_path: Path) -> None:
    assert read_current_session_id(tmp_path) is None


def test_read_current_session_id_happy_path(tmp_path: Path) -> None:
    (tmp_path / SESSION_STATE_FILENAME).write_text(
        json.dumps({"session_id": "sid-1"}), encoding="utf-8"
    )
    assert read_current_session_id(tmp_path) == "sid-1"


def test_read_current_session_id_malformed(tmp_path: Path) -> None:
    (tmp_path / SESSION_STATE_FILENAME).write_text("not json", encoding="utf-8")
    assert read_current_session_id(tmp_path) is None


def test_read_current_session_id_empty_string(tmp_path: Path) -> None:
    (tmp_path / SESSION_STATE_FILENAME).write_text(
        json.dumps({"session_id": ""}), encoding="utf-8"
    )
    assert read_current_session_id(tmp_path) is None


def test_read_current_session_id_non_string(tmp_path: Path) -> None:
    (tmp_path / SESSION_STATE_FILENAME).write_text(
        json.dumps({"session_id": 42}), encoding="utf-8"
    )
    assert read_current_session_id(tmp_path) is None


@pytest.mark.parametrize("bad_root", ["[]", "null", "42", '"string"'])
def test_read_current_session_id_non_dict_root(
    tmp_path: Path, bad_root: str
) -> None:
    (tmp_path / SESSION_STATE_FILENAME).write_text(bad_root, encoding="utf-8")
    assert read_current_session_id(tmp_path) is None
