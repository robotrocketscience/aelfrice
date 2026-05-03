"""Tests for the #218 AC6 collapse_duplicate_hashes flag.

Coverage:
- Default config has collapse OFF.
- When OFF, duplicate content hashes are NOT deduped.
- When ON, duplicate content hashes are deduped (first occurrence wins).
- Order of unique hits preserved after collapse.
- Telemetry n_returned records pre-collapse count.
- Telemetry total_chars records post-collapse size.
- n_unique_content_hashes equals number of distinct hashes pre-collapse.
- load_user_prompt_submit_config returns defaults when no .aelfrice.toml.
- load_user_prompt_submit_config reads collapse_duplicate_hashes from toml.
- Malformed TOML degrades to defaults.
- Wrong type for collapse_duplicate_hashes (non-bool) degrades to False.
"""
from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aelfrice.hook import (
    UserPromptSubmitConfig,
    _dedup_by_content_hash,
    load_user_prompt_submit_config,
    read_user_prompt_submit_telemetry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_belief(content: str, lock_level: int = 0, belief_id: str | None = None) -> object:
    b = MagicMock()
    b.content = content
    b.lock_level = lock_level
    b.id = belief_id or ("b" + content[:4].encode().hex())
    return b


def _write_toml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# UserPromptSubmitConfig defaults
# ---------------------------------------------------------------------------


def test_default_config_collapse_is_false() -> None:
    cfg = UserPromptSubmitConfig()
    assert cfg.collapse_duplicate_hashes is False


# ---------------------------------------------------------------------------
# _dedup_by_content_hash
# ---------------------------------------------------------------------------


def test_dedup_empty_list() -> None:
    assert _dedup_by_content_hash([]) == []


def test_dedup_no_duplicates_preserves_all() -> None:
    hits = [_make_belief("alpha"), _make_belief("beta"), _make_belief("gamma")]
    result = _dedup_by_content_hash(hits)
    assert len(result) == 3


def test_dedup_removes_exact_content_duplicates() -> None:
    hits = [
        _make_belief("same content"),
        _make_belief("unique"),
        _make_belief("same content"),  # duplicate
    ]
    result = _dedup_by_content_hash(hits)
    assert len(result) == 2
    assert result[0].content == "same content"
    assert result[1].content == "unique"


def test_dedup_preserves_first_occurrence_order() -> None:
    hits = [
        _make_belief("c"),
        _make_belief("a"),
        _make_belief("b"),
        _make_belief("a"),  # duplicate of index 1
        _make_belief("c"),  # duplicate of index 0
    ]
    result = _dedup_by_content_hash(hits)
    assert [h.content for h in result] == ["c", "a", "b"]


def test_dedup_all_same_content_returns_one() -> None:
    hits = [_make_belief("identical") for _ in range(5)]
    result = _dedup_by_content_hash(hits)
    assert len(result) == 1
    assert result[0].content == "identical"


# ---------------------------------------------------------------------------
# load_user_prompt_submit_config
# ---------------------------------------------------------------------------


def test_load_config_no_file_returns_defaults(tmp_path: Path) -> None:
    cfg = load_user_prompt_submit_config(tmp_path / "nonexistent")
    assert cfg.collapse_duplicate_hashes is False


def test_load_config_reads_collapse_true(tmp_path: Path) -> None:
    _write_toml(
        tmp_path / ".aelfrice.toml",
        "[user_prompt_submit_hook]\ncollapse_duplicate_hashes = true\n",
    )
    cfg = load_user_prompt_submit_config(tmp_path)
    assert cfg.collapse_duplicate_hashes is True


def test_load_config_reads_collapse_false(tmp_path: Path) -> None:
    _write_toml(
        tmp_path / ".aelfrice.toml",
        "[user_prompt_submit_hook]\ncollapse_duplicate_hashes = false\n",
    )
    cfg = load_user_prompt_submit_config(tmp_path)
    assert cfg.collapse_duplicate_hashes is False


def test_load_config_missing_section_returns_defaults(tmp_path: Path) -> None:
    _write_toml(tmp_path / ".aelfrice.toml", "[rebuilder]\nturn_window_n = 5\n")
    cfg = load_user_prompt_submit_config(tmp_path)
    assert cfg.collapse_duplicate_hashes is False


def test_load_config_malformed_toml_returns_defaults(tmp_path: Path) -> None:
    (tmp_path / ".aelfrice.toml").write_text("not: [valid toml", encoding="utf-8")
    serr = StringIO()
    cfg = load_user_prompt_submit_config(tmp_path, stderr=serr)
    assert cfg.collapse_duplicate_hashes is False
    assert "malformed TOML" in serr.getvalue()


def test_load_config_wrong_type_for_collapse_key(tmp_path: Path) -> None:
    _write_toml(
        tmp_path / ".aelfrice.toml",
        "[user_prompt_submit_hook]\ncollapse_duplicate_hashes = 1\n",
    )
    serr = StringIO()
    cfg = load_user_prompt_submit_config(tmp_path, stderr=serr)
    assert cfg.collapse_duplicate_hashes is False
    assert "expected bool" in serr.getvalue()


def test_load_config_walks_up_to_parent(tmp_path: Path) -> None:
    """Config file in parent dir is found when child has none."""
    child = tmp_path / "sub" / "project"
    child.mkdir(parents=True)
    _write_toml(
        tmp_path / ".aelfrice.toml",
        "[user_prompt_submit_hook]\ncollapse_duplicate_hashes = true\n",
    )
    cfg = load_user_prompt_submit_config(child)
    assert cfg.collapse_duplicate_hashes is True


# ---------------------------------------------------------------------------
# Integration: end-to-end collapse + telemetry accounting
# ---------------------------------------------------------------------------


def test_collapse_off_keeps_duplicates_in_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With collapse OFF, duplicates pass through to formatted output."""
    from aelfrice.hook import user_prompt_submit
    from aelfrice.models import LOCK_USER

    tel = tmp_path / "user_prompt_submit.jsonl"
    hits = [
        _make_belief("duplicate text"),
        _make_belief("unique text"),
        _make_belief("duplicate text"),  # same as first
    ]

    monkeypatch.setattr("aelfrice.hook._retrieve", lambda p, b, **_: hits)
    monkeypatch.setattr("aelfrice.hook._telemetry_path_for_db", lambda p: tel)
    monkeypatch.setattr("aelfrice.hook.db_path", lambda: tmp_path / "memory.db")
    monkeypatch.setattr(
        "aelfrice.hook.load_user_prompt_submit_config",
        lambda **_kw: UserPromptSubmitConfig(collapse_duplicate_hashes=False),
    )

    sin = StringIO(json.dumps({"prompt": "test"}))
    sout = StringIO()
    user_prompt_submit(stdin=sin, stdout=sout, stderr=StringIO())

    output = sout.getvalue()
    # All 3 should appear in the output (duplicates NOT removed).
    assert output.count("duplicate text") == 2

    records = read_user_prompt_submit_telemetry(tel)
    assert records[0]["n_returned"] == 3
    assert records[0]["n_unique_content_hashes"] == 2


def test_collapse_on_deduplicates_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With collapse ON, duplicates are removed before formatting."""
    from aelfrice.hook import user_prompt_submit

    tel = tmp_path / "user_prompt_submit.jsonl"
    hits = [
        _make_belief("duplicate text"),
        _make_belief("unique text"),
        _make_belief("duplicate text"),  # same as first
    ]

    monkeypatch.setattr("aelfrice.hook._retrieve", lambda p, b, **_: hits)
    monkeypatch.setattr("aelfrice.hook._telemetry_path_for_db", lambda p: tel)
    monkeypatch.setattr("aelfrice.hook.db_path", lambda: tmp_path / "memory.db")
    monkeypatch.setattr(
        "aelfrice.hook.load_user_prompt_submit_config",
        lambda **_kw: UserPromptSubmitConfig(collapse_duplicate_hashes=True),
    )

    sin = StringIO(json.dumps({"prompt": "test"}))
    sout = StringIO()
    user_prompt_submit(stdin=sin, stdout=sout, stderr=StringIO())

    output = sout.getvalue()
    # Only 1 instance of the duplicate content in output.
    assert output.count("duplicate text") == 1
    assert "unique text" in output


def test_telemetry_n_returned_is_pre_collapse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """n_returned always reflects the raw retrieval count, not post-collapse."""
    from aelfrice.hook import user_prompt_submit

    tel = tmp_path / "user_prompt_submit.jsonl"
    # 4 hits but only 2 unique content hashes
    hits = [
        _make_belief("a"),
        _make_belief("a"),
        _make_belief("b"),
        _make_belief("b"),
    ]

    monkeypatch.setattr("aelfrice.hook._retrieve", lambda p, b, **_: hits)
    monkeypatch.setattr("aelfrice.hook._telemetry_path_for_db", lambda p: tel)
    monkeypatch.setattr("aelfrice.hook.db_path", lambda: tmp_path / "memory.db")
    monkeypatch.setattr(
        "aelfrice.hook.load_user_prompt_submit_config",
        lambda **_kw: UserPromptSubmitConfig(collapse_duplicate_hashes=True),
    )

    sin = StringIO(json.dumps({"prompt": "test collapse accounting"}))
    user_prompt_submit(stdin=sin, stdout=StringIO(), stderr=StringIO())

    records = read_user_prompt_submit_telemetry(tel)
    r = records[0]
    # Pre-collapse count preserved.
    assert r["n_returned"] == 4
    assert r["n_unique_content_hashes"] == 2
    # total_chars is post-collapse: 2 unique beliefs × 1 char each.
    assert r["total_chars"] == 2  # len("a") + len("b")
