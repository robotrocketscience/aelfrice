"""#1658: `aelf doctor` counts ingest_log rows by rule-set digest.

The section reports how many rows carry a digest other than the current
one (written under other classifier rules) and how many carry none
(written before ingest stamped one). It reads the store read-only.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from aelfrice.classification_core import rule_set_hash
from aelfrice.doctor import (
    diagnose,
    diagnose_ingest_rule_set,
    format_report,
)
from aelfrice.models import INGEST_SOURCE_CLI_REMEMBER
from aelfrice.store import MemoryStore

_CURRENT = "c" * 64
_OLD = "0" * 64


def _seed(tmp_path: Path) -> Path:
    path = tmp_path / "doctor-rule-set.db"
    store = MemoryStore(str(path))
    try:
        for i, digest in enumerate((_CURRENT, _CURRENT, _OLD, None)):
            store.record_ingest(
                source_kind=INGEST_SOURCE_CLI_REMEMBER,
                raw_text=f"row {i}",
                rule_set_hash=digest,
            )
    finally:
        store.close()
    return path


def test_counts_a_mismatched_and_a_null_row(tmp_path: Path) -> None:
    st = diagnose_ingest_rule_set(str(_seed(tmp_path)), _CURRENT)
    assert st is not None
    assert (st.total, st.mismatched, st.missing) == (4, 1, 1)
    assert st.current_hash == _CURRENT


def test_empty_string_counts_as_no_digest(tmp_path: Path) -> None:
    path = _seed(tmp_path)
    conn = sqlite3.connect(path)
    try:
        conn.execute("UPDATE ingest_log SET rule_set_hash = '' WHERE raw_text = 'row 2'")
        conn.commit()
    finally:
        conn.close()
    st = diagnose_ingest_rule_set(str(path), _CURRENT)
    assert st is not None
    assert (st.mismatched, st.missing) == (0, 2)


def test_defaults_to_this_process_digest(tmp_path: Path) -> None:
    st = diagnose_ingest_rule_set(str(_seed(tmp_path)))
    assert st is not None
    assert st.current_hash == rule_set_hash()
    # Neither seeded digest is the real one, so all three stamped rows differ.
    assert (st.mismatched, st.missing) == (3, 1)


def test_does_not_write_the_store(tmp_path: Path) -> None:
    path = _seed(tmp_path)
    before = path.read_bytes()
    diagnose_ingest_rule_set(str(path), _CURRENT)
    assert path.read_bytes() == before


def test_fail_soft_on_missing_store_or_table(tmp_path: Path) -> None:
    assert diagnose_ingest_rule_set(":memory:", _CURRENT) is None
    assert diagnose_ingest_rule_set(str(tmp_path / "nope.db"), _CURRENT) is None
    empty = tmp_path / "empty.db"
    sqlite3.connect(empty).close()
    assert diagnose_ingest_rule_set(str(empty), _CURRENT) is None


def test_doctor_report_renders_the_counts(tmp_path: Path) -> None:
    report = diagnose(
        user_settings=tmp_path / "missing-user.json",
        project_root=tmp_path / "missing-project",
        store_path=str(_seed(tmp_path)),
    )
    st = report.ingest_rule_set
    assert st is not None
    assert (st.total, st.mismatched, st.missing) == (4, 3, 1)
    text = format_report(report)
    assert "ingest log rule set" in text
    assert "4 row(s) in total" in text
    assert "3 row(s) carry a different digest" in text
    assert "1 row(s) carry no digest" in text


def test_doctor_report_renders_the_counts_with_settings_scanned(
    tmp_path: Path,
) -> None:
    """The section also renders on the path that scanned a settings.json."""
    user = tmp_path / "user-settings.json"
    user.write_text("{}", encoding="utf-8")
    report = diagnose(
        user_settings=user,
        project_root=tmp_path / "missing-project",
        store_path=str(_seed(tmp_path)),
    )
    assert report.scopes_scanned
    assert "3 row(s) carry a different digest" in format_report(report)


def test_section_absent_without_a_store_path(tmp_path: Path) -> None:
    report = diagnose(
        user_settings=tmp_path / "missing-user.json",
        project_root=tmp_path / "missing-project",
    )
    assert report.ingest_rule_set is None
    assert "ingest log rule set" not in format_report(report)
