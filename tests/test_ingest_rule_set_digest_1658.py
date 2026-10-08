"""#1658: the ingest classifier version and rule-set digest.

`ingest_log.classifier_version` and `ingest_log.rule_set_hash` stayed
NULL on every row because no production writer passed them. These tests
pin the digest's contract (deterministic in and across processes,
sensitive to every rule table it covers, None rather than an exception
when an input is missing).
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice import classification_core, correction, llm_classifier
from aelfrice.classification import (
    HostClassification,
    accept_classifications,
    start_onboard_session,
)
from aelfrice.classification_core import (
    INGEST_CLASSIFIER_VERSION,
    compute_rule_set_hash,
    rule_set_hash,
)
from aelfrice.cli import main as cli_main
from aelfrice.ingest import ingest_turn
from aelfrice.models import BELIEF_FACTUAL
from aelfrice.scanner import scan_repo
from aelfrice.store import MemoryStore
from aelfrice.triple_extractor import Triple, ingest_triples

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


# --- Digest contract --------------------------------------------------------


def test_digest_is_sha256_hex_and_stable_across_calls() -> None:
    first = compute_rule_set_hash()
    second = compute_rule_set_hash()
    assert first is not None
    assert _SHA256_HEX.match(first)
    assert first == second
    assert rule_set_hash() == first


def test_classifier_version_is_semver() -> None:
    assert re.match(r"^\d+\.\d+\.\d+$", INGEST_CLASSIFIER_VERSION)


@pytest.mark.timeout(120)
def test_digest_is_stable_across_processes_and_hash_seeds() -> None:
    """Two fresh interpreters with different hash seeds agree with this one.

    A digest built from set iteration order or `hash()` would differ
    between seeds.
    """
    code = (
        "from aelfrice.classification_core import rule_set_hash;"
        "print(rule_set_hash())"
    )
    seen: list[str] = []
    for seed in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, env=env, timeout=60, check=True,
        )
        seen.append(proc.stdout.strip())
    assert seen == [compute_rule_set_hash()] * 2


@pytest.mark.parametrize(
    ("module", "name", "value"),
    [
        (classification_core, "_PREFERENCE_KEYWORDS", ("prefer",)),
        (classification_core, "_REQUIREMENT_RE", re.compile(r"\bmust\b")),
        (classification_core, "_QUESTION_PREFIXES", ("what ",)),
        (classification_core, "_FLOAT_LEADING_HEDGES", ("maybe ",)),
        (classification_core, "_FLOAT_INTERNAL_HEDGES", ("maybe we ",)),
        (classification_core, "TYPE_PRIORS", {"factual": (1.0, 1.0)}),
        (classification_core, "_AGENT_INFERRED_DEFLATION", 0.5),
        (correction, "_NEGATION_RE", re.compile(r"\bnot\b")),
        (correction, "_IMPERATIVE_RE", re.compile(r"^use\b")),
        (correction, "CORRECTION_SIGNAL_THRESHOLD", 3),
        (llm_classifier, "_SYSTEM_PROMPT", "a different prompt"),
    ],
)
def test_digest_changes_when_a_rule_table_changes(
    monkeypatch: pytest.MonkeyPatch, module: object, name: str, value: object,
) -> None:
    before = compute_rule_set_hash()
    monkeypatch.setattr(module, name, value)
    after = compute_rule_set_hash()
    assert after is not None
    assert after != before


def test_digest_changes_when_the_user_message_format_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = compute_rule_set_hash()
    monkeypatch.setattr(
        llm_classifier, "build_user_message", lambda cands: "[]",
    )
    assert compute_rule_set_hash() != before


def test_digest_is_none_when_an_input_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing LLM classifier module degrades to None, never raises."""
    monkeypatch.setitem(sys.modules, "aelfrice.llm_classifier", None)
    monkeypatch.delattr("aelfrice.llm_classifier", raising=False)
    assert compute_rule_set_hash() is None


def test_cached_digest_is_computed_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rule_set_hash.cache_clear()
    calls: list[int] = []
    real = classification_core.compute_rule_set_hash

    def counting() -> str | None:
        calls.append(1)
        return real()

    monkeypatch.setattr(classification_core, "compute_rule_set_hash", counting)
    try:
        a = classification_core.rule_set_hash()
        b = classification_core.rule_set_hash()
    finally:
        rule_set_hash.cache_clear()
    assert a == b
    assert calls == [1]


# --- Every production writer stamps both columns ----------------------------

_SRC = Path(classification_core.__file__).resolve().parent
_EXPECTED_WRITERS = 6


def _record_ingest_calls() -> list[tuple[str, int, set[str]]]:
    calls: list[tuple[str, int, set[str]]] = []
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "record_ingest"
            ):
                kws = {k.arg for k in node.keywords if k.arg is not None}
                calls.append((path.name, node.lineno, kws))
    return calls


def test_every_record_ingest_call_passes_version_and_digest() -> None:
    """A new writer, or an old one that drops a kwarg, fails here."""
    calls = _record_ingest_calls()
    assert len(calls) == _EXPECTED_WRITERS, calls
    missing = [
        (name, line) for name, line, kws in calls
        if not {"classifier_version", "rule_set_hash"} <= kws
    ]
    assert missing == []


def _stamps(
    store: MemoryStore, where: str = "1", *args: object,
) -> list[tuple[object, object]]:
    rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        f"SELECT classifier_version, rule_set_hash FROM ingest_log WHERE {where}",
        args,
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def _assert_stamped(rows: list[tuple[object, object]]) -> None:
    assert rows, "the path wrote no ingest_log row"
    expected = (INGEST_CLASSIFIER_VERSION, rule_set_hash())
    assert expected[1] is not None
    assert all(r == expected for r in rows), rows


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(tmp_path / "rule-set-digest.db"))
    yield s
    s.close()


def _write_doc(root: Path) -> None:
    (root / "DESIGN.md").write_text(
        "The configuration file lives at /etc/aelfrice/conf.\n\n"
        "Aelfrice stores beliefs in a SQLite database.\n",
        encoding="utf-8",
    )


def test_onboard_accept_stamps_rows(store: MemoryStore, tmp_path: Path) -> None:
    _write_doc(tmp_path)
    started = start_onboard_session(store, tmp_path, now="2026-10-07T00:00:00Z")
    accept_classifications(
        store, started.session_id,
        [
            HostClassification(index=s.index, belief_type=BELIEF_FACTUAL, persist=True)
            for s in started.sentences
        ],
        now="2026-10-07T00:00:00Z",
    )
    _assert_stamped(_stamps(store))


def test_scan_repo_stamps_rows(store: MemoryStore, tmp_path: Path) -> None:
    _write_doc(tmp_path)
    scan_repo(store, tmp_path, now="2026-10-07T00:00:00Z")
    _assert_stamped(_stamps(store))


def test_ingest_turn_stamps_rows(store: MemoryStore) -> None:
    ingest_turn(
        store,
        "The worker stamps every ingest row. The store keeps beliefs in SQLite.",
        source="transcript:test",
    )
    _assert_stamped(_stamps(store))


@pytest.mark.parametrize("side", ["subject", "object"])
def test_ingest_triples_stamps_both_rows(store: MemoryStore, side: str) -> None:
    triple = Triple(
        subject="the derivation worker", relation="supports",
        object="the ingest log", anchor_text="the worker supports the log",
    )
    ingest_triples(store, [triple])
    _assert_stamped(_stamps(store, "raw_text = ?", getattr(triple, side)))


def test_cli_lock_stamps_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "cli-lock.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    assert cli_main(["lock", "Every commit is signed."]) == 0
    s = MemoryStore(str(db))
    try:
        _assert_stamped(_stamps(s))
    finally:
        s.close()
