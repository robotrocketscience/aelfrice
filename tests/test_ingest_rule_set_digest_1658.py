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

from aelfrice import classification_core, correction, llm_classifier, llm_prompt
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


# Every value `_rule_set_payload` hashes, keyed by its path in the payload,
# mapped to the module attribute it reads. The user-message sample is
# derived from `build_user_message`, so its own test covers it below.
_INPUTS: dict[str, tuple[object, str]] = {
    "classification_core.type_priors": (classification_core, "TYPE_PRIORS"),
    "classification_core.agent_inferred_deflation": (
        classification_core, "_AGENT_INFERRED_DEFLATION",
    ),
    "classification_core.deflated_alpha_floor": (
        classification_core, "_DEFLATED_ALPHA_FLOOR",
    ),
    "classification_core.user_source": (classification_core, "USER_SOURCE"),
    "classification_core.requirement_keywords": (
        classification_core, "_REQUIREMENT_KEYWORDS",
    ),
    "classification_core.requirement_re": (classification_core, "_REQUIREMENT_RE"),
    "classification_core.preference_keywords": (
        classification_core, "_PREFERENCE_KEYWORDS",
    ),
    "classification_core.question_prefixes": (
        classification_core, "_QUESTION_PREFIXES",
    ),
    "classification_core.float_leading_hedges": (
        classification_core, "_FLOAT_LEADING_HEDGES",
    ),
    "classification_core.float_internal_hedges": (
        classification_core, "_FLOAT_INTERNAL_HEDGES",
    ),
    "classification_core.sentence_boundary_re": (
        classification_core, "_SENTENCE_BOUNDARY_RE",
    ),
    "correction.imperative_re": (correction, "_IMPERATIVE_RE"),
    "correction.correction_anchor_re": (correction, "_CORRECTION_ANCHOR_RE"),
    "correction.requirement_anchor_re": (correction, "_REQUIREMENT_ANCHOR_RE"),
    "correction.declarative_re": (correction, "_DECLARATIVE_RE"),
    "correction.always_never_re": (correction, "_ALWAYS_NEVER_RE"),
    "correction.negation_re": (correction, "_NEGATION_RE"),
    "correction.emphasis_re": (correction, "_EMPHASIS_RE"),
    "correction.prior_ref_re": (correction, "_PRIOR_REF_RE"),
    "correction.signal_threshold": (correction, "CORRECTION_SIGNAL_THRESHOLD"),
    "correction.confidence_per_signal": (correction, "_CONFIDENCE_PER_SIGNAL"),
    "llm_classifier.system_prompt": (llm_prompt, "SYSTEM_PROMPT"),
}
_DERIVED_INPUTS = {"llm_classifier.user_message_sample"}
_REGEX_INPUTS = sorted(
    key for key, (module, name) in _INPUTS.items()
    if isinstance(getattr(module, name), re.Pattern)
)
_MARKER = "zz-digest-perturbation-zz"


def _edit_str(value: str) -> str:
    """Return `value` with its first character replaced, same length."""
    if not value:
        return _MARKER
    return ("Y" if value[0] == "X" else "X") + value[1:]


def _edit_pattern(value: re.Pattern[str]) -> re.Pattern[str]:
    """Return a pattern of the same length and flags with one character changed.

    It swaps the case of the first letter that still compiles once
    swapped, for example `\\b` to `\\B` or `use` to `Use`.
    """
    text = value.pattern
    for i, char in enumerate(text):
        if not char.isalpha() or char.swapcase() == char:
            continue
        candidate = text[:i] + char.swapcase() + text[i + 1:]
        try:
            return re.compile(candidate, value.flags)
        except re.error:
            continue
    raise AssertionError(f"no same-length edit compiles for {text!r}")


def _changed(value: object) -> object:
    """Return a value of the same type and length that differs from `value`."""
    if isinstance(value, str):
        return _edit_str(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value + 1
    raise TypeError(f"no perturbation for {type(value).__name__}")


def _variants(value: object) -> Iterator[tuple[str, object]]:
    """Yield labelled values that each differ from `value` in one way.

    Every input gets at least one variant that changes an existing value
    and keeps its length: a string or regex pattern gets one character
    changed, and a container gets one variant per element, or per
    component of a tuple value, changed in place under the same index
    or key. A digest that hashes only the keys, the lengths, or a
    constant in place of the values misses these. Containers, strings
    and patterns also get one variant that adds to the value.
    """
    if isinstance(value, re.Pattern):
        yield "edit pattern", _edit_pattern(value)
        yield "extend pattern", re.compile(f"{value.pattern}|{_MARKER}", value.flags)
    elif isinstance(value, tuple):
        for i, item in enumerate(value):
            yield f"[{i}]", (*value[:i], _changed(item), *value[i + 1:])
        yield "append", (*value, _MARKER)
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, tuple):
                for i, part in enumerate(item):
                    new_item = (*item[:i], _changed(part), *item[i + 1:])
                    yield f"[{key!r}][{i}]", {**value, key: new_item}
            else:
                yield f"[{key!r}]", {**value, key: _changed(item)}
        yield "add key", {**value, _MARKER: (1.0, 1.0)}
    elif isinstance(value, str):
        yield "edit", _edit_str(value)
        yield "extend", value + _MARKER
    else:
        yield "value", _changed(value)


def test_payload_hashes_exactly_the_listed_inputs() -> None:
    """Dropping or adding a payload entry fails here.

    Each regex hashes both its pattern and its flags.
    """
    payload = classification_core._rule_set_payload()  # pyright: ignore[reportPrivateUsage]
    keys = {
        f"{section}.{key}"
        for section, entries in payload.items()
        for key in entries  # pyright: ignore[reportGeneralTypeIssues]
    }
    assert keys == set(_INPUTS) | _DERIVED_INPUTS
    for key in _REGEX_INPUTS:
        section, name = key.split(".")
        assert payload[section][name].keys() == {"pattern", "flags"}  # pyright: ignore[reportIndexIssue]


def test_every_module_level_regex_is_a_digest_input() -> None:
    """A new rule regex in either classifier module must join the digest."""
    covered = {(id(m), n) for m, n in _INPUTS.values()}
    unhashed = [
        f"{module.__name__}.{name}"
        for module in (classification_core, correction)
        for name, value in vars(module).items()
        if isinstance(value, re.Pattern) and (id(module), name) not in covered
    ]
    assert unhashed == []


@pytest.mark.parametrize("key", sorted(_INPUTS))
def test_digest_changes_when_any_input_changes(
    monkeypatch: pytest.MonkeyPatch, key: str,
) -> None:
    """Changing any element of any input, or adding one, changes the digest."""
    module, name = _INPUTS[key]
    original = getattr(module, name)
    before = compute_rule_set_hash()
    assert before is not None
    unnoticed: list[str] = []
    variants = list(_variants(original))
    for label, changed in variants:
        assert changed != original, label
        monkeypatch.setattr(module, name, changed)
        if compute_rule_set_hash() == before:
            unnoticed.append(label)
    monkeypatch.setattr(module, name, original)
    assert compute_rule_set_hash() == before
    assert variants
    assert unnoticed == []


@pytest.mark.parametrize("key", _REGEX_INPUTS)
def test_digest_changes_when_only_regex_flags_change(
    monkeypatch: pytest.MonkeyPatch, key: str,
) -> None:
    module, name = _INPUTS[key]
    pattern = getattr(module, name)
    changed = re.compile(pattern.pattern, pattern.flags ^ re.DOTALL)
    assert changed.flags != pattern.flags
    before = compute_rule_set_hash()
    monkeypatch.setattr(module, name, changed)
    after = compute_rule_set_hash()
    assert after is not None
    assert after != before


@pytest.mark.parametrize("edit", ["same length", "replaced"])
def test_digest_changes_when_the_user_message_format_changes(
    monkeypatch: pytest.MonkeyPatch, edit: str,
) -> None:
    before = compute_rule_set_hash()
    real = llm_prompt.build_user_message

    def edited(candidates: list[llm_prompt.CandidateInput]) -> str:
        if edit == "replaced":
            return "[]"
        return _edit_str(real(candidates))

    monkeypatch.setattr(llm_prompt, "build_user_message", edited)
    assert compute_rule_set_hash() != before


def test_digest_is_none_when_an_input_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing LLM classifier module degrades to None, never raises."""
    monkeypatch.setitem(sys.modules, "aelfrice.llm_prompt", None)
    monkeypatch.delattr("aelfrice.llm_prompt", raising=False)
    assert compute_rule_set_hash() is None


def test_classifier_sends_the_hashed_prompt() -> None:
    """The digest hashes the text `llm_classifier` actually sends."""
    assert llm_classifier._SYSTEM_PROMPT is llm_prompt.SYSTEM_PROMPT  # pyright: ignore[reportPrivateUsage]
    assert llm_classifier.build_user_message is llm_prompt.build_user_message
    assert llm_classifier.CandidateInput is llm_prompt.CandidateInput


@pytest.mark.timeout(120)
def test_digest_does_not_import_the_llm_classifier() -> None:
    """The ingest path pays for the leaf prompt module only (#1658)."""
    code = (
        "import sys;"
        "from aelfrice.classification_core import rule_set_hash;"
        "assert rule_set_hash() is not None;"
        "print('aelfrice.llm_classifier' in sys.modules)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=60, check=True,
    )
    assert proc.stdout.strip() == "False"


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
