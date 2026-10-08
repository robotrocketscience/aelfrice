"""#1658: the ingest classifier version and rule-set digest.

`ingest_log.classifier_version` and `ingest_log.rule_set_hash` stayed
NULL on every row because no production writer passed them. These tests
pin the digest's contract (deterministic in and across processes,
sensitive to every rule table it covers, None rather than an exception
when an input is missing).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

from aelfrice import classification_core, correction, llm_classifier
from aelfrice.classification_core import (
    INGEST_CLASSIFIER_VERSION,
    compute_rule_set_hash,
    rule_set_hash,
)

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
