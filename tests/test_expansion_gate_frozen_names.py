"""#2018: the expansion gate matches a frozen list of edge-type names.

``_EDGE_TYPE_RE`` in ``aelfrice.expansion_gate`` decides which prompts
count as narrow edge-type prompts and keep the BFS expansion lane. These
tests pin its source text to a literal, so a rename, addition, or removal
in ``models.EDGE_TYPES`` can't change which prompts skip BFS, and the text
is the same in every process whatever ``PYTHONHASHSEED`` is.
"""
from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice import detector_thresholds, expansion_gate, models
from aelfrice.detector_thresholds import pin_value

# Today's 10 edge-type names, in the fixed order the gate must use: longest
# name first, and names of equal length in alphabetical order. The fix
# builds `_EDGE_TYPE_RE` from a literal tuple in exactly this order, so the
# source below is the one string it can produce.
FROZEN_EDGE_TYPE_NAMES: tuple[str, ...] = (
    "TEMPORAL_NEXT",
    "DERIVED_FROM",
    "CONTRADICTS",
    "IMPLEMENTS",
    "RELATES_TO",
    "SUPERSEDES",
    "RESOLVES",
    "SUPPORTS",
    "CITES",
    "TESTS",
)

PINNED_SOURCE: str = (
    r"\b(?:TEMPORAL_NEXT|DERIVED_FROM|CONTRADICTS|IMPLEMENTS|RELATES_TO"
    r"|SUPERSEDES|RESOLVES|SUPPORTS|CITES|TESTS)\b"
)

# The flags a plain `re.compile` gives a str pattern today. The gate
# compiles with no extra flags, and changing them would change matching.
_DEFAULT_FLAGS: int = re.compile("").flags

# Edge-type names used to patch `models.EDGE_TYPES`. The added name is not
# one of today's names, and the removed name is.
_ADDED_NAME = "REFINES"
_REMOVED_NAME = "SUPPORTS"


def test_regex_source_equals_the_pinned_string() -> None:
    assert expansion_gate._EDGE_TYPE_RE.pattern == PINNED_SOURCE
    assert expansion_gate._EDGE_TYPE_RE.flags == _DEFAULT_FLAGS


@pytest.fixture
def _restore_expansion_gate() -> Iterator[None]:
    """Put back the original module contents after a test reloads it.

    A reload rebinds every name in the module, including
    `ExpansionDecision` and `should_run_expansion`, which other modules
    imported by reference. Restoring the saved namespace puts back those
    original objects, so tests that run afterwards see the module as it
    was first imported.
    """
    saved = dict(expansion_gate.__dict__)
    try:
        yield
    finally:
        expansion_gate.__dict__.clear()
        expansion_gate.__dict__.update(saved)


def test_reload_with_patched_edge_types_keeps_the_pinned_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _restore_expansion_gate: None,
) -> None:
    """Adding and removing an edge type doesn't move the gate.

    `models.EDGE_TYPES` gains one name and loses another, and the gate
    module is reloaded so any import-time read of `EDGE_TYPES` sees the
    patched set. The gate must still compile the pinned source. A prompt
    naming only the added type stays broad, and one naming only the removed
    type stays narrow.
    """
    for name in (
        expansion_gate.ENV_FORCE_EXPANSION,
        expansion_gate.ENV_NO_EXPANSION_GATE,
    ):
        monkeypatch.delenv(name, raising=False)
    patched = (set(models.EDGE_TYPES) | {_ADDED_NAME}) - {_REMOVED_NAME}
    monkeypatch.setattr(models, "EDGE_TYPES", frozenset(patched))

    gate = importlib.reload(expansion_gate)

    assert gate._EDGE_TYPE_RE.pattern == PINNED_SOURCE
    assert gate._EDGE_TYPE_RE.flags == _DEFAULT_FLAGS

    added = gate.should_run_expansion(f"edge {_ADDED_NAME} here", start=tmp_path)
    assert added.run_bfs is False, added.reason
    assert "no-markers" in added.reason

    removed = gate.should_run_expansion(
        f"edge {_REMOVED_NAME} here", start=tmp_path,
    )
    assert removed.run_bfs is True, removed.reason
    assert removed.reason == "narrow"


def _regex_source_in_subprocess(hash_seed: str) -> str:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hash_seed
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from aelfrice import expansion_gate as g; "
            "print(g._EDGE_TYPE_RE.pattern)",
        ],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=True,
    )
    return proc.stdout.strip()


@pytest.mark.timeout(120)
def test_regex_source_is_the_same_under_different_hash_seeds() -> None:
    """Two processes with different `PYTHONHASHSEED` values agree.

    On CPython 3.12, the set-built regex orders `CITES` and `TESTS`
    differently under seeds 0 and 1, so this pair of seeds tells the two
    builds apart.
    """
    first = _regex_source_in_subprocess("0")
    second = _regex_source_in_subprocess("1")
    assert first == second
    assert first == PINNED_SOURCE


def _reference_edge_type_re() -> re.Pattern[str]:
    """Today's names sorted longest first, the sort `main` used before #2018.

    This sorts the frozen tuple, not a frozenset, so names of equal length
    keep the tuple's order instead of a set's. Order doesn't change what
    the regex matches, because every name sits between `\\b` anchors.
    """
    return re.compile(
        r"\b(?:"
        + "|".join(sorted(FROZEN_EDGE_TYPE_NAMES, key=len, reverse=True))
        + r")\b",
    )


def _probe_texts() -> list[str]:
    probes: list[str] = []
    for name in FROZEN_EDGE_TYPE_NAMES:
        probes += [
            name,
            name.lower(),
            f"a {name} b",
            f"{name}S",
            f"X{name}",
            f"{name}_X",
            f"{name}.",
            f"({name})",
            f"#{name}",
        ]
    probes += [
        "",
        "plain prose with no edge names",
        "SUPPORTS and CONTRADICTS together",
        "TEMPORAL",
        "NEXT",
        "DERIVED",
        "RELATES",
        "POTENTIALLY_STALE",
        _ADDED_NAME,
    ]
    return probes


def test_matching_is_unchanged_from_the_length_sorted_regex() -> None:
    """Retrieval output can't change, because the gate matches the same text.

    The gate's only use of `_EDGE_TYPE_RE` is a `search` in
    `_has_structural_markers`, so identical `search` results on every probe
    mean identical gate verdicts and identical retrieval output.
    """
    reference = _reference_edge_type_re()
    live = expansion_gate._EDGE_TYPE_RE
    for text in _probe_texts():
        expected = reference.search(text)
        actual = live.search(text)
        assert (actual is None) == (expected is None), text
        if expected is not None and actual is not None:
            assert actual.span() == expected.span(), text
            assert actual.group(0) == expected.group(0), text


def _expansion_gate_pins() -> list[detector_thresholds.PinnedThreshold]:
    return [
        t
        for t in detector_thresholds.THRESHOLDS
        if t.module == "aelfrice.expansion_gate"
    ]


def test_thresholds_manifest_pins_the_regex_source() -> None:
    """The manifest has exactly one pin for this regex, equal to its source.

    The pin may name the compiled pattern or a string constant holding its
    source. Either way its value must be what `pin_value` gives for the
    pinned source, so the manifest records this exact text.
    """
    accepted = {
        pin_value(re.compile(PINNED_SOURCE)),
        pin_value(PINNED_SOURCE),
    }
    pins = _expansion_gate_pins()
    matching = [t for t in pins if t.value in accepted]
    assert len(matching) == 1, [(t.name, t.value) for t in pins]

    entry = matching[0]
    live = getattr(expansion_gate, entry.name)
    assert pin_value(live) == entry.value


def test_thresholds_version_was_bumped_with_a_history_row() -> None:
    """The new pin ships under a new version with its own history row."""
    version = detector_thresholds.DETECTOR_THRESHOLDS_VERSION
    assert version >= 10
    assert version in detector_thresholds.DIGEST_HISTORY
    assert (
        detector_thresholds.DIGEST_HISTORY[version]
        == detector_thresholds.manifest_digest()
    )
