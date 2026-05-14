"""Regression suite for the ingest subfloor-noise filter (#809).

`_ingest_turn_ids` (the sentence-level transcript / commit ingest path)
filters out sentences matching `_looks_like_subfloor_noise`. The
pattern set is defense-in-depth across the three noise classes the
lab campaign named:

1. **Code-fence boundaries** (` ```bash`, `` ``` ``). Already handled
   upstream by `extract_sentences` (it strips paired triple-backtick
   regions wholesale); the gate here is a backstop for malformed /
   unpaired fence text that survives the strip.
2. **Bullet stubs** (`- run tests`, `* foo`). Already handled upstream
   by `extract_sentences` (it strips line-leading list markers); the
   gate here is a backstop for bullets that appear mid-line and
   survive.
3. **Header stubs ending in `:`** ("Acceptance criteria:", "Pipeline
   composition, in order of evidence:"). NOT handled by
   `extract_sentences` — this is the load-bearing pattern in the
   normal pipeline.

Below the gate:

- matched sentences do not become freestanding belief rows;
- if a matched sentence sits *between* two full-length beliefs in the
  same turn, it attaches as `anchor_text` on an intra-turn
  DERIVED_FROM edge between those two beliefs;
- if unanchored (no surrounding full-length belief in the same turn),
  it is silently dropped.

Empirical basis: `retrieval-corpus-bloat` R0/R2 (lab campaign,
2026-05-11) attributed 19% of short-reinforced beliefs in the
alpha+beta >= 10 stratum to these three pattern classes. Operator-
ratified scope (#809) is pattern-based rather than length-based so
legit short factual claims survive the gate.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice.ingest import _ingest_turn_ids, _looks_like_subfloor_noise
from aelfrice.models import EDGE_DERIVED_FROM
from aelfrice.store import MemoryStore


# Full-length fixtures (real claims, ingest-eligible under
# `is_transcript_noise`). Each sentence terminates with `.` so
# `extract_sentences` splits on it cleanly.
FULL_A = "The configuration file lives at /etc/aelfrice/conf."
FULL_B = "Astronomers process supernova imagery nightly using clusters."
FULL_C = "The pipeline orders feedback events ahead of corroboration."

# Sub-floor (header-ending-in-`:`) fixtures. These must be placed on
# their own newline-separated lines in fixture text — `:` is not a
# sentence boundary in `extract_sentences`, so without the newline
# they merge with surrounding prose and don't reach the gate.
SUB_HEADER = "Acceptance criteria:"
SUB_HEADER_LONG = "Pipeline composition, in order of evidence:"


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(tmp_path / "subfloor.db"))
    yield s
    s.close()


# --- _looks_like_subfloor_noise unit checks (all three patterns) ---------


def test_helper_flags_header_ending_with_colon() -> None:
    assert _looks_like_subfloor_noise("Acceptance criteria:") is True
    assert _looks_like_subfloor_noise(
        "Pipeline composition, in order of evidence:"
    ) is True


def test_helper_flags_codefence_prefix() -> None:
    """Backstop: `extract_sentences` already strips paired code-fence
    regions, but the helper catches unpaired / malformed fence text
    that survives."""
    assert _looks_like_subfloor_noise("```bash") is True
    assert _looks_like_subfloor_noise("```") is True
    assert _looks_like_subfloor_noise("```python") is True


def test_helper_flags_bullet_stubs() -> None:
    """Backstop: `extract_sentences` already strips line-leading list
    markers, but the helper catches bullets that survive (e.g.,
    mid-line `- foo` after sentence-split)."""
    assert _looks_like_subfloor_noise("- run tests") is True
    assert _looks_like_subfloor_noise("* foo bar") is True
    assert _looks_like_subfloor_noise("+ another bullet") is True


def test_helper_preserves_legit_short_claims() -> None:
    """Pattern-gate (per #809 operator-ratified scope) does NOT flag
    short factual claims that a strict length floor would have
    dropped."""
    assert _looks_like_subfloor_noise(
        "The configuration file lives at /etc/aelfrice/conf."
    ) is False
    assert _looks_like_subfloor_noise(
        "The default port is 8080 for the dashboard."
    ) is False
    assert _looks_like_subfloor_noise(
        "Hubble observes galaxies nightly."
    ) is False


def test_helper_strips_before_checking() -> None:
    """Whitespace around the noise marker doesn't bypass the gate."""
    assert _looks_like_subfloor_noise("   ```bash   ") is True
    assert _looks_like_subfloor_noise("\t- bullet\t") is True
    assert _looks_like_subfloor_noise("  Acceptance criteria:  ") is True


def test_helper_empty_sentence_not_flagged() -> None:
    """Empty / whitespace-only sentences are not noise-marked here —
    they're filtered upstream by `extract_sentences` (< 10 chars)."""
    assert _looks_like_subfloor_noise("") is False
    assert _looks_like_subfloor_noise("   ") is False


# --- Filter: header stubs do not become belief rows ----------------------


def test_header_stub_alone_in_turn_dropped(store: MemoryStore) -> None:
    """A header-ending-in-`:` sentence alone produces zero beliefs."""
    ids = _ingest_turn_ids(store, SUB_HEADER, source="user")
    assert ids == []
    assert store.count_beliefs() == 0


def test_all_subfloor_turn_drops_all_silently(
    store: MemoryStore,
) -> None:
    """A turn composed entirely of subfloor sentences produces zero
    beliefs AND zero edges — no surrounding full-length belief to
    anchor to."""
    text = f"{SUB_HEADER}\n\n{SUB_HEADER_LONG}"
    ids = _ingest_turn_ids(store, text, source="user")
    assert ids == []
    assert store.count_beliefs() == 0
    assert store.count_edges() == 0


def test_subfloor_at_start_dropped_when_no_preceding_full(
    store: MemoryStore,
) -> None:
    """Sub-floor leading a turn has no preceding full-length belief to
    pair with — unanchored, silently dropped."""
    text = f"{SUB_HEADER}\n\n{FULL_A}"
    ids = _ingest_turn_ids(store, text, source="user")
    assert len(ids) == 1
    assert store.count_edges() == 0


def test_subfloor_at_end_dropped_when_no_following_full(
    store: MemoryStore,
) -> None:
    """Sub-floor trailing the last full-length sentence has no
    surrounding belief on its right side — silently dropped."""
    text = f"{FULL_A}\n\n{SUB_HEADER}"
    ids = _ingest_turn_ids(store, text, source="user")
    assert len(ids) == 1
    assert store.count_edges() == 0


# --- Demotion: subfloor between two full sentences -> edge anchor --------


def test_header_stub_between_full_demotes_to_edge_anchor(
    store: MemoryStore,
) -> None:
    """Spec § 3 primary test (codebase-adapted): a paragraph with
    (full_A, header_stub, full_B) creates two beliefs and one
    intra-turn DERIVED_FROM edge full_B -> full_A whose anchor_text
    carries the sub-floor header."""
    text = f"{FULL_A}\n\n{SUB_HEADER}\n\n{FULL_B}"
    ids = _ingest_turn_ids(store, text, source="user")
    assert len(ids) == 2
    assert store.count_beliefs() == 2

    later_id, earlier_id = ids[1], ids[0]
    edge = store.get_edge(later_id, earlier_id, EDGE_DERIVED_FROM)
    assert edge is not None
    assert SUB_HEADER in edge.anchor_text


def test_multiple_header_stubs_between_two_full_concatenated(
    store: MemoryStore,
) -> None:
    """Two header stubs between the same pair of full-length sentences
    both land in the edge's anchor_text (joined by ' | ')."""
    text = f"{FULL_A}\n\n{SUB_HEADER}\n\n{SUB_HEADER_LONG}\n\n{FULL_B}"
    ids = _ingest_turn_ids(store, text, source="user")
    assert len(ids) == 2

    edge = store.get_edge(ids[1], ids[0], EDGE_DERIVED_FROM)
    assert edge is not None
    assert SUB_HEADER in edge.anchor_text
    assert SUB_HEADER_LONG in edge.anchor_text


def test_subfloor_between_each_pair_of_three_full_sentences(
    store: MemoryStore,
) -> None:
    """Three full sentences with sub-floor between each consecutive
    pair create three beliefs and two demotion edges, each carrying
    the appropriate sub-floor clause."""
    text = (
        f"{FULL_A}\n\n{SUB_HEADER}\n\n{FULL_B}\n\n{SUB_HEADER_LONG}\n\n"
        f"{FULL_C}"
    )
    ids = _ingest_turn_ids(store, text, source="user")
    assert len(ids) == 3

    edge_b_to_a = store.get_edge(ids[1], ids[0], EDGE_DERIVED_FROM)
    assert edge_b_to_a is not None
    assert SUB_HEADER in edge_b_to_a.anchor_text

    edge_c_to_b = store.get_edge(ids[2], ids[1], EDGE_DERIVED_FROM)
    assert edge_c_to_b is not None
    assert SUB_HEADER_LONG in edge_c_to_b.anchor_text


# --- Full-length sentences: existing behavior unchanged ------------------


def test_full_length_consecutive_sentences_no_intra_turn_edge(
    store: MemoryStore,
) -> None:
    """Two full-length sentences adjacent with no sub-floor between
    them produce no intra-turn DERIVED_FROM edge — the demotion path
    only fires when sub-floor sits between full-length beliefs."""
    text = f"{FULL_A} {FULL_B}"
    ids = _ingest_turn_ids(store, text, source="user")
    assert len(ids) == 2
    assert store.count_edges() == 0


def test_legit_short_factual_claim_still_ingested(
    store: MemoryStore,
) -> None:
    """The pattern-gate (vs length-floor) trade-off: short factual
    claims that a length floor would have dropped survive here."""
    text = "The default port is 8080."  # 25 chars, no noise markers
    ids = _ingest_turn_ids(store, text, source="user")
    assert len(ids) == 1
    assert store.count_beliefs() == 1


def test_ingest_idempotent_on_repeat_under_demotion(
    store: MemoryStore,
) -> None:
    """Re-ingesting the same turn does not create duplicate beliefs or
    duplicate intra-turn DERIVED_FROM edges (deduped via `get_edge`
    check)."""
    text = f"{FULL_A}\n\n{SUB_HEADER}\n\n{FULL_B}"
    ids_first = _ingest_turn_ids(store, text, source="user")
    n_beliefs = store.count_beliefs()
    n_edges = store.count_edges()
    _ = _ingest_turn_ids(store, text, source="user")
    assert store.count_beliefs() == n_beliefs
    assert store.count_edges() == n_edges
    assert len(ids_first) == 2
