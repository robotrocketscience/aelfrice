"""Tests for the Obsidian vault exporter (#630).

Covers:
  * filename slug shape (deterministic, ascii-safe).
  * YAML front-matter shape (every edge key always present, even empty).
  * Body shape (content + provenance + Connections section).
  * Round-trip determinism (same store + same flags → byte-identical
    note set across runs).
  * Wipe-and-emit semantics (`<vault>/aelfrice/` is removed and rewritten).
  * --max-notes cap enforcement.
  * Disclaimer-string presence on the module surface (the acceptance
    criterion says the perf + edge-type disclaimers must ship; this
    test pins that they do).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CITES,
    EDGE_RELATES_TO,
    EDGE_TESTS,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_UNKNOWN,
    ORIGIN_USER_VALIDATED,
    Belief,
    Edge,
)
from aelfrice.obsidian_export import (
    DEFAULT_MAX_NOTES,
    EDGE_TYPE_DISCLAIMER,
    HARD_MAX_NOTES,
    PERF_DISCLAIMER,
    SHORT_ID_LEN,
    _EDGE_YAML_KEYS,
    note_filename,
    render_note,
    render_yaml_frontmatter,
    select_beliefs,
    slugify,
    write_vault,
)
from aelfrice.store import MemoryStore


def _belief(
    bid: str,
    content: str,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    lock_level: str = LOCK_NONE,
    origin: str = ORIGIN_UNKNOWN,
    locked_at: str | None = None,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"hash_{bid}",
        alpha=alpha,
        beta=beta,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=locked_at,
        created_at="2026-05-15T00:00:00Z",
        last_retrieved_at=None,
        origin=origin,
    )


def _seed_store(beliefs: list[Belief], edges: list[Edge]) -> MemoryStore:
    store = MemoryStore(":memory:")
    for b in beliefs:
        store.insert_belief(b)
    for e in edges:
        store.insert_edge(e)
    return store


# ---------- slugify ----------------------------------------------------


def test_slugify_basic_kebab():
    assert slugify("Hello, World!") == "hello-world"


def test_slugify_collapses_whitespace_and_punctuation():
    assert slugify("Two-repo  workflow / public + private") == (
        "two-repo-workflow-public-private"
    )


def test_slugify_empty_falls_back_to_belief():
    assert slugify("") == "belief"
    assert slugify("!!!") == "belief"


def test_slugify_truncates_to_max_len():
    s = slugify("a" * 200, max_len=20)
    assert len(s) <= 20
    assert not s.startswith("-") and not s.endswith("-")


# ---------- note_filename ----------------------------------------------


def test_note_filename_uses_short_id_prefix():
    b = _belief("abcdef1234567890longid", "Some content")
    fname = note_filename(b)
    assert fname.startswith("abcdef123456-")  # SHORT_ID_LEN = 12
    assert fname.endswith(".md")


def test_note_filename_deterministic():
    b = _belief("idA", "content X")
    assert note_filename(b) == note_filename(b)


# ---------- YAML front-matter ------------------------------------------


def test_frontmatter_includes_every_edge_key_even_empty():
    b = _belief("b1", "lonely belief")
    fm = render_yaml_frontmatter(b, [])
    for key in _EDGE_YAML_KEYS:
        assert f"{key}: []" in fm, f"missing empty key: {key}"


def test_frontmatter_groups_edges_by_type():
    b = _belief("src", "source")
    edges = [
        Edge(src="src", dst="dst-cites", type=EDGE_CITES, weight=1.0),
        Edge(src="src", dst="dst-rel-1", type=EDGE_RELATES_TO, weight=1.0),
        Edge(src="src", dst="dst-rel-2", type=EDGE_RELATES_TO, weight=1.0),
        Edge(src="src", dst="dst-tests", type=EDGE_TESTS, weight=1.0),
    ]
    fm = render_yaml_frontmatter(b, edges)
    # cites: single entry
    assert 'cites:\n  - "[[dst-cites' in fm
    # relates_to: two entries, sorted
    assert 'relates_to:\n  - "[[dst-rel-1' in fm
    assert '  - "[[dst-rel-2' in fm
    # tests: single entry
    assert 'tests:\n  - "[[dst-tests' in fm


def test_frontmatter_emits_provenance_fields():
    b = _belief(
        "p1", "provenance test",
        alpha=3.0, beta=2.0, origin=ORIGIN_USER_VALIDATED,
    )
    fm = render_yaml_frontmatter(b, [])
    assert "belief_id: p1" in fm
    assert f'origin: "{ORIGIN_USER_VALIDATED}"' in fm
    assert "alpha: 3.0" in fm
    assert "beta: 2.0" in fm
    assert "posterior_mean: 0.6" in fm  # 3/(3+2)


# ---------- body -------------------------------------------------------


def test_render_note_contains_content_and_wikilinks():
    b = _belief("src", "the body content goes here")
    edges = [
        Edge(src="src", dst="d-long-id-aaaa", type=EDGE_CITES, weight=1.0),
    ]
    note = render_note(b, edges)
    assert "the body content goes here" in note
    assert "## Provenance" in note
    assert "## Connections" in note
    assert "### cites" in note
    # body wikilink uses short-id prefix
    assert f"- [[{('d-long-id-aaaa')[:SHORT_ID_LEN]}]]" in note


def test_render_note_no_edges_shows_placeholder():
    b = _belief("solo", "alone")
    note = render_note(b, [])
    assert "_(no outbound edges)_" in note


def test_render_note_locked_belief_shows_locked_at():
    b = _belief(
        "locked1", "ground truth",
        lock_level=LOCK_USER, locked_at="2026-05-14T10:00:00Z",
    )
    note = render_note(b, [])
    assert "2026-05-14T10:00:00Z" in note


# ---------- write_vault: determinism + wipe ----------------------------


def test_write_vault_round_trip_determinism(tmp_path: Path):
    beliefs = [
        _belief("a1", "first", alpha=2.0, beta=1.0),
        _belief("b2", "second"),
        _belief("c3", "third"),
    ]
    edges = [
        Edge(src="a1", dst="b2", type=EDGE_CITES, weight=1.0),
        Edge(src="b2", dst="c3", type=EDGE_RELATES_TO, weight=1.0),
    ]
    store = _seed_store(beliefs, edges)
    try:
        vault = tmp_path / "vault"
        vault.mkdir()
        res1 = write_vault(beliefs, store, vault)
        # Capture all file contents.
        files1 = {p.name: p.read_text() for p in res1.vault_dir.iterdir()}

        # Re-run on same inputs.
        res2 = write_vault(beliefs, store, vault)
        files2 = {p.name: p.read_text() for p in res2.vault_dir.iterdir()}
    finally:
        store.close()

    assert files1 == files2
    # Three beliefs → three notes.
    assert len(files1) == 3
    assert res1.notes_written == 3


def test_write_vault_wipes_aelfrice_subdir(tmp_path: Path):
    beliefs = [_belief("a1", "first")]
    store = _seed_store(beliefs, [])
    try:
        vault = tmp_path / "vault"
        vault.mkdir()
        # Seed pre-existing junk under <vault>/aelfrice/
        target = vault / "aelfrice"
        target.mkdir()
        (target / "stale.md").write_text("delete me")

        write_vault(beliefs, store, vault)
    finally:
        store.close()

    assert not (vault / "aelfrice" / "stale.md").exists()
    assert (vault / "aelfrice").is_dir()
    # The one belief's note IS present.
    assert any(
        p.name.startswith("a1") for p in (vault / "aelfrice").iterdir()
    )


def test_write_vault_does_not_touch_sibling_dirs(tmp_path: Path):
    """Wipe-and-emit must stay inside <vault>/aelfrice/."""
    beliefs = [_belief("a1", "first")]
    store = _seed_store(beliefs, [])
    try:
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "user-notes").mkdir()
        (vault / "user-notes" / "important.md").write_text("don't touch")

        write_vault(beliefs, store, vault)
    finally:
        store.close()

    assert (vault / "user-notes" / "important.md").read_text() == "don't touch"


# ---------- select_beliefs ---------------------------------------------


def test_select_beliefs_all_respects_cap():
    beliefs = [_belief(f"b{i:02d}", f"belief {i}") for i in range(10)]
    store = _seed_store(beliefs, [])
    try:
        out = select_beliefs(
            store, scope="all", query=None, max_notes=3,
            neighborhood_hops=1, k_seeds=8,
        )
    finally:
        store.close()
    assert len(out) == 3


def test_select_beliefs_query_requires_query_text():
    store = MemoryStore(":memory:")
    try:
        with pytest.raises(ValueError, match="--scope query requires"):
            select_beliefs(
                store, scope="query", query=None, max_notes=10,
                neighborhood_hops=1, k_seeds=8,
            )
    finally:
        store.close()


def test_select_beliefs_unknown_scope_raises():
    store = MemoryStore(":memory:")
    try:
        with pytest.raises(ValueError, match="unknown scope"):
            select_beliefs(
                store, scope="bogus", query=None, max_notes=10,
                neighborhood_hops=1, k_seeds=8,
            )
    finally:
        store.close()


# ---------- acceptance: disclaimer strings ship ------------------------


def test_perf_disclaimer_references_aelf_graph_escape_hatch():
    # The proposal locks this: the disclaimer must point users at
    # `aelf graph` (#629) for the visualisation use case Obsidian's
    # graph view cannot handle.
    assert "aelf graph" in PERF_DISCLAIMER
    assert "graph view" in PERF_DISCLAIMER.lower()


def test_edge_type_disclaimer_mentions_dataview_and_untyped():
    assert "untyped" in EDGE_TYPE_DISCLAIMER.lower()
    assert "Dataview" in EDGE_TYPE_DISCLAIMER


def test_caps_match_ratified_decisions():
    assert HARD_MAX_NOTES == 5000
    assert DEFAULT_MAX_NOTES == 500
