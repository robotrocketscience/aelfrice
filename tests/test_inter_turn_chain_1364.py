"""#1364 — the inter-turn DERIVED_FROM chain must not skip a corroborating turn.

`_ingest_turn_ids` returns newly-inserted ids, not per-sentence resolved
ids. `ingest_jsonl` read it as the latter, so a turn whose sentences all
corroborated existing beliefs returned `[]`, was skipped by the
`continue`, and left `last_per_session` pointing at the turn before it.
The next turn then linked back across it.

The edge that results is not merely missing — it is *wrong*: it asserts
that turn N+1 is derived from turn N-1, a claim the transcript does not
support.

Operator ruling 2026-08-05: `_ingest_turn_ids` keeps its contract
(`ingest_turn` returns its length as the public count of newly-inserted
beliefs, predating #264); the chain gets `TurnIngest.resolved` / `.head`
instead.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice.ingest import _ingest_turn, ingest_jsonl
from aelfrice.models import EDGE_DERIVED_FROM
from aelfrice.store import MemoryStore

_SESSION = "s-1364"

# Three distinct full-length sentences the classifier persists as facts.
_TURN_A = "The configuration file lives at /etc/aelfrice/conf."
_TURN_B = "Astronomers process supernova imagery nightly using clusters."
_TURN_C = "Radio telescopes calibrate against known pulsar timings."


@pytest.fixture(autouse=True)
def _pinned_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's repo-local live store out of every test."""
    monkeypatch.setenv("AELFRICE_DOTDIR", str(tmp_path / "dotdir"))
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "pinned.db"))


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(tmp_path / "chain.db"))
    yield s
    s.close()


def _write_turns(path: Path, texts: list[str]) -> None:
    with path.open("w") as f:
        for i, t in enumerate(texts):
            f.write(json.dumps({
                "role": "user",
                "text": t,
                "session_id": _SESSION,
                "ts": f"2026-08-0{i + 1}T00:00:00Z",
            }) + "\n")


def _derived_from(store: MemoryStore) -> set[tuple[str, str]]:
    return {
        (e.src, e.dst)
        for e in store.iter_all_edges()
        if e.type == EDGE_DERIVED_FROM
    }


def _bid(store: MemoryStore, text: str) -> str:
    """The belief id a turn's text resolved to, via a dry re-ingest.

    Re-ingesting is idempotent — the content hash resolves to the same
    row — so this reads the id back rather than recomputing it, which
    would duplicate the id scheme under test.
    """
    head = _ingest_turn(
        store, text, "transcript", session_id=_SESSION,
        created_at="2026-08-09T00:00:00Z", role="user",
    ).head
    assert head is not None
    return head


# --- the two readings are genuinely different ---------------------------


def test_a_corroborating_turn_inserts_nothing_but_resolves_something(
    store: MemoryStore,
) -> None:
    """Hypothesis: on a turn whose sentences all corroborate, `inserted`
    is empty while `resolved` names the beliefs.

    This is the whole premise. If the two agreed here there would be no
    defect, so every other assertion in this file rests on it.
    Falsifiable by `resolved` being built from the same `was_inserted`
    filter as `inserted`.
    """
    first = _ingest_turn(
        store, _TURN_A, "transcript", session_id=_SESSION,
        created_at="2026-08-01T00:00:00Z", role="user",
    )
    assert len(first.inserted) == 1
    assert first.head is not None

    again = _ingest_turn(
        store, _TURN_A, "transcript", session_id=_SESSION,
        created_at="2026-08-02T00:00:00Z", role="user",
    )
    assert again.inserted == []          # nothing new
    assert again.resolved == first.resolved   # but it resolved
    assert again.head == first.head


def test_head_is_the_last_resolved_not_the_last_inserted(
    store: MemoryStore,
) -> None:
    """Hypothesis: on a turn whose LAST sentence corroborates and whose
    first is new, `head` names the last sentence, not the first.

    Falsifiable by `head` reading `inserted[-1]` — which would name the
    earlier, newly-inserted sentence and anchor the next turn's edge to
    the wrong belief. A turn where every sentence is new cannot tell the
    two apart.
    """
    seeded = _ingest_turn(
        store, _TURN_C, "transcript", session_id="other",
        created_at="2026-08-01T00:00:00Z", role="user",
    )
    assert len(seeded.inserted) == 1

    mixed = _ingest_turn(
        store, f"{_TURN_B}\n\n{_TURN_C}", "transcript", session_id=_SESSION,
        created_at="2026-08-02T00:00:00Z", role="user",
    )
    assert len(mixed.inserted) == 1                 # only _TURN_B was new
    assert mixed.head == seeded.head                # but _TURN_C is the head
    assert mixed.head != mixed.inserted[0]


# --- the chain itself ---------------------------------------------------


def test_the_chain_does_not_skip_a_fully_corroborating_turn(
    store: MemoryStore, tmp_path: Path,
) -> None:
    """Hypothesis: over three turns where the middle one fully
    corroborates, the chain links C->B and B->A, never C->A.

    This is the distinguishing test. Before #1364 turn B returned `[]`,
    the `continue` fired, `last_per_session` still pointed at A, and the
    chain linked C->A — an edge asserting a derivation the transcript
    does not support. Asserting merely that *some* edge exists passes on
    both behaviours, so the C->A absence is the load-bearing assertion.
    """
    # Seed B's belief in a DIFFERENT session, so that when it appears as
    # the middle turn here it corroborates instead of inserting.
    pre = tmp_path / "pre.jsonl"
    with pre.open("w") as f:
        f.write(json.dumps({
            "role": "user", "text": _TURN_B,
            "session_id": "seed-session", "ts": "2026-07-01T00:00:00Z",
        }) + "\n")
    ingest_jsonl(store, pre, source_label="transcript")

    path = tmp_path / "turns.jsonl"
    _write_turns(path, [_TURN_A, _TURN_B, _TURN_C])
    ingest_jsonl(store, path, source_label="transcript")

    a, b, c = _bid(store, _TURN_A), _bid(store, _TURN_B), _bid(store, _TURN_C)
    assert len({a, b, c}) == 3, "fixture must produce three distinct beliefs"

    edges = _derived_from(store)
    assert (b, a) in edges, "B must link to A"
    assert (c, b) in edges, "C must link to B, its real predecessor"
    assert (c, a) not in edges, (
        "C linked across the corroborating turn to A — the #1364 defect"
    )


def test_the_chain_is_unchanged_when_no_turn_corroborates(
    store: MemoryStore, tmp_path: Path,
) -> None:
    """Hypothesis: with three all-new turns the chain is exactly C->B->A,
    the same as before #1364.

    The regression guard. The fix must not add, drop or redirect an edge
    on the path that was already correct — which is every turn in a
    transcript of novel content. Pairs with the test above: that one
    fails on main, this one passes on main, and both must pass here.
    """
    path = tmp_path / "turns.jsonl"
    _write_turns(path, [_TURN_A, _TURN_B, _TURN_C])
    ingest_jsonl(store, path, source_label="transcript")

    a, b, c = _bid(store, _TURN_A), _bid(store, _TURN_B), _bid(store, _TURN_C)
    assert _derived_from(store) == {(b, a), (c, b)}
