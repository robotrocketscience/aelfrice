"""Atomic tests for the polymorphic onboard state machine.

`start_onboard_session` produces a session_id + sentences for the host
to classify; `accept_classifications` applies host verdicts and closes
the session. One property per test, fixture repos in tmp_path so the
extractors run on real on-disk content but stay deterministic.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from aelfrice.classification import (
    HostClassification,
    accept_classifications,
    start_onboard_session,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    BELIEF_PREFERENCE,
    BELIEF_REQUIREMENT,
    LOCK_NONE,
    ONBOARD_STATE_COMPLETED,
    ONBOARD_STATE_PENDING,
    Belief,
    OnboardSession,
)
from aelfrice.store import MemoryStore


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(":memory:")


def _populate_repo(root: Path) -> None:
    """Write enough content that the three extractors find candidates."""
    (root / "README.md").write_text(
        "This project must use uv for environment management.\n\n"
        "We always prefer atomic commits over batched commits.\n\n"
        "The system follows a Bayesian feedback loop with locks.\n"
    )
    (root / "module.py").write_text(
        '"""Top-level module docstring describing the module purpose."""\n\n'
        "def f():\n"
        '    """A top-level function that returns a constant value."""\n'
        "    return 1\n"
    )


def test_start_returns_nonempty_session_id(store: MemoryStore, tmp_path: Path) -> None:
    _populate_repo(tmp_path)
    result = start_onboard_session(store, tmp_path, now="2026-04-26T00:00:00Z")
    assert result.session_id
    assert len(result.session_id) >= 16


def test_start_persists_session_in_pending_state(
    store: MemoryStore, tmp_path: Path
) -> None:
    _populate_repo(tmp_path)
    result = start_onboard_session(store, tmp_path, now="2026-04-26T00:00:00Z")
    session = store.get_onboard_session(result.session_id)
    assert session is not None
    assert session.state == ONBOARD_STATE_PENDING


def test_start_session_records_repo_path(store: MemoryStore, tmp_path: Path) -> None:
    _populate_repo(tmp_path)
    result = start_onboard_session(store, tmp_path, now="2026-04-26T00:00:00Z")
    session = store.get_onboard_session(result.session_id)
    assert session is not None
    assert session.repo_path == str(tmp_path)


def test_start_returns_at_least_one_sentence(store: MemoryStore, tmp_path: Path) -> None:
    _populate_repo(tmp_path)
    result = start_onboard_session(store, tmp_path, now="2026-04-26T00:00:00Z")
    assert len(result.sentences) > 0


def test_sentences_have_contiguous_indices_from_zero(
    store: MemoryStore, tmp_path: Path
) -> None:
    _populate_repo(tmp_path)
    result = start_onboard_session(store, tmp_path, now="2026-04-26T00:00:00Z")
    indices = [s.index for s in result.sentences]
    assert indices == list(range(len(indices)))


def test_candidates_json_round_trips_sentences(
    store: MemoryStore, tmp_path: Path
) -> None:
    _populate_repo(tmp_path)
    result = start_onboard_session(store, tmp_path, now="2026-04-26T00:00:00Z")
    session = store.get_onboard_session(result.session_id)
    assert session is not None
    parsed = json.loads(session.candidates_json)
    assert len(parsed) == len(result.sentences)
    assert {p["index"] for p in parsed} == {s.index for s in result.sentences}


def test_start_on_empty_dir_returns_empty_sentence_list(
    store: MemoryStore, tmp_path: Path
) -> None:
    result = start_onboard_session(store, tmp_path, now="2026-04-26T00:00:00Z")
    assert result.sentences == []


def test_start_on_empty_dir_still_creates_session(
    store: MemoryStore, tmp_path: Path
) -> None:
    result = start_onboard_session(store, tmp_path, now="2026-04-26T00:00:00Z")
    session = store.get_onboard_session(result.session_id)
    assert session is not None


def test_start_skips_already_present_beliefs(
    store: MemoryStore, tmp_path: Path
) -> None:
    _populate_repo(tmp_path)
    first = start_onboard_session(store, tmp_path, now="2026-04-26T00:00:00Z")
    classifications = [
        HostClassification(index=s.index, belief_type=BELIEF_FACTUAL, persist=True)
        for s in first.sentences
    ]
    accept_classifications(store, first.session_id, classifications,
                            now="2026-04-26T01:00:00Z")
    second = start_onboard_session(store, tmp_path, now="2026-04-26T02:00:00Z")
    assert second.sentences == []
    assert second.n_already_present > 0


def test_accept_unknown_session_raises(store: MemoryStore) -> None:
    with pytest.raises(ValueError, match="unknown session"):
        accept_classifications(store, "no-such-session", [],
                                now="2026-04-26T01:00:00Z")


def test_accept_already_completed_session_raises(
    store: MemoryStore, tmp_path: Path
) -> None:
    _populate_repo(tmp_path)
    result = start_onboard_session(store, tmp_path, now="2026-04-26T00:00:00Z")
    accept_classifications(store, result.session_id, [],
                            now="2026-04-26T01:00:00Z")
    with pytest.raises(ValueError, match="not pending"):
        accept_classifications(store, result.session_id, [],
                                now="2026-04-26T02:00:00Z")


def test_accept_invalid_belief_type_raises(
    store: MemoryStore, tmp_path: Path
) -> None:
    _populate_repo(tmp_path)
    result = start_onboard_session(store, tmp_path, now="2026-04-26T00:00:00Z")
    bad = [HostClassification(index=0, belief_type="not_a_real_type", persist=True)]
    with pytest.raises(ValueError, match="unknown belief_type"):
        accept_classifications(store, result.session_id, bad,
                                now="2026-04-26T01:00:00Z")


def test_accept_marks_session_completed(store: MemoryStore, tmp_path: Path) -> None:
    _populate_repo(tmp_path)
    result = start_onboard_session(store, tmp_path, now="2026-04-26T00:00:00Z")
    accept_classifications(store, result.session_id, [],
                            now="2026-04-26T01:00:00Z")
    session = store.get_onboard_session(result.session_id)
    assert session is not None
    assert session.state == ONBOARD_STATE_COMPLETED


def test_accept_writes_completed_at_timestamp(
    store: MemoryStore, tmp_path: Path
) -> None:
    _populate_repo(tmp_path)
    result = start_onboard_session(store, tmp_path, now="2026-04-26T00:00:00Z")
    accept_classifications(store, result.session_id, [],
                            now="2026-04-26T01:00:00Z")
    session = store.get_onboard_session(result.session_id)
    assert session is not None
    assert session.completed_at == "2026-04-26T01:00:00Z"


def test_accept_inserts_one_belief_per_persisting_classification(
    store: MemoryStore, tmp_path: Path
) -> None:
    _populate_repo(tmp_path)
    result = start_onboard_session(store, tmp_path, now="2026-04-26T00:00:00Z")
    cls = [
        HostClassification(index=s.index, belief_type=BELIEF_FACTUAL, persist=True)
        for s in result.sentences
    ]
    outcome = accept_classifications(
        store, result.session_id, cls, now="2026-04-26T01:00:00Z"
    )
    assert outcome.inserted == len(result.sentences)


def test_accept_stamps_origin_agent_inferred(
    store: MemoryStore, tmp_path: Path
) -> None:
    """Beliefs from accept_classifications land with origin=agent_inferred,
    not the model default ORIGIN_UNKNOWN. Regression for #224."""
    _populate_repo(tmp_path)
    result = start_onboard_session(store, tmp_path, now="2026-04-26T00:00:00Z")
    cls = [
        HostClassification(index=s.index, belief_type=BELIEF_FACTUAL, persist=True)
        for s in result.sentences
    ]
    accept_classifications(store, result.session_id, cls, now="2026-04-26T01:00:00Z")
    ids = store.list_belief_ids()
    assert ids, "expected at least one belief inserted"
    rows = [store.get_belief(b) for b in ids]
    assert all(r is not None and r.origin == "agent_inferred" for r in rows), (
        f"unexpected origins: {sorted({r.origin for r in rows if r})}"
    )


def test_accept_skips_non_persisting_classifications(
    store: MemoryStore, tmp_path: Path
) -> None:
    _populate_repo(tmp_path)
    result = start_onboard_session(store, tmp_path, now="2026-04-26T00:00:00Z")
    if not result.sentences:
        pytest.skip("fixture produced no sentences")
    cls = [HostClassification(
        index=result.sentences[0].index, belief_type=BELIEF_FACTUAL, persist=False
    )]
    outcome = accept_classifications(
        store, result.session_id, cls, now="2026-04-26T01:00:00Z"
    )
    assert outcome.skipped_non_persisting == 1
    assert outcome.inserted == 0


def test_accept_counts_unclassified_sentences(
    store: MemoryStore, tmp_path: Path
) -> None:
    _populate_repo(tmp_path)
    result = start_onboard_session(store, tmp_path, now="2026-04-26T00:00:00Z")
    if len(result.sentences) < 2:
        pytest.skip("fixture produced < 2 sentences")
    # Classify only the first sentence; everything else is "unclassified".
    cls = [HostClassification(
        index=result.sentences[0].index, belief_type=BELIEF_FACTUAL, persist=True
    )]
    outcome = accept_classifications(
        store, result.session_id, cls, now="2026-04-26T01:00:00Z"
    )
    assert outcome.skipped_unclassified == len(result.sentences) - 1


def test_accept_assigns_host_provided_belief_type(
    store: MemoryStore, tmp_path: Path
) -> None:
    _populate_repo(tmp_path)
    result = start_onboard_session(store, tmp_path, now="2026-04-26T00:00:00Z")
    if not result.sentences:
        pytest.skip("fixture produced no sentences")
    s0 = result.sentences[0]
    cls = [HostClassification(
        index=s0.index, belief_type=BELIEF_REQUIREMENT, persist=True
    )]
    accept_classifications(
        store, result.session_id, cls, now="2026-04-26T01:00:00Z"
    )
    bid = _derive_id_for_sentence(s0.text, s0.source)
    inserted = store.get_belief(bid)
    assert inserted is not None
    assert inserted.type == BELIEF_REQUIREMENT


def test_accept_skips_existing_beliefs_inserted_after_session_started(
    store: MemoryStore, tmp_path: Path
) -> None:
    _populate_repo(tmp_path)
    result = start_onboard_session(store, tmp_path, now="2026-04-26T00:00:00Z")
    if not result.sentences:
        pytest.skip("fixture produced no sentences")
    s0 = result.sentences[0]
    # Pre-insert a belief with the deterministic id that the session
    # would derive for sentence 0. Race we are protecting against:
    # session A starts, session B starts and completes for the same
    # tree, A's accept must skip rather than collide.
    bid = _derive_id_for_sentence(s0.text, s0.source)
    store.insert_belief(Belief(
        id=bid, content=s0.text, content_hash="precomputed",
        alpha=1.0, beta=1.0, type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE, locked_at=None,
        created_at="2026-04-26T00:30:00Z", last_retrieved_at=None,
    ))
    cls = [HostClassification(
        index=s0.index, belief_type=BELIEF_PREFERENCE, persist=True
    )]
    outcome = accept_classifications(
        store, result.session_id, cls, now="2026-04-26T01:00:00Z"
    )
    assert outcome.skipped_existing == 1
    assert outcome.inserted == 0


def test_two_starts_produce_distinct_session_ids(
    store: MemoryStore, tmp_path: Path
) -> None:
    _populate_repo(tmp_path)
    a = start_onboard_session(store, tmp_path, now="2026-04-26T00:00:00Z")
    b = start_onboard_session(store, tmp_path, now="2026-04-26T00:00:01Z")
    assert a.session_id != b.session_id


def _derive_id_for_sentence(text: str, source: str) -> str:
    """Mirror classification._derive_belief_id for the precomputed-id test."""
    import hashlib
    return hashlib.sha256(f"{source}\x00{text}".encode("utf-8")).hexdigest()[:16]


# --- session_id propagation (#192) -------------------------------------


def test_accept_classifications_tags_beliefs_with_session_id(
    store: MemoryStore, tmp_path: Path
) -> None:
    """Beliefs inserted via accept_classifications carry the onboard session_id."""
    _populate_repo(tmp_path)
    result = start_onboard_session(store, tmp_path, now="2026-04-26T00:00:00Z")
    cls = [
        HostClassification(index=s.index, belief_type=BELIEF_FACTUAL, persist=True)
        for s in result.sentences
    ]
    outcome = accept_classifications(
        store, result.session_id, cls, now="2026-04-26T01:00:00Z"
    )
    assert outcome.inserted > 0
    for bid in store.list_belief_ids():
        b = store.get_belief(bid)
        assert b is not None
        assert b.session_id == result.session_id


# --- check_onboard_candidates (#761) -----------------------------------


def test_check_returns_zero_counts_on_empty_dir(
    store: MemoryStore, tmp_path: Path
) -> None:
    from aelfrice.classification import check_onboard_candidates

    result = check_onboard_candidates(store, tmp_path)
    assert result.n_already_present == 0
    assert result.n_new == 0


def test_check_counts_new_candidates_on_fresh_repo(
    store: MemoryStore, tmp_path: Path
) -> None:
    from aelfrice.classification import check_onboard_candidates

    _populate_repo(tmp_path)
    result = check_onboard_candidates(store, tmp_path)
    assert result.n_new > 0
    assert result.n_already_present == 0


def test_check_records_repo_path(store: MemoryStore, tmp_path: Path) -> None:
    from aelfrice.classification import check_onboard_candidates

    _populate_repo(tmp_path)
    result = check_onboard_candidates(store, tmp_path)
    assert result.repo_path == str(tmp_path)


def test_check_does_not_persist_session(
    store: MemoryStore, tmp_path: Path
) -> None:
    """Pre-scan must not write an onboard_sessions row."""
    from aelfrice.classification import check_onboard_candidates

    _populate_repo(tmp_path)
    check_onboard_candidates(store, tmp_path)
    assert store.count_onboard_sessions() == 0


def test_check_does_not_insert_beliefs(
    store: MemoryStore, tmp_path: Path
) -> None:
    """Pre-scan must not insert any beliefs."""
    from aelfrice.classification import check_onboard_candidates

    _populate_repo(tmp_path)
    check_onboard_candidates(store, tmp_path)
    cur = store._conn.execute("SELECT COUNT(*) FROM beliefs")
    assert cur.fetchone()[0] == 0


def test_check_flips_to_already_present_after_accept(
    store: MemoryStore, tmp_path: Path
) -> None:
    """After accept_classifications inserts beliefs, a second check
    must classify those same candidates as already-present."""
    from aelfrice.classification import check_onboard_candidates

    _populate_repo(tmp_path)
    first = check_onboard_candidates(store, tmp_path)
    started = start_onboard_session(store, tmp_path, now="2026-05-13T00:00:00Z")
    classifications = [
        HostClassification(index=s.index, belief_type=BELIEF_FACTUAL, persist=True)
        for s in started.sentences
    ]
    accept_classifications(
        store, started.session_id, classifications, now="2026-05-13T00:01:00Z"
    )
    second = check_onboard_candidates(store, tmp_path)
    assert second.n_already_present == first.n_new
    assert second.n_new == 0


def test_check_is_idempotent_under_repeat_calls(
    store: MemoryStore, tmp_path: Path
) -> None:
    """Repeated pre-scans against the same tree return identical counts."""
    from aelfrice.classification import check_onboard_candidates

    _populate_repo(tmp_path)
    a = check_onboard_candidates(store, tmp_path)
    b = check_onboard_candidates(store, tmp_path)
    assert (a.n_already_present, a.n_new) == (b.n_already_present, b.n_new)


# --- commit_date propagation (#1609) ------------------------------------
#
# `scan_repo` (scanner.py) already writes `created_at = commit_date or
# timestamp` for direct ingestion. The onboard handshake serializes
# candidates into `onboard_sessions.candidates_json` between the two
# calls below and, before this fix, carried only `index`/`text`/`source`
# — dropping `SentenceCandidate.commit_date` on the floor, so every
# accepted belief got the handshake-completion time instead.


def _git_init(repo: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, timeout=30)


def _git_commit(
    repo: Path,
    *files: str,
    message: str,
    author_date: str = "2024-01-01T00:00:00+00:00",
) -> None:
    subprocess.run(["git", "add", *files], cwd=repo, check=True, timeout=30)
    subprocess.run(
        [
            "git",
            "-c", "user.email=t@t",
            "-c", "user.name=t",
            "-c", "commit.gpgsign=false",
            "commit", "-q", "-m", message,
            "--date", author_date,
        ],
        cwd=repo,
        check=True,
        env={
            **os.environ,
            "GIT_AUTHOR_DATE": author_date,
            "GIT_COMMITTER_DATE": author_date,
        },
        timeout=30,
    )


@pytest.mark.timeout(30)
def test_accept_uses_doc_commit_date_as_created_at(
    store: MemoryStore, tmp_path: Path
) -> None:
    """A doc candidate whose file has a git commit lands with
    `belief.created_at == commit author date`, not the handshake
    completion time (`now`) — mirroring `scan_repo`'s behaviour."""
    repo = tmp_path / "r"
    repo.mkdir()
    _git_init(repo)
    (repo / "RULES.md").write_text(
        "lock fast and ship small commits with conventional prefixes",
        encoding="utf-8",
    )
    _git_commit(repo, "RULES.md", message="add rules",
                author_date="2024-08-01T00:00:00+00:00")

    result = start_onboard_session(store, repo, now="2099-01-01T00:00:00Z")
    target = [s for s in result.sentences if "lock fast" in s.text]
    assert target, "expected the RULES.md paragraph among the candidates"
    s = target[0]
    cls = [HostClassification(index=s.index, belief_type=BELIEF_FACTUAL, persist=True)]
    accept_classifications(store, result.session_id, cls, now="2099-01-01T00:00:00Z")

    bid = _derive_id_for_sentence(s.text, s.source)
    belief = store.get_belief(bid)
    assert belief is not None
    # Must be the commit date, not the 2099 handshake-completion time.
    assert belief.created_at.startswith("2024-08-01")


@pytest.mark.timeout(30)
def test_accept_uses_git_commit_candidate_date_as_created_at(
    store: MemoryStore, tmp_path: Path
) -> None:
    """A `git:commit:*` candidate lands with `belief.created_at` equal to
    that commit's own author date."""
    repo = tmp_path / "r"
    repo.mkdir()
    _git_init(repo)
    (repo / "README.md").write_text("hello", encoding="utf-8")
    _git_commit(repo, "README.md",
                message="a distinctly worded commit subject for lookup",
                author_date="2023-03-14T00:00:00+00:00")

    result = start_onboard_session(store, repo, now="2099-01-01T00:00:00Z")
    target = [s for s in result.sentences if s.source.startswith("git:commit:")]
    assert target, "expected at least one git:commit: candidate"
    s = target[0]
    cls = [HostClassification(index=s.index, belief_type=BELIEF_FACTUAL, persist=True)]
    accept_classifications(store, result.session_id, cls, now="2099-01-01T00:00:00Z")

    bid = _derive_id_for_sentence(s.text, s.source)
    belief = store.get_belief(bid)
    assert belief is not None
    assert belief.created_at.startswith("2023-03-14")


def test_accept_falls_back_to_handshake_timestamp_without_commit_date(
    store: MemoryStore, tmp_path: Path
) -> None:
    """Outside any git work-tree, a candidate has no `commit_date`, and
    the accepted belief falls back to the handshake completion time —
    the pre-#1609 behaviour, preserved for the no-date case."""
    _populate_repo(tmp_path)
    result = start_onboard_session(store, tmp_path, now="2026-04-26T00:00:00Z")
    assert result.sentences, "fixture must produce at least one candidate"
    s = result.sentences[0]
    cls = [HostClassification(index=s.index, belief_type=BELIEF_FACTUAL, persist=True)]
    accept_classifications(store, result.session_id, cls, now="2026-04-26T01:00:00Z")

    bid = _derive_id_for_sentence(s.text, s.source)
    belief = store.get_belief(bid)
    assert belief is not None
    assert belief.created_at == "2026-04-26T01:00:00Z"


def test_accept_loads_legacy_candidates_json_without_commit_date_key(
    store: MemoryStore,
) -> None:
    """A session persisted before #1609 — whose `candidates_json` entries
    carry no `commit_date` key at all, not even `null` — must still load
    and accept cleanly, falling back to the handshake timestamp."""
    legacy_json = json.dumps([
        {"index": 0, "text": "a legacy candidate with no commit_date key", "source": "doc:LEGACY.md:p0"},
    ])
    store.insert_onboard_session(OnboardSession(
        session_id="legacy-session",
        repo_path=str(Path("/tmp/legacy")),
        state=ONBOARD_STATE_PENDING,
        candidates_json=legacy_json,
        created_at="2026-01-01T00:00:00Z",
        completed_at=None,
    ))
    cls = [HostClassification(index=0, belief_type=BELIEF_FACTUAL, persist=True)]
    outcome = accept_classifications(
        store, "legacy-session", cls, now="2026-01-02T00:00:00Z"
    )
    assert outcome.inserted == 1

    bid = _derive_id_for_sentence(
        "a legacy candidate with no commit_date key", "doc:LEGACY.md:p0"
    )
    belief = store.get_belief(bid)
    assert belief is not None
    assert belief.created_at == "2026-01-02T00:00:00Z"


@pytest.mark.timeout(30)
def test_accept_uses_ast_commit_date_as_created_at(
    store: MemoryStore, tmp_path: Path
) -> None:
    """An `ast:` candidate lands on its file's commit date too.

    The doc-candidate test above covers `extract_filesystem`'s `recency=`
    kwarg and nothing else. `extract_ast` takes the same kwarg on its own
    call, and neither git-backed fixture here contains a `.py` file, so
    dropping `recency=recency` from the `extract_ast` call left the whole
    onboard suite green while silently un-dating every docstring-derived
    belief — 58.7% of the candidates this repo produces. This test is the
    arm that fails under that mutation.
    """
    repo = tmp_path / "r"
    repo.mkdir()
    _git_init(repo)
    (repo / "mod.py").write_text(
        '"""Retry the upload twice before surfacing a transport error."""\n',
        encoding="utf-8",
    )
    _git_commit(repo, "mod.py", message="add mod",
                author_date="2022-05-09T00:00:00+00:00")

    result = start_onboard_session(store, repo, now="2099-01-01T00:00:00Z")
    target = [
        s
        for s in result.sentences
        if s.source.startswith("ast:") and "Retry the upload" in s.text
    ]
    assert target, (
        "expected the module docstring among the candidates as an `ast:` "
        f"source; got {sorted({s.source.split(':')[0] for s in result.sentences})}"
    )
    s = target[0]
    cls = [HostClassification(index=s.index, belief_type=BELIEF_FACTUAL, persist=True)]
    accept_classifications(store, result.session_id, cls, now="2099-01-01T00:00:00Z")

    bid = _derive_id_for_sentence(s.text, s.source)
    belief = store.get_belief(bid)
    assert belief is not None
    # The commit date, not the 2099 handshake-completion time.
    assert belief.created_at.startswith("2022-05-09"), (
        "an ast: candidate must carry its file's git commit date; "
        f"got {belief.created_at!r}"
    )
