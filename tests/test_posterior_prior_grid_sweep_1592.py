"""The #1592 sweep must be able to see an updated posterior.

`benchmarks/posterior_prior_grid_sweep.py` answers #1592 AC1 by counting
beliefs whose `(alpha, beta)` is *not* one of the insertion priors. A
sweep that reports zero because it cannot recognise an update would
publish the headline it was written to test, so both arms are pinned
here: a store straight off the prior grid reports zero, and the same
store with one posterior bumped reports one.

The grid itself is also pinned against drift. It is built by calling
`get_source_adjusted_prior` rather than by transcribing its arithmetic,
and that has to stay true — a hand-copied grid that fell behind a change
to `TYPE_PRIORS` or `_AGENT_INFERRED_DEFLATION` would reclassify every
belief in the store as updated and invert the finding.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from aelfrice.classification_core import (
    TYPE_PRIORS,
    USER_SOURCE,
    get_source_adjusted_prior,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.store import MemoryStore

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "benchmarks" / "posterior_prior_grid_sweep.py"

_spec = importlib.util.spec_from_file_location("_prior_grid_sweep", _SCRIPT)
assert _spec and _spec.loader
# `Any` for the reason `tests/test_soak_producer_figures.py` gives: pyright
# runs `tests/` in strict mode and an implicitly-typed module object turns
# every attribute read into an `Unknown`.
sweep: Any = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)


def _belief(bid: str, alpha: float, beta: float) -> Belief:
    return Belief(
        id=bid,
        content=f"content for {bid}",
        content_hash=f"h_{bid}",
        alpha=alpha,
        beta=beta,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-09-22T00:00:00Z",
        last_retrieved_at=None,
    )


def _store_on_the_grid(path: Path) -> None:
    """Three beliefs, each created at a real insertion prior."""
    store = MemoryStore(str(path))
    try:
        store.insert_belief(_belief("U", *get_source_adjusted_prior(
            BELIEF_FACTUAL, USER_SOURCE)))
        store.insert_belief(_belief("A", *get_source_adjusted_prior(
            BELIEF_FACTUAL, "agent")))
        store.insert_belief(_belief("G", 1.0, 1.0))
    finally:
        store.close()


def test_a_store_straight_off_the_prior_grid_reports_nothing_updated(
    tmp_path: Path,
) -> None:
    db = tmp_path / "memory.db"
    _store_on_the_grid(db)

    row = sweep.probe(db, sweep.insertion_priors())

    assert row["beliefs"] == 3
    assert row["off_prior"] == 0, row["off_pairs"]
    assert row["on_prior"] == 3


def test_one_bumped_posterior_is_counted_off_the_grid(tmp_path: Path) -> None:
    """The non-vacuity arm: without it, a blind sweep reports the headline.

    One success observation, which `apply_evidence` spends as
    `alpha += 1`. The other two beliefs are untouched, so the count
    distinguishes "saw the update" from "classified everything as
    updated".
    """
    db = tmp_path / "memory.db"
    _store_on_the_grid(db)

    store = MemoryStore(str(db))
    try:
        assert store.bump_posterior("A", 1.0, 0.0) is not None
    finally:
        store.close()

    row = sweep.probe(db, sweep.insertion_priors())

    assert row["beliefs"] == 3
    assert row["off_prior"] == 1, row["off_pairs"]
    assert row["on_prior"] == 2
    (pair,) = row["off_pairs"]
    alpha, beta = pair
    prior_alpha, prior_beta = get_source_adjusted_prior(BELIEF_FACTUAL, "agent")
    assert alpha == prior_alpha + 1.0
    assert beta == prior_beta


def test_a_fractional_evidence_step_is_still_off_the_grid(
    tmp_path: Path,
) -> None:
    """`feedback._bayesian_delta` honours fractional valences.

    Weighted sources (a propagated signal attenuated by broker
    confidence) move a posterior by well under 1.0, so the step the
    classifier has to see is much smaller than one observation. The
    tolerance must not absorb it.

    The delta is one-sided on purpose: positive valence goes entirely to
    alpha, so a split like `(0.05, 0.05)` is a shape this writer cannot
    produce and would not test the real step.
    """
    db = tmp_path / "memory.db"
    _store_on_the_grid(db)

    store = MemoryStore(str(db))
    try:
        assert store.bump_posterior("A", 0.1, 0.0) is not None
    finally:
        store.close()

    row = sweep.probe(db, sweep.insertion_priors())

    assert row["off_prior"] == 1, row["off_pairs"]
    assert row["mean_moved"] == 1


def test_dedupe_mass_is_separated_from_a_real_move(tmp_path: Path) -> None:
    """Off-grid is not the same as informative, and the split is the point.

    `MemoryStore` dedupe sums a duplicate group's alphas and betas onto
    the canonical row, so `n` copies of one prior land at `n * prior` —
    off the grid, with the posterior mean unchanged. Retrieval blends
    `log(posterior_mean)`, so such a belief scores exactly as it did
    before. Counting it as evidence that the feedback loop ran is the
    error this arm exists to catch.
    """
    prior_alpha, prior_beta = get_source_adjusted_prior(
        BELIEF_FACTUAL, "agent")
    db = tmp_path / "memory.db"
    store = MemoryStore(str(db))
    try:
        # Three duplicates collapsed: the dedupe signature.
        store.insert_belief(_belief("D", 3 * prior_alpha, 3 * prior_beta))
        # A real positive-valence event: alpha only, so the mean rises.
        store.insert_belief(_belief("M", prior_alpha + 0.1, prior_beta))
    finally:
        store.close()

    row = sweep.probe(db, sweep.insertion_priors())

    assert row["off_prior"] == 2
    assert row["mean_preserving"] == 1
    assert row["mean_moved"] == 1


def test_an_alpha_only_move_sharing_a_prior_mean_is_not_dedupe(
    tmp_path: Path,
) -> None:
    """The collision an adversarial review found, pinned.

    `(3.6000000000000005, 1.0)` is the factual non-user prior plus 3.0 of
    alpha with beta untouched — a real move. Its mean is 0.7826, which is
    exactly `(1.8, 0.5)`'s, and `2 * 1.8` rounds to the same four decimal
    places. Classifying by mean, or by a tolerant ratio, calls it dedupe
    mass and hides it. Only the exact float product separates the two:
    `2 * 1.8 == 3.6`, but this belief carries `6 * 0.6000000000000001 ==
    3.6000000000000005`.
    """
    db = tmp_path / "memory.db"
    store = MemoryStore(str(db))
    try:
        store.insert_belief(_belief("C", 3.6000000000000005, 1.0))
    finally:
        store.close()

    row = sweep.probe(db, sweep.insertion_priors())

    assert row["off_prior"] == 1
    assert row["mean_preserving"] == 0, "an alpha-only move read as dedupe"
    assert row["mean_moved"] == 1


def test_the_write_ahead_log_is_copied_with_the_database(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    """Dropping the `-wal` copy must red, not pass quietly.

    In WAL mode a committed belief lives in `memory.db-wal` until a
    checkpoint. Copying only `memory.db` loses exactly the most recent
    writes — the ones most likely to carry a moved posterior — and the
    sweep under-reports while looking healthy. An adversarial review
    emptied the sidecar loop and every other arm still passed.

    This pins the mechanism rather than the effect, deliberately. Closing
    a `MemoryStore` checkpoints the log, so an in-process fixture cannot
    reliably leave a belief WAL-resident at probe time; a test that tried
    would pass for the wrong reason. Recording what gets copied fails the
    moment the sidecar loop stops running.
    """
    db = tmp_path / "memory.db"
    _store_on_the_grid(db)
    wal = db.with_name(db.name + "-wal")
    wal.write_bytes(b"")  # present, so the probe has something to copy

    copied: list[str] = []
    real_copyfile = sweep.shutil.copyfile

    def spy(src: Any, dst: Any, **kwargs: Any) -> Any:
        copied.append(Path(src).name)
        return real_copyfile(src, dst, **kwargs)

    monkeypatch.setattr(sweep.shutil, "copyfile", spy)
    sweep.probe(db, sweep.insertion_priors())

    assert "memory.db" in copied, "the database itself was never copied"
    assert "memory.db-wal" in copied, (
        "the write-ahead log was not copied; beliefs committed since the "
        f"last checkpoint would be missed. copied: {copied}"
    )


def test_the_probe_never_opens_the_path_it_was_given(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    """The store handed in must never be opened, read-only or otherwise.

    Opening a `MemoryStore` read-write runs schema DDL and migrations,
    which is a write, so the probe copies first and opens the copy. Both
    halves matter and neither was pinned: a review mutated
    `read_only=True` away and every arm still passed. This asserts the
    path actually opened is not the one passed in, which fails if the
    copy is skipped, and that it is opened read-only.
    """
    db = tmp_path / "memory.db"
    _store_on_the_grid(db)

    opened: list[tuple[str, bool]] = []
    real = MemoryStore

    def spy(path: str, *args: Any, **kwargs: Any) -> Any:
        opened.append((path, bool(kwargs.get("read_only", False))))
        return real(path, *args, **kwargs)

    monkeypatch.setattr("aelfrice.store.MemoryStore", spy)
    sweep.probe(db, sweep.insertion_priors())

    assert opened, "probe opened no store at all"
    for path, read_only in opened:
        assert Path(path) != db, (
            f"probe opened the caller's store at {path} instead of a copy"
        )
        assert read_only, f"probe opened {path} read-write"


def test_the_prior_grid_is_derived_from_the_shipped_function() -> None:
    """Every prior the shipped resolver can return must be in the grid.

    This is what stops the sweep inverting its own finding after a change
    to `TYPE_PRIORS`, `_AGENT_INFERRED_DEFLATION`, or
    `_DEFLATED_ALPHA_FLOOR`: a grid that fell behind any of them would
    classify ordinary insertions as updates.
    """
    grid = sweep.insertion_priors()
    for belief_type in TYPE_PRIORS:
        for source in (USER_SOURCE, "agent"):
            pair = get_source_adjusted_prior(belief_type, source)
            assert sweep._matches_prior(pair, grid) is not None, (
                f"{belief_type}/{source} resolves to {pair}, which the "
                f"sweep would report as an updated posterior"
            )


def test_an_unknown_pair_is_reported_rather_than_silently_counted_on_grid(
    tmp_path: Path,
) -> None:
    """A pair matching no prior must land in `off_pairs`, not be dropped.

    The sweep's conclusion rests on `off_prior == 0` meaning "nothing was
    updated". That reading is only sound if every unrecognised pair is
    surfaced, so the reader can tell an update from a prior the script
    does not know about.
    """
    db = tmp_path / "memory.db"
    store = MemoryStore(str(db))
    try:
        store.insert_belief(_belief("X", 4.25, 2.75))
    finally:
        store.close()

    row = sweep.probe(db, sweep.insertion_priors())

    assert row["off_prior"] == 1
    assert row["off_pairs"] == {(4.25, 2.75): 1}
