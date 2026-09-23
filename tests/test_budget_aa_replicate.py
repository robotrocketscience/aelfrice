"""#1546 K3 — the mutation guard on the A/A replicate itself.

`scripts/budget_discriminability_aa_replicate.py` reports `A/A_band`, the
spread the census statistic shows when only the belief insertion order moves.
On shipped code it reports 0.0pp, and a zero reported by an instrument that
cannot report anything else is a print statement rather than a measurement.

The load-bearing test here is therefore the positive control: it drives the
replicate against a fake `retrieval.retrieve` whose output genuinely depends on
insertion order — it drops the pool member with the highest rowid, which is the
belief the store saw last — and requires the band to go non-zero on every lane.
Without it, nothing in this file distinguishes "the retrieval paths this census
exercises do not read insertion order" from "the replicate never looks".

The second guard is one level down. A band of zero over a statistic whose
inputs never moved is a different finding from a band of zero over inputs that
moved and cancelled, so `arm_outputs` sweeps every lane x query x grid cell and
asks whether the retrieved list itself moved. That sweep answers 0 of 2070 on
shipped code, and the mechanism is visible in `store.py`: the FTS search orders
by `bm25(beliefs_fts), b.id`, a content-derived tie-break, and the one lane
known to order by rowid — the temporal spine — never runs on the census's
edgeless `:memory:` stores.

The third pins the perturbation. If permuting the insertion order did not move
a rowid, the whole replicate would be vacuous, so that is asserted against the
store rather than assumed from the code.

Most tests pass `order_sensitivity_sweep=False`, which changes nothing about
how the band is computed and skips only the diagnostic; the sweep has its own
test. Every run here builds `:memory:` stores through the census's own
`_open_store` and opens no user store.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from aelfrice.store import MemoryStore

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "budget_discriminability_aa_replicate.py"
)

# Small on purpose: the band is a sample range, so two permuted replicates plus
# the committed order is the smallest population that can exhibit a spread at
# all. Every assertion below is on whether a spread is zero or non-zero, never
# on its width, so a larger count would buy runtime and no power.
TEST_SEEDS = 2


def _load_replicate() -> Any:
    """Import the replicate as a module, the `test_budget_census.py` way."""
    spec = importlib.util.spec_from_file_location(
        "budget_discriminability_aa_replicate", str(_SCRIPT_PATH)
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["budget_discriminability_aa_replicate"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def aa() -> Any:
    return _load_replicate()


# --- The perturbation is not inert ------------------------------------


def test_the_permutation_actually_moves_the_rowids(aa: Any) -> None:
    """If insertion order did not move a rowid, the replicate is vacuous.

    Asserted against a real store rather than read off `_open_store`, because
    the claim is about what SQLite assigns, not about what the loop looks like.
    Every belief id must still be present in every replicate: a permutation
    that dropped or added a belief would be a different corpus, not a different
    order.
    """
    census = aa.census
    base = census.corpora()
    orders: list[dict[str, list[tuple[int, str]]]] = []
    for replicate in range(0, TEST_SEEDS + 1):
        population = aa.permuted_corpora(base, replicate)
        per_query: dict[str, list[tuple[int, str]]] = {}
        for q in population:
            store = census._open_store(q)
            try:
                rows = store._conn.execute(
                    "SELECT rowid, id FROM beliefs ORDER BY rowid"
                ).fetchall()
            finally:
                store.close()
            per_query[f"{q.corpus}/{q.qid}"] = [
                (int(r[0]), str(r[1])) for r in rows
            ]
        orders.append(per_query)

    keys = sorted(orders[0])
    assert keys, "no labelled queries, so nothing was permuted"

    # Same beliefs everywhere: only their rowids may differ.
    for per_query in orders[1:]:
        assert sorted(per_query) == keys
        for key in keys:
            assert {bid for _, bid in per_query[key]} == {
                bid for _, bid in orders[0][key]
            }, key

    moved = [
        key
        for key in keys
        if len({tuple(o[key]) for o in orders}) > 1
    ]
    assert moved == keys, (
        "the permutation left the rowid assignment unchanged on "
        f"{sorted(set(keys) - set(moved))}, so the A/A perturbation is inert "
        "there and any band measured over those queries is vacuous"
    )


def test_a_permuted_replicate_keeps_the_query_gold_and_content(aa: Any) -> None:
    """Only the order moves. The population itself must be identical.

    A replicate that also changed the gold join or the query text would be an
    A/B, and its spread would not be noise.
    """
    census = aa.census
    base = census.corpora()
    permuted = aa.permuted_corpora(base, 1)
    assert len(permuted) == len(base)
    for before, after in zip(base, permuted, strict=True):
        assert (after.corpus, after.qid, after.query) == (
            before.corpus,
            before.qid,
            before.query,
        )
        assert after.gold_ids == before.gold_ids
        assert sorted(b.id for b in after.beliefs) == sorted(
            b.id for b in before.beliefs
        )
        assert {b.id: b.content for b in after.beliefs} == {
            b.id: b.content for b in before.beliefs
        }


def test_the_reference_replicate_is_the_committed_order(aa: Any) -> None:
    """Replicate 0 must be the census's own population, byte for byte.

    It is the row that ties the A/A report to the published K0 figures. A
    reference replicate that was itself permuted would make the band a spread
    around an unpublished baseline.
    """
    base = aa.census.corpora()
    assert aa.permuted_corpora(base, aa.REFERENCE_REPLICATE) == list(base)


def test_the_census_report_measures_the_population_it_is_given(aa: Any) -> None:
    """The seam the whole replicate rests on.

    `census.report(queries=...)` is the one change this work makes to the K0
    producer. If it ignored its argument, every replicate would measure the
    committed order, the band would be zero for a reason that has nothing to do
    with insertion order, and nothing else in this file would notice.

    Checked by giving it a strictly smaller population and requiring the
    published counts to shrink with it, rather than by reading the signature.
    """
    census = aa.census
    base = census.corpora()
    subset = [q for q in base if q.corpus == "posterior_ranking"]
    assert 0 < len(subset) < len(base)

    whole = census.report()
    part = census.report(queries=subset)
    assert part["n"] < whole["n"], (part["n"], whole["n"])
    assert set(part["lanes"]["ups"]["by_corpus"]) == {"posterior_ranking"}
    assert set(whole["lanes"]["ups"]["by_corpus"]) == {
        "benchmark",
        "posterior_ranking",
    }


# --- The control, and then the positive control -----------------------


@pytest.mark.timeout(180)
def test_the_band_is_zero_on_shipped_code(aa: Any) -> None:
    """The control arm. On its own this proves nothing; see the arm below."""
    rep = aa.replicate(TEST_SEEDS, order_sensitivity_sweep=False)
    assert rep["violations"] == []
    assert rep["n"] > 0
    assert rep["n_band"] == 0
    assert rep["aa_band_pp"] == 0.0
    assert rep["bands"], "no cell produced a band at all"
    assert set(rep["bands"].values()) == {0.0}


@pytest.mark.timeout(300)
def test_the_replicate_reports_a_non_zero_band_when_retrieval_reads_the_rowid(
    aa: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The positive control, and the load-bearing test in this file.

    The fake reads the store's `rowid` column — the one thing the permutation
    moves — and drops the pool member the store saw last, on the grid cells
    whose L2.5 sub-budget is 200. That is a genuine insertion-order dependence:
    which belief it drops is decided by the ingest order and by nothing else,
    so whether the dropped belief is gold varies from replicate to replicate
    and EC moves with it.

    The assertion is on the replicate's own `aa_band_pp`, not on a spread the
    test computes itself, because what is under test is the replicate's ability
    to report a spread. It is also per lane: a band carried by one lane while
    five report zero would read as covered while five lanes' bands were never
    exercised.

    The trigger sits on the sub-budget axis rather than on `token_budget` so it
    is lane-independent — 200 is a pre-registered grid value on every lane,
    the shipped cell is 400, and `POOL_PROBE_BUDGET` is 1e9, so the unbudgeted
    probe is untouched and every arm stays a subset of it.

    The census also reports block-identity violations under this fake, and it
    is right to: every pool in these corpora prices below every budget in the
    grid, so a discordant footprint is also a broken bound here. That is why
    the replicate computes the band whatever the violation list says — a band
    gated on a clean violation list could only ever report zero on this corpus.
    """
    census = aa.census
    shipped = census.retrieval.retrieve
    trigger_sub = 200
    assert trigger_sub in census.L25_SUBBUDGETS
    assert trigger_sub != census.SHIPPED_L25_SUBBUDGET
    assert trigger_sub < census.POOL_PROBE_BUDGET

    def rowid_sensitive_retrieve(
        store: MemoryStore, query: str, **kwargs: Any
    ) -> list[Any]:
        hits = shipped(store, query, **kwargs)
        if kwargs.get("l25_token_subbudget") != trigger_sub or not hits:
            return hits
        rowid_by_id = {
            str(row[0]): int(row[1])
            for row in store._conn.execute("SELECT id, rowid FROM beliefs")
        }
        last = max(hits, key=lambda b: rowid_by_id[b.id])
        return [b for b in hits if b.id != last.id]

    monkeypatch.setattr(
        census.retrieval, "retrieve", rowid_sensitive_retrieve
    )
    rep = aa.replicate(TEST_SEEDS, order_sensitivity_sweep=False)

    assert rep["aa_band_pp"], (
        "the replicate reported a zero band against a retrieval path whose "
        "output is decided by insertion order; its 0.0pp on shipped code is "
        "then a print statement and not a measurement"
    )
    flat = {
        lane.name: rep["bands"][lane.name] for lane in census.lanes()
    }
    assert all(v > 0 for v in flat.values()), (
        "the band is non-zero on only some lanes, so the lanes reporting "
        f"zero are unmutation-tested: {flat}"
    )
    assert rep["n_band"] == 0, (
        "the fake moved N as well as EC, so the band mixes populations and "
        f"the arm proves less than it looks: {rep['n_band']}"
    )

    # And the same call reports a zero band once the shipped path is back, so
    # the spread is attributable to the mutation and not to the fixture.
    monkeypatch.undo()
    clean = aa.replicate(TEST_SEEDS, order_sensitivity_sweep=False)
    assert clean["aa_band_pp"] == 0.0
    assert clean["violations"] == []


@pytest.mark.timeout(300)
def test_the_order_sensitivity_sweep_can_see_an_order_sensitive_arm(
    aa: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The diagnostic one level below the band, mutation-checked the same way.

    `arm_cells_order_sensitive` is 0 of 2070 on shipped code, and that zero is
    what turns "the band is zero" into "nothing the statistic reads moved at
    all". A counter that cannot count is worth less than no counter, so the
    same rowid-reading fake must drive it above zero.
    """
    census = aa.census
    shipped = census.retrieval.retrieve

    def rowid_sensitive_retrieve(
        store: MemoryStore, query: str, **kwargs: Any
    ) -> list[Any]:
        hits = shipped(store, query, **kwargs)
        if kwargs.get("l25_token_subbudget") != 200 or not hits:
            return hits
        rowid_by_id = {
            str(row[0]): int(row[1])
            for row in store._conn.execute("SELECT id, rowid FROM beliefs")
        }
        last = max(hits, key=lambda b: rowid_by_id[b.id])
        return [b for b in hits if b.id != last.id]

    baseline = aa.replicate(1)
    assert baseline["arm_cells_examined"] > 0
    assert baseline["arm_cells_order_sensitive"] == 0, (
        "an arm already moves with insertion order on shipped code; the "
        "result document's mechanism claim needs re-deriving before this "
        "test is edited"
    )

    monkeypatch.setattr(
        census.retrieval, "retrieve", rowid_sensitive_retrieve
    )
    mutated = aa.replicate(1)
    assert mutated["arm_cells_examined"] == baseline["arm_cells_examined"]
    assert mutated["arm_cells_order_sensitive"] > 0, (
        "the sweep counted no order-sensitive arm against a retrieval path "
        "decided by insertion order, so its zero on shipped code is vacuous"
    )
    assert mutated["arm_cells_order_sensitive_examples"]


# --- Determinism, and the published surface ---------------------------


@pytest.mark.timeout(180)
def test_the_replicate_is_deterministic_at_a_fixed_seed(aa: Any) -> None:
    """#605. Two runs at the pinned seed must hash identically.

    The control is the seed itself: a different seed must produce a different
    population, otherwise the hash agreeing proves only that the permutation
    never ran.
    """
    first = aa.replicate(TEST_SEEDS, order_sensitivity_sweep=False)
    second = aa.replicate(TEST_SEEDS, order_sensitivity_sweep=False)
    assert aa._stable_hash(first) == aa._stable_hash(second)

    base = aa.census.corpora()
    one = aa.permuted_corpora(base, 1)
    two = aa.permuted_corpora(base, 2)
    assert [tuple(b.id for b in q.beliefs) for q in one] != [
        tuple(b.id for b in q.beliefs) for q in two
    ], "two replicate indices produced the same population"


def test_the_permutation_seed_is_pinned_and_content_addressed(aa: Any) -> None:
    """The seed depends on the pinned base seed and the query id, on nothing else.

    Pinned by value, not by shape: a seed derived from a clock, a path, or
    `hash()` would still be an int, and only re-deriving the digest here shows
    it is none of those.
    """
    import hashlib

    for replicate, corpus, qid in (
        (1, "benchmark", "cook_01"),
        (7, "posterior_ranking", "q3"),
    ):
        payload = f"{aa.BASE_SEED}:{replicate}:{corpus}:{qid}".encode()
        want = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        assert aa.permutation_seed(replicate, corpus, qid) == want
    assert aa.BASE_SEED == 1546


@pytest.mark.timeout(300)
def test_emit_figures_is_flat_json_and_names_every_lane(aa: Any) -> None:
    """The `scripts/check_derived_figures.py` protocol, at the shape it reads."""
    rep = aa.replicate(1)
    figs = aa.figures(rep)
    assert json.loads(json.dumps(figs)) == figs
    for value in figs.values():
        assert not isinstance(value, dict | list), figs
    for lane in aa.census.lanes():
        assert f"aa_band_pp.{lane.name}" in figs
    assert figs["aa_base_seed"] == aa.BASE_SEED
    assert figs["aa_band_pp"] == rep["aa_band_pp"]
    assert figs["aa_arm_cells_order_sensitive"] == 0


def test_figures_omits_the_sweep_keys_when_the_sweep_was_skipped(
    aa: Any,
) -> None:
    """A skipped diagnostic must never reach a published figure.

    The sweep costs 50s of a 57s run while the band costs 8s, and the
    derived-figures gate's markers cover only the band — so the sweep is
    off by default and `--sweep` turns it on. The guarantee is unchanged
    and is now carried by omission rather than by refusing: there is no
    key to read, rather than a key holding a number nothing measured.
    Emitting `aa_arm_cells_order_sensitive` as null would read as "no
    order-sensitive arm" instead of "not measured", which is the failure
    this pins.
    """
    rep = aa.replicate(1, order_sensitivity_sweep=False)
    assert rep["arm_sweep"] is False
    assert rep["arm_cells_order_sensitive"] is None

    figs = aa.figures(rep)
    assert figs["aa_arm_sweep"] is False, (
        "a consumer must be able to tell a skipped sweep from a clean one"
    )
    assert "aa_arm_cells_order_sensitive" not in figs
    assert "aa_arm_cells_examined" not in figs
    # The band is still published: skipping the diagnostic changes nothing
    # about how it is computed.
    assert figs["aa_band_pp"] == rep["aa_band_pp"]


def test_figures_publishes_the_sweep_keys_when_the_sweep_ran(aa: Any) -> None:
    """The other half: omission must be caused by the skip, not by a bug.

    Without this, a `figures()` that dropped the sweep keys unconditionally
    would pass the arm above while silently never publishing the diagnostic.
    """
    rep = aa.replicate(1, order_sensitivity_sweep=True)
    assert rep["arm_sweep"] is True

    figs = aa.figures(rep)
    assert figs["aa_arm_sweep"] is True
    assert figs["aa_arm_cells_order_sensitive"] == rep["arm_cells_order_sensitive"]
    assert figs["aa_arm_cells_examined"] == rep["arm_cells_examined"]


def test_the_replicate_exits_non_zero_when_a_replicate_violates(
    aa: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reported violation must also be a failing exit code (#1160)."""
    monkeypatch.setattr(
        aa,
        "replicate",
        lambda *a, **k: {**_MINIMAL_REPORT, "violations": ["synthetic"]},
    )
    assert aa.main([]) == 2
    assert aa.main(["--emit-figures"]) == 2


def test_a_seed_count_below_one_is_refused(aa: Any) -> None:
    """`--seeds 0` has no spread to measure and must not report 0.0pp."""
    assert aa.main(["--seeds", "0"]) == 1
    with pytest.raises(ValueError, match="at least 1"):
        aa.replicate(0, order_sensitivity_sweep=False)


_MINIMAL_REPORT: dict[str, Any] = {
    "instrument": "x",
    "replicates_of": "y",
    "pre_registration": "z",
    "statistic": "s",
    "perturbation": "p",
    "base_seed": 1546,
    "seeds": 1,
    "replicates": 2,
    "reference_replicate": 0,
    "n": 7,
    "n_band": 0,
    "cells": ["ups"],
    "unbandable_cells": [],
    "rows": [],
    "bands": {"ups": 0.0},
    "aa_band_pp": 0.0,
    "distinct_insertion_orders": {},
    "queries_with_one_insertion_order": [],
    "arm_sweep": True,
    "arm_cells_examined": 0,
    "arm_cells_order_sensitive": 0,
    "arm_cells_order_sensitive_examples": [],
    "resolvers": {},
    "violations": [],
}
