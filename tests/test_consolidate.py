"""Unit tests for `aelfrice.consolidate` (#1312, #1176 proposal 4).

The audit is read-only, so every test that runs a pass also asserts the
store was not mutated.

Two of these are deliberately *distinguishing* rather than smoke: the
medoid tiebreak and the size->=3 restriction both have a plausible wrong
implementation that a happy-path assertion would not catch, so each is
paired with a case whose expected value differs under the wrong rule.
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest

from aelfrice.consolidate import (
    DEFAULT_MAX_CANDIDATE_PAIRS,
    MIN_COMPONENT_SIZE,
    ConsolidationReport,
    consolidation_audit,
    format_consolidation_report,
    shingles,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_USER,
    Belief,
)
from aelfrice.store import MemoryStore


def _insert(
    store: MemoryStore, bid: str, content: str, *, locked: bool = False
) -> None:
    store.insert_belief(
        Belief(
            id=bid,
            content=content,
            content_hash=f"h_{bid}",
            alpha=1.0,
            beta=1.0,
            type=BELIEF_FACTUAL,
            lock_level=LOCK_USER if locked else LOCK_NONE,
            locked_at="2026-08-03T00:00:00Z" if locked else None,
            created_at="2026-08-03T00:00:00Z",
            last_retrieved_at=None,
        )
    )


_PRE = (
    "the alpha config sets retries and timeouts for the staging deploy to"
)
"""Long shared prefix: one differing token still clears Jaccard 0.8."""

_ROLL = (
    "rollback the staging release whenever the deploy gate reports"
)
"""Same, for the medoid-centrality fixture."""


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(":memory:")


class TestShingles:
    def test_empty_tokens_yield_nothing(self) -> None:
        assert shingles([]) == frozenset()

    def test_short_belief_yields_one_whole_tuple(self) -> None:
        assert shingles(["a", "b"]) == frozenset({("a", "b")})

    def test_window_is_order_four(self) -> None:
        got = shingles(["a", "b", "c", "d", "e"])
        assert got == frozenset({("a", "b", "c", "d"), ("b", "c", "d", "e")})


class TestThresholdValidation:
    @pytest.mark.parametrize(
        ("kwargs", "needle"),
        [
            ({"jaccard_min": 1.5}, "jaccard_min"),
            ({"levenshtein_min": -0.1}, "levenshtein_min"),
            ({"max_candidate_pairs": 0}, "max_candidate_pairs"),
        ],
    )
    def test_malformed_thresholds_raise(
        self, store: MemoryStore, kwargs: dict, needle: str
    ) -> None:
        _insert(store, "b1", "deploy via terraform on aws today")
        _insert(store, "b2", "deploy via terraform on aws today.")
        _insert(store, "b3", "deploy via terraform on aws today!")
        with pytest.raises(ValueError, match=needle):
            consolidation_audit(store, **kwargs)


class TestComponentSizeFloor:
    """size >= 3 is the rule; a pair must NOT be reported.

    Distinguishing: the two cases below differ only by one member, and
    an implementation that reported every component would return 1
    cluster for both. The pair case asserting 0 is what fails then.
    """

    def test_a_duplicate_pair_is_not_a_cluster(
        self, store: MemoryStore
    ) -> None:
        _insert(store, "b1", "deploy via terraform on aws today")
        _insert(store, "b2", "deploy via terraform on aws today.")
        report = consolidation_audit(store)
        assert report.n_duplicate_pairs == 1, "the pair must still be found"
        assert report.n_clusters == 0
        assert report.n_beliefs_in_clusters == 0
        assert report.n_would_remove == 0

    def test_a_triple_is_a_cluster(self, store: MemoryStore) -> None:
        _insert(store, "b1", "deploy via terraform on aws today")
        _insert(store, "b2", "deploy via terraform on aws today.")
        _insert(store, "b3", "deploy via terraform on aws today!")
        report = consolidation_audit(store)
        assert report.n_clusters == 1
        assert report.n_beliefs_in_clusters == 3
        assert report.largest_cluster == 3

    def test_floor_constant_is_three(self) -> None:
        assert MIN_COMPONENT_SIZE == 3


class TestMedoidTiebreak:
    def test_tie_breaks_on_id_ascending(self, store: MemoryStore) -> None:
        """Three mutually equidistant members -> smallest id wins.

        Every pairwise Levenshtein distance here is equal, so the summed
        cost is identical for all three and the tiebreak is the only
        thing deciding the medoid. An implementation that kept the last
        best-scoring candidate (`<=` instead of `<`) returns "c3"; one
        that used `min(member_ids)` by luck also returns "c1", so the
        sibling test below separates those two.
        """
        _insert(store, "c1", f"{_PRE} one")
        _insert(store, "c2", f"{_PRE} two")
        _insert(store, "c3", f"{_PRE} six")
        report = consolidation_audit(store)
        assert report.n_clusters == 1
        assert report.clusters[0].medoid_id == "c1"

    def test_medoid_is_central_not_smallest_id(
        self, store: MemoryStore
    ) -> None:
        """The centre wins even when it does not sort first.

        "z_mid" sits between the other two, so it minimises summed
        distance while sorting last. This is the case that fails if the
        medoid is really just `min(member_ids)` — which is what
        `dedup.DuplicateCluster` uses and what a copy-paste would
        inherit.
        """
        _insert(store, "a_low", f"{_ROLL} red")
        _insert(store, "z_mid", f"{_ROLL} redd")
        _insert(store, "m_far", f"{_ROLL} reddd")
        report = consolidation_audit(store)
        assert report.n_clusters == 1
        members = report.clusters[0].member_ids
        assert members == ("a_low", "m_far", "z_mid"), "members sort ASC"
        assert report.clusters[0].medoid_id == "z_mid"


class TestMedoidSampleCapIsBounded:
    """The cap itself, not just the medoid rule (AC4/AC5 of #1312).

    `test_medoid_is_central_not_smallest_id` covers the *criterion* — it
    fails at `MEDOID_SAMPLE_CAP = 1`. It does not cover the *bound*:
    raising the cap to 10,000,000, i.e. deleting the limit this module
    partly exists to add, leaves every other test in this file green. A
    bound nothing distinguishes is a bound nothing protects.

    So: build a component larger than the cap in which the exact medoid
    provably lies past it, and assert the returned medoid comes from
    inside the sampled pool. Unbounded, the exact medoid wins and the
    assertion fails.
    """

    def test_the_shipped_cap_is_the_documented_value(self) -> None:
        """Pins the constant, not only the mechanism.

        The two arms below monkeypatch the cap, so they measure whether
        slicing happens — they stay green if someone raises the shipped
        value to 10,000,000 and effectively deletes the bound. This is
        the arm that makes such a change deliberate rather than silent;
        the 74s pass the cap exists to prevent is a real cost, and the
        docstring's stated tradeoff is calibrated to 64.
        """
        from aelfrice.consolidate import MEDOID_SAMPLE_CAP

        assert MEDOID_SAMPLE_CAP == 64

    @pytest.mark.timeout(60)
    def test_the_medoid_comes_from_the_sampled_prefix(
        self, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Terminates by construction: 12 short beliefs at cap 4."""
        import aelfrice.consolidate as consolidate_mod

        monkeypatch.setattr(consolidate_mod, "MEDOID_SAMPLE_CAP", 4)
        # Ids sort ASC, so "a00".."a11" — the sampled pool is a00..a03.
        # Distances grow with the index, so the *central* member is near
        # a05/a06, i.e. outside the pool. An unbounded medoid returns one
        # of those; a bounded one cannot.
        for i in range(12):
            _insert(store, f"a{i:02d}", f"{_PRE} {'z' * (i + 1)}")

        report = consolidation_audit(store)
        assert report.n_clusters == 1
        assert report.largest_cluster == 12
        pool = set(report.clusters[0].member_ids[:4])
        assert report.clusters[0].medoid_id in pool, (
            f"medoid {report.clusters[0].medoid_id!r} came from outside "
            "the sampled prefix: MEDOID_SAMPLE_CAP is not bounding "
            "anything"
        )

    @pytest.mark.timeout(60)
    def test_an_unbounded_cap_would_pick_a_different_member(
        self, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The control that keeps the arm above from passing vacuously.

        If the exact medoid happened to sit inside the first four
        members, the bounded assertion would hold for the wrong reason.
        Running the same corpus uncapped must therefore return a
        *different* member — which is what makes the pair a measurement
        of the bound rather than of the fixture.
        """
        import aelfrice.consolidate as consolidate_mod

        for i in range(12):
            _insert(store, f"a{i:02d}", f"{_PRE} {'z' * (i + 1)}")

        monkeypatch.setattr(consolidate_mod, "MEDOID_SAMPLE_CAP", 4)
        bounded = consolidation_audit(store).clusters[0].medoid_id
        monkeypatch.setattr(consolidate_mod, "MEDOID_SAMPLE_CAP", 10_000_000)
        exact = consolidation_audit(store).clusters[0].medoid_id
        assert bounded != exact, (
            f"bounded and exact both chose {bounded!r}: this fixture no "
            "longer distinguishes the cap"
        )


class TestWouldRemoveArithmetic:
    def test_one_medoid_survives_per_cluster(
        self, store: MemoryStore
    ) -> None:
        for i, suffix in enumerate(("", ".", "!", "?")):
            _insert(
                store, f"b{i}", f"deploy via terraform on aws today{suffix}"
            )
        for i, suffix in enumerate((" now", " now.", " now!")):
            _insert(
                store, f"c{i}", f"rotate the signing key every quarter{suffix}"
            )
        report = consolidation_audit(store)
        assert report.n_clusters == 2
        assert report.n_beliefs_in_clusters == 7
        assert report.n_would_remove == 5
        assert report.n_would_remove == (
            report.n_beliefs_in_clusters - report.n_clusters
        )
        # Two differently-sized clusters (4 and 3), so max and min
        # disagree: the only fixture in this file where an accidental
        # `min(...)` for largest_cluster is visible. AC2 of #1312 turns
        # on this field.
        assert report.largest_cluster == 4

    def test_share_of_store_denominator_is_the_whole_store(
        self, store: MemoryStore
    ) -> None:
        """Percent of beliefs scanned, not percent of clustered beliefs.

        Distinguishing on both axes. The isolated belief makes
        `n_beliefs_scanned` (8) differ from `n_beliefs_in_clusters` (7),
        so swapping the denominator moves the value from 62.5 to 71.43;
        and the literal band pins the `100.0` factor, so demoting the
        percentage to a fraction (0.625) fails too. The headline share
        the issue is priced on (3.19% of active, re-derived at #1316)
        is this property.
        """
        for i, suffix in enumerate(("", ".", "!", "?")):
            _insert(
                store, f"b{i}", f"deploy via terraform on aws today{suffix}"
            )
        for i, suffix in enumerate((" now", " now.", " now!")):
            _insert(
                store, f"c{i}", f"rotate the signing key every quarter{suffix}"
            )
        _insert(store, "solo", "an unrelated singleton with no near twin")

        report = consolidation_audit(store)
        assert report.n_beliefs_scanned == 8
        assert report.n_beliefs_in_clusters == 7
        assert report.n_would_remove == 5
        assert report.share_of_store == pytest.approx(100.0 * 5 / 8)
        assert 62.0 < report.share_of_store < 63.0

    def test_share_of_store_is_zero_on_empty(self) -> None:
        empty = ConsolidationReport(
            n_beliefs_scanned=0,
            n_candidate_pairs=0,
            n_duplicate_pairs=0,
            n_clusters=0,
            n_beliefs_in_clusters=0,
            n_locked_in_clusters=0,
            n_beliefs_rescued=0,
            largest_cluster=0,
            jaccard_min=0.8,
            levenshtein_min=0.85,
            max_candidate_pairs=DEFAULT_MAX_CANDIDATE_PAIRS,
        )
        assert empty.share_of_store == 0.0
        assert empty.n_would_remove == 0


_FAMILY_BODY = (
    "the staging deploy pipeline rotates the signing key and then "
    "revalidates every downstream artifact before promotion to prod"
)


def _homogeneous_family(store: MemoryStore, k: int) -> None:
    """`k` near-identical beliefs, each with a unique trailing token."""
    for i in range(k):
        _insert(store, f"f{i:03d}", f"{_FAMILY_BODY} variant {i}")


class TestLargeFamiliesDoNotVanish:
    """The regression that shipped in the first cut of this module.

    Blocking used to skip any 4-gram above a `df` cap, justified by the
    claim that real near-duplicates also share rarer shingles. In a
    *homogeneous* family every member shares every shingle, so they all
    sit at `df = K` and none is rarer: past the cap the family lost all
    its postings and the audit reported **zero** duplicates exactly when
    the duplicate family was largest. At the shipped cap of 32, K=32
    reported 31 removable and K=33 reported 0.
    """

    @pytest.mark.timeout(30)
    @pytest.mark.parametrize("k", [3, 32, 33])
    def test_family_of_any_size_is_one_cluster(
        self, store: MemoryStore, k: int
    ) -> None:
        """Terminates by construction: <=60 short beliefs, no waiting."""
        _homogeneous_family(store, k)
        report = consolidation_audit(store)
        assert report.n_clusters == 1
        assert report.largest_cluster == k
        assert report.n_would_remove == k - 1

    @pytest.mark.timeout(30)
    def test_no_cliff_between_32_and_33(self, store: MemoryStore) -> None:
        """The specific discontinuity, pinned as its own case.

        Parametrised coverage would still pass if both sides regressed
        together; this asserts the *relation*, which is what broke.
        """
        _homogeneous_family(store, 32)
        at_32 = consolidation_audit(store).n_would_remove
        store.close()
        bigger = MemoryStore(":memory:")
        try:
            _homogeneous_family(bigger, 33)
            at_33 = consolidation_audit(bigger).n_would_remove
        finally:
            bigger.close()
        assert at_33 == at_32 + 1, (
            f"a family of 33 priced {at_33} against 32's {at_32}: "
            "blocking is discarding whole families again"
        )


def _heterogeneous_family(
    store: MemoryStore, k: int, *, minority: int = 5
) -> None:
    """`k` over-cap near-duplicates whose *minimum-df* shingles differ.

    All `k` share `_FAMILY_BODY` (`df = k`); all but `minority` of them
    also share a trailing clause (`df = k - minority`). Both frequencies
    sit above the shipped cap of 32 at the sizes used here, so every
    member takes the rescue path — yet the majority's minimum is
    `k - minority` and the minority's is `k`.

    That difference is the whole fixture, and it is why the per-member
    discriminator goes at the **front**. Appending it instead creates a
    body-to-suffix boundary shingle shared by only the `minority`, which
    is a sub-cap posting: those members would then never reach the
    rescue at all and the split would be the cap's doing rather than the
    rescue rule's, testing something other than what it claims to.

    Under "post a rescued belief to its rarest shingles only" the two
    groups land in disjoint buckets and the family splits in two. Only
    posting to *every* shared shingle keeps it whole.
    """
    # Three tokens: long enough to make the majority's boundary
    # shingles a distinct, over-cap df, short enough that a
    # majority/minority pair still clears Jaccard 0.8 (0.857) and
    # Levenshtein 0.85 (0.886) — otherwise the family would split
    # on the *predicate* and the test would pass or fail for a
    # reason that has nothing to do with blocking.
    tail = " and the ledger"
    for i in range(k):
        extra = "" if i < minority else tail
        _insert(store, f"g{i:03d}", f"variant {i} {_FAMILY_BODY}{extra}")


class TestHeterogeneousFamiliesDoNotShatter:
    """The regression the rescue-as-*replacement* variant introduced.

    Fixing #1316 by posting every belief to its own rarest shingles
    trades the homogeneous cliff for a heterogeneous one. Two genuine
    near-duplicates whose minimum `df` differs never share a bucket, so
    a real family fragments into 2-components and the audit again
    reports **zero** on a large family. Measured on the development
    store that variant dropped 490 beliefs and 81 whole clusters that
    the flat cap had found, including a 46-member clique in which all
    1,035 pairs satisfy the shipped predicate.

    `TestLargeFamiliesDoNotVanish` cannot catch this — its family is
    homogeneous, so every member's minimum is the same and the shattering
    never triggers. That is why this class exists separately.
    """

    @pytest.mark.timeout(30)
    @pytest.mark.parametrize("k", [40, 50])
    def test_mixed_minima_still_form_one_cluster(
        self, store: MemoryStore, k: int
    ) -> None:
        """Terminates by construction: <=50 short beliefs, no waiting."""
        _heterogeneous_family(store, k)
        report = consolidation_audit(store)
        assert report.n_clusters == 1, (
            f"a heterogeneous family of {k} produced "
            f"{report.n_clusters} clusters: blocking is shattering it"
        )
        assert report.largest_cluster == k

    @pytest.mark.timeout(30)
    def test_the_rescue_is_a_fallback_not_a_replacement(
        self, store: MemoryStore
    ) -> None:
        """Sub-cap beliefs must keep their full posting set.

        The distinguishing arm: a family small enough to sit entirely
        under the cap must be clustered *without* anything being
        rescued. If the rescue had replaced the cap rather than backing
        it up, this family would route through the rarest-shingle path
        and `n_beliefs_rescued` would be non-zero.
        """
        _homogeneous_family(store, 6)
        report = consolidation_audit(store)
        assert report.n_clusters == 1
        assert report.n_beliefs_rescued == 0

    @pytest.mark.timeout(30)
    def test_an_over_cap_family_reports_that_it_was_rescued(
        self, store: MemoryStore
    ) -> None:
        """The fallback is never silent.

        Paired with the arm above so the two together distinguish which
        path ran, rather than only that a cluster came out.
        """
        _heterogeneous_family(store, 40)
        report = consolidation_audit(store)
        assert report.n_clusters == 1
        assert report.n_beliefs_rescued > 0


class TestCandidateBudgetIsDisclosed:
    @pytest.mark.timeout(30)
    def test_budget_reached_sets_truncated(self, store: MemoryStore) -> None:
        """A bound that does not announce itself reads as full coverage.

        Distinguishing: the same corpus without the budget reports
        `truncated is False`, so this cannot pass by always being True.

        Terminates by construction: the budget IS the bound.
        """
        _homogeneous_family(store, 40)
        capped = consolidation_audit(store, max_candidate_pairs=10)
        assert capped.truncated is True
        assert capped.n_candidate_pairs == 10

        full = consolidation_audit(store)
        assert full.truncated is False
        assert full.n_candidate_pairs > 10

    @pytest.mark.timeout(30)
    def test_truncation_is_named_in_the_report(
        self, store: MemoryStore
    ) -> None:
        _homogeneous_family(store, 40)
        text = format_consolidation_report(
            consolidation_audit(store, max_candidate_pairs=10)
        )
        assert "BUDGET REACHED" in text

    @pytest.mark.timeout(30)
    def test_the_budget_bounds_work_not_just_the_result_set(
        self, store: MemoryStore
    ) -> None:
        """The budget counts pairs *attempted*, not distinct pairs kept.

        `candidates` is a set, so a bucket re-proposing a pair it
        already holds does not grow it. A guard on `len(candidates)`
        can therefore stay false forever while the nested loop keeps
        running — the bound reads as real and is not, on exactly the
        near-identical-boilerplate shape it is documented to bound.

        A homogeneous family is that shape: every member shares every
        shingle, so each of the many buckets re-proposes the same
        `k*(k-1)/2` pairs. Setting the budget above that distinct total
        but below the attempted total must still truncate.

        Distinguishing both ways: the same corpus at a budget above the
        attempted total does *not* truncate, so this cannot pass by
        always being True.
        """
        _homogeneous_family(store, 12)
        distinct_total = consolidation_audit(store).n_candidate_pairs
        assert distinct_total == 66, distinct_total

        # Above every distinct pair, so a len(candidates) guard never
        # fires; below the attempted total, so an attempts guard does.
        capped = consolidation_audit(store, max_candidate_pairs=100)
        assert capped.truncated is True, (
            "the budget did not fire at 100 with only 66 distinct pairs: "
            "it is bounding the result set rather than the work"
        )

        roomy = consolidation_audit(store, max_candidate_pairs=1_000_000)
        assert roomy.truncated is False
        assert roomy.n_candidate_pairs == distinct_total


class TestLockedMembersAreNotPriced:
    def test_locked_non_medoid_members_are_excluded(
        self, store: MemoryStore
    ) -> None:
        """`aelf retire` refuses a locked belief without --force.

        Counting locks as removable prices work the product will not do.
        Distinguishing: the same five beliefs unlocked price 4.

        Terminates by construction: five beliefs, in-process.
        """
        for i in range(5):
            _insert(
                store,
                f"f{i:03d}",
                f"{_FAMILY_BODY} variant {i}",
                locked=i < 2,
            )
        report = consolidation_audit(store)
        assert report.n_beliefs_in_clusters == 5
        assert report.n_clusters == 1
        # one of the two locks is the medoid and survives anyway
        assert report.n_locked_in_clusters == 1
        assert report.n_would_remove == 3

    def test_same_family_unlocked_prices_higher(
        self, store: MemoryStore
    ) -> None:
        _homogeneous_family(store, 5)
        report = consolidation_audit(store)
        assert report.n_locked_in_clusters == 0
        assert report.n_would_remove == 4


class TestReadOnly:
    def test_audit_writes_nothing(self, store: MemoryStore) -> None:
        """No row is inserted, updated, or deleted — in any table.

        Row counts alone cannot see an in-place UPDATE, and they look at
        two tables out of the whole schema: a posterior bump, an
        `ingest_log` append, a `lock_level` flip or a
        `belief_corroborations` re-point would all pass a count check
        silently. `total_changes` is the connection's cumulative count
        of INSERT/UPDATE/DELETE rows, so a delta of zero is the actual
        read-only claim rather than a proxy for it.
        """
        _insert(store, "b1", "deploy via terraform on aws today")
        _insert(store, "b2", "deploy via terraform on aws today.")
        _insert(store, "b3", "deploy via terraform on aws today!")
        before_beliefs = len(store.list_beliefs_for_indexing())
        before_edges = store._conn.execute(
            "SELECT COUNT(*) FROM edges"
        ).fetchone()[0]
        before_changes = store._conn.total_changes

        consolidation_audit(store)

        assert store._conn.total_changes == before_changes, (
            "the audit executed a write; it is specified as read-only "
            "and contraction is deliberately not funded (#1312)"
        )
        assert len(store.list_beliefs_for_indexing()) == before_beliefs
        assert (
            store._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
            == before_edges
        )


class TestDeterminism:
    def test_two_passes_agree(self, store: MemoryStore) -> None:
        for i, suffix in enumerate(("", ".", "!", "?", ";")):
            _insert(
                store, f"b{i}", f"deploy via terraform on aws today{suffix}"
            )
        first = consolidation_audit(store)
        second = consolidation_audit(store)
        assert first == second
        assert format_consolidation_report(
            first
        ) == format_consolidation_report(second)


class TestReportRendering:
    def test_renders_no_ids_or_content(self, store: MemoryStore) -> None:
        """The report is pasteable into an issue: counts only."""
        _insert(store, "secret_id_1", "deploy via terraform on aws today")
        _insert(store, "secret_id_2", "deploy via terraform on aws today.")
        _insert(store, "secret_id_3", "deploy via terraform on aws today!")
        text = format_consolidation_report(consolidation_audit(store))
        assert "secret_id_1" not in text
        assert "terraform" not in text
        assert "would remove" in text
        assert "2 (" in text  # 3 members - 1 medoid

    def test_names_that_it_wrote_nothing(self, store: MemoryStore) -> None:
        text = format_consolidation_report(consolidation_audit(store))
        assert "read-only" in text


def _assert_no_ambient_config(start: Path) -> None:
    """Fail loudly if a `.aelfrice.toml` is visible from `start` upward.

    `_cmd_doctor_consolidate` resolves its thresholds through
    `dedup.load_dedup_config()`, which walks up from the working
    directory. A developer with `[dedup]` set at or above the repo would
    otherwise see a CLI test fail for a reason that has nothing to do
    with the code — CI green, local red (#1295 is the same shape).

    Resolve first: `load_dedup_config` resolves its start and
    `Path.parents` is lexical, so an unresolved walk checks a different
    chain than the one actually read.
    """
    resolved = start.resolve()
    stray = [
        str(parent / ".aelfrice.toml")
        for parent in (resolved, *resolved.parents)
        if (parent / ".aelfrice.toml").is_file()
    ]
    assert not stray, (
        "this test pins the TOML tier by running from a scratch "
        f"directory, but that directory is not clean: {stray}"
    )


class TestCliSurface:
    def test_doctor_consolidate_runs(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default path. Pinned against ambient `[dedup]` config.

        Terminates by construction: pure in-process computation over
        three beliefs, no subprocess, no waiting, no retry.
        """
        from aelfrice.cli import main

        db = str(tmp_path / "brain.db")
        monkeypatch.setenv("AELFRICE_DB", db)
        monkeypatch.chdir(tmp_path)
        _assert_no_ambient_config(tmp_path)

        s = MemoryStore(db)
        _insert(s, "b1", "deploy via terraform on aws today")
        _insert(s, "b2", "deploy via terraform on aws today.")
        _insert(s, "b3", "deploy via terraform on aws today!")
        s.close()

        out = io.StringIO()
        rc = main(["doctor", "--consolidate"], out=out)
        assert rc == 0
        text = out.getvalue()
        assert "Consolidation audit" in text
        assert "would remove          : 2" in text

    def test_ambient_toml_tier_is_honoured(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The `[dedup]` TOML tier reaches the consolidation audit.

        Distinguishing rather than smoke: the three beliefs cluster at
        the shipped defaults (the test above asserts exactly that), and
        `levenshtein_min = 0.995` is above their pairwise ratio, so the
        cluster must disappear. A build that ignored the TOML tier would
        still report a cluster here.

        Terminates by construction: no subprocess, no waiting.
        """
        from aelfrice.cli import main

        db = str(tmp_path / "brain.db")
        monkeypatch.setenv("AELFRICE_DB", db)
        monkeypatch.chdir(tmp_path)
        _assert_no_ambient_config(tmp_path)

        (tmp_path / ".aelfrice.toml").write_text(
            "[dedup]\nlevenshtein_min = 0.995\n", encoding="utf-8"
        )

        s = MemoryStore(db)
        _insert(s, "b1", "deploy via terraform on aws today")
        _insert(s, "b2", "deploy via terraform on aws today.")
        _insert(s, "b3", "deploy via terraform on aws today!")
        s.close()

        out = io.StringIO()
        rc = main(["doctor", "--consolidate"], out=out)
        assert rc == 0
        text = out.getvalue()
        assert "Levenshtein ratio >= 0.995" in text, text
        assert "would remove          : 0" in text, text

    def test_cli_overrides_beat_the_toml_tier(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`--consolidate-*` outrank `[dedup]`, and the df cap is wired.

        The same config that suppresses the cluster above is present
        here, so a build that ignored the CLI overrides would report 0.

        Terminates by construction: no subprocess, no waiting.
        """
        from aelfrice.cli import main

        db = str(tmp_path / "brain.db")
        monkeypatch.setenv("AELFRICE_DB", db)
        monkeypatch.chdir(tmp_path)
        _assert_no_ambient_config(tmp_path)

        (tmp_path / ".aelfrice.toml").write_text(
            "[dedup]\nlevenshtein_min = 0.995\n", encoding="utf-8"
        )

        s = MemoryStore(db)
        _insert(s, "b1", "deploy via terraform on aws today")
        _insert(s, "b2", "deploy via terraform on aws today.")
        _insert(s, "b3", "deploy via terraform on aws today!")
        s.close()

        out = io.StringIO()
        rc = main(
            [
                "doctor",
                "--consolidate",
                "--consolidate-levenshtein",
                "0.85",
                "--consolidate-max-pairs",
                "1",
            ],
            out=out,
        )
        assert rc == 0
        text = out.getvalue()
        assert "Levenshtein ratio >= 0.85" in text, text
        assert "BUDGET REACHED" in text, text

    def test_malformed_threshold_exits_one(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aelfrice.cli import main

        db = str(tmp_path / "brain.db")
        monkeypatch.setenv("AELFRICE_DB", db)
        MemoryStore(db).close()

        out = io.StringIO()
        rc = main(
            ["doctor", "--consolidate", "--consolidate-jaccard", "1.5"],
            out=out,
        )
        assert rc == 1
