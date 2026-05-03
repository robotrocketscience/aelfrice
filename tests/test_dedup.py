"""Unit tests for `aelfrice.dedup` (#197 R1).

Covers the four similarity primitives (jaccard / levenshtein /
levenshtein_ratio / cluster_pairs) plus the top-level `dedup_audit`
entry point against a real `MemoryStore` fixture.

The audit is read-only — every test asserts that no edges and no
beliefs were inserted/mutated by the audit pass.
"""
from __future__ import annotations

import pytest

from aelfrice.dedup import (
    DEFAULT_JACCARD_MIN,
    DEFAULT_LEVENSHTEIN_MIN,
    DuplicateCluster,
    DuplicatePair,
    cluster_pairs,
    dedup_audit,
    format_audit_report,
    jaccard,
    levenshtein_distance,
    levenshtein_ratio,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.store import MemoryStore


# --- Similarity primitives ---------------------------------------------


class TestJaccard:
    def test_both_empty_is_one(self) -> None:
        assert jaccard(frozenset(), frozenset()) == 1.0

    def test_one_empty_is_zero(self) -> None:
        assert jaccard(frozenset({"a"}), frozenset()) == 0.0
        assert jaccard(frozenset(), frozenset({"a"})) == 0.0

    def test_identical_is_one(self) -> None:
        s = frozenset({"the", "cat", "sat"})
        assert jaccard(s, s) == 1.0

    def test_disjoint_is_zero(self) -> None:
        a = frozenset({"x", "y"})
        b = frozenset({"u", "v"})
        assert jaccard(a, b) == 0.0

    def test_partial_overlap(self) -> None:
        a = frozenset({"a", "b", "c"})
        b = frozenset({"b", "c", "d"})
        # |a∩b|=2, |a∪b|=4
        assert jaccard(a, b) == 0.5


class TestLevenshtein:
    def test_distance_identical(self) -> None:
        assert levenshtein_distance("kitten", "kitten") == 0

    def test_distance_empty(self) -> None:
        assert levenshtein_distance("", "abc") == 3
        assert levenshtein_distance("abc", "") == 3
        assert levenshtein_distance("", "") == 0

    def test_distance_classic(self) -> None:
        # textbook: kitten -> sitting is distance 3
        assert levenshtein_distance("kitten", "sitting") == 3

    def test_distance_substitution(self) -> None:
        assert levenshtein_distance("cat", "bat") == 1

    def test_distance_insertion(self) -> None:
        assert levenshtein_distance("cat", "cats") == 1

    def test_distance_deletion(self) -> None:
        assert levenshtein_distance("cats", "cat") == 1

    def test_ratio_identical(self) -> None:
        assert levenshtein_ratio("hello", "hello") == 1.0

    def test_ratio_empty_pair(self) -> None:
        assert levenshtein_ratio("", "") == 1.0
        assert levenshtein_ratio("", "abc") == 0.0
        assert levenshtein_ratio("abc", "") == 0.0

    def test_ratio_classic(self) -> None:
        # kitten/sitting: dist=3, max_len=7 → ratio = 1 - 3/7 ≈ 0.571
        ratio = levenshtein_ratio("kitten", "sitting")
        assert abs(ratio - (1 - 3 / 7)) < 1e-9

    def test_ratio_close_paraphrase_clears_default(self) -> None:
        a = "don't push to main"
        b = "do not push to main"
        # 18 vs 19 chars; dist 3 (don't → do not). ratio = 1 - 3/19 ≈ 0.842
        assert levenshtein_ratio(a, b) >= 0.84


# --- Union-find + cluster collapse -------------------------------------


class TestClusterPairs:
    def test_no_pairs_returns_empty(self) -> None:
        assert cluster_pairs([]) == ()

    def test_single_pair(self) -> None:
        pair = DuplicatePair("a", "b", 1.0, 1.0)
        clusters = cluster_pairs([pair])
        assert clusters == (
            DuplicateCluster(representative_id="a", member_ids=("a", "b")),
        )

    def test_chain_collapses(self) -> None:
        # a~b, b~c, c~d should produce one cluster {a,b,c,d}
        pairs = [
            DuplicatePair("a", "b", 1.0, 1.0),
            DuplicatePair("b", "c", 1.0, 1.0),
            DuplicatePair("c", "d", 1.0, 1.0),
        ]
        clusters = cluster_pairs(pairs)
        assert len(clusters) == 1
        assert clusters[0].representative_id == "a"
        assert clusters[0].member_ids == ("a", "b", "c", "d")

    def test_disjoint_clusters(self) -> None:
        pairs = [
            DuplicatePair("a", "b", 1.0, 1.0),
            DuplicatePair("c", "d", 1.0, 1.0),
        ]
        clusters = cluster_pairs(pairs)
        assert len(clusters) == 2
        reps = sorted(c.representative_id for c in clusters)
        assert reps == ["a", "c"]

    def test_representative_is_min_member(self) -> None:
        pair = DuplicatePair("zeta", "alpha", 1.0, 1.0)
        clusters = cluster_pairs([pair])
        # member_ids is sorted: alpha < zeta
        assert clusters[0].representative_id == "alpha"
        assert clusters[0].member_ids == ("alpha", "zeta")


# --- Top-level audit against a real store -------------------------------


def _insert(store: MemoryStore, bid: str, content: str) -> None:
    """Helper: minimal Belief insertion for audit-pass test fixtures."""
    store.insert_belief(
        Belief(
            id=bid,
            content=content,
            content_hash=f"h_{bid}",
            alpha=1.0,
            beta=1.0,
            type=BELIEF_FACTUAL,
            lock_level=LOCK_NONE,
            locked_at=None,
            demotion_pressure=0,
            created_at="2026-05-03T00:00:00Z",
            last_retrieved_at=None,
        )
    )


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(":memory:")


class TestDedupAudit:
    def test_empty_store(self, store: MemoryStore) -> None:
        report = dedup_audit(store)
        assert report.n_beliefs_scanned == 0
        assert report.n_duplicate_pairs == 0
        assert report.n_clusters == 0
        assert report.pairs == ()
        assert report.clusters == ()

    def test_single_belief(self, store: MemoryStore) -> None:
        _insert(store, "b1", "alone in the store")
        report = dedup_audit(store)
        assert report.n_beliefs_scanned == 1
        assert report.n_candidate_pairs == 0
        assert report.n_duplicate_pairs == 0

    def test_finds_near_duplicate_pair(self, store: MemoryStore) -> None:
        # Trailing punctuation only — tokens identical (Jaccard = 1.0),
        # 1-char edit distance (Levenshtein ratio ~0.97).
        _insert(store, "b1", "deploy via terraform on aws")
        _insert(store, "b2", "deploy via terraform on aws.")
        _insert(store, "b3", "the cat sat on the mat")
        report = dedup_audit(store)
        assert report.n_beliefs_scanned == 3
        assert report.n_duplicate_pairs == 1
        p = report.pairs[0]
        assert (p.belief_a_id, p.belief_b_id) == ("b1", "b2")
        assert p.jaccard_score >= DEFAULT_JACCARD_MIN
        assert p.levenshtein_score >= DEFAULT_LEVENSHTEIN_MIN
        assert report.n_clusters == 1

    def test_distinct_beliefs_no_duplicates(self, store: MemoryStore) -> None:
        _insert(store, "b1", "deploy via terraform on aws")
        _insert(store, "b2", "the database is postgres 15")
        _insert(store, "b3", "monitoring dashboards live in grafana")
        report = dedup_audit(store)
        assert report.n_duplicate_pairs == 0
        assert report.n_clusters == 0

    def test_threshold_floor_rejects_jaccard_only_match(
        self, store: MemoryStore
    ) -> None:
        # Same words, very different word order → high Jaccard but
        # the Levenshtein floor should reject this as too far apart.
        _insert(
            store, "b1",
            "main branch push policy is no direct pushes ever allowed",
        )
        _insert(
            store, "b2",
            "ever allowed pushes direct no is policy push branch main",
        )
        report = dedup_audit(store)
        # Jaccard would be ~1.0 (same word set) but Levenshtein
        # ratio on the strings is far below 0.85.
        assert report.n_duplicate_pairs == 0

    def test_audit_is_read_only(self, store: MemoryStore) -> None:
        _insert(store, "b1", "deploy via terraform on aws")
        _insert(store, "b2", "deploy via terraform on aws.")
        ids_before = sorted(store.list_belief_ids())
        _ = dedup_audit(store)
        # No new beliefs, no edges inserted. (#197 audit-only contract.)
        assert sorted(store.list_belief_ids()) == ids_before

    def test_cluster_chain(self, store: MemoryStore) -> None:
        # Three near-identical (punctuation-only diffs) → one cluster of 3.
        _insert(store, "b1", "deploy via terraform on aws")
        _insert(store, "b2", "deploy via terraform on aws.")
        _insert(store, "b3", "deploy via terraform on aws!")
        report = dedup_audit(store)
        assert report.n_clusters == 1
        cluster = report.clusters[0]
        assert set(cluster.member_ids) == {"b1", "b2", "b3"}

    def test_paraphrase_below_jaccard_floor_is_not_a_duplicate(
        self, store: MemoryStore
    ) -> None:
        # Documents the threshold semantics: "don't" / "do not"
        # tokenise into disjoint sets ({don, t} vs {do, not}) so even
        # though Levenshtein clears, Jaccard does not. The ratified
        # 0.8 Jaccard floor is strict by design; the write-path hook
        # (deferred, bench-gated) is the place to reconsider.
        _insert(store, "b1", "don't push directly to main")
        _insert(store, "b2", "do not push directly to main")
        report = dedup_audit(store)
        assert report.n_duplicate_pairs == 0

    def test_invalid_thresholds_raise(self, store: MemoryStore) -> None:
        with pytest.raises(ValueError):
            dedup_audit(store, jaccard_min=-0.1)
        with pytest.raises(ValueError):
            dedup_audit(store, jaccard_min=1.5)
        with pytest.raises(ValueError):
            dedup_audit(store, levenshtein_min=-0.1)
        with pytest.raises(ValueError):
            dedup_audit(store, max_candidate_pairs=0)

    def test_threshold_relaxation_finds_more(self, store: MemoryStore) -> None:
        # Same intent, different conjunction word — tokens diverge enough
        # at default 0.8 Jaccard to be rejected; relaxed Jaccard finds it.
        _insert(store, "b1", "do not push directly to main branch")
        _insert(store, "b2", "never push directly to main branch")
        strict = dedup_audit(store)
        relaxed = dedup_audit(store, jaccard_min=0.5, levenshtein_min=0.7)
        assert strict.n_duplicate_pairs == 0
        assert relaxed.n_duplicate_pairs == 1


class TestFormatAuditReport:
    def test_empty_report_renders(self) -> None:
        from aelfrice.dedup import DedupAuditReport
        r = DedupAuditReport(
            n_beliefs_scanned=0,
            n_candidate_pairs=0,
            n_duplicate_pairs=0,
            n_clusters=0,
            truncated=False,
        )
        out = format_audit_report(r)
        assert "aelf doctor dedup" in out
        assert "Beliefs scanned" in out
        assert "No near-duplicates" in out

    def test_clustered_report_lists_members(
        self, store: MemoryStore
    ) -> None:
        _insert(store, "b1", "deploy via terraform on aws")
        _insert(store, "b2", "deploy via terraform on aws.")
        report = dedup_audit(store)
        out = format_audit_report(report)
        assert "b1" in out
        assert "b2" in out
        assert "Clusters:" in out


# --- TOML config loader -------------------------------------------------


class TestLoadDedupConfig:
    def test_no_config_returns_defaults(self, tmp_path) -> None:
        from aelfrice.dedup import (
            DEFAULT_JACCARD_MIN,
            DEFAULT_LEVENSHTEIN_MIN,
            DEFAULT_MAX_CANDIDATE_PAIRS,
            load_dedup_config,
        )
        cfg = load_dedup_config(tmp_path)
        assert cfg.jaccard_min == DEFAULT_JACCARD_MIN
        assert cfg.levenshtein_min == DEFAULT_LEVENSHTEIN_MIN
        assert cfg.max_candidate_pairs == DEFAULT_MAX_CANDIDATE_PAIRS

    def test_well_formed_config_overrides(self, tmp_path) -> None:
        from aelfrice.dedup import load_dedup_config
        (tmp_path / ".aelfrice.toml").write_text(
            "[dedup]\n"
            "jaccard_min = 0.7\n"
            "levenshtein_min = 0.9\n"
            "max_candidate_pairs = 1000\n"
        )
        cfg = load_dedup_config(tmp_path)
        assert cfg.jaccard_min == 0.7
        assert cfg.levenshtein_min == 0.9
        assert cfg.max_candidate_pairs == 1000

    def test_out_of_range_falls_back(self, tmp_path) -> None:
        from aelfrice.dedup import (
            DEFAULT_JACCARD_MIN,
            load_dedup_config,
        )
        (tmp_path / ".aelfrice.toml").write_text(
            "[dedup]\njaccard_min = 1.5\n"
        )
        cfg = load_dedup_config(tmp_path)
        assert cfg.jaccard_min == DEFAULT_JACCARD_MIN

    def test_wrong_type_falls_back(self, tmp_path) -> None:
        from aelfrice.dedup import (
            DEFAULT_MAX_CANDIDATE_PAIRS,
            load_dedup_config,
        )
        (tmp_path / ".aelfrice.toml").write_text(
            '[dedup]\nmax_candidate_pairs = "lots"\n'
        )
        cfg = load_dedup_config(tmp_path)
        assert cfg.max_candidate_pairs == DEFAULT_MAX_CANDIDATE_PAIRS

    def test_malformed_toml_falls_back(self, tmp_path) -> None:
        from aelfrice.dedup import DedupConfig, load_dedup_config
        (tmp_path / ".aelfrice.toml").write_text(
            "[dedup\nthis is not valid toml\n"
        )
        cfg = load_dedup_config(tmp_path)
        assert cfg == DedupConfig()


# --- CLI integration ----------------------------------------------------


class TestCLIDoctorDedup:
    def test_clean_store_exit_0(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import io
        from aelfrice.cli import main

        db = str(tmp_path / "brain.db")
        monkeypatch.setenv("AELFRICE_DB", db)
        s = MemoryStore(db)
        _insert(s, "b1", "the cat sat on the mat")
        _insert(s, "b2", "deploy via terraform")
        s.close()

        out = io.StringIO()
        rc = main(["doctor", "--dedup"], out=out)
        assert rc == 0
        assert "aelf doctor dedup" in out.getvalue()
        assert "No near-duplicates" in out.getvalue()

    def test_finds_duplicate_via_cli(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import io
        from aelfrice.cli import main

        db = str(tmp_path / "brain.db")
        monkeypatch.setenv("AELFRICE_DB", db)
        s = MemoryStore(db)
        _insert(s, "b1", "deploy via terraform on aws")
        _insert(s, "b2", "deploy via terraform on aws.")
        s.close()

        out = io.StringIO()
        rc = main(["doctor", "--dedup"], out=out)
        assert rc == 0
        text = out.getvalue()
        assert "Duplicate pairs         : 1" in text
        assert "b1" in text and "b2" in text

    def test_threshold_overrides_via_flags(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import io
        from aelfrice.cli import main

        db = str(tmp_path / "brain.db")
        monkeypatch.setenv("AELFRICE_DB", db)
        s = MemoryStore(db)
        # Pair fails at default but passes when both thresholds drop.
        _insert(s, "b1", "do not push directly to main branch")
        _insert(s, "b2", "never push directly to main branch")
        s.close()

        out_strict = io.StringIO()
        rc1 = main(["doctor", "--dedup"], out=out_strict)
        assert rc1 == 0
        assert "Duplicate pairs         : 0" in out_strict.getvalue()

        out_relaxed = io.StringIO()
        rc2 = main(
            [
                "doctor", "--dedup",
                "--dedup-jaccard", "0.5",
                "--dedup-levenshtein", "0.7",
            ],
            out=out_relaxed,
        )
        assert rc2 == 0
        assert "Duplicate pairs         : 1" in out_relaxed.getvalue()
