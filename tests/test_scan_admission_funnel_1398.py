"""The admission-funnel harness measures what it claims to (#1398).

Three properties are worth gating, and each fails on a different mutation:

* the per-rule noise split reproduces `noise_filter.is_noise` exactly — so the
  columns sum to `skipped_noise` and the split means something;
* the drift guard actually fires, rather than being a branch nothing reaches;
* the run is deterministic and writes to a throwaway store, never the ambient
  development one.

The tree under test is a handful of synthetic files rather than the repo: the
harness's own default scans ~10k candidates, which is a benchmark, not a unit
test.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aelfrice.noise_filter import NoiseConfig, is_noise
from benchmarks import scan_admission_funnel as funnel


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    """A small tree that exercises several noise buckets and still admits."""
    (tmp_path / "README.md").write_text(
        "# Heading One\n\n"
        "The scanner walks the tree and emits one candidate per paragraph, "
        "which the classifier then types before anything is written.\n\n"
        "- [ ] a checklist item\n- [ ] another checklist item\n\n"
        "Three word fragment\n\n"
        "## Heading Two\n\n"
        "Beliefs are deduplicated by a content hash so a rescan of the same "
        "tree inserts nothing new the second time around.\n",
        encoding="utf-8",
    )
    (tmp_path / "mod.py").write_text(
        '"""Module docstring that is long enough to survive the fragment '
        'filter and reach the classifier intact."""\n\n'
        "def f():\n"
        '    """A top-level function docstring, also comfortably long enough '
        'to be treated as a real candidate."""\n'
        "    return 1\n",
        encoding="utf-8",
    )
    return tmp_path


def test_noise_bucket_agrees_with_is_noise_and_names_the_arm() -> None:
    """The replica must return the SAME verdict and the right bucket.

    Asserting only the verdict would pass on a replica that lumps everything
    into one arm, which is precisely the failure that makes the per-rule split
    worthless.
    """
    cfg = NoiseConfig.discover(Path("."))
    cases = [
        ("", "empty"),
        ("   \n  ", "empty"),
        ("# Just A Heading", "headings"),
        ("- [ ] one item\n- [ ] two item", "checklists"),
        ("only three words", "fragments"),
    ]
    for text, expected in cases:
        assert funnel._noise_bucket(text, cfg) == expected, text
        assert is_noise(text, cfg) is True, text

    survivor = (
        "The derivation worker stamps every unstamped ingest row once per "
        "scan rather than once per candidate."
    )
    assert funnel._noise_bucket(survivor, cfg) is None
    assert is_noise(survivor, cfg) is False


def test_every_declared_bucket_is_reachable_in_order() -> None:
    """`NOISE_BUCKETS` must list exactly the arms `_noise_bucket` can return.

    A bucket in the tuple that the replica never returns prints a permanent
    zero column and reads as "this rule never fires"; an arm missing from the
    tuple is dropped from the printed total while still counting toward
    `skipped_noise`, so the sum check fails with no indication of which rule
    is unaccounted for.
    """
    import inspect

    src = inspect.getsource(funnel._noise_bucket)
    returned = {
        line.split("return")[1].strip().strip('"')
        for line in src.splitlines()
        if "return " in line and '"' in line
    }
    assert returned == set(funnel.NOISE_BUCKETS)


def test_replica_drift_is_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard fires when the product filter and the replica disagree.

    Without this the guard is an unreachable branch: `_noise_bucket` is a copy
    of `is_noise`, so they agree by construction until someone edits one of
    them, and no existing test would notice the guard had stopped working.
    """
    cfg = NoiseConfig.discover(Path("."))
    cand = funnel.SentenceCandidate(
        text="A perfectly ordinary sentence that survives every noise rule.",
        source="doc:test.md:p0",
    )
    assert funnel._noise_bucket(cand.text, cfg) is None

    monkeypatch.setattr(funnel, "is_noise", lambda _text, _cfg=None: True)
    with pytest.raises(SystemExit, match="drifted from noise_filter.is_noise"):
        funnel._assert_replica_agrees([cand], cfg)


@pytest.mark.timeout(30)
def test_noise_columns_sum_to_skipped_noise(corpus: Path) -> None:
    """The per-rule split accounts for every candidate `scan_repo` dropped.

    This is the assertion that catches a first-match-wins violation: testing
    the four categories independently double-counts (a heading block is also
    usually a short fragment), so the columns would exceed `skipped_noise`.
    """
    m = funnel.measure(corpus)
    assert sum(m["noise_by_bucket"].values()) == m["funnel"]["skipped_noise"]
    assert m["funnel"]["total_candidates"] > 0
    assert m["funnel"]["skipped_noise"] > 0, "corpus must exercise the filter"
    assert m["funnel"]["inserted"] > 0, "corpus must admit something"


@pytest.mark.timeout(30)
def test_funnel_arithmetic_separates_survival_from_admission(corpus: Path) -> None:
    """Both rates are reported, and the gap between them is accounted for.

    The 8,428-vs-8,388 discrepancy in #1159 is exactly this gap left implicit.
    `survived - inserted` must equal the reported convergence gap, so a run can
    never present one rate as though it were the other.
    """
    m = funnel.measure(corpus)
    f = m["funnel"]
    survived = (
        f["total_candidates"] - f["skipped_noise"] - f["skipped_non_persisting"]
    )
    assert m["rates"]["convergence_gap"] == survived - f["inserted"]
    assert m["rates"]["survival"] >= m["rates"]["admission"]
    assert f["beliefs_in_store"] == f["inserted"]


@pytest.mark.timeout(60)
def test_measure_is_deterministic(corpus: Path) -> None:
    """Two runs over one tree agree on every count.

    `scan_repo` defaults `now` to wall-clock time; the harness pins it. Reverting
    that pin leaves this green only if the clock happens not to cross a second
    boundary, so the assertion covers the funnel counts, which is where a
    non-deterministic id derivation would actually show up.
    """
    first = funnel.measure(corpus)
    second = funnel.measure(corpus)
    assert first["funnel"] == second["funnel"]
    assert first["noise_by_bucket"] == second["noise_by_bucket"]
    assert first["types"] == second["types"]


@pytest.mark.timeout(30)
def test_scan_writes_to_a_throwaway_store_not_the_corpus(corpus: Path) -> None:
    """AC7: the ambient store is unreachable, structurally rather than by flag.

    Asserts the store path the harness opens is a fresh temp dir and that it
    leaves nothing behind under the scanned tree — a harness that scanned into
    `<root>/.git/aelfrice/memory.db` would report `skipped_existing` on the
    second run and silently understate admission.
    """
    opened: list[str] = []
    real_store = funnel.MemoryStore

    def spy(path: str, **kw: object) -> object:
        opened.append(path)
        return real_store(path, **kw)  # type: ignore[arg-type]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(funnel, "MemoryStore", spy)
        funnel.measure(corpus)

    assert len(opened) == 1
    store_path = Path(opened[0])
    assert corpus not in store_path.parents, store_path
    assert "aelf-scan-admission-" in str(store_path)
    # The temp dir is removed after the run, so nothing accumulates per run.
    assert not store_path.exists()
    assert not (corpus / ".git" / "aelfrice").exists()
