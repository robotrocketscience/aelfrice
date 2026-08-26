"""The published-figure gate does what it claims (#1469).

Six of eight PRs in the 2026-08-10 board sweep shipped a stale or false
published figure, and CI saw none of them. `scripts/check_derived_figures.py`
is the gate; this module is the reason a green run of it means something.

Three properties are load-bearing and each has a distinguishing arm here:

  * a **store-free** figure is held to the producer's actual output, so
    changing the code without changing the prose is red;
  * a figure published in two files must agree in both, whatever class it is
    in -- the only #1449-shaped check a public runner can perform without the
    store;
  * an **unmarked** figure is allowed, and does not crash the scanner. The
    annotation backlog is the whole corpus; a gate that failed on it would be
    disabled the same day. The single exception is the overclaim sentence,
    which is hard, and which has arms on both sides of the boundary.

The live-repo arms at the bottom would pass over an empty scan, so each one
asserts the scanner actually found something before asserting it found nothing
wrong.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "check_derived_figures.py"

_spec = importlib.util.spec_from_file_location("_cdf", _SCRIPT)
assert _spec and _spec.loader
# Declared `Any` rather than left implicit: pyright runs `tests/` in strict
# mode, where an implicitly-typed module object makes every attribute read an
# `Unknown` and buries a real type error in 37 lines of noise.
cdf: Any = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cdf)

CORPUS = "corpus=repo-local-store/44687@2026-08-10"
SHA = "producer-sha=0123456789ab"


def _marker(ident: str, value: object, *attrs: str) -> str:
    tail = ("" if not attrs else " " + " ".join(attrs))
    return f"<!-- derived: {ident} = {value}{tail} -->"


@pytest.fixture()
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway tree the scanner treats as the repo root."""
    monkeypatch.setattr(cdf, "REPO_ROOT", tmp_path)
    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "benchmarks" / "p.py").write_text("# a producer\n")
    (tmp_path / "CHANGELOG").mkdir()
    return tmp_path


def _run(repo: Path, *names: str) -> Any:
    report = cdf.Report(github=False)
    cdf.check_text([repo / n for n in names], report)
    return report


# --- marker grammar ------------------------------------------------------


def test_absence_of_corpus_is_what_makes_a_figure_store_free() -> None:
    """The class is readable off the marker, which is the #1469 requirement."""
    free = cdf.parse_markers(Path("x.md"), _marker("benchmarks/p.py#k", 20))[0]
    backed = cdf.parse_markers(
        Path("x.md"), _marker("benchmarks/p.py#k", 20, CORPUS, SHA)
    )[0]
    assert free.store_backed is False
    assert backed.store_backed is True
    assert backed.corpus == "repo-local-store/44687@2026-08-10"
    assert free.errors == [] and backed.errors == []


def test_store_backed_marker_without_a_sha_is_a_grammar_error() -> None:
    """Without the stamp there is nothing staleness could be measured against."""
    m = cdf.parse_markers(Path("x.md"), _marker("benchmarks/p.py#k", 20, CORPUS))[0]
    assert any("producer-sha" in e for e in m.errors)


def test_a_corpus_without_a_date_is_a_grammar_error() -> None:
    """#1449's two counts are two real snapshots; only the date says so."""
    m = cdf.parse_markers(
        Path("x.md"), _marker("benchmarks/p.py#k", 20, "corpus=repo-local-store", SHA)
    )[0]
    assert any("YYYY-MM-DD" in e for e in m.errors)


def test_a_sha_without_a_corpus_is_a_grammar_error() -> None:
    """A store-free figure is checked by re-running; a stamp there misleads."""
    m = cdf.parse_markers(Path("x.md"), _marker("benchmarks/p.py#k", 20, SHA))[0]
    assert any("without corpus" in e for e in m.errors)


def test_an_unknown_attribute_is_reported_rather_than_ignored() -> None:
    m = cdf.parse_markers(Path("x.md"), _marker("benchmarks/p.py#k", 20, "when=friday"))[0]
    assert any("unknown attribute" in e for e in m.errors)


@pytest.mark.parametrize(
    ("published", "emitted"),
    [("11,508", 11508), ("299.7x", 299.7), ("8.69%", 8.69), ("20", 20.0), ("41_984", 41984)],
)
def test_normalise_equates_a_rendering_with_its_value(published: str, emitted: object) -> None:
    """Prose writes `11,508`; a producer emits `11508`. One figure."""
    assert cdf.normalise(published) == cdf.normalise(emitted)


def test_normalise_keeps_distinct_figures_distinct() -> None:
    """The distinguishing half -- a normaliser that collapsed everything would
    pass every test above and catch nothing."""
    assert cdf.normalise("11,508") != cdf.normalise("11,388")
    assert cdf.normalise("44,683") != cdf.normalise("44,687")


# --- self-consistency ----------------------------------------------------


def test_one_key_published_with_two_values_is_a_hard_failure(repo: Path) -> None:
    """#1449 exactly: 44,683 in one file, 44,687 in four others, one PR."""
    (repo / "a.md").write_text(_marker("benchmarks/p.py#n", 44683, CORPUS, SHA))
    (repo / "b.md").write_text(_marker("benchmarks/p.py#n", 44687, CORPUS, SHA))
    report = _run(repo, "a.md", "b.md")
    assert len(report.hard) == 2
    assert all("one figure, one value" in h for h in report.hard)


def test_one_key_published_with_one_value_twice_is_clean(repo: Path) -> None:
    (repo / "a.md").write_text(_marker("benchmarks/p.py#n", 44687, CORPUS, SHA))
    (repo / "b.md").write_text(_marker("benchmarks/p.py#n", "44,687", CORPUS, SHA))
    report = _run(repo, "a.md", "b.md")
    assert report.hard == []


def test_a_marker_naming_a_producer_that_does_not_exist_fails(repo: Path) -> None:
    (repo / "a.md").write_text(_marker("benchmarks/gone.py#n", 1))
    report = _run(repo, "a.md")
    assert any("does not exist" in h for h in report.hard)


# --- staleness is advisory, never hard -----------------------------------


def test_a_moved_producer_warns_and_does_not_fail(repo: Path) -> None:
    """A producer edit does not prove the figure moved, and re-measuring needs
    a store no public runner has (#1456). Advisory is the honest verdict."""
    (repo / "a.md").write_text(_marker("benchmarks/p.py#n", 5, CORPUS, SHA))
    report = _run(repo, "a.md")
    assert report.hard == []
    assert len(report.advisory) == 1
    assert "producer changed since this figure was measured" in report.advisory[0]


def test_a_current_stamp_does_not_warn(repo: Path) -> None:
    sha = cdf.producer_sha("benchmarks/p.py")
    (repo / "a.md").write_text(_marker("benchmarks/p.py#n", 5, CORPUS, f"producer-sha={sha}"))
    report = _run(repo, "a.md")
    assert report.advisory == [] and report.hard == []


def test_the_stamp_is_excluded_from_the_bytes_it_stamps(repo: Path) -> None:
    """Otherwise a producer documenting its own figures has no fixed point.

    Observed, not theorised: `--restamp` on
    `benchmarks/spine_fan_in_baseline.py` reported a different current hash on
    each of two consecutive runs, because writing the stamp changed the bytes
    the stamp was over.
    """
    producer = repo / "benchmarks" / "p.py"
    before = cdf.producer_sha("benchmarks/p.py")
    producer.write_text(producer.read_text() + f"# {_marker('benchmarks/p.py#n', 5, CORPUS, SHA)}\n")
    with_stamp = cdf.producer_sha("benchmarks/p.py")
    producer.write_text(
        producer.read_text().replace("producer-sha=0123456789ab", "producer-sha=ffffffffffff")
    )
    restamped = cdf.producer_sha("benchmarks/p.py")
    assert with_stamp != before, "adding a line must move the hash"
    assert restamped == with_stamp, "changing only the stamp must not"


# --- grandfathering: an unmarked figure is allowed ------------------------


def test_an_unmarked_figure_is_not_an_error_and_does_not_crash(repo: Path) -> None:
    """The whole existing corpus is unmarked. A gate that failed on it would be
    switched off, which is how #1160 happened."""
    (repo / "a.md").write_text(
        "- **An entry with figures and no markers.** It measured 44,687 beliefs,\n"
        "  a 299.7x reduction, 8.69% of fires and 2,102 recomputed-only links.\n"
    )
    report = _run(repo, "a.md")
    assert report.hard == [] and report.advisory == []


def test_unmarked_figures_are_enumerable(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Grandfathered is not invisible: the backlog has to be countable."""
    (repo / "a.md").write_text("- **Entry.** 44,687 beliefs and a 299.7x reduction.\n")
    assert cdf.list_unmarked([repo / "a.md"]) == 0
    out = capsys.readouterr().out
    assert "44,687" in out and "299.7x" in out
    assert "2 unmarked figures" in out


def test_issue_refs_versions_dates_and_code_are_not_figures() -> None:
    """A false positive here makes the overclaim check unsatisfiable, which is
    the same as deleting it."""
    block = (
        "- **Title ([#1445](https://github.com/x/y/issues/1445)).** Shipped in "
        "v1.0.3 on 2026-08-10, see § 3, with `STOP_PROMPT_MAX_ITEMS = 20` "
        "and a real figure of 11,508 bytes.\n"
    )
    assert cdf.extract_figures(block) == ["11,508"]


# --- the overclaim sentence is hard --------------------------------------


# Written with the emphasis #1445 actually shipped: a pattern blind to `**`
# would miss the instance the rule exists for, and every arm below would still
# pass. The bare-word form is exercised too.
OVERCLAIM = "`benchmarks/p.py` re-derives **every** figure here"
OVERCLAIM_PLAIN = "benchmarks/p.py re-derives all figures here"


def test_the_overclaim_fails_when_a_figure_is_unmarked(repo: Path) -> None:
    """#1445 and #1447 both shipped this sentence over a script emitting about
    a third of the entry's numbers."""
    (repo / "CHANGELOG" / "v9.md").write_text(
        f"- **Entry.** The bound is 11,508 bytes and the share is 93.7%. {OVERCLAIM}.\n"
        f"  {_marker('benchmarks/p.py#bounded_max', 11508, CORPUS, SHA)}\n"
    )
    report = _run(repo, "CHANGELOG/v9.md")
    assert len(report.hard) == 1
    assert "carry no marker" in report.hard[0]
    assert "93.7%" in report.hard[0]
    assert "11,508" not in report.hard[0], "the marked figure must not be listed"


def test_the_overclaim_passes_when_every_figure_is_marked(repo: Path) -> None:
    (repo / "CHANGELOG" / "v9.md").write_text(
        f"- **Entry.** The bound is 11,508 bytes and the share is 93.7%. {OVERCLAIM}.\n"
        f"  {_marker('benchmarks/p.py#bounded_max', 11508, CORPUS, SHA)}\n"
        f"  {_marker('benchmarks/p.py#share', 93.7, CORPUS, SHA)}\n"
    )
    report = _run(repo, "CHANGELOG/v9.md")
    assert report.hard == []


def test_the_overclaim_is_scoped_to_its_own_entry(repo: Path) -> None:
    """A neighbouring bullet's unmarked figures are not this bullet's problem;
    scoping it to the file would make the claim unmakeable anywhere."""
    (repo / "CHANGELOG" / "v9.md").write_text(
        f"- **Clean entry.** The bound is 11,508 bytes. {OVERCLAIM}.\n"
        f"  {_marker('benchmarks/p.py#bounded_max', 11508, CORPUS, SHA)}\n"
        "- **Neighbour.** 44,687 beliefs, unmarked and unclaimed.\n"
    )
    report = _run(repo, "CHANGELOG/v9.md")
    assert report.hard == []


def test_the_bare_word_form_of_the_claim_is_caught_too(repo: Path) -> None:
    """The emphasis is optional in the source sentence, not in the pattern."""
    (repo / "CHANGELOG" / "v9.md").write_text(
        f"- **Entry.** The share is 93.7%. {OVERCLAIM_PLAIN}.\n"
    )
    report = _run(repo, "CHANGELOG/v9.md")
    assert len(report.hard) == 1 and "93.7%" in report.hard[0]


def test_a_claim_quoted_as_inline_code_is_a_citation_not_a_claim(repo: Path) -> None:
    """This file, the checker's docstring and the CHANGELOG entry introducing
    the rule all quote the sentence. A gate that fired on every mention of
    itself would need an exemption list, which is worse than the escape."""
    (repo / "CHANGELOG" / "v9.md").write_text(
        f"- **Entry.** The share is 93.7%. The banned shape is ``{OVERCLAIM_PLAIN}``.\n"
    )
    report = _run(repo, "CHANGELOG/v9.md")
    assert report.hard == []


def test_an_unmarked_entry_without_the_sentence_is_clean(repo: Path) -> None:
    """The distinguishing arm for the two above: without the claim, the same
    unmarked figures pass. The sentence is what is being priced."""
    (repo / "CHANGELOG" / "v9.md").write_text(
        "- **Entry.** The bound is 11,508 bytes and the share is 93.7%.\n"
    )
    report = _run(repo, "CHANGELOG/v9.md")
    assert report.hard == []


# --- the producer protocol -----------------------------------------------


@pytest.mark.timeout(60)
def test_a_producer_emitting_a_different_value_is_a_hard_failure(
    repo: Path,
) -> None:
    (repo / "benchmarks" / "emitter.py").write_text(
        "import json, sys\nprint(json.dumps({'k': 21}))\n"
    )
    (repo / "a.md").write_text(_marker("benchmarks/emitter.py#k", 20))
    report = cdf.Report(github=False)
    markers = cdf.check_text([repo / "a.md"], report)
    cdf.check_producers(markers, report)
    assert len(report.hard) == 1
    assert "now emits 21" in report.hard[0]


def test_a_store_backed_marker_is_never_executed(repo: Path) -> None:
    """The crux of #1469: these producers need a belief store, public CI has
    none, and the lab corpus must not go there (#1456). Running one would fail
    for the wrong reason and the gate would be turned off."""
    (repo / "benchmarks" / "emitter.py").write_text("raise SystemExit('needs a store')\n")
    (repo / "a.md").write_text(_marker("benchmarks/emitter.py#k", 20, CORPUS, SHA))
    report = cdf.Report(github=False)
    markers = cdf.check_text([repo / "a.md"], report)
    cdf.check_producers(markers, report)
    assert report.hard == []


@pytest.mark.timeout(60)
def test_a_producer_missing_the_key_is_a_hard_failure(repo: Path) -> None:
    (repo / "benchmarks" / "emitter.py").write_text(
        "import json\nprint(json.dumps({'other': 1}))\n"
    )
    (repo / "a.md").write_text(_marker("benchmarks/emitter.py#k", 20))
    report = cdf.Report(github=False)
    markers = cdf.check_text([repo / "a.md"], report)
    cdf.check_producers(markers, report)
    assert any("emits no key" in h for h in report.hard)


# --- the live repo -------------------------------------------------------


@pytest.mark.timeout(60)
def test_published_constants_speaks_the_emit_figures_protocol() -> None:
    """Spawns a child (#1307): the 5s unit default would report a loaded
    machine as a hang."""
    proc = subprocess.run(
        [sys.executable, str(_REPO / "benchmarks" / "published_constants.py"), "--emit-figures"],
        capture_output=True, text=True, cwd=str(_REPO), timeout=120, check=True,
    )
    payload: object = json.loads(proc.stdout)
    assert isinstance(payload, dict), "the protocol is a JSON object, not a scalar"
    keys = [k for k in cast("dict[object, object]", payload)]
    assert keys, "an empty object would pass every marker check vacuously"
    assert all(isinstance(k, str) for k in keys)


def test_the_repo_passes_the_text_checks() -> None:
    files = cdf.iter_files(list(cdf.DEFAULT_ROOTS))
    report = cdf.Report(github=False)
    markers = cdf.check_text(files, report)
    assert len(markers) >= 10, "an empty scan would pass this test vacuously"
    assert {m.store_backed for m in markers} == {True, False}, (
        "both classes must be exercised, or the live scan proves only one half"
    )
    assert report.hard == [], "\n".join(report.hard)


@pytest.mark.timeout(120)
def test_the_repo_passes_the_producer_checks() -> None:
    """Spawns one child per store-free producer (#1307)."""
    files = cdf.iter_files(list(cdf.DEFAULT_ROOTS))
    report = cdf.Report(github=False)
    markers = cdf.check_text(files, report)
    store_free = [m for m in markers if not m.store_backed]
    assert store_free, "an empty store-free set would pass this test vacuously"
    producer_report = cdf.Report(github=False)
    cdf.check_producers(markers, producer_report)
    assert producer_report.hard == [], "\n".join(producer_report.hard)
