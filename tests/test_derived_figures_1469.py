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
import sqlite3
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


def _published(ident: str, value: object, *attrs: str) -> str:
    """A marker together with the sentence it annotates.

    A bare marker is a hard failure now: a value that appears in no surrounding
    text is guarding nothing. A fixture exercising some *other* rule therefore
    has to publish the figure as well, which is what a real file does anyway.
    """
    return f"The measured figure is {value}.\n{_marker(ident, value, *attrs)}\n"


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
    (repo / "a.md").write_text(_published("benchmarks/p.py#n", 44683, CORPUS, SHA))
    (repo / "b.md").write_text(_published("benchmarks/p.py#n", 44687, CORPUS, SHA))
    report = _run(repo, "a.md", "b.md")
    assert len(report.hard) == 2
    assert all("one figure, one value" in h for h in report.hard)


def test_one_key_published_with_one_value_twice_is_clean(repo: Path) -> None:
    (repo / "a.md").write_text(_published("benchmarks/p.py#n", 44687, CORPUS, SHA))
    (repo / "b.md").write_text(_published("benchmarks/p.py#n", "44,687", CORPUS, SHA))
    report = _run(repo, "a.md", "b.md")
    assert report.hard == []


def test_a_marker_naming_a_producer_that_does_not_exist_fails(repo: Path) -> None:
    (repo / "a.md").write_text(_published("benchmarks/gone.py#n", 1))
    report = _run(repo, "a.md")
    assert any("does not exist" in h for h in report.hard)


@pytest.mark.parametrize("producer", ["/bin/sh", "../../../../bin/sh"])
def test_a_producer_outside_the_repository_is_rejected(
    repo: Path, producer: str
) -> None:
    """`REPO_ROOT / producer` is not containment: an absolute path discards
    REPO_ROOT and `..` walks out. `--mode producers` execs the result under
    `sys.executable`, so the marker is committed input to a subprocess argv."""
    (repo / "a.md").write_text(_published(f"{producer}#n", 1))
    report = _run(repo, "a.md")
    assert any("not inside the repository" in h for h in report.hard)
    assert cdf.producer_path(producer) is None


# --- staleness is advisory, never hard -----------------------------------


def test_a_moved_producer_warns_and_does_not_fail(repo: Path) -> None:
    """A producer edit does not prove the figure moved, and re-measuring needs
    a store no public runner has (#1456). Advisory is the honest verdict."""
    (repo / "a.md").write_text(_published("benchmarks/p.py#n", 5, CORPUS, SHA))
    report = _run(repo, "a.md")
    assert report.hard == []
    assert len(report.advisory) == 1
    assert "producer changed since this figure was measured" in report.advisory[0]


def test_a_current_stamp_does_not_warn(repo: Path) -> None:
    sha = cdf.producer_sha("benchmarks/p.py")
    (repo / "a.md").write_text(_published("benchmarks/p.py#n", 5, CORPUS, f"producer-sha={sha}"))
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


def test_a_figure_inside_a_double_backtick_span_is_code_not_a_figure() -> None:
    """Both paths share one inline-code regex. They did not at first: figure
    extraction used the single-backtick form and claim detection the
    double-aware one, so a number quoted inside a citation counted as a
    published figure in one path and not the other."""
    assert cdf.extract_figures("- **Entry.** See ``foo 1234 bar`` and 11,508.\n") == [
        "11,508"
    ]


def test_a_code_span_that_is_wholly_a_number_is_still_a_figure() -> None:
    """The house style for a measured value is code font, so masking every
    span put published figures outside every check the gate makes."""
    assert cdf.extract_figures("- **E.** the share is `93.69%`.") == ["93.69%"]
    assert cdf.extract_figures("- **E.** the share is 93.69%.") == ["93.69%"]


def test_a_code_span_carrying_anything_else_stays_masked() -> None:
    """The distinguishing arm. A false positive here makes the overclaim rule
    unsatisfiable, which is the same as deleting it."""
    for span in ("`STOP_PROMPT_MAX_ITEMS = 20`", "`[:16]`", "`93.69% of rows`"):
        assert cdf.extract_figures(f"- **E.** {span} and 11,508.") == ["11,508"], span


def test_backticks_do_not_buy_out_of_the_overclaim_rule(repo: Path) -> None:
    """The hard rule was satisfiable by typing two backticks.

    Byte-for-byte the same entry, the figures once bare and once in code font:
    before the narrowing the first reported one hard failure and the second
    reported none.
    """
    bare = (
        f"- **Entry.** The bound is 11,508 bytes and the share is 93.7%. "
        f"{OVERCLAIM}.\n"
    )
    fenced = (
        f"- **Entry.** The bound is `11,508` bytes and the share is `93.7%`. "
        f"{OVERCLAIM}.\n"
    )
    (repo / "CHANGELOG" / "bare.md").write_text(bare)
    (repo / "CHANGELOG" / "fenced.md").write_text(fenced)
    bare_report = _run(repo, "CHANGELOG/bare.md")
    fenced_report = _run(repo, "CHANGELOG/fenced.md")
    assert len(bare_report.hard) == 1, "\n".join(bare_report.hard)
    assert len(fenced_report.hard) == 1, (
        "backticking the figures cleared the hard rule: " + "\n".join(fenced_report.hard)
    )
    assert "93.7%" in fenced_report.hard[0] and "11,508" in fenced_report.hard[0]


def test_mask_delta_ranks_the_three_candidate_rules(repo: Path) -> None:
    """`--mask-delta` prices the narrowing instead of asserting its cost.

    The ordering is the property: the shipped rule sees more than masking every
    span and much less than masking none. The numbers themselves move with the
    corpus, which is why nothing publishes them.
    """
    (repo / "CHANGELOG" / "v9.md").write_text(
        "- **Entry.** The share is `93.7%`, the cap is `STOP_PROMPT_MAX = 20`, "
        "and `retrieve(k=7)` is a call.\n"
    )
    files = [repo / "CHANGELOG" / "v9.md"]
    strict = cdf.unmarked_total(files, cdf._mask_every_code_span)
    shipped = cdf.unmarked_total(files, cdf._mask_code_spans)
    loose = cdf.unmarked_total(files, cdf._mask_no_code_span)
    assert strict == 0, "every figure in this fixture is inside a code span"
    assert shipped == 1, "only the bare-number span is a figure"
    assert loose == 3, "unmasked, the identifier and the kwarg count too"
    assert strict < shipped < loose


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


# --- entry scoping: the rule has to be able to see the line --------------
#
# The first splitter put a whole file into changelog-bullet mode on a single
# `- ` line anywhere in it, and in that mode discarded everything above the
# first bullet. Every scanned Python file carrying a top-level `- ` line went
# that way, and the overclaim rule -- the one hard non-producer check --
# silently did not run on any of them. The count is not published: its
# denominator is the scanned corpus and it moves on every merge.
# A check that reports green because it never looked is the #1160 defect
# this repo has already paid for once.


def _docstring_module(list_line: str) -> str:
    """A module whose docstring overclaims, plus one line of the caller's."""
    return (
        '"""A module.\n'
        "\n"
        f"The bound is 11,508 bytes and the share is 93.7%. {OVERCLAIM_PLAIN}.\n"
        "\n"
        f"{list_line}\n"
        '"""\n'
    )


def test_a_dash_line_in_a_python_docstring_does_not_switch_the_rule_off(
    repo: Path,
) -> None:
    """The reviewer's live replay, in miniature.

    Inserting the same #1445 sentence into two benchmark docstrings gave exit 1
    on one and exit 0 on the other. The only difference between the files was
    that the silent one contained a `- ` line somewhere in it -- and it was one
    of the three producers this feature's own CHANGELOG entry names.
    """
    (repo / "benchmarks" / "x.py").write_text(_docstring_module("- a list item"))
    report = _run(repo, "benchmarks/x.py")
    assert len(report.hard) == 1, "the claim is in the file; the rule must see it"
    assert "carry no marker" in report.hard[0]
    assert "93.7%" in report.hard[0] and "11,508" in report.hard[0]


def test_the_same_module_without_the_dash_line_fails_identically(
    repo: Path,
) -> None:
    """The distinguishing arm: identical text, identical unmarked figures, and
    the verdict must not depend on an unrelated list item three lines down."""
    (repo / "benchmarks" / "x.py").write_text(_docstring_module("* a list item"))
    with_dash = _run(repo, "benchmarks/x.py")
    (repo / "benchmarks" / "x.py").write_text(_docstring_module("- a list item"))
    without = _run(repo, "benchmarks/x.py")
    assert with_dash.hard == without.hard


def test_a_markdown_claim_above_the_first_bullet_is_checked(repo: Path) -> None:
    """Case 1 of the same bug: in a `.md` file the old splitter dropped every
    paragraph above the first bullet, which is most of `docs/` and all of
    `README.md`."""
    (repo / "CHANGELOG" / "v9.md").write_text(
        "## Heading\n"
        "\n"
        f"The share is 93.7%. {OVERCLAIM_PLAIN}.\n"
        "\n"
        "- **A bullet.** Nothing to see.\n"
    )
    report = _run(repo, "CHANGELOG/v9.md")
    assert len(report.hard) == 1 and "93.7%" in report.hard[0]


def test_markdown_bullets_are_still_scoped_one_entry_at_a_time(repo: Path) -> None:
    """Paragraph coverage must not cost the bullet scoping: a CHANGELOG entry
    is still the unit, or a neighbour's unmarked figures sink a clean claim."""
    (repo / "CHANGELOG" / "v9.md").write_text(
        f"- **Clean.** The bound is 11,508 bytes. {OVERCLAIM}.\n"
        f"  {_marker('benchmarks/p.py#bounded_max', 11508, CORPUS, SHA)}\n"
        "\n"
        "- **Neighbour.** 44,687 beliefs, unmarked and unclaimed.\n"
    )
    report = _run(repo, "CHANGELOG/v9.md")
    assert report.hard == []


def test_a_dash_line_in_a_python_file_is_not_a_changelog_entry(repo: Path) -> None:
    """Bullet scoping is markdown-only. In a `.py` file the unit is always the
    paragraph, so the list item is scoped with the text around it."""
    body = "- an item\n\n  and 44,687 more.\n"
    src = repo / "benchmarks" / "x.py"
    src.write_text(f'"""D.\n\n{body}"""\n')
    md = repo / "CHANGELOG" / "v9.md"
    md.write_text(body)
    py = dict(cdf.split_entries(src, src.read_text()))
    entries_md = dict(cdf.split_entries(md, md.read_text()))
    assert sorted(py) == [1, 3, 5], "a `.py` list item does not open a changelog entry"
    assert "44,687" not in py[3], "the item is scoped as a paragraph, not a bullet"
    assert sorted(entries_md) == [1], "markdown bullets still absorb their continuation"
    assert "44,687" in entries_md[1]


@pytest.mark.timeout(120)
def test_every_non_blank_line_of_the_corpus_lands_in_exactly_one_entry() -> None:
    """The invariant the splitter's docstring claims, asserted rather than
    claimed. `--list-unmarked` and the overclaim rule both iterate entries, so
    a line in no entry is a line neither of them can ever guard."""
    files = cdf.iter_files(list(cdf.DEFAULT_ROOTS))
    assert len(files) > 100, "an empty scan would pass this test vacuously"
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        seen: dict[int, int] = {}
        for start, block in cdf.split_entries(path, text):
            for off in range(block.count("\n") + 1):
                seen[start + off] = seen.get(start + off, 0) + 1
        for idx, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            assert seen.get(idx, 0) == 1, f"{path}:{idx} is in {seen.get(idx, 0)} entries"


# --- binding: the marker's value is the figure the prose publishes -------
#
# The link the rest of the gate does not make. Producer checks bind the
# producer to the marker; self-consistency binds markers to each other. The
# number a reader sees is the prose, and before this rule nothing compared the
# two -- so #1445's failure mode was reproducible straight through #1469's own
# gate.


def test_editing_a_published_figure_and_leaving_its_marker_is_a_hard_failure(
    repo: Path,
) -> None:
    """Reproduced on the live tree: changing `**3,448,428 bytes**` in
    `CHANGELOG/v4.md` to `**9,999,999 bytes**` while leaving the marker at
    3448428 printed the marker count and exited 0."""
    (repo / "CHANGELOG" / "v9.md").write_text(
        "- **Entry.** Unbounded, the worst session rendered **9,999,999 bytes**.\n"
        f"  {_marker('benchmarks/p.py#unbounded_max', 3448428, CORPUS, SHA)}\n"
    )
    report = _run(repo, "CHANGELOG/v9.md")
    assert len(report.hard) == 1
    assert "no figure reading 3448428 appears" in report.hard[0]


def test_a_marker_sitting_against_its_figure_is_clean(repo: Path) -> None:
    """The distinguishing arm: the same entry with the published figure intact
    passes, so the rule is reading the number and not the shape."""
    (repo / "CHANGELOG" / "v9.md").write_text(
        "- **Entry.** Unbounded, the worst session rendered **3,448,428 bytes**.\n"
        f"  {_marker('benchmarks/p.py#unbounded_max', 3448428, CORPUS, SHA)}\n"
    )
    report = _run(repo, "CHANGELOG/v9.md")
    assert report.hard == []


def test_bumping_every_marker_and_leaving_the_prose_is_a_hard_failure(
    repo: Path,
) -> None:
    """The repair path a producer move opens, and the reason this rule is hard.

    When the constant legitimately changes, an author can satisfy the producer
    check by editing the marker values alone -- leaving every surrounding
    sentence reading the old number. Green CI, false published figure, which is
    #1445 again through the new gate.
    """
    (repo / "CHANGELOG" / "v9.md").write_text(
        "- **Entry.** A cap of 20 leaves the median session whole.\n"
        f"  {_marker('benchmarks/p.py#cap', 21)}\n"
    )
    report = _run(repo, "CHANGELOG/v9.md")
    assert len(report.hard) == 1 and "no figure reading 21 appears" in report.hard[0]


def test_a_comment_marker_is_held_to_its_comment_not_the_statement_below(
    repo: Path,
) -> None:
    """Reproduced on the live tree: editing `# A cap of 20 leaves the median
    session whole` to read 25, with the constant and all four markers untouched,
    left `--mode all` at exit 0 and the suite at 33 passed.

    The assignment below is what the producer already re-runs. The sentence is
    the figure a reader sees, so the comment run is the scope.
    """
    (repo / "benchmarks" / "x.py").write_text(
        "# A cap of 25 leaves the median session whole.\n"
        f"# {_marker('benchmarks/p.py#cap', 20)}\n"
        "CAP = 20\n"
    )
    report = _run(repo, "benchmarks/x.py")
    assert len(report.hard) == 1 and "no figure reading 20 appears" in report.hard[0]


def test_a_comment_marker_agreeing_with_its_comment_is_clean(repo: Path) -> None:
    """The distinguishing arm for the narrowing."""
    (repo / "benchmarks" / "x.py").write_text(
        "# A cap of 20 leaves the median session whole.\n"
        f"# {_marker('benchmarks/p.py#cap', 20)}\n"
        "CAP = 20\n"
    )
    report = _run(repo, "benchmarks/x.py")
    assert report.hard == []


def test_a_marker_quoted_as_inline_code_publishes_nothing(repo: Path) -> None:
    """Same convention as the overclaim sentence: inline code is a citation.
    Without it, the CHANGELOG entry that documents the marker syntax parses as
    carrying a marker, and documenting the format publishes a figure."""
    (repo / "CHANGELOG" / "v9.md").write_text(
        "- **Entry.** A marker reads "
        f"`{_marker('benchmarks/p.py#cap', 20)}` and names its producer.\n"
    )
    report = _run(repo, "CHANGELOG/v9.md")
    assert report.hard == []
    assert cdf.parse_markers(repo / "CHANGELOG" / "v9.md", (repo / "CHANGELOG" / "v9.md").read_text()) == []


def test_a_multiline_code_span_does_not_shift_the_reported_line(
    repo: Path,
) -> None:
    """Blanking a citation must keep its newlines.

    A `` ` `` span straddles lines all over `src/`, and blanking one to spaces
    ate its newlines: every marker after it reported two lines early, which
    sends the reader -- and the `::error` annotation -- to the wrong place.
    """
    (repo / "CHANGELOG" / "v9.md").write_text(
        "- **Entry.** A span `over\ntwo lines` and more text.\n"
        "\n"
        "- **Second.** The cap is 20.\n"
        f"  {_marker('benchmarks/p.py#cap', 20)}\n"
    )
    text = (repo / "CHANGELOG" / "v9.md").read_text()
    markers = cdf.parse_markers(repo / "CHANGELOG" / "v9.md", text)
    assert [m.line for m in markers] == [5]
    assert text.splitlines()[4].strip().startswith("<!-- derived:")


def test_an_unmarked_figure_beside_a_marked_one_stays_grandfathered(
    repo: Path,
) -> None:
    """The direction is marker -> prose, never prose -> marker. Requiring the
    reverse would fail on the whole existing corpus, and the gate would be off
    within a day -- the reason grandfathering exists at all."""
    (repo / "CHANGELOG" / "v9.md").write_text(
        "- **Entry.** The bound is 11,508 bytes and the share is 93.7%.\n"
        f"  {_marker('benchmarks/p.py#bounded_max', 11508, CORPUS, SHA)}\n"
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
    (repo / "a.md").write_text(_published("benchmarks/emitter.py#k", 20))
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
    (repo / "a.md").write_text(_published("benchmarks/emitter.py#k", 20, CORPUS, SHA))
    report = cdf.Report(github=False)
    markers = cdf.check_text([repo / "a.md"], report)
    cdf.check_producers(markers, report)
    assert report.hard == []


@pytest.mark.timeout(60)
def test_a_producer_missing_the_key_is_a_hard_failure(repo: Path) -> None:
    (repo / "benchmarks" / "emitter.py").write_text(
        "import json\nprint(json.dumps({'other': 1}))\n"
    )
    (repo / "a.md").write_text(_published("benchmarks/emitter.py#k", 20))
    report = cdf.Report(github=False)
    markers = cdf.check_text([repo / "a.md"], report)
    cdf.check_producers(markers, report)
    assert any("emits no key" in h for h in report.hard)


# --- #1445's reduction factor now has a producer -------------------------


_BOUNDS = _REPO / "benchmarks" / "stop_prompt_block_bounds.py"
_bounds_spec = importlib.util.spec_from_file_location("_spbb", _BOUNDS)
assert _bounds_spec and _bounds_spec.loader
bounds: Any = importlib.util.module_from_spec(_bounds_spec)
_bounds_spec.loader.exec_module(bounds)


def test_the_reduction_factor_is_the_ratio_the_entry_publishes() -> None:
    """`CHANGELOG/v4.md` published "a 299.7x reduction" with no producer, while
    both numbers it divides carried markers. A ratio of two measured values is
    arithmetic, not a third measurement, so it belongs in the producer."""
    assert bounds.reduction_factor(11508, 3448428) == 299.7


def test_the_reduction_factor_refuses_an_empty_or_zero_arm() -> None:
    """An arm with no sessions must report no figure rather than divide by
    zero or, worse, publish a 0 that reads as a measurement."""
    assert bounds.reduction_factor(0, 3448428) is None
    assert bounds.reduction_factor(None, 3448428) is None
    assert bounds.reduction_factor(11508, None) is None


def test_measure_emits_the_factor_beside_the_maxima_it_divides(tmp_path: Path) -> None:
    """The key has to survive in the report, not only in the helper.

    A synthetic store rather than the real one: `measure()` reads a store
    through a read-only URI, and the figure this guards is the *presence and
    consistency* of the key, which one fabricated session shows as well as
    44,687 real beliefs do.
    """
    db = tmp_path / "memory.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE beliefs (id TEXT PRIMARY KEY, content TEXT, type TEXT, "
        "origin TEXT, session_id TEXT, valid_to TEXT, lock_level TEXT)"
    )
    con.executemany(
        "INSERT INTO beliefs VALUES (?,?,?,?,?,?,?)",
        [
            (f"b{i}", "correct the thing: " + "x" * (40 * i + 5), "correction",
             "user_transcript", "sess-1", None, None)
            for i in range(1, 60)
        ],
    )
    con.commit()
    con.close()

    arm = cast("dict[str, Any]", bounds.measure([str(db)])["post_1315"])
    assert arm["sessions"] == 1, "the fixture must produce a non-empty arm"
    bounded = arm["rendered_bytes_bounded"]["max"]
    unbounded = arm["rendered_bytes_unbounded"]["max"]
    assert unbounded > bounded, "an unbounded render must exceed the capped one"
    assert arm["worst_case_reduction_factor"] == round(unbounded / bounded, 1)


def test_the_published_factor_agrees_with_the_two_maxima_it_divides() -> None:
    """The three markers on the #1442 entry have to be one statement.

    Read straight out of the shipped file: the factor marker must be the
    rounded ratio of the two maxima markers beside it, so editing one of the
    three and leaving the others is caught here as well as by the gate.
    """
    files = cdf.iter_files(["CHANGELOG"])
    report = cdf.Report(github=False)
    published = {
        m.key: m.value
        for m in cdf.check_text(files, report)
        if m.producer == "benchmarks/stop_prompt_block_bounds.py"
    }
    wanted = {
        "post_1315.rendered_bytes_bounded.max",
        "post_1315.rendered_bytes_unbounded.max",
        "post_1315.worst_case_reduction_factor",
    }
    assert wanted <= set(published), (
        f"missing markers: {sorted(wanted - set(published))}"
    )
    bounded = int(published["post_1315.rendered_bytes_bounded.max"])
    unbounded = int(published["post_1315.rendered_bytes_unbounded.max"])
    assert bounds.reduction_factor(bounded, unbounded) == float(
        published["post_1315.worst_case_reduction_factor"]
    )


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


@pytest.mark.timeout(120)
def test_every_store_free_key_is_named_by_a_marker() -> None:
    """A producer key nobody cites guards nothing.

    `published_constants.py` says adding a key is only half a change: the
    figure it re-derives has to gain a marker naming it. Nothing enforced
    that, so a key could be added, pass every producer check by being
    compared against nothing, and leave the prose it was written for
    unguarded -- the #1160 shape, where a check reports green over a figure
    it never looked at.
    """
    proc = subprocess.run(
        [sys.executable, str(_REPO / "benchmarks" / "published_constants.py"), "--emit-figures"],
        capture_output=True, text=True, cwd=str(_REPO), timeout=120, check=True,
    )
    emitted = {str(k) for k in cast("dict[object, object]", json.loads(proc.stdout))}
    assert emitted, "an empty producer would pass this test vacuously"

    files = cdf.iter_files(list(cdf.DEFAULT_ROOTS))
    report = cdf.Report(github=False)
    cited = {
        m.key
        for m in cdf.check_text(files, report)
        if m.producer == "benchmarks/published_constants.py"
    }
    assert cited, "no marker names the store-free producer at all"
    assert emitted <= cited, (
        "these published_constants keys are named by no marker, so nothing "
        f"in the repo is held to them: {sorted(emitted - cited)}"
    )


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
