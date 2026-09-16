"""#1526 — the injection packers charge what the renderer emits.

The budgets on the UserPromptSubmit path used to be denominated in
belief-content characters while the block that reached the model was XML,
so each block overran its cap by the share of the rendered line its
`<belief>` element takes up.

Two kinds of test live here, and they are not interchangeable.

**Pins against the live renderer.** `aelfrice.render_cost` holds wrapper
widths that the renderers do not read — a packer cannot import the
formatter it feeds without a cycle — so each constant is asserted against
a line the shipped renderer actually produced. These are what stop the
copy drifting from the original.

**Pins on the property.** A budget is a bound on emitted bytes, so the
tests that matter assert emitted bytes against the budget, not one cost
function against another. The old accounting fails those by construction,
which is why each carries the old cost function beside it as a control:
`0 failures` on a guarded behaviour is not evidence, and a control that
must go the other way is.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

import aelfrice
import aelfrice.hook_search_tool
from aelfrice.hook import (
    _build_session_start_subblock,
    _core_belief_cost,
    _core_belief_line,
    _CORE_CHARS_PER_TOKEN,
    _escape_for_hook_block,
    _FRAMING_HEADER,
    _pack_core_candidates,
    _split_belief_lines,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_TIER_REFERENCE,
    LOCK_USER,
    ORIGIN_SPECULATIVE,
    Belief,
)
from aelfrice.render_cost import (
    BELIEF_LINE_WRAPPER_CHARS,
    MANIFEST_LINE_WRAPPER_CHARS,
    SPECULATIVE_ATTR_CHARS,
    belief_line_chars,
    chars_to_tokens,
)
from aelfrice.retrieval import (
    _belief_tokens,
    ENV_RETRIEVAL_TOKEN_BUDGET,
    lock_manifest_line,
    lock_injection_tokens,
    resolve_token_budget,
)
from aelfrice.store import MemoryStore

# A 16-character belief id: `derivation._belief_id` truncates the sha256 hex
# to 16, and `BELIEF_LINE_WRAPPER_CHARS` is stated for that width.
_ID = "0123456789abcdef"

# Naming shapes this repo gives a character bound on injected text. A
# constant is discovered by its name rather than listed by hand because the
# point of the discovery is to catch the cap nobody thought to list: sizing
# the SessionStart fixture against one named constant is what let
# `STOP_PROMPT_MAX_CONTENT` — 1000, five times the cap the fixture was built
# against — sit above it unnoticed.
_CONTENT_CAP_SUFFIXES = ("CHAR_CAP", "MAX_CONTENT", "_CHARS")

# The two named caps that must be in the discovered set. Over-inclusion is
# safe here (a cap that bounds something other than belief text only raises
# the floor), but a *rename* would shrink the set silently, so both are
# required by name.
_KNOWN_CONTENT_CAPS = frozenset(
    {
        "aelfrice.hook.STOP_PROMPT_MAX_CONTENT",
        "aelfrice.hook_search_tool.PER_LINE_CHAR_CAP",
    }
)


def _shipped_content_caps() -> dict[str, int]:
    """Every per-belief character cap the two injection modules ship.

    `aelfrice.hook` renders the SessionStart block, so a cap added there is
    the one that would trim it; `aelfrice.hook_search_tool` is the sibling
    lane that already truncates, and is what proves this class of regression
    ships rather than being hypothetical.

    Discovered, not enumerated. The fixture below has to be larger than the
    largest of these, and a maximum taken over a discovered set rises on its
    own when a cap is added — which is the whole difference between this and
    a fixture sized for one constant.
    """
    caps: dict[str, int] = {}
    for mod in (aelfrice.hook, aelfrice.hook_search_tool):
        for name, value in vars(mod).items():
            if not name.isupper() or isinstance(value, bool):
                continue
            if not isinstance(value, int):
                continue
            if name.endswith(_CONTENT_CAP_SUFFIXES):
                caps[f"{mod.__name__}.{name}"] = value
    return caps


# Padding for the SessionStart fixture's belief content, sized above every
# cap `_shipped_content_caps()` finds (the largest is
# `hook.STOP_PROMPT_MAX_CONTENT = 1000`). A lock shorter than a cap is
# invisible to it, so the guard that asserts no lock is trimmed would pass
# on a block where every line had been cut. The test asserts the resulting
# length against that discovered maximum rather than trusting this number,
# so a cap added later fails the assert instead of slipping under it.
_PAD_CHARS = 1200


def _mk(
    bid: str = _ID,
    content: str = "the scanner rejects duplicate content_hash inserts",
    *,
    lock_level: str = LOCK_NONE,
    lock_tier: str = "frozen",
    origin: str = "agent_inferred",
    alpha: float = 5.0,
    beta: float = 1.0,
    corroboration_count: int = 2,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=alpha,
        beta=beta,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at="2026-09-01T00:00:00Z" if lock_level == LOCK_USER else None,
        created_at="2026-09-01T00:00:00Z",
        last_retrieved_at=None,
        origin=origin,
        lock_tier=lock_tier,
        corroboration_count=corroboration_count,
    )


def _legacy_belief_tokens(b: Belief) -> int:
    """`retrieval._belief_tokens` as it stood before #1526."""
    if not b.content:
        return 0
    return int((len(b.content) + 3) // 4)


def _legacy_core_cost(b: Belief) -> int:
    """The `<core>` packer's per-belief cost before #1526."""
    return max(1, len(b.content) // 4)


# ---------------------------------------------------------------------------
# Pins against the live renderer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lock_level", [LOCK_NONE, LOCK_USER])
def test_belief_line_wrapper_matches_the_renderer(lock_level: str) -> None:
    """The constant equals what `_split_belief_lines` actually wraps with.

    Asserted for both `lock` values the renderer emits, because the
    constant claims the wrapper does not vary between them.
    """
    content = "x" * 40
    b = _mk(content=content, lock_level=lock_level)
    lines, manifest = _split_belief_lines([b], order_policy="lane",
                                          provenance_render=False)
    assert manifest == []
    assert len(lines) == 1
    # +1 for the newline `"\n".join` contributes between lines.
    assert len(lines[0]) + 1 - len(content) == BELIEF_LINE_WRAPPER_CHARS


def test_speculative_attr_chars_matches_the_renderer() -> None:
    """The speculative marker's width is the delta between the two lines."""
    content = "y" * 30
    plain = _mk(content=content)
    spec = _mk(content=content, origin=ORIGIN_SPECULATIVE)
    plain_lines, _ = _split_belief_lines([plain], order_policy="lane",
                                         provenance_render=False)
    spec_lines, _ = _split_belief_lines([spec], order_policy="lane",
                                        provenance_render=False)
    assert len(spec_lines[0]) - len(plain_lines[0]) == SPECULATIVE_ATTR_CHARS


def test_manifest_line_wrapper_matches_the_renderer() -> None:
    """A reference lock's manifest entry costs the entry text plus indent."""
    b = _mk(
        content="the store is opened read-write on every hook fire",
        lock_level=LOCK_USER,
        lock_tier=LOCK_TIER_REFERENCE,
    )
    lines, manifest = _split_belief_lines([b], order_policy="lane",
                                          provenance_render=False)
    assert lines == []
    assert len(manifest) == 1
    entry = lock_manifest_line(b)
    assert len(manifest[0]) + 1 - len(entry) == MANIFEST_LINE_WRAPPER_CHARS


def test_core_line_width_is_not_a_constant() -> None:
    """`<core>`'s wrapper varies, which is why nothing transcribes it.

    `corr` and `posterior` are interpolated numbers. A single transcribed
    width for this line shape would be wrong for some beliefs, so
    `_core_belief_cost` measures the rendered line instead of adding a
    constant — and this test is the reason that choice cannot be
    simplified away.
    """
    content = "z" * 20
    narrow = _mk(content=content, corroboration_count=2, alpha=5.0, beta=1.0)
    wide = _mk(content=content, corroboration_count=1234, alpha=3.0, beta=0.0)
    narrow_wrapper = len(_core_belief_line(narrow)) - len(content)
    wide_wrapper = len(_core_belief_line(wide)) - len(content)
    assert narrow_wrapper != wide_wrapper


# ---------------------------------------------------------------------------
# Pins on the cost functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 7, 81, 92, 300])
def test_belief_tokens_charges_the_rendered_line(n: int) -> None:
    """Cost equals the rendered line rounded up, at every length tried.

    The control is the old cost function: at each length tried
    the new cost must be strictly larger, or this test would pass against
    the accounting it exists to replace.
    """
    b = _mk(content="c" * n)
    lines, _ = _split_belief_lines([b], order_policy="lane",
                                   provenance_render=False)
    assert _belief_tokens(b) == chars_to_tokens(len(lines[0]) + 1)
    assert _belief_tokens(b) > _legacy_belief_tokens(b)


def test_belief_tokens_charges_the_speculative_marker() -> None:
    """A speculative belief costs the wider element it renders as."""
    content = "w" * 44
    plain = _mk(content=content)
    spec = _mk(content=content, origin=ORIGIN_SPECULATIVE)
    assert _belief_tokens(spec) > _belief_tokens(plain)
    assert _belief_tokens(spec) == chars_to_tokens(
        belief_line_chars(len(content), speculative=True)
    )


def test_lock_injection_tokens_charges_the_manifest_wrapper() -> None:
    """The manifest arm charges the indent and newline around the entry."""
    b = _mk(
        content="the live store is repo-local rather than in the dotdir",
        lock_level=LOCK_USER,
        lock_tier=LOCK_TIER_REFERENCE,
    )
    entry = lock_manifest_line(b)
    assert lock_injection_tokens(b, manifest_reference_locks=True) == (
        chars_to_tokens(len(entry) + MANIFEST_LINE_WRAPPER_CHARS)
    )
    # The other arm is the full rendered line, not the manifest entry.
    assert lock_injection_tokens(
        b, manifest_reference_locks=False
    ) == _belief_tokens(b)


def test_core_cost_charges_the_rendered_core_line() -> None:
    """`<core>`'s cost is its own rendered line, newline included."""
    b = _mk(content="the rebuilder self-report counted content, not lines")
    line = _core_belief_line(b)
    expected = -(-(len(line) + 1) // _CORE_CHARS_PER_TOKEN)
    assert _core_belief_cost(b) == expected
    assert _core_belief_cost(b) > _legacy_core_cost(b)


# ---------------------------------------------------------------------------
# Pins on the property: a budget bounds emitted bytes
# ---------------------------------------------------------------------------


def _core_bytes(packed: list[Belief]) -> int:
    return sum(len(_core_belief_line(b)) + 1 for b in packed)


def test_core_pack_keeps_emitted_bytes_inside_the_budget() -> None:
    """The property the old accounting broke, with that accounting as control.

    A budget of N tokens is a claim about at most 4N emitted characters.
    Packed under the shipped cost function the claim holds; packed under
    the pre-#1526 cost function, over the same candidates at the same
    budget, it does not — so this asserts the fix rather than restating
    the implementation.
    """
    budget = 200
    candidates = [
        _mk(bid=f"{i:016d}", content=f"belief {i} " + "s" * 30)
        for i in range(60)
    ]
    packed = _pack_core_candidates(candidates, budget)
    assert packed, "the fixture must pack something"
    assert _core_bytes(packed) <= budget * _CORE_CHARS_PER_TOKEN

    legacy_packed = _pack_core_candidates(candidates, budget, _legacy_core_cost)
    assert _core_bytes(legacy_packed) > budget * _CORE_CHARS_PER_TOKEN


def test_core_section_bytes_stay_inside_the_budget_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same property through the shipped builder, not the packer alone.

    `_build_session_start_subblock` is what the hook calls; this pins that
    the packer it uses is the one under test and that the section it
    renders is bounded in the bytes the budget is denominated in. `cwd` is
    a non-git tmp_path so `<recent-work>` stays out of the block, and the
    budget is set through the documented override so the assertion does not
    depend on the shipped default staying where it is.
    """
    budget = 150
    monkeypatch.setenv("AELFRICE_SESSION_START_CORE_BUDGET", str(budget))
    store = MemoryStore(str(tmp_path / "memory.db"))
    try:
        for i in range(40):
            store.insert_belief(
                _mk(bid=f"{i:016d}", content=f"core belief {i} " + "k" * 40)
            )
        block = _build_session_start_subblock(store, cwd=tmp_path)
    finally:
        store.close()
    section = block.split("<core>", 1)[1].split("</core>", 1)[0]
    body = section.strip("\n")
    assert body, "the fixture must render a non-empty <core> section"
    assert len(body) + 1 <= budget * _CORE_CHARS_PER_TOKEN


def test_escaping_is_the_residual_this_change_does_not_close() -> None:
    """Angle brackets still cost more to render than they cost to pack.

    Stated as a test rather than only in a docstring: `render_cost` calls
    this out as a known residual, and a reader who later "fixes" the
    docstring without the code should find out here. If this ever starts
    failing because the packers learned to charge escaped content, delete
    the test and the residual paragraph together.
    """
    raw = "a <tag> in content"
    b = _mk(content=raw)
    rendered = _escape_for_hook_block(raw)
    assert len(rendered) > len(raw)
    assert _belief_tokens(b) == chars_to_tokens(belief_line_chars(len(raw)))
    assert _belief_tokens(b) < chars_to_tokens(belief_line_chars(len(rendered)))


# ---------------------------------------------------------------------------
# The acceptance evidence
# ---------------------------------------------------------------------------


def _producer_module() -> object:
    """`benchmarks/injection_budget_bytes.py`, imported by path.

    `benchmarks/` is not a package on the install path, so the producer is
    reached the same way the other benchmark-backed tests reach theirs.
    """
    import sys

    root = Path(__file__).resolve().parent.parent
    if str(root / "benchmarks") not in sys.path:
        sys.path.insert(0, str(root / "benchmarks"))
    import injection_budget_bytes  # type: ignore[import-not-found]

    return injection_budget_bytes


@pytest.fixture(scope="module")
def producer_figures() -> dict[str, object]:
    """One producer run, shared by the tests below.

    A module-scoped fixture rather than `functools.lru_cache`: a cache does
    not memoise a call that raised, so under `lru_cache` the second test
    re-ran the whole producer from scratch and the file blew the 30-second
    per-test timeout CI pins with `AELF_TEST_TIMEOUT_SCALE=1`. A fixture
    errors once and reports the rest as errors rather than re-running.

    The run itself is fast because there is nothing to search: #1526 ships
    unchanged budgets, so the producer renders two arms per cell instead of
    sweeping a budget range for a byte-neutral band. Measured at about 1.4
    seconds for the full grid.
    """
    return _producer_module().figures()  # type: ignore[attr-defined]


def _producer_lanes() -> tuple[str, ...]:
    return _producer_module().LANES  # type: ignore[attr-defined]


def test_the_corrected_accounting_shrinks_what_each_budget_buys(
    producer_figures: dict[str, object],
) -> None:
    """The headline claim of #1526, measured on every lane the producer packs.

    The budgets are unchanged, so what moves is what a budget buys. At the
    committed replay-soak corpus's median belief length the corrected cost
    functions charge more per belief than the old ones on every lane whose
    pack a budget ends, so no lane's block may grow there.

    `session_start` is exempt and is asserted to be exempt rather than
    skipped quietly: its arms both end on the candidate pool, because no
    budget can bind on that lane at all -- which is why #1546 deleted the
    one it used to carry (see
    `test_session_start_lane_never_trims_its_l0_pool`).
    """
    fig = producer_figures
    assert fig["store_beliefs"] > 0
    exempt: list[str] = []
    for lane in _producer_lanes():
        before = fig[f"{lane}_bytes_before"]
        after = fig[f"{lane}_bytes_after"]
        assert before > 0, f"{lane}: the before arm must emit something"
        if fig[f"{lane}_pool_equality"]:
            exempt.append(lane)
            assert after == before, (
                f"{lane}: both arms ended on the candidate pool, so they must "
                f"render the same pool: {before} vs {after}"
            )
            continue
        assert after <= before, (
            f"{lane}: emitted bytes grew from {before} to {after} at "
            f"{fig[f'{lane}_headline_chars']} content chars"
        )
    assert exempt == ["session_start"], exempt
    # A control: if every lane came out flat this would assert nothing.
    assert any(
        fig[f"{lane}_pct"] < 0 for lane in _producer_lanes()
    ), {lane: fig[f"{lane}_pct"] for lane in _producer_lanes()}


def test_the_producer_names_which_budget_ended_every_pack(
    producer_figures: dict[str, object],
) -> None:
    """No cell is suppressed, and each one says what ended it.

    This is the round-2 defect, pinned. The earlier producer probed binding
    by raising `token_budget` alone while `l25_token_subbudget` stayed at the
    module default, so a pack the L2.5 sub-cap had ended read as a budget
    that does not bind — and the cell was replaced by a dash whose published
    legend claimed the two arms had returned equal pools. Deviations of
    -21.1%, -20.1% and +53.0% were hidden that way.

    What is asserted: every cell carries both byte counts, and every arm
    carries a `binds_on` naming which cap ended it — including the
    `l25_subbudget` value the earlier probe could not produce, which must
    actually occur somewhere in the grid or the fix is untested.
    """
    fig = producer_figures
    m = _producer_module()
    seen: set[str] = set()
    for lane in _producer_lanes():
        curve = fig[f"{lane}_curve"]
        assert set(curve) == {str(c) for c in fig["lengths"]}, lane
        for chars, row in curve.items():
            for arm in ("before", "after"):
                assert isinstance(row[arm], int) and row[arm] > 0, (lane, chars)
                binds_on = row[f"{arm}_binds_on"]
                assert binds_on in {
                    "token_budget", "l25_subbudget", "both", "pool",
                }, (lane, chars, binds_on)
                seen.add(binds_on)
            assert row["pool_equality"] == (
                row["before_binds_on"] == "pool"
                and row["after_binds_on"] == "pool"
            ), (lane, chars)
    assert "l25_subbudget" in seen, (
        "no arm in the grid ended on the L2.5 sub-budget, so the "
        "multi-budget binding probe is not exercised by this run"
    )
    assert "token_budget" in seen, seen
    assert m.SATURATION_PROBE_FACTOR > 1


def test_the_effect_is_length_dependent_and_changes_sign(
    producer_figures: dict[str, object],
) -> None:
    """There is no single multiplier for this change, and the curve shows it.

    Two claims the CHANGELOG makes, asserted rather than described:

    * On the `<belief …>` lanes the correction removes a wrapper whose share
      of the line shrinks as content grows, so the deviation shrinks with
      length.
    * On the Grep|Glob lane it changes sign. Past `PER_LINE_CHAR_CAP` the old
      accounting charged the untruncated content, which is *more* than that
      lane emits, so the corrected cost lets more beliefs through.
    """
    fig = producer_figures
    core = fig["core_curve"]
    assert core["40"]["pct"] < core["300"]["pct"] < 0, core

    st = fig["search_tool_curve"]
    assert st["92"]["pct"] < 0, st["92"]
    assert st["300"]["pct"] > 0, st["300"]


def test_the_producer_says_the_session_start_number_is_its_own_probe(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The one lane whose budget column is not a setting says so, in the run.

    Every other lane in `benchmarks/injection_budget_bytes.py` is measured at
    a shipped constant held equal in both arms, and the printed legend reads
    `budget N unchanged`. `session_start` has no shipped constant to hold —
    #1546 deleted it, and the production lane passes none — so the number in
    that row is `SESSION_START_PROBE_BUDGET`, a literal of the producer. The
    same legend on that row would tell a reader the lane ships a 1500-token
    budget: the misreading the CHANGELOG spent an entry removing.

    **Asserted on `main`'s stdout, not on `_budget_label` in isolation.**
    The helper is not what a reader sees; its two call sites are. Reverting
    only those call sites to the pre-branch f-strings — leaving the helper,
    its docstring and a helper-level assert all in place — puts
    `session_start: budget 1500 unchanged` back on stdout with the full
    suite green. So the run is executed here and its lines are read.

    `--curve` because both call sites print: the per-lane summary line and
    the per-lane curve header. One run covers both.

    Both directions are asserted: a legend that said "passes none" on every
    lane would be just as wrong, and every lane must be found in the output
    or a lane that silently stopped printing would pass vacuously.
    """
    m = _producer_module()
    assert m.main(["--curve"]) == 0
    out = capsys.readouterr().out

    for lane in _producer_lanes():
        summary = [ln for ln in out.splitlines() if ln.startswith(f"{lane}: ")]
        header = [ln for ln in out.splitlines() if ln.startswith(f"--- {lane} (")]
        assert len(summary) == 1, (lane, summary)
        assert len(header) == 1, (lane, header)
        for line in (*summary, *header):
            if lane == "session_start":
                assert "passes none" in line, (
                    f"the producer printed {line!r}, which presents this "
                    "module's probe value as a budget the lane ships. It "
                    "ships none (#1546)."
                )
                assert "unchanged" not in line, line
            else:
                assert "passes none" not in line, (lane, line)
                assert "unchanged" in line, (lane, line)


def test_the_producer_reads_the_session_start_budget_off_its_own_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`shipped_budget` is wired to the constant, not spelling the same number.

    The probe value is published — it is the `session_start` row of the
    CHANGELOG's budget column — and a derived-figure marker on that row
    re-derives it from this module through `shipped_budget`. Written as
    `"session_start": 1500`, that marker compares a literal against a
    literal: the suite stays green, the `derived-figures` job exits 0, and
    moving `SESSION_START_PROBE_BUDGET` to 3000 leaves the CHANGELOG
    publishing 1500 with nothing failing. So the wire is asserted with a
    value no other constant on this module carries, and the published value
    is pinned separately.
    """
    m = _producer_module()
    assert m.SESSION_START_PROBE_BUDGET == 1500, (
        "benchmarks/injection_budget_bytes.py SESSION_START_PROBE_BUDGET is "
        f"{m.SESSION_START_PROBE_BUDGET}, but CHANGELOG/unreleased/"
        "1526-render-line-budgets.md publishes 1500 in the SessionStart row "
        "of its budget column. Move both together (#1546)."
    )
    monkeypatch.setattr(m, "SESSION_START_PROBE_BUDGET", 4242)
    assert m.shipped_budget("session_start") == 4242, (
        "shipped_budget('session_start') does not read "
        "SESSION_START_PROBE_BUDGET, so the derived-figure marker on the "
        "CHANGELOG's SessionStart budget row is checking a literal against "
        "a literal (#1546)."
    )


def test_the_acceptance_corpus_carries_locks_and_speculative_beliefs(
    producer_figures: dict[str, object],
) -> None:
    """Two arms this change rewrote are exercised by the published figures.

    `retrieval.lock_injection_tokens` and the `speculative="1"` attribute
    width are dead code against a corpus of plain agent-inferred beliefs, so
    the producer's store carries both and reports the counts it actually
    holds. Read back off the store rather than from the generator's
    arithmetic.
    """
    shape = producer_figures["corpus_shape"]
    assert shape["locked"] > 0, shape
    assert shape["speculative"] > 0, shape


# ---------------------------------------------------------------------------
# The rebuilder's self-report
# ---------------------------------------------------------------------------


def test_rebuild_block_budget_used_reports_the_rendered_lines() -> None:
    """`budget_used`'s numerator is what the section emits, not the text in it.

    It used to sum belief content, which is not a number any reader of the
    block can check against the block, and which understated the section by
    the `<belief>` elements around it. The control is that same old sum:
    the reported figure must be strictly larger, or the report is still the
    pre-#1526 one.
    """
    from aelfrice.context_rebuilder import _format_block

    hits = [
        _mk(bid=f"{i:016d}", content=f"rebuild belief {i} about the packer")
        for i in range(4)
    ]
    block = _format_block([], hits, set(), token_budget=4000)
    open_tag = [
        ln for ln in block.splitlines() if "<retrieved-beliefs" in ln
    ][0]
    reported = int(
        open_tag.split('budget_used="', 1)[1].split('"', 1)[0].split("/", 1)[0]
    )
    belief_lines = [
        ln for ln in block.splitlines() if ln.lstrip().startswith("<belief ")
    ]
    assert len(belief_lines) == len(hits)
    assert reported == sum(len(ln) + 1 for ln in belief_lines)
    assert reported > sum(len(b.content) for b in hits)


@pytest.mark.parametrize("n", [0, 12, 92, 400])
def test_clustering_cost_mirrors_the_retrieval_cost(n: int) -> None:
    """`clustering._belief_tokens` is a mirror, and mirrors must match.

    `clustering` duplicates the estimator rather than importing it, to keep
    the wiring direction retrieval -> clustering. A duplicate is free to
    drift, so the claim that it mirrors is asserted here rather than only
    in its docstring. The control is the content-only estimate the mirror
    used to be: matching that instead would mean the copy was left behind.
    """
    from aelfrice.clustering import _belief_tokens as cluster_cost

    b = _mk(content="m" * n)
    assert cluster_cost(b) == _belief_tokens(b)
    assert cluster_cost(b) > _legacy_belief_tokens(b)


# ---------------------------------------------------------------------------
# The seventh renderer: `[L0] <prefix>: <content>`, truncated (#1526 item 4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 12, 92, 150, 500, 4000])
def test_search_tool_charges_the_line_its_own_renderer_emits(n: int) -> None:
    """The PreToolUse lane's cost is the line `_format_results` puts out.

    Not a transcribed width: the cost function builds the line. The control
    is `retrieval._belief_tokens`, the `<belief …>` cost this lane was
    charged before #1526 and does not emit — on a belief past the per-line
    cap the two must disagree, or the truncation is still uncharged.
    """
    st = aelfrice.hook_search_tool
    PER_LINE_CHAR_CAP = st.PER_LINE_CHAR_CAP
    _belief_line = st._belief_line
    _belief_line_cost = st._belief_line_cost
    _format_results = st._format_results

    b = _mk(content="m" * n)
    line = _belief_line(b, frozenset())
    if not n:
        # `_format_results` drops a contentless belief, so it costs nothing.
        assert line is None
        assert _belief_line_cost(b) == 0
        return
    assert line is not None
    assert len(line) <= PER_LINE_CHAR_CAP
    assert _belief_line_cost(b) == chars_to_tokens(len(line) + 1)
    # The line really is the one the renderer emits.
    assert line in _format_results("q", [b], set())
    if n > PER_LINE_CHAR_CAP:
        assert _belief_line_cost(b) < _belief_tokens(b)


def test_search_tool_cost_may_ignore_the_locked_set_but_not_the_lock_tier() -> None:
    """Both halves of `_belief_line_cost`'s stated shortcut, pinned.

    It passes an empty locked set, on the claim that `[L0]` and `[L1]` are
    the same width so the tier it assumes cannot change the count. The
    reference tier `[L0-ref]` is wider and is detected off the belief
    itself, not off the set — so the shortcut must not extend to it.
    """
    _belief_line = aelfrice.hook_search_tool._belief_line
    _belief_line_cost = aelfrice.hook_search_tool._belief_line_cost

    b = _mk(content="a belief long enough to render a full line of text")
    free = _belief_line(b, frozenset()) or ""
    locked = _belief_line(b, frozenset({b.id})) or ""
    assert free.startswith("[L1] ") and locked.startswith("[L0] ")
    assert len(free) == len(locked)

    ref = _mk(
        content="a reference-tier lock whose topic is bounded at eighty chars",
        lock_level=LOCK_USER,
        lock_tier=LOCK_TIER_REFERENCE,
    )
    ref_line = _belief_line(ref, frozenset())
    assert ref_line is not None and ref_line.startswith("[L0-ref] ")
    assert _belief_line_cost(ref) == chars_to_tokens(len(ref_line) + 1)


# ---------------------------------------------------------------------------
# The `belief_cost_fn` seam
# ---------------------------------------------------------------------------


def _seam_store(tmp_path: Path, *, n_locked: int = 0) -> MemoryStore:
    store = MemoryStore(str(tmp_path / "seam.db"))
    for i in range(20):
        store.insert_belief(
            _mk(
                bid=f"{i:016d}",
                content=f"packer budget belief number {i} about retrieval",
                lock_level=LOCK_USER if i < n_locked else LOCK_NONE,
            )
        )
    return store


def test_belief_cost_fn_reaches_the_pack(tmp_path: Path) -> None:
    """A lane's own cost function decides how many beliefs the pack keeps.

    Control: the same call with `belief_cost_fn=None`, which must keep more.
    Without it, "the parameter exists" would be the only thing asserted.
    """
    from aelfrice.retrieval import retrieve

    store = _seam_store(tmp_path)
    try:
        default = retrieve(store, "packer budget retrieval", token_budget=400)
        expensive = retrieve(
            store, "packer budget retrieval", token_budget=400,
            belief_cost_fn=lambda b: 200,
        )
    finally:
        store.close()
    assert len(default) > len(expensive)
    assert len(expensive) <= 2


def test_belief_cost_fn_reaches_the_lock_arm(tmp_path: Path) -> None:
    """Locks are charged through the override too.

    Locks are never trimmed, so what an expensive lock changes is how much
    budget is left for relevance. The control is the same store costed
    normally: it must surface more non-locked hits.
    """
    from aelfrice.retrieval import retrieve

    store = _seam_store(tmp_path, n_locked=4)
    try:
        default = retrieve(store, "packer budget retrieval", token_budget=400)
        costly_locks = retrieve(
            store, "packer budget retrieval", token_budget=400,
            belief_cost_fn=lambda b: (
                5000 if b.lock_level == LOCK_USER else _belief_tokens(b)
            ),
        )
    finally:
        store.close()
    n_free_default = sum(1 for b in default if b.lock_level == LOCK_NONE)
    n_free_costly = sum(1 for b in costly_locks if b.lock_level == LOCK_NONE)
    assert n_free_default > n_free_costly


# ---------------------------------------------------------------------------
# What the round-1 enumeration missed, and what it cannot reach
# ---------------------------------------------------------------------------


def test_belief_cost_fn_reaches_the_l25_sub_pack(tmp_path: Path) -> None:
    """The L2.5 sub-pack charges the caller's line shape too.

    `_l25_hits` runs a second budget loop against `l25_token_subbudget`. Until
    #1526 threaded `cost_fn` into it, that loop read `_belief_tokens` off the
    module global, so a lane passing `belief_cost_fn` selected its L2.5 slice
    against the `<belief …>` shape it never emits.

    Driven through `_l25_hits` directly so the sub-budget is the only cap in
    play. The control is the same call with `cost_fn=None`: an expensive
    override must keep strictly fewer beliefs, or the parameter is inert.
    """
    from aelfrice.retrieval import _l25_hits

    store = MemoryStore(str(tmp_path / "l25.db"))
    try:
        for i in range(12):
            store.insert_belief(
                _mk(
                    bid=f"{i:016d}",
                    content=f"MemoryStore EntityIndex belief {i} about packers",
                )
            )
        kwargs = dict(
            locked_ids=set(),
            l25_limit=20,
            l25_token_subbudget=200,
            query_entity_cap=16,
        )
        default = _l25_hits(store, "MemoryStore EntityIndex", **kwargs)
        costly = _l25_hits(
            store, "MemoryStore EntityIndex", cost_fn=lambda b: 100, **kwargs,
        )
    finally:
        store.close()
    assert len(default) > len(costly), (len(default), len(costly))
    assert len(costly) == 2


def test_core_pack_skips_an_oversized_belief_instead_of_breaking(
    tmp_path: Path,
) -> None:
    """`_pack_core_candidates`'s stated invariant, pinned.

    Its docstring calls skip-don't-break load-bearing — an oversized FIRST
    candidate would otherwise empty the whole section — and until now nothing
    asserted it. Ranked first is a belief no budget here can fit; the ones
    behind it must still be packed.

    The control is the break-on-first-miss loop the docstring rules out,
    written out here so the assertion distinguishes the two behaviours rather
    than merely passing.
    """
    del tmp_path
    budget = 200
    oversized = _mk(bid=f"{0:016d}", content="x" * 4000)
    rest = [
        _mk(bid=f"{i:016d}", content=f"core belief {i}") for i in range(1, 12)
    ]
    candidates = [oversized, *rest]

    packed = _pack_core_candidates(candidates, budget)
    assert packed, "an oversized first candidate must not empty the section"
    assert oversized.id not in {b.id for b in packed}
    assert _core_bytes(packed) <= budget * _CORE_CHARS_PER_TOKEN

    # Control: break-on-first-miss over the same candidates keeps nothing.
    broken: list[Belief] = []
    used = 0
    for b in candidates:
        cost = _core_belief_cost(b)
        if used + cost > budget:
            break
        broken.append(b)
        used += cost
    assert broken == []


@pytest.mark.parametrize(
    ("n_locked", "n_free"),
    [(10, 60), (300, 1800)],
    ids=["decoys-reachable", "locks-above-any-plausible-cap"],
)
def test_session_start_lane_never_trims_its_l0_pool(
    n_locked: int,
    n_free: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#379: the SessionStart block is bounded by the lock count, not a budget.

    This is the property #1546 deleted
    `hook.DEFAULT_SESSION_START_TOKEN_BUDGET` on, so it is asserted here
    directly rather than against that constant. The lane must keep emitting
    every lock whatever budget it is run under; a future change that gives
    the SessionStart pack a trim fails this.

    Driven through the real `_retrieve_baseline_with_block`, at the
    production call shape — no budget argument, which is what `session_start`
    now passes. The branch's round-2 guard instead monkeypatched that
    function and asserted the integer handed to it, which pins the ends and
    not the edge: it would have passed unchanged while the budget did
    nothing.

    The budget is varied through both knobs that reach this lane, and
    through neither. `AELFRICE_RETRIEVAL_TOKEN_BUDGET` outranks everything
    and always reached it; `[retrieval] token_budget` is the one #1546
    newly exposed, because the resolver ranks TOML below an explicit
    argument and this caller used to pass one. The unset arm is the shipped
    default. `session_start` retrieves with an empty query; in
    `retrieve_with_tiers` every relevance lane is gated on `query.strip()`,
    so only L0 contributes and L0 is appended unconditionally.

    The TOML arm asserts its own resolution first. A config file the
    resolver never discovers would leave that arm running at the default and
    agreeing with every other arm for the wrong reason.

    **Each arm is held absolutely, to the store's own lock list.** An
    identity across arms is blind to a trim that does not vary with the
    budget: `hits = hits[:5]` in `_retrieve_baseline_with_block` agrees with
    every arm and still breaks #379. Both the hit list and the rendered block
    are checked, because a formatter that packed the block would leave the
    hit list whole.

    **The block is held to content, not to ids, and the fixture is sized
    above every cap that could trim it.** An id-presence assert passes on a
    block whose every line has been truncated: the id sits in the
    `<belief id="…"` attribute ahead of the content, so a per-belief cap
    leaves every id in place, leaves `hits` whole, and leaves the block
    length identical across arms. Each lock's content is therefore required
    in the block verbatim, and the locks are sized above
    `max(_shipped_content_caps().values())` rather than above one named
    constant. Sizing against one constant is how this arm was wrong before:
    it was built against `hook_search_tool.PER_LINE_CHAR_CAP = 200`, the
    truncation a sibling lane already ships, while `hook` — the module that
    renders this very block — ships `STOP_PROMPT_MAX_CONTENT = 1000` for the
    identical reason. Truncating every lock at 1000 characters here passed
    the whole suite. A maximum over a discovered set rises on its own when
    the next cap lands.

    **The two store sizes are not a sweep; each catches a mutation the other
    cannot, so neither may be dropped.** Any fixture is blind to a cap above
    its own lock count, so the large store sets the ceiling: at 10 locks
    `hits = hits[:50]` changes nothing, and at 300 it drops 250. 300 is the
    lock count #1546's measurement is published at, and it sits above every
    cap this lane could plausibly acquire. The small store is what keeps the
    decoys reachable. They exist for the mutation that gives this lane a
    non-empty query, and `resolve_l1_limit`'s default candidate slice is 50:
    at 10 locks that slice has room for decoys and this test fails, and at
    300 the locks fill it and dedupe away, so no decoy is admitted and the
    mutation passes. Measured with the empty query replaced by "locked
    baseline belief", at this fixture's content length — at (10, 60): 10 hits
    and 0 decoys at budgets 1 and 10, 12 hits and 2 decoys at 1500, 50 hits
    and 40 decoys at 100000; at (300, 1800): 300 hits and 0 decoys at all
    four.
    """
    db = tmp_path / "memory.db"
    store = MemoryStore(str(db))
    try:
        for i in range(n_locked):
            store.insert_belief(
                _mk(
                    bid=f"{i:016d}",
                    content=f"locked baseline belief {i} " + "b" * _PAD_CHARS,
                    lock_level=LOCK_USER,
                )
            )
        # Unlocked decoys, and the fixture's proportions are load-bearing.
        # They carry the locks' words verbatim so that an L1 pass -- on any
        # query a defect might substitute for the empty one -- would reach
        # them, and they outnumber the locks six to one so they are not all
        # pushed out of the candidate slice by locks that dedupe away. Only
        # the small store has room for them in that slice; see the docstring.
        for i in range(n_free):
            store.insert_belief(
                _mk(
                    bid=f"{i + 100_000:016d}",
                    content=(
                        f"locked baseline belief spare {i} " + "b" * _PAD_CHARS
                    ),
                    lock_level=LOCK_NONE,
                )
            )
        # The control comes from the store, not from the loop above: it is
        # the set the #379 contract says every arm must emit in full, and the
        # text it says must reach the model unabridged.
        locked = {b.id: b.content for b in store.list_locked_beliefs()}
    finally:
        store.close()
    locked_ids = set(locked)
    assert len(locked_ids) == n_locked, len(locked_ids)
    # The fixture has to be able to see every cap it is guarding against,
    # so it is sized against the largest one the injection modules ship and
    # not against any single named constant.
    caps = _shipped_content_caps()
    missing_caps = _KNOWN_CONTENT_CAPS - set(caps)
    assert not missing_caps, (
        f"{sorted(missing_caps)} was not discovered by "
        f"_shipped_content_caps(), which found {sorted(caps)}. A rename "
        "shrinks that set silently and drops this fixture's floor."
    )
    tightest_name, largest_cap = max(caps.items(), key=lambda kv: (kv[1], kv[0]))
    shortest = min(len(c) for c in locked.values())
    assert shortest > largest_cap, (
        f"the locks are {shortest} characters, which is at or under "
        f"{tightest_name} = {largest_cap}, the largest per-belief character "
        f"cap these modules ship (all of them: {caps}). A cap of that size "
        "added to this lane would leave every id in place and pass this "
        "test. Raise _PAD_CHARS above it."
    )
    monkeypatch.setattr(
        aelfrice.hook, "_open_store", lambda: MemoryStore(str(db))
    )

    seen: list[tuple[str, list[Belief], str]] = []
    for budget in (1, 10, 1500, 100_000):
        monkeypatch.setenv(ENV_RETRIEVAL_TOKEN_BUDGET, str(budget))
        hits, block = aelfrice.hook._retrieve_baseline_with_block()
        seen.append((f"env={budget}", hits, block))
    # The shipped path: nobody sets the env var, so `retrieve()` falls
    # through to its own default.
    monkeypatch.delenv(ENV_RETRIEVAL_TOKEN_BUDGET, raising=False)
    hits, block = aelfrice.hook._retrieve_baseline_with_block()
    seen.append(("unset", hits, block))
    # The knob the deletion exposed. `resolve_token_budget` walks up from the
    # process's cwd, so the file has to sit in a directory this test owns and
    # the chdir has to happen before the call.
    (tmp_path / ".aelfrice.toml").write_text(
        "[retrieval]\ntoken_budget = 1\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    assert resolve_token_budget() == 1, (
        "the TOML arm resolved to something else, so it is not measuring "
        f"what it claims: {resolve_token_budget()}"
    )
    hits, block = aelfrice.hook._retrieve_baseline_with_block()
    seen.append(("toml=1", hits, block))

    sizes = {label: len(block) for label, _, block in seen}
    assert len(set(sizes.values())) == 1, sizes
    for label, hits, block in seen:
        # Absolute, not an identity across arms: a trim that does not vary
        # with the budget agrees with every arm and still breaks #379.
        missing = locked_ids - {b.id for b in hits}
        assert not missing, (
            f"{label}: the lane dropped {len(missing)} of {n_locked} locks, "
            "so the SessionStart block no longer carries every locked "
            "belief. That is the #379 contract, and it is what "
            "docs/user/PRIVACY.md promises the user (#1546). "
            f"First dropped: {sorted(missing)[:3]}"
        )
        # Only the locks reach the block, at every budget: no relevance lane
        # runs on an empty query, so there is nothing for a budget to trim.
        assert all(b.lock_level == LOCK_USER for b in hits), (
            label,
            [b.id for b in hits if b.lock_level != LOCK_USER],
        )
        # The render edge, which the hit list alone cannot see: a formatter
        # that packs the block would leave `hits` whole and still ship less.
        # Content, not the id: the id is an attribute ahead of the content, so
        # a per-line cap keeps every id and truncates every lock.
        absent = [
            bid for bid in sorted(locked_ids) if locked[bid] not in block
        ]
        assert not absent, (
            f"{label}: {len(absent)} of {n_locked} locks reached the "
            "formatter and not the rendered block in full, so the block "
            "trims content docs/user/PRIVACY.md promises is never trimmed "
            f"(#379, #1546). First absent: {absent[:3]}"
        )
        # The id arm is kept beside it: it is the one that fails when a lock
        # is dropped outright rather than shortened.
        idless = [bid for bid in sorted(locked_ids) if bid not in block]
        assert not idless, (
            f"{label}: {len(idless)} locks reached the formatter and not the "
            f"rendered block. First absent: {idless[:3]}"
        )


def test_the_search_tool_lane_passes_its_own_cost_function_to_retrieve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wiring, not just the two ends of it.

    `_belief_line_cost` being correct and `retrieve()` honouring
    `belief_cost_fn` are both pinned above, and neither notices if the lane
    stops connecting them -- the producer builds its own call and would go on
    measuring a cost function the shipped lane no longer passes. Driven
    through `_do_search` with `retrieve` replaced, and asserted by identity,
    so a lookalike closure would not satisfy it.
    """
    import io


    db = tmp_path / "memory.db"
    MemoryStore(str(db)).close()
    monkeypatch.setenv("AELFRICE_DB", str(db))
    seen: list[object] = []

    def _capture(store: object, query: str, **kwargs: object) -> list[Belief]:
        seen.append(kwargs.get("belief_cost_fn"))
        return []

    monkeypatch.setattr(aelfrice.retrieval, "retrieve", _capture)
    aelfrice.hook_search_tool._do_search(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "Grep",
            "tool_input": {"pattern": "packer budget retrieval"},
            "cwd": str(tmp_path),
            "session_id": "s-cost-fn",
        },
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert seen, "the lane never reached retrieval"
    assert seen[0] is aelfrice.hook_search_tool._belief_line_cost, seen[0]


def test_every_block_that_emits_the_framing_header_is_enumerated() -> None:
    """Four formatters emit `_FRAMING_HEADER`. There must be four.

    #1526 measured this header (502 characters, 126 tokens) and deliberately
    left it uncharged — reserving it out of the retrieval budget is a
    separable, uncompensated change carried into the follow-up issue. The
    enumeration is what that issue rests on, so it is asserted here rather
    than written down: a fourth emitter in another module is how a grep-free
    enumeration loses one.
    """
    from aelfrice.hook import (
        _format_baseline_hits,
        _format_hits,
        _format_hits_with_session_start,
    )
    from aelfrice.hook_agent_context import _build_block

    # The enumeration has to come off the tree, not off a list written here.
    # A hand-written list of four formatters makes `len(...) == 4` true by
    # construction: it stayed green when a fifth emitter was added, which is
    # the only event it exists to catch.
    #
    # Two scans, because neither alone is enough. The AST scan says WHERE the
    # header is emitted, which is the fact the follow-up issue cites. It is a
    # name match on the call, so it sees a direct call and an attribute call
    # and misses a call through an alias, a `getattr`, or a local. The module
    # scan closes those: every one of them still has to spell
    # `_framing_header_for` somewhere in its source to reach the function, so
    # the set of modules mentioning the name at all is the wider net. A name
    # assembled at run time from pieces defeats both, and nothing here claims
    # otherwise.
    src_root = Path(aelfrice.__file__).parent
    call_sites: list[tuple[str, int]] = []
    mentioning: set[str] = set()
    for path in sorted(src_root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "_framing_header_for" in source:
            mentioning.add(path.name)
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name):
                called = func.id
            elif isinstance(func, ast.Attribute):
                called = func.attr
            else:
                continue
            if called == "_framing_header_for":
                call_sites.append((path.name, node.lineno))
    # `ast.walk` is breadth-first, so the sites come out in node order rather
    # than source order. Sort them for a stable failure message.
    call_sites.sort()

    assert mentioning == {"hook.py", "hook_agent_context.py"}, (
        "a module names `_framing_header_for` that this enumeration does not "
        f"know about; #1526's follow-up rests on the emitter set: {mentioning}"
    )
    assert [name for name, _ in call_sites] == [
        "hook.py",
        "hook.py",
        "hook.py",
        "hook_agent_context.py",
    ], (
        "the framing header is emitted from a call site this enumeration does "
        f"not know about; #1526's follow-up rests on there being four: {call_sites}"
    )

    hits = [_mk(bid=f"{i:016d}", content=f"framing belief {i}") for i in range(2)]
    rendered = [
        _format_hits(hits),
        _format_hits_with_session_start(hits, "<session-start/>"),
        _format_baseline_hits(hits),
        _build_block(list(hits)),
    ]
    assert all(_FRAMING_HEADER in block for block in rendered)
    assert len(_FRAMING_HEADER) == 502
