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
import os
from pathlib import Path

import pytest

import aelfrice
import aelfrice.hook_search_tool
from aelfrice.hook import (
    _build_session_start_subblock,
    BELIEF_CONTENT_CHAR_CAP,
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
# cap `_shipped_content_caps()` finds. The largest was
# `hook.STOP_PROMPT_MAX_CONTENT = 1000` until #1551 added
# `hook.BELIEF_CONTENT_CHAR_CAP = 1200`, which the name shapes below
# discover; the lock content is 1225 characters, so it clears that by 25.
# A lock shorter than a cap is
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

    There is nothing to search: #1526 ships unchanged budgets, so the producer
    renders two arms per cell instead of sweeping a budget range for a
    byte-neutral band. That still costs **15.5 seconds** for the full grid
    after #1547 extended it to 18,600 characters, reported as the `setup` row
    for the first test below by

        uv run pytest tests/test_render_cost_1526.py -q -p no:randomly \
            --durations=0

    and ranging 14.7 to 16.4 seconds over seven runs on an unloaded machine.
    An earlier revision of this docstring published "about 8 seconds", which
    was measured before the grid reached 18,600 characters and which no run on
    this tree reproduces.

    That figure is half the budget, not a footnote, because the timeout covers
    setup: a `@pytest.mark.timeout(5)` on the first test below errors *inside*
    this fixture with `Failed: Timeout (>5.0s) from pytest-timeout`, charged to
    the item that requested it. So the first test to ask for these figures
    spends about 15.5 of its 30 seconds here and has roughly 14 left for its
    own body, which is why the grid is nine lengths and not more, and why
    `_CORE_CROSSING` and the emptied-lengths list in
    `test_the_core_section_empties_once_one_belief_exceeds_its_budget` pin the
    ninth length rather than leaving it to be trimmed under that pressure. The
    figure is not asserted anywhere: a wall-clock assertion would be the
    non-deterministic thing this suite forbids, so it is published here and
    re-measured by hand.

    Every `AELFRICE_` variable is removed for the duration. `figures()` is
    already hermetic against `.aelfrice.toml` — it chdirs into a tempdir with
    no ancestor config — but a dozen of the resolvers it reaches check an
    environment variable first, `AELFRICE_TYPE_AWARE_COMPRESSION` among them,
    and a value exported in the operator's shell must not change a published
    figure or the assertions below. The prefix is cleared as a class rather
    than a list of names, so a resolver added later is covered too.
    """
    mp = pytest.MonkeyPatch()
    try:
        for name in [k for k in os.environ if k.startswith("AELFRICE_")]:
            mp.delenv(name, raising=False)
        return _producer_module().figures()  # type: ignore[attr-defined]
    finally:
        mp.undo()


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


# The `<core>` crossing edge this branch publishes, as content characters.
#
# At or above it, one `<core>` line is charged at or below
# `DEFAULT_SESSION_START_CORE_TOKEN_BUDGET` **as shipped** and above it in the
# pre-#1526 currency, so the two accountings disagree about whether the section
# can hold a line. It is a half-line and not a window, which is the #1552
# correction: `_core_belief_line` truncates content at
# `hook.BELIEF_CONTENT_CHAR_CAP`, so the shipped charge plateaus at 320 tokens
# for any content past 1,200 characters and can never reach a 1,500-token
# budget again. The pre-#1526 charge `max(1, len(content) // 4)` is uncapped
# and crosses at exactly 6,004 — 1,500 at 6,003, 1,501 here — so every length
# from here up is on the disagreeing side and every length below it is not.
#
# The relation used to run the other way. Before #1552 the shipped charge was
# the larger of the two and overran the budget first, at 5,934, and the
# pre-#1526 charge caught up at 6,004, which made a 70-character window that
# `LENGTH_GRID` held open with an entry at 5,950. `legacy <= budget < shipped`
# is now unsatisfiable at every content length, so that entry moved to the edge
# itself (see `LENGTH_GRID`): 6,004 is a grid length, and it is the first grid
# length whose before arm packs no `<core>` line, which is what
# `test_the_core_section_empties_once_one_belief_exceeds_its_budget` pins it
# by. The edge is asserted here too, at 6,003 and 6,004, because a grid
# assertion alone cannot see an edge that moves within a grid step.
#
# Re-derive with `_core_pack_costs_at`: the edge is the first length whose
# pre-#1526 charge exceeds the budget while the shipped charge does not.
_CORE_CROSSING = 6004


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

    What is asserted: every cell carries both byte counts, every arm carries a
    `binds_on` naming which cap ended it — including the `l25_subbudget` value
    the earlier probe could not produce, which must actually occur somewhere in
    the grid or the fix is untested — and every arm carries the probe budget
    that label was established at.

    The per-cell floor is the zero branch itself: a cell that is zero has to
    name itself as the one measured zero, and every other lane and length
    reaches that branch and fails. A zero is admitted by re-deriving the
    condition that empties it rather than by widening the floor. #1547
    extended the grid past every lane's own
    budget, and above `DEFAULT_SESSION_START_CORE_TOKEN_BUDGET` a single
    `<core>` line no longer fits — **in the before arm**.
    `_pack_core_candidates` skips an oversized belief rather than breaking, so
    at 6,004 content characters the before arm packs none of 300 candidates and
    `<core>` emits nothing, while the after arm packs four. Zero is that
    lane's measured value there, not a suppressed cell — the property this
    test exists to defend — so a zero is accepted only from `<core>`, and only
    where `<core>`'s own line at that length costs more than `<core>`'s own
    budget **in the currency the emitting arm packs under**: the pre-#1526
    content-character charge for the before arm, the rendered line for the
    after arm. `_core_pack_costs_at` returns the pair and the branch picks the
    arm's half.

    Reading the arm's own half is the whole content of the exemption, and
    #1552 inverted which half is the strict one. `_core_belief_line` truncates
    content at `hook.BELIEF_CONTENT_CHAR_CAP`, so the shipped charge
    `chars_to_tokens(len(line) + 1)` plateaus at 320 tokens for any content
    past 1,200 characters while the pre-#1526 charge `max(1, len(content) //
    4)` keeps growing: the pair is 10/27 at 40 content characters, 250/267 at
    1,000, and then 1,501/320 at 6,004, 1,792/320 at 7,170 and 4,650/320 at
    18,600. The shipped charge crosses `core_budget = 1500` nowhere; the
    pre-#1526 charge crosses it at 6,004. So `legacy <= budget < shipped` —
    what this test asserted before the rebase — is now unsatisfiable at every
    content length, and the true relation is `shipped <= budget < legacy` for
    every length from 6,004 up. It is a half-line, not a window.

    What is asserted is that *some* grid length charges one `<core>` line at or
    below `core_budget` as shipped and above it in the pre-#1526 currency, read
    off the budget rather than named. Three grid lengths satisfy it — 6,004,
    7,170 and 18,600 — which is why the edge is not left to the grid: any one
    of the three could be dropped and the `any(...)` would still pass. The edge
    itself is charged directly at `_CORE_CROSSING` and one character below it,
    so an edge that moves within a grid step is still caught. The sibling guard
    `test_the_core_section_empties_once_one_belief_exceeds_its_budget` is what
    holds the grid entry at 6,004 specifically: it asserts the *exact list* of
    lengths whose before arm is empty, so a producer whose edge moved off 6,004
    reds there.

    Those assertions are arithmetic. They re-derive the two charges from the
    budget scalar and never read the curve the producer emitted, so a producer
    that stopped emptying the crossing cell would restore the inert state above
    with the grid untouched. The asymmetry is therefore also asserted off the
    produced `<core>` curve: some cell must pack a line in the after arm and
    none in the before arm. That condition is flipped from what this test
    asserted before #1552, for the same reason the inequality is.

    **The `pool` label is falsifiable now, and it was not.** `binds_on` used to
    be checked only for membership in its own four-value set, which no false
    label can fail. `pool` means "this arm is not evidence about any budget",
    and a reader takes that from the probe the producer raised the caps to. The
    probe was `budget * SATURATION_PROBE_FACTOR`, sized for a grid topping out
    at 300 characters; at 18,600 one belief costs more than four times `ups`'s
    whole budget, so a 4x probe could not admit even one more belief and 16 of
    the 41 `pool` labels on the extended grid were false. Two assertions
    replace the membership check:

    1. Every published `{arm}_probe_budget` at the top of the grid must exceed
       the floor `SATURATION_PROBE_FACTOR` leaves under it, so the label rests
       on `_probe_budget`'s bound and not on the factor. Reverting
       `_probe_budget` to the factor makes probe equal floor and fails here.
    2. A probe too small to admit one more belief must be a producer crash and
       not a published label. That one is
       `test_a_probe_too_small_to_admit_a_belief_is_a_crash_not_a_label`
       below rather than an assertion here, because it re-runs the producer
       and stacking a second 5-second run on top of this module's 15.5-second
       fixture would put one item close to the 30-second per-test timeout CI
       pins.

    Mutations, re-measured on this branch rebased onto post-#1552
    `github/main`, with `uv run pytest tests/test_render_cost_1526.py -q`.
    Names and assertions rather than pass counts.

    * Drop 6,004 from `LENGTH_GRID` — the sibling guard
      `test_the_core_section_empties_once_one_belief_exceeds_its_budget`, on
      `assert 7170 == 6004`. It does **not** red the `any(...)` here: 7,170
      and 18,600 satisfy the condition too, which is why the grid entry is
      pinned by the first-emptied length over there and not by this test's
      existence check.
    * Raise `DEFAULT_SESSION_START_CORE_TOKEN_BUDGET` to 2,000 — this test on
      the edge assertion, `assert (2000 < 1501) is True` at 6,004 content
      characters, and the sibling on `assert 18600 == 6004`. The edge moves to
      around 8,004 characters and the published one stops being enforced.
    * Raise it to 5,000 instead — this test on the `any(...)` itself, `assert
      False`, because no grid length's pre-#1526 charge reaches 5,000. That is
      the mutation the existence check is for; the budget rise to 2,000 is
      caught one assertion later.
    * Refill an empty pack from the top candidate — `if not packed and
      candidates: packed = candidates[:1]` in `_render_core` — this test on the
      curve assertion `assert []`, because the before arm stops emptying and
      the asymmetry disappears, and the sibling guard on `assert [] == [6004,
      7170, 18600]`.
    * Revert `_probe_budget` to the old factor — `probe = max(budget * f, sub
      * f)` in `_measure` — the **fixture**, not an assertion here:
      `PoolProbeTooSmall` is raised for `agent_context` at 7,170 content
      characters (43,887 bytes held at a probe of 2,400, 45,152 at twice it),
      so 10 items error and one fails. The fail-closed render gets there before
      any assertion can, which is what it is for.
    * Publish the factor instead of the probe — `row[f"{name}_probe_budget"] =
      budget * SATURATION_PROBE_FACTOR` in `_curve` — this test on the probe
      assertion, `assert 6000 > 6000`. That is the mutation the probe assertion
      is for: the label stays correct and only its audit trail is wrong.
    * Delete the fail-closed `confirm` render in `_measure` —
      `test_a_probe_too_small_to_admit_a_belief_is_a_crash_not_a_label` on
      `DID NOT RAISE <class 'PoolProbeTooSmall'>`.
    * An oversize guard on `_render_session_start` returning `Arm(0, 0)` above
      100,000 bytes — this test on `assert 'session_start' == 'core'`.

    **The newline charge is no longer distinguishing here, and that is a
    measurement rather than an omission.** Two earlier passes fought to make
    the `+ 1` in `_core_pack_costs_at` — the character `"\\n".join` puts after
    a `<core>` line, which `_core_belief_cost` charges — load-bearing, because
    it moved the shipped charge's crossing point by one content character.
    #1552 removed that crossing: the shipped charge is capped at 320 tokens
    and never approaches the budget, so one character in it cannot change any
    comparison this file makes. The charge is kept because it is what the
    packer bills, and it is pinned by a direct equality against
    `hook._core_belief_cost` in the sibling guard instead — dropping it reds
    there, at the lengths where the two round differently.

    The before arm's half of the exemption is now the only half that fires.
    Post-#1552 every zero in the grid is a before-arm zero, because the after
    arm cannot empty at any content length; the after-arm half of the branch
    is kept for the same reason the `isinstance` check above it is, as the
    statement of what would have to be true if it ever did.

    A floor relaxed to `>= 0` across all eight lanes and all nine lengths, on
    the strength of that one cell, is what the zero branch replaces. It let
    any lane emit nothing at any length with the suite green. An oversize
    guard on `_render_session_start` that returns `Arm(0, 0)` above 100,000
    bytes turns that lane's 112,456-byte cell at 18,600 content characters
    into a published zero, and the relaxed floor reported it green; the same
    mutation now reds this test on `assert 'session_start' == 'core'`.

    An `else: assert row[arm] > 0` beside that branch is not written, because
    it could not fail. The `isinstance` check above and the zero branch beside
    it leave it only a negative int, and every value it would guard is an
    `Arm.n_bytes` summed from lengths, so replacing its body with `pass` left
    the file green.
    """
    fig = producer_figures
    m = _producer_module()
    seen: set[str] = set()
    for lane in _producer_lanes():
        curve = fig[f"{lane}_curve"]
        assert set(curve) == {str(c) for c in fig["lengths"]}, lane
        for chars, row in curve.items():
            for arm in ("before", "after"):
                assert isinstance(row[arm], int), (lane, chars, arm, row[arm])
                if row[arm] == 0:
                    assert lane == "core", (
                        f"{lane}: the {arm} arm emitted no bytes at {chars} "
                        f"content chars; only <core> has a measured zero"
                    )
                    legacy_cost, shipped_cost = _core_pack_costs_at(int(chars))
                    charged = legacy_cost if arm == "before" else shipped_cost
                    assert charged > fig["core_budget"], (
                        f"<core> emitted nothing at {chars} content chars in "
                        f"the {arm} arm, but that arm charges {charged} for "
                        f"one line against a budget of {fig['core_budget']}, "
                        f"so a line fits in the currency the arm packs under "
                        f"(the other arm charges "
                        f"{shipped_cost if arm == 'before' else legacy_cost})"
                    )
                seen.add(row[f"{arm}_binds_on"])
            assert row["pool_equality"] == (
                row["before_binds_on"] == "pool"
                and row["after_binds_on"] == "pool"
            ), (lane, chars)
    costs = {int(c): _core_pack_costs_at(int(c)) for c in fig["lengths"]}
    assert any(
        shipped <= fig["core_budget"] < legacy
        for legacy, shipped in costs.values()
    ), (
        "no grid length charges one <core> line at or below "
        f"{fig['core_budget']} tokens as shipped and above it in the pre-#1526 "
        "currency, so the two accountings agree at every length and the "
        f"per-arm half of the exemption above decides nothing: {costs}"
    )
    # ...and the edge that condition turns on is the edge this branch
    # publishes. Asserted directly on either side of it, because the grid steps
    # 1,000 -> 6,004 -> 7,170: an edge that moves within a step passes every
    # assertion written over `LENGTH_GRID` while the published 6,004 stops
    # being true.
    for chars, past in ((_CORE_CROSSING - 1, False), (_CORE_CROSSING, True)):
        legacy, shipped = _core_pack_costs_at(chars)
        assert (shipped <= fig["core_budget"] < legacy) is past, (
            f"at {chars} content characters one <core> line is charged "
            f"{legacy} in the pre-#1526 currency and {shipped} as shipped "
            f"against a budget of {fig['core_budget']}, so the published "
            f"crossing edge {_CORE_CROSSING} is not the edge the two charges "
            "enforce"
        )
    core_curve = fig["core_curve"]
    crossing = [
        int(c)
        for c in fig["lengths"]
        if core_curve[str(c)]["after"] > 0 and core_curve[str(c)]["before"] == 0
    ]
    assert crossing, (
        "no <core> cell packs a line in the after arm and none in the before "
        "arm, so the produced curve no longer shows the asymmetry the "
        "exemption above reads: "
        f"{ {int(c): core_curve[str(c)] for c in fig['lengths']} }"
    )
    assert "l25_subbudget" in seen, (
        "no arm in the grid ended on the L2.5 sub-budget, so the "
        "multi-budget binding probe is not exercised by this run"
    )
    assert "token_budget" in seen, seen
    assert seen <= {"token_budget", "l25_subbudget", "both", "pool"}, seen
    # Falsifiability 1 for the `pool` label. At the top of the grid one belief
    # costs more than four times `ups`'s whole budget, so a probe that is a
    # multiple of the cap cannot admit even one more belief and every label it
    # produces is arithmetic rather than evidence. Every published probe there
    # must therefore be above the floor `SATURATION_PROBE_FACTOR` leaves under
    # `_probe_budget`: reverting the probe to the factor makes the two equal.
    assert m.SATURATION_PROBE_FACTOR > 1
    top = str(max(int(c) for c in fig["lengths"]))
    for lane in _producer_lanes():
        row = fig[f"{lane}_curve"][top]
        floor = m.SATURATION_PROBE_FACTOR * max(
            fig[f"{lane}_budget"], fig["l25_token_subbudget"],
        )
        for arm in ("before", "after"):
            probe = row[f"{arm}_probe_budget"]
            assert isinstance(probe, int), (lane, arm, probe)
            assert probe > floor, (
                f"{lane}'s {arm} arm at {top} content chars was probed at "
                f"{probe}, which is no more than the {floor} floor "
                f"`SATURATION_PROBE_FACTOR` puts under `_probe_budget`, so "
                f"its `binds_on` label rests on a multiple of the cap and not "
                f"on a bound read off the store"
            )


def test_a_probe_too_small_to_admit_a_belief_is_a_crash_not_a_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second half of the `pool` repair: the label fails closed.

    `pool` is defined in the producer as "this arm is not evidence about any
    budget". Everything downstream of that label — the `pool_equality` legend,
    the #1546 finding that no budget can bind on `session_start` — is only as
    good as the probe the caps were raised to, and the probe used to be
    `budget * SATURATION_PROBE_FACTOR`. At 18,600 content characters one
    belief costs more than four times `ups`'s whole budget, so that probe
    could not admit one more belief anywhere in the cell and returned `pool`
    for arms a larger probe moves: 16 of 41 labels were false and the emitted
    figures showed nothing, because the probe behind them was published
    nowhere.

    `_probe_budget` is stubbed to 0, which drops `_measure` back onto exactly
    that old probe — `max(0, budget * 4, sub * 4)` — and the producer must
    raise rather than emit the label. Deleting the `confirm` render in
    `_measure` reds this on `DID NOT RAISE`.

    One grid length, so the run is about 5 seconds. `producer_figures` is
    deliberately not requested: this test needs its own producer run at its
    own monkeypatched module state, and requesting the module fixture as well
    would charge it 15.5 seconds it cannot use.
    """
    m = _producer_module()
    for name in [k for k in os.environ if k.startswith("AELFRICE_")]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(m, "_probe_budget", lambda store: 0)
    with pytest.raises(m.PoolProbeTooSmall) as excinfo:  # type: ignore[attr-defined]
        m.figures(lengths=(18600,))  # type: ignore[attr-defined]
    # The exception carries the cell, so the crash names what to look at
    # rather than only that something moved.
    assert excinfo.value.chars == 18600, excinfo.value
    assert excinfo.value.moved_bytes != excinfo.value.arm_bytes, excinfo.value


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
# #1547 AC4 — the producer that can see the charge-vs-emit defect
#
# These assert on the same producer and the same `producer_figures` run as the
# block above, which is why they live in this file rather than beside
# `tests/test_envelope_dedupe_1547.py`: a second module-scoped fixture would
# pay for that run twice to assert on the same numbers. What it costs, and how
# much of the per-test timeout that leaves, is published once — in the
# `producer_figures` docstring above — so this file carries one figure for it.
# ---------------------------------------------------------------------------


def test_the_snapshot_arm_is_a_second_corpus_and_the_control_holds_none(
    producer_figures: dict[str, object],
) -> None:
    """The arm is one field's difference between two otherwise-equal stores.

    The control corpus is the one this producer shipped with: every belief on
    the `RETENTION_UNKNOWN` default, which `compress_for_retrieval` maps to
    `verbatim`, so every charge equals its emission by construction. That zero
    is asserted, not assumed — it is the property that made a 150x undercharge
    invisible here, and a producer that quietly grew a snapshot belief into its
    control would stop being a control without saying so.

    `snapshot_unlocked` is strictly smaller than `snapshot`, because the
    stride lands on belief 0, which `LOCK_EVERY` also locks, and a locked
    snapshot renders verbatim — locks override retention class. Reading both
    counts off the store is what distinguishes "the class was written" from
    "the generator meant to write it".

    The same question is then asked of the *text* change, which is the half
    that had no assertion: `sentence_headline` counts the beliefs carrying a
    sentence boundary at or before `MAX_HEADLINE_CHARS`, the condition
    `compression._headline` keys on. A prose corpus built without the boundaries
    is byte-identical to the control at every length and satisfies every other
    assertion here, so without this count the middle column could quietly
    collapse into the column it exists to be compared against.
    """
    fig = producer_figures
    control = fig["corpus_shape"]
    arm = fig["snapshot_corpus_shape"]
    assert control["snapshot"] == 0, control
    assert control["snapshot_unlocked"] == 0, control
    assert arm["snapshot"] > 0, arm
    assert 0 < arm["snapshot_unlocked"] < arm["snapshot"], arm
    # The middle corpus is the attribution control: sentence-bearing text, no
    # class, so it must hold no snapshot belief either.
    prose = fig["prose_corpus_shape"]
    assert prose["snapshot"] == 0, prose
    # Everything except the class is identical across the three corpora.
    for other in (prose, arm):
        assert other["locked"] == control["locked"], (other, control)
        assert other["speculative"] == control["speculative"], (other, control)
    # The text change is real, and the control does not carry it. All three
    # shapes below are read at the same grid length, so the comparison is not
    # a comparison of two lengths.
    top_control = fig["control_corpus_shape_at_arm_top"]
    assert top_control["sentence_headline"] == 0, top_control
    assert prose["sentence_headline"] > 0, prose
    assert arm["sentence_headline"] > 0, arm
    assert prose["sentence_headline"] == arm["sentence_headline"], (prose, arm)


def test_a_snapshot_belief_is_charged_less_than_the_lane_emits(
    producer_figures: dict[str, object],
) -> None:
    """The defect, at every grid length, in the producer's own numbers.

    This table is `agent_context`'s and no other lane's. `_emitted_chars`
    renders through `hook._split_belief_lines`, which is the shape `ups`,
    `first_prompt` and `agent_context` all emit, but #1551 and #1552 wired
    `hook._ups_belief_line_cost` onto the first two as a `belief_cost_fn`, so
    they short-circuit `retrieve_with_tiers._cost` before compression is
    consulted and charge exactly what they emit. `agent_context` passes no
    cost function, so it still pays `_charged_tokens` — the compressed branch —
    for a line it renders in full. See `SNAPSHOT_ARM_HEADLINE_LANE`.

    `fact` and `unknown` render verbatim, and the control they provide has two
    halves now. Below `hook.BELIEF_CONTENT_CHAR_CAP` they charge exactly what
    they emit — without that a table where every ratio was large would be
    evidence of a broken measurement rather than of a defect. Above it the
    renderer truncates and the charge does not, so they **over**charge: 4,663
    tokens against 317 emitted at 18,600 content characters. That is the
    mirror of the defect this file is about and it is asserted rather than
    tolerated, because it is what a capped renderer and an uncapped cost
    function do to each other.

    The overcharge is asserted through the two facts that produce it rather
    than through `charged > emitted` at every length above the cap: the emitted
    side is flat above the cap and the charged side is strictly increasing.
    Between about 1,201 and 1,270 content characters the growing charge has not
    yet passed the capped emission, and no grid length falls in that band, so
    a direct inequality written over the grid would be asserting something the
    grid cannot see the edge of.

    `snapshot` must charge less than it emits wherever the headline strategy
    fires, and the ratio must grow with belief length — up to a ceiling.
    **That ceiling is the #1552 finding.** The headline charge is a fixed 44
    tokens and the emitted side is capped at 1,265 characters, so the ratio
    plateaus at 7.2x and cannot grow past it however long the belief is. It is
    asserted as an equality across every above-cap grid length: an uncapped
    renderer would make those three ratios differ, and 18,600 characters would
    read 106.0x, which is what this table published before the rebase.

    The order-of-magnitude claim therefore no longer belongs to `snapshot`, and
    the threshold is not moved down to fit it. It belongs to `transient`, whose
    stub strategy charges 25 tokens against the same 317 — 12.7x, over the same
    unchanged 10.0. Asserting it on `snapshot` at 7.2x would be publishing a
    ratio from a code path the cap has closed.

    The two shortest grid points are the arm's inert control: below
    `SENTENCE_CHARS` a belief carries no sentence boundary, `_headline`
    returns it unchanged, and the ratio is 1.0 for every class that is not
    `transient`. A grid that stopped there would measure nothing, which is
    what the pre-#1547 grid did.

    Mutation, measured with `uv run pytest tests/test_render_cost_1526.py -q`:
    removing `_cap_belief_content` from `_belief_element_line` reds this test
    on `assert 3 == 1`, the emitted side going back to
    `{6004: 6056, 7170: 7222, 18600: 18652}` characters. The ceiling assertion
    below pins the same cap on the `snapshot` column and is not separately
    falsifiable — the only mutation that moves it is the one that moves the
    emission, and that reaches the verbatim assertion first.
    """
    fig = producer_figures
    m = _producer_module()
    assert fig["compression"] is True, (
        "the charge this measures only exists with type-aware compression on"
    )
    table = fig["undercharge"]
    lengths = [int(c) for c in fig["lengths"]]
    cap = BELIEF_CONTENT_CHAR_CAP
    for chars in lengths:
        row = table[str(chars)]
        for verbatim in ("fact", "unknown"):
            cell = row[verbatim]
            assert cell["strategy"] == "verbatim", (chars, verbatim)
            if chars <= cap:
                assert cell["ratio"] == 1.0, (chars, cell)
                assert cell["charged_tokens"] == cell["emitted_tokens"], (
                    chars, cell,
                )
        snap = row["snapshot"]
        if chars < m.SENTENCE_CHARS:
            assert snap["strategy"] == "verbatim", (chars, snap)
            assert snap["ratio"] == 1.0, (chars, snap)
        else:
            assert snap["strategy"] == "headline", (chars, snap)
            assert snap["ratio"] > 1.0, (chars, snap)
            assert snap["charged_tokens"] < snap["emitted_tokens"], (chars, snap)
    # Above the cap the renderer stops growing and the verbatim charge does
    # not, which is what turns the control into an overcharge. Both halves are
    # asserted, because either one alone is consistent with no cap at all.
    above = [c for c in lengths if c > cap]
    assert len(above) > 1, (cap, lengths)
    emitted_above = {table[str(c)]["fact"]["emitted_chars"] for c in above}
    assert len(emitted_above) == 1, (
        "the emitted side still grows above "
        f"BELIEF_CONTENT_CHAR_CAP = {cap}: "
        f"{ {c: table[str(c)]['fact']['emitted_chars'] for c in above} }"
    )
    charged_above = [table[str(c)]["fact"]["charged_tokens"] for c in above]
    assert charged_above == sorted(charged_above) and len(
        set(charged_above)
    ) == len(charged_above), dict(zip(above, charged_above))
    top = max(lengths)
    for verbatim in ("fact", "unknown"):
        cell = table[str(top)][verbatim]
        assert cell["charged_tokens"] > cell["emitted_tokens"], (top, cell)
    compressing = [c for c in lengths if c >= m.SENTENCE_CHARS]
    ratios = [table[str(c)]["snapshot"]["ratio"] for c in compressing]
    assert ratios == sorted(ratios), dict(zip(compressing, ratios))
    # The ceiling. `snapshot`'s charge is a fixed-size headline and its
    # emission is capped, so the ratio stops growing once the cap binds; every
    # above-cap length must report the same one. Removing the cap makes these
    # three differ and 18,600 characters read 106.0x again.
    ceiling = {table[str(c)]["snapshot"]["ratio"] for c in above}
    assert len(ceiling) == 1, (
        "the snapshot undercharge still grows above "
        f"BELIEF_CONTENT_CHAR_CAP = {cap}, so it is not bounded by the cap: "
        f"{ {c: table[str(c)]['snapshot']['ratio'] for c in above} }"
    )
    assert ceiling.pop() > 1.0, table[str(top)]["snapshot"]
    # The grid reaches a range where the undercharge is an order of magnitude.
    # Not on `snapshot`, which the cap holds at its ceiling, but on
    # `transient`: the threshold is the same 10.0 read against the class that
    # still reaches it.
    worst = max(
        (table[str(c)][cls]["ratio"], c, cls)
        for c in compressing
        for cls in ("snapshot", "transient")
    )
    assert worst[0] > 10.0, {
        c: {
            cls: table[str(c)][cls]["ratio"]
            for cls in ("snapshot", "transient")
        }
        for c in compressing
    }


def test_the_snapshot_arm_admits_beliefs_the_control_cannot_afford(
    producer_figures: dict[str, object],
) -> None:
    """A pack that believes it is under budget while emitting more than it charged.

    **The three lanes no longer say the same thing, and that is the finding.**
    `ups` and `first_prompt` pass `hook._ups_belief_line_cost` as their
    `belief_cost_fn` (#1551, #1552), which is the line they emit, so their pack
    charge equals their pack emission in every retention class at every arm
    length and the retention class buys them nothing: same items, same bytes,
    ratio 1.0. That is asserted here as an equality rather than dropped,
    because it is the thing that would regress if the cost function were ever
    unwired — a producer charging `_charged_tokens` on those lanes would
    publish a discount from a branch they short-circuit past.

    `agent_context` is where the gap survives: it passes no cost function, so
    its pack still charges the compressed form. The class changes what it
    admits at exactly one arm length, 300 characters, and that length is
    selected by a stated rule — the largest arm length at which the snapshot
    corpus admits a belief the prose corpus does not — rather than named. At
    7,170 and 18,600 the lane is lock-starved: its 600-token budget is spent by
    six user locks whose content `BELIEF_CONTENT_CHAR_CAP` exempts, before the
    pack loop reaches a candidate, so all three corpora return the same six
    locks, there is no non-locked hit to sum over and the ratio is `None`. That
    starvation is asserted too, because those are the cells `_flat_1547_keys`
    publishes and a reader of the emitted figures needs to know the zeros are
    measured.

    The comparison is prose against snapshot, not control against snapshot.
    Those two corpora differ in exactly one field; the control differs in two,
    and attributing a ratio to the class while the text also moved is the
    confound this arm was built to avoid. The control is asserted where the
    attribution is actually made — at the length the class moves the pack, the
    text change must have moved nothing — rather than at the top of the grid,
    where post-#1552 there is no class effect to attribute.

    All three sides go through `_measure`, so both budgets are varied on each,
    and the `binds_on` each reports is the #1546 property applied to the arm.

    Mutation, measured with `uv run pytest tests/test_render_cost_1526.py -q`:
    dropping `"ups"` from `_lane_belief_cost`'s table — putting that lane back
    on the compressed charge the producer used to model — reds this test on
    `assert 20 == 17` at 300 content characters, the snapshot pack admitting
    three beliefs the prose pack does not because it is charging them a
    headline price. That is the producer defect #1559 fixed, and it is what the
    equalities above exist to keep fixed.
    """
    fig = producer_figures
    m = _producer_module()
    arm = fig["snapshot_arm"]
    # Named literally, not read back off `SNAPSHOT_ARM_LANES`. Comparing the
    # produced dict against the constant that produced it cannot fail:
    # `snapshot_arm` builds its dict by iterating that constant and nothing
    # mutates it afterwards, so `set(arm) == set(SNAPSHOT_ARM_LANES)` is an
    # invariant of that function under every mutation of every file. Dropping
    # `first_prompt` from the constant drops the composed envelope — the
    # intersection of #1547 AC4's two deliverables — out of the arm entirely and
    # keeps the equality true, and no other edit can make it false. What the
    # lane set can be held to is something the arm is not generated from: every
    # lane measured here must be one this module also publishes a `LANES` curve
    # for, or the arm reports on a lane no #1526 figure covers.
    assert "ups" in arm, sorted(arm)
    assert "first_prompt" in arm, sorted(arm)
    assert "agent_context" in arm, sorted(arm)
    assert m.SNAPSHOT_ARM_HEADLINE_LANE in arm, (
        m.SNAPSHOT_ARM_HEADLINE_LANE, sorted(arm),
    )
    assert set(arm) <= set(m.LANES), (sorted(arm), sorted(m.LANES))
    lengths = sorted(int(c) for c in fig["snapshot_arm_lengths"])
    for lane, rows in arm.items():
        for c in lengths:
            row = rows[str(c)]
            for side in (
                "control_binds_on", "prose_binds_on", "snapshot_binds_on",
            ):
                assert row[side] in {
                    "token_budget", "l25_subbudget", "both", "pool",
                }, (lane, c, side, row[side])
        # The inert control length: below `SENTENCE_CHARS` no belief carries a
        # sentence boundary, so the class has nothing to shorten and all three
        # corpora must render byte-identically.
        short = rows[str(lengths[0])]
        assert short["control_bytes"] == short["prose_bytes"] == short[
            "snapshot_bytes"
        ], (lane, short)
        assert short["snapshot_pack_ratio"] == 1.0, (lane, short)
    # The middle corpus has to move something somewhere: a corpus that renders
    # byte-identically to the control everywhere *is* the control, and the
    # attribution below would be a comparison of a corpus with itself. It is
    # asserted over the arm and not per lane because it is a property of the
    # corpus: `ups` and `first_prompt` differ at 300 content characters, where
    # the headline the sentence boundary makes available is longer than the
    # hard truncation it replaces, while `agent_context` renders the two
    # identically at every arm length — which is what makes its own class
    # effect at 300 attributable to the class alone.
    assert [
        (lane, c)
        for lane, rows in arm.items()
        for c in lengths
        if rows[str(c)]["prose_bytes"] != rows[str(c)]["control_bytes"]
    ], {
        lane: {c: (rows[str(c)]["control_bytes"], rows[str(c)]["prose_bytes"])
               for c in lengths}
        for lane, rows in arm.items()
    }
    # The two lanes #1552 closed. The class changes nothing they admit, and
    # their charge is their emission, at every arm length.
    for lane in ("ups", "first_prompt"):
        for c in lengths:
            row = arm[lane][str(c)]
            assert row["snapshot_items"] == row["prose_items"], (lane, c, row)
            assert row["snapshot_bytes"] == row["prose_bytes"], (lane, c, row)
            assert row["snapshot_pack_ratio"] == 1.0, (lane, c, row)
            assert row["snapshot_charged_tokens"] == row[
                "snapshot_emitted_tokens"
            ], (lane, c, row)
    # The lane that still charges through compression. The arm length is
    # selected by a rule read off the produced figures — the largest at which
    # the class admits a belief the prose corpus does not — so nothing here is
    # a length chosen for its number.
    head = m.SNAPSHOT_ARM_HEADLINE_LANE
    rows = arm[head]
    moved = [
        c for c in lengths
        if rows[str(c)]["snapshot_items"] > rows[str(c)]["prose_items"]
    ]
    assert moved, (
        f"the retention class changes nothing {head} admits at any arm "
        f"length, so the arm measures no class effect on the one lane that "
        f"still charges the compressed form: "
        f"{ {c: (rows[str(c)]['prose_items'], rows[str(c)]['snapshot_items']) for c in lengths} }"
    )
    row = rows[str(max(moved))]
    assert row["snapshot_bytes"] > row["prose_bytes"], (head, row)
    assert row["snapshot_unlocked_hits"] > row["prose_unlocked_hits"], (
        head, row,
    )
    assert row["snapshot_charged_tokens"] < row["snapshot_emitted_tokens"], (
        head, row,
    )
    assert row["snapshot_pack_ratio"] > 1.0, (head, row)
    # ...and the text change on its own moved nothing there, which is what
    # lets the line above be attributed to the class rather than to the
    # sentence boundaries the class needs.
    assert row["prose_bytes"] == row["control_bytes"], (head, row)
    # At the top of the grid this lane is lock-starved rather than measured.
    # Its six user locks are exempt from `BELIEF_CONTENT_CHAR_CAP` and spend
    # the 600-token budget before the pack loop reaches a candidate, so all
    # three corpora return the same six and there is nothing to charge. Those
    # are the cells `_flat_1547_keys` publishes, so the zeros are pinned here.
    top = rows[str(lengths[-1])]
    assert top["snapshot_items"] == top["prose_items"] == top[
        "control_items"
    ], (head, top)
    assert top["snapshot_unlocked_hits"] == 0, (head, top)
    assert top["snapshot_charged_tokens"] == 0, (head, top)
    assert top["snapshot_pack_ratio"] is None, (head, top)


_1526_COST_NAME_SUFFIXES = ("_tokens", "_cost")


def _scan_1526_cost_functions() -> set[tuple[str, str]]:
    """Every `#1526` cost function in the shipped tree, as `(module, name)`.

    The rule, stated so a reader can apply it by hand: a module-level function
    in `src/aelfrice` whose name ends in `_tokens` or `_cost` and whose
    docstring names #1526. Read off the AST rather than by import, so a
    function nothing here happens to import is still in the set, and so the
    scan cannot be satisfied by a name the producer defines for itself.
    """
    src = Path(aelfrice.__file__).resolve().parent
    found: set[tuple[str, str]] = set()
    for path in sorted(src.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.endswith(_1526_COST_NAME_SUFFIXES):
                continue
            if "#1526" in (ast.get_docstring(node) or ""):
                found.add((path.stem, node.name))
    return found


def test_every_1526_cost_function_is_rebound_or_named_as_an_exception() -> None:
    """A before arm that misses one cost function measures a hybrid.

    The producer's before arm rebinds the pre-#1526 bodies onto module
    globals. Nothing checked that the set it rebinds was the set the lanes
    reach, and the `first_prompt` lane shipped without
    `hook._core_belief_cost` in it: that lane packs `<core>` through
    `_build_session_start_subblock`, which has no `cost_fn` parameter to pass
    a legacy body into, so every composed before cell mixed pre-#1526
    retrieval cost with post-#1526 `<core>` cost. Measured at 92 content
    characters, the before arm read 15,861 bytes against a true legacy 19,999
    and the change was published as -15.7% where it is -33.2%.

    So the set is scanned out of the shipped tree and must partition exactly
    into the names the producer rebinds and the names it declares it does not,
    each with a reason. A sixth cost function added to #1526's set lands in
    neither table and fails here, which is the check that was missing.
    """
    import importlib

    m = _producer_module()
    scanned = _scan_1526_cost_functions()
    rebound = {(stem, attr) for stem, attr, _body in m.LEGACY_COST_REBINDS}
    excused = {(stem, attr) for stem, attr, _why in m.LEGACY_COST_NOT_REBOUND}
    assert scanned, "the scan matched no #1526 cost function at all"
    assert not rebound & excused, sorted(rebound & excused)
    assert scanned == rebound | excused, {
        "scanned but in neither table": sorted(scanned - rebound - excused),
        "tabled but not in the tree": sorted((rebound | excused) - scanned),
    }
    # Each rebind must name a live attribute, or it would restore `None` on
    # exit and leave the shipped tree broken for every later test.
    for stem, attr, body in m.LEGACY_COST_REBINDS:
        mod = importlib.import_module(f"aelfrice.{stem}")
        assert callable(getattr(mod, attr)), (stem, attr)
        assert callable(body), (stem, attr)
    # An exception without a reason is a tolerance.
    for stem, attr, why in m.LEGACY_COST_NOT_REBOUND:
        assert len(why) > 60, (stem, attr, why)


def test_the_before_arm_rebinds_the_core_cost_the_composed_lane_packs_with(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rebind reaches `<core>`, which is what the table check cannot show.

    `test_every_1526_cost_function_is_rebound_or_named_as_an_exception` pins
    the membership of the set; this pins that membership does anything.
    `_pack_core_candidates` resolves its default `cost_fn` to
    `hook._core_belief_cost` per call, so rebinding the module global changes
    what the session-start sub-block packs — and the legacy body charges
    content where the shipped one charges the rendered line, so the legacy arm
    must fit strictly more `<core>` beliefs into the same unchanged budget.

    `corr="` counts the `<core>` lines specifically: `_core_belief_line`
    renders that attribute and the `<locked>` section renders `lock="`
    instead, so the count separates the section the budget governs from the
    one #379 exempts.
    """
    m = _producer_module()
    for name in [k for k in os.environ if k.startswith("AELFRICE_")]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    store = m._synthetic_store(  # type: ignore[attr-defined]
        tmp_path / "core.db", m.CORPUS_MEDIAN_CHARS,  # type: ignore[attr-defined]
    )
    try:
        shipped = m._session_start_block(store)  # type: ignore[attr-defined]
        with m._legacy_accounting():  # type: ignore[attr-defined]
            legacy = m._session_start_block(store)  # type: ignore[attr-defined]
        after = m._session_start_block(store)  # type: ignore[attr-defined]
    finally:
        store.close()
    assert shipped.count('corr="') > 0, "the corpus packed no <core> belief"
    assert legacy.count('corr="') > shipped.count('corr="'), (
        legacy.count('corr="'), shipped.count('corr="'),
    )
    assert len(legacy) > len(shipped)
    # The context restores what it rebound; a before arm that leaked would
    # make every test that ran after it a measurement of the wrong tree.
    assert after == shipped


def test_the_composed_lane_renders_both_halves_of_the_first_prompt(
    producer_figures: dict[str, object],
) -> None:
    """The lane this producer did not have, and the #1546 property on it.

    `first_prompt` composes what the other lanes render one at a time, so it
    must be strictly larger than the per-turn lane alone at the same budget on
    the same corpus, and the difference must be the session-start sub-block.

    Because it carries `<core>` as well as the per-turn pack, and #1526
    changed the accounting of both, the change it measures must be strictly
    larger than the per-turn lane's. That is the assertion that fails on a
    before arm which rebinds the retrieval cost functions and not
    `hook._core_belief_cost`: the composed delta collapses to exactly the
    per-turn delta, because the `<core>` half renders identically in both
    arms.

    `first_prompt_recent_work_chars` is read out of the sub-block this lane
    renders, not from a second call to `hook._build_recent_work_subblock`, so
    asserting it is zero constrains the lane. The producer builds the
    sub-block on a non-git cwd on purpose: that section resolves from git
    plumbing and would otherwise make every composed byte count a function of
    the branch name of whatever checkout the producer ran in.

    The #1546 property — vary both budgets, name the cap that ended the arm —
    is asserted on this lane by name rather than left to the all-lanes sweep,
    because a lane registered in `_RENDERERS` but not in `LANES` would pass
    that sweep by not being in it.
    """
    fig = producer_figures
    m = _producer_module()
    assert "first_prompt" in m.LANES, m.LANES
    assert fig["first_prompt_recent_work_chars"] == 0, fig[
        "first_prompt_recent_work_chars"
    ]
    assert fig["first_prompt_headline_chars"] == fig["ups_headline_chars"]
    assert fig["first_prompt_budget"] == fig["ups_budget"]
    for arm in ("before", "after"):
        assert fig[f"first_prompt_bytes_{arm}"] > fig[f"ups_bytes_{arm}"], (
            "the composed envelope must be larger than its per-turn half alone"
        )
    composed_delta = fig["first_prompt_bytes_before"] - fig[
        "first_prompt_bytes_after"
    ]
    per_turn_delta = fig["ups_bytes_before"] - fig["ups_bytes_after"]
    assert composed_delta > per_turn_delta, (
        "the composed lane's before arm did not carry the <core> accounting: "
        f"{composed_delta} against the per-turn lane's {per_turn_delta}"
    )
    curve = fig["first_prompt_curve"]
    assert set(curve) == {str(c) for c in fig["lengths"]}, set(curve)
    binds = {
        curve[chars][f"{arm}_binds_on"]
        for chars in curve
        for arm in ("before", "after")
    }
    assert binds <= {"token_budget", "l25_subbudget", "both", "pool"}, binds
    assert binds - {"pool"}, (
        "every cell of the composed lane ended on the candidate pool, so no "
        "budget was measured on it"
    )


def _init_probe_repo(path: Path) -> None:
    """A one-commit git repository, so `<recent-work>` has something to render.

    A throwaway repo rather than this checkout: `_resolve_branch` reads git
    plumbing under the cwd it is handed, and pointing it at the tree the tests
    run in would make the assertion below depend on the tester's branch name
    and on whether the source was unpacked from a checkout at all.
    """
    import subprocess

    def run(*args: str) -> None:
        subprocess.run(
            ["git", *args], cwd=str(path), check=True,
            capture_output=True, timeout=30,
        )

    path.mkdir(parents=True, exist_ok=True)
    run("init", "-q", "-b", "main")
    run("config", "user.email", "t@t")
    run("config", "user.name", "t")
    run("config", "commit.gpgsign", "false")
    (path / "seed.txt").write_text("seed", encoding="utf-8")
    run("add", "seed.txt")
    run("commit", "-q", "-m", "feat: seed the probe repo")


@pytest.mark.timeout(30)
def test_the_recent_work_reader_finds_the_section_the_lane_suppresses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The zero in `first_prompt_recent_work_chars` is a suppression, not a stub.

    `test_the_composed_lane_renders_both_halves_of_the_first_prompt` asserts
    that figure is zero. On its own that assertion cannot fail for the right
    reason: `_recent_work_chars_in` replaced by `return 0` keeps it green, and
    so does the cwd mutation the reader exists to catch, because a reader that
    always returns zero cannot see the lane move. What is missing is a case
    where the answer must not be zero.

    So both directions are asserted against the same store. The lane's own
    block is built on a non-git cwd and must read 0; the same builder handed a
    git cwd emits the section, and the reader must find it and report its
    tag-to-tag span exactly — cross-checked against `_build_recent_work_subblock`
    on that cwd, which renders the identical text standalone. A reader that
    returns a constant fails the second assertion; a lane that starts resolving
    a real cwd fails the first.
    """
    from aelfrice import hook

    m = _producer_module()
    for name in [k for k in os.environ if k.startswith("AELFRICE_")]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    repo = tmp_path / "probe-repo"
    _init_probe_repo(repo)
    store = m._synthetic_store(  # type: ignore[attr-defined]
        tmp_path / "recent.db", m.CORPUS_MEDIAN_CHARS,  # type: ignore[attr-defined]
    )
    try:
        hermetic = m._session_start_block(store)  # type: ignore[attr-defined]
        carrying = hook._build_session_start_subblock(store, cwd=repo)
    finally:
        store.close()
    # The lane's own cwd carries no git, so the section is absent.
    assert hook.RECENT_WORK_OPEN_TAG not in hermetic
    assert m._recent_work_chars_in(hermetic) == 0  # type: ignore[attr-defined]
    # The same builder on a git cwd emits it, and the reader must find it.
    section = hook._build_recent_work_subblock(cwd=repo)
    assert section.startswith(hook.RECENT_WORK_OPEN_TAG), section[:40]
    assert "<branch>main</branch>" in section, section
    chars = m._recent_work_chars_in(carrying)  # type: ignore[attr-defined]
    assert chars == len(section), (chars, len(section))
    assert chars > 0
    assert len(carrying) > len(hermetic)


def test_the_envelope_dedupe_is_measurable_only_on_the_composed_lane(
    producer_figures: dict[str, object],
) -> None:
    """#1547's AC2 shipped with no producer able to measure it. This is it.

    The dedupe fires where a belief appears in both halves of one envelope, so
    a lane that renders one half in isolation has no repeat to find. The
    measured effect is a byte reduction and a positive count of repeated ids:
    a zero count would mean the corpus never puts the same belief in both
    halves, and then the byte figure would be a tautology rather than a
    measurement.
    """
    d = producer_figures["dedupe"]
    assert d["hits"] > 0, d
    assert d["repeated_ids"] > 0, d
    assert d["session_start_chars"] > 0, d
    assert d["bytes_after"] < d["bytes_before"], d
    assert d["pct"] < 0, d


def test_the_core_section_empties_once_one_belief_exceeds_its_budget(
    producer_figures: dict[str, object],
) -> None:
    """A zero in the grid is a measurement, and this is what produces it.

    Two facts, and #1552 moved the zeros from the first to the second.

    **1. The shipped arm cannot empty, at any content length.**
    `_core_belief_line` truncates content at `hook.BELIEF_CONTENT_CHAR_CAP`,
    so `hook._core_belief_cost` — the function `_pack_core_candidates` charges
    with — plateaus at 320 tokens for anything past 1,200 characters, measured
    flat out to 120,000. The pathological case is content that is entirely
    angle brackets, which `_escape_for_hook_block` expands fourfold: that costs
    1,220. Both are under `DEFAULT_SESSION_START_CORE_TOKEN_BUDGET = 1500`, so
    no belief of any length or content can make the packer skip every
    candidate, and the `<core>` after-arm curve bottoms out at 5,120 bytes
    rather than at zero.

    This is *stronger* than the guard it replaces, which asserted that the
    shipped arm does empty somewhere and re-derived the threshold it emptied
    at. It fails the moment the cap is raised or removed — uncapped, one line
    at 18,600 content characters costs 4,667 against the same budget — and it
    fails if the core budget drops below the 1,220-token pathological bound.

    **2. The before arm still empties, and that is where the grid's zeros
    are.** The pre-#1526 charge `max(1, len(content) // 4)` is uncapped and
    crosses the budget at `_CORE_CROSSING`, so the before arm packs none of 300
    candidates at 6,004 content characters and above. The emptied set is
    asserted as an exact list against the lengths whose pre-#1526 charge
    exceeds the budget, and the first member of that list is asserted to be
    `_CORE_CROSSING` itself. That is what holds the 6,004 entry in
    `LENGTH_GRID`: drop it and the list starts at 7,170.

    The line is charged with the newline `"\\n".join` puts after it, because
    that is what `_core_belief_cost` charges. It is no longer the crossing
    point that pins the character — the shipped charge is capped and crosses
    nothing — so the transcription is pinned directly instead: `charged()` must
    equal `hook._core_belief_cost` at every grid length. The two differ by a
    token at 150 content characters, where the line is 216 characters and the
    newline rounds it up from 54 to 55.

    Mutations, re-measured on this branch rebased onto post-#1552
    `github/main`, with `uv run pytest tests/test_render_cost_1526.py -q`:

    * Drop the `+ 1` from `charged()` — this test, on `assert 54 == 55` at 150
      content characters.
    * Remove `_cap_belief_content` from `_core_belief_line` — this test, on the
      shipped-arm bound `assert 30017 < 1500`, which is one line at the top of
      the sweep.
    * Raise `DEFAULT_SESSION_START_CORE_TOKEN_BUDGET` to 2,000 — this test, on
      `assert 18600 == 6004`: the crossing moves to about 8,004 characters and
      6,004 stops being the edge. The emptied list itself still agrees with the
      re-derived one, which is why the first member is asserted separately.
    * Drop 6,004 from `LENGTH_GRID` — this test, on
      `assert 7170 == 6004`, the first emptied length.
    * Refill an empty `<core>` pack from the top candidate — `if not packed
      and candidates: packed = candidates[:1]` in `_render_core` — this test
      on `assert [] == [6004, 7170, 18600]`, and the sibling on its curve
      assertion.
    """
    fig = producer_figures
    m = _producer_module()
    curve = fig["core_curve"]
    lengths = sorted(int(c) for c in fig["lengths"])
    budget = fig["core_budget"]

    def charged(content_chars: int) -> int:
        """The line the section emits, plus the newline that joins it on.

        A cost function is not consulted: the claim is about what `<core>`
        emits, so `_core_belief_cost` must not be the thing that decides it.
        The newline is charged because `_core_belief_cost` charges it, and
        that sum is what would empty the section if anything could.
        """
        return chars_to_tokens(len(_core_line_at(content_chars)) + 1)

    # 1. The shipped arm cannot empty. Measured past the cap by two orders of
    # magnitude, because the claim is about every content length and not about
    # the grid: uncapped, `charged(18600)` is 4,667.
    cap = BELIEF_CONTENT_CHAR_CAP
    sweep = sorted(set(lengths) | {1, cap, cap + 1, 20000, 120000})
    worst = max(charged(n) for n in sweep)
    assert worst < budget, (
        f"one <core> line costs up to {worst} tokens against a budget of "
        f"{budget} over content lengths {sweep}, so the shipped arm can empty "
        "and the after-arm zeros this guard says are impossible are not"
    )
    # ...including the pathological content: every character an angle bracket,
    # which `_escape_for_hook_block` expands fourfold past the cap.
    brackets = m._probe_belief("<" * (cap * 4), "unknown")  # type: ignore[attr-defined]
    worst_escaped = _core_belief_cost(brackets)
    assert worst_escaped < budget, (
        f"a belief of {cap * 4} angle brackets is charged {worst_escaped} for "
        f"one <core> line against a budget of {budget}, so the escape "
        "expansion can still empty the shipped arm"
    )
    # The transcription above must be the charge the packer actually bills, or
    # the bound is a bound on the wrong number.
    for c in lengths:
        assert charged(c) == _core_belief_cost(_core_probe_at(c)), (
            c, charged(c), _core_belief_cost(_core_probe_at(c)),
        )
    # ...and the produced curve agrees: no after-arm cell is zero.
    for c in lengths:
        assert curve[str(c)]["after"] > 0, (c, curve[str(c)])
    # 2. The before arm still empties, at exactly the lengths whose pre-#1526
    # charge overruns the budget, starting at the published edge.
    emptied_before = [c for c in lengths if curve[str(c)]["before"] == 0]
    expected = [c for c in lengths if _core_pack_costs_at(c)[0] > budget]
    assert emptied_before == expected, (
        f"the before arm emits nothing at {emptied_before} but the pre-#1526 "
        f"charge overruns the {budget}-token budget at {expected}: "
        f"{ {c: _core_pack_costs_at(c) for c in lengths} }"
    )
    assert expected, {c: _core_pack_costs_at(c) for c in lengths}
    assert expected[0] == _CORE_CROSSING, (
        f"the first grid length whose before arm empties is {expected[0]}, not "
        f"the published crossing edge {_CORE_CROSSING}, so `LENGTH_GRID` no "
        "longer carries that edge"
    )
    # `pct` is None exactly where the before arm is zero.
    for c in lengths:
        row = curve[str(c)]
        assert (row["pct"] is None) == (row["before"] == 0), (c, row)


def _core_probe_at(content_chars: int) -> Belief:
    """One `<core>` candidate of that content length, built as the producer builds them.

    `synthetic_content` returns content of exactly `content_chars`, so every
    candidate in a producer store at a given grid length is the same length and
    this one stands for all 300 of them.
    """
    import random

    m = _producer_module()
    content = m.synthetic_content(  # type: ignore[attr-defined]
        random.Random(m.UNDERCHARGE_SEED), 0, content_chars,  # type: ignore[attr-defined]
    )
    return m._probe_belief(content, "unknown")  # type: ignore[attr-defined]


def _core_line_at(content_chars: int) -> str:
    """The `<core>` line the shipped renderer emits for one belief of that size."""
    from aelfrice.hook import _core_belief_line

    return _core_belief_line(_core_probe_at(content_chars))


def _core_pack_costs_at(content_chars: int) -> tuple[int, int]:
    """What `<core>`'s packer charges for one belief of that length, both arms.

    Returns `(legacy, shipped)`: the pre-#1526 `max(1, len(content) // 4)` the
    before arm packs under, and the rendered line the after arm packs under.
    Both are returned because the caller picks the half belonging to the arm
    whose cell it is reading — a zero cell is legitimate only when the arm
    that produced it could not afford one line in its own currency, and from
    6,004 content characters up the two halves disagree about that. See
    `test_the_producer_names_which_budget_ended_every_pack`. The shipped side
    is written as the emitted line and not as a call to `_core_belief_cost`,
    for the reason
    `test_the_core_section_empties_once_one_belief_exceeds_its_budget`
    gives: the claim is about what the section emits, so a cost function must
    not be the thing that decides it. That guard asserts the two agree at
    every grid length, which is what keeps the transcription honest without
    letting the cost function decide the claim.
    """
    b = _core_probe_at(content_chars)
    return _legacy_core_cost(b), chars_to_tokens(len(_core_belief_line(b)) + 1)


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
    the next cap lands **under one of the name shapes in
    `_CONTENT_CAP_SUFFIXES`**. A cap named otherwise -- `SESSION_START_CAP`,
    `BASELINE_MAX_BYTES` -- is not discovered, the floor does not rise, and a
    trim at it above this fixture's padding would pass. The two known caps
    are required by name so a rename fails loudly rather than shrinking the
    set, but discovery is a name convention, not a proof.

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
