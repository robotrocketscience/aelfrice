"""#1382 — turn-differential lock rendering.

Within one session epoch a belief already rendered verbatim is re-emitted as a
one-line `seen` reference instead of the identical block. The epoch boundary is
a SessionStart fire: `hook.session_start` renders the baseline unconditionally
*before* it reads `source`, so `source == "compact"` is neither necessary nor
sufficient (#1382 premise 1).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aelfrice import hook, injection_ledger
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_TIER_REFERENCE,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_USER_STATED,
    Belief,
)


def _b(
    bid: str,
    content: str = "a locked policy statement about branching",
    *,
    lock: str = LOCK_USER,
    tier: str | None = None,
    origin: str = ORIGIN_USER_STATED,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock,
        lock_tier=tier,
        locked_at=None,
        origin=origin,
        created_at="2026-01-01T00:00:00Z",
        last_retrieved_at=None,
    )


# --- AC1: the differential ------------------------------------------------


def test_an_unseen_belief_renders_verbatim() -> None:
    lines, manifest = hook._split_belief_lines([_b("aa")], already_rendered=frozenset())
    assert any("a locked policy statement" in ln for ln in lines)
    assert manifest == []


def test_a_seen_belief_renders_as_a_one_line_reference() -> None:
    """AC1: within an epoch, an already-injected lock is not re-injected.

    The saving is asserted as a size ratio rather than an absence of the text:
    `_lock_topic` returns short content whole, so a short belief's reference
    legitimately contains its content and an absence check would pin nothing.
    The prize only exists on beliefs long enough to truncate — which is what
    the 32% is made of.
    """
    long_content = (
        "The merge train is concurrency-1 and verifies that the head SHA still "
        "matches the labeled event, the branch is fast-forward on current main, "
        "every commit is signed, and all required checks are green before it "
        "pushes; on any failure it unlabels and posts a rejection comment."
    )
    verbatim, _ = hook._split_belief_lines([_b("aa", long_content)])
    lines, manifest = hook._split_belief_lines(
        [_b("aa", long_content)], already_rendered=frozenset({"aa"})
    )
    assert lines == []
    assert len(manifest) == 1
    assert manifest[0].strip().startswith("seen aa:")
    assert len(manifest[0]) < len(verbatim[0]) / 2, (
        f"reference is {len(manifest[0])} chars against {len(verbatim[0])} "
        "verbatim — the differential is not saving anything"
    )


def test_the_empty_ledger_is_byte_identical_to_the_old_render() -> None:
    """The default must not move a single byte, or every existing envelope
    assertion in the suite is silently re-baselined."""
    hits = [_b("aa"), _b("bb", "second statement", lock=LOCK_NONE)]
    assert hook._split_belief_lines(hits) == hook._split_belief_lines(
        hits, already_rendered=frozenset()
    )


def test_a_reference_lock_stays_a_ref_entry_even_when_seen() -> None:
    """`ref` is the stronger claim — the full text was never injected at all —
    and it is what the #1016-B block note documents."""
    ref = _b("cc", lock=LOCK_USER, tier=LOCK_TIER_REFERENCE)
    _, manifest = hook._split_belief_lines([ref], already_rendered=frozenset({"cc"}))
    assert manifest[0].strip().startswith("ref cc:")


# --- AC3: the differential is not locked-only -----------------------------


def test_a_non_locked_core_belief_is_also_differentiated() -> None:
    """AC3: 14% of the prize is `<core>`, not locked. The predicate keys on the
    belief id, not the lock tier, so the coverage is structural."""
    core = _b("dd", "an unlocked corroborated belief", lock=LOCK_NONE)
    lines, manifest = hook._split_belief_lines(
        [core], already_rendered=frozenset({"dd"})
    )
    assert lines == []
    assert manifest[0].strip().startswith("seen dd:")


# --- AC4: one shared predicate --------------------------------------------


def test_trust_tier_grouping_survives_a_differential_render() -> None:
    """AC4, and the whole reason the predicate is shared.

    `_group_by_provenance` positionally zips its hit list against the rendered
    lines and returns **ungrouped** on a length mismatch, behind a
    `# pragma: no cover` guard. If it filtered on `is_reference_lock` alone
    while the splitter also diverted already-seen hits, the lists would differ
    in length and trust-tier grouping would vanish with no error and no
    coverage.

    Asserting on the section tags is what makes this distinguishing: a test
    that only checked the seen belief was absent passes under that exact
    regression.
    """
    hits = [
        _b("aa", "a locked statement"),
        _b("bb", "an inferred claim", lock=LOCK_NONE, origin=ORIGIN_AGENT_INFERRED),
    ]
    lines, manifest = hook._split_belief_lines(
        hits, provenance_render=True, already_rendered=frozenset({"aa"})
    )
    body = "\n".join(lines)
    # the seen one is gone from the verbatim body ...
    assert "a locked statement" not in body
    assert manifest[0].strip().startswith("seen aa:")
    # ... and the survivor is still inside a trust-tier section, i.e. grouping
    # was not silently disabled by a length mismatch.
    assert "<" in body and "an inferred claim" in body
    assert any(ln.startswith("<") and "<!--" in ln for ln in lines), (
        "trust-tier section headers are absent — _group_by_provenance fell "
        "through its length guard and returned ungrouped"
    )


def test_the_predicate_is_the_union_of_both_reasons() -> None:
    ref = _b("cc", lock=LOCK_USER, tier=LOCK_TIER_REFERENCE)
    plain = _b("dd", lock=LOCK_NONE)
    assert hook._renders_as_manifest(ref, frozenset())
    assert hook._renders_as_manifest(plain, frozenset({"dd"}))
    assert not hook._renders_as_manifest(plain, frozenset())
    assert not hook._renders_as_manifest(plain, frozenset({"other"}))


# --- the ledger -----------------------------------------------------------


def test_a_fresh_ledger_reads_empty(tmp_path: Path) -> None:
    assert injection_ledger.read_rendered("s1", path=tmp_path / "l.json") == frozenset()


def test_begin_epoch_replaces_rather_than_unions(tmp_path: Path) -> None:
    """A SessionStart opens a new context window; the previous epoch's verbatim
    text is not in it, so carrying its ids forward would suppress content the
    model has never seen."""
    p = tmp_path / "l.json"
    injection_ledger.begin_epoch("s1", frozenset({"aa", "bb"}), path=p)
    injection_ledger.begin_epoch("s1", frozenset({"cc"}), path=p)
    assert injection_ledger.read_rendered("s1", path=p) == frozenset({"cc"})


def test_record_rendered_unions(tmp_path: Path) -> None:
    p = tmp_path / "l.json"
    injection_ledger.begin_epoch("s1", frozenset({"aa"}), path=p)
    injection_ledger.record_rendered("s1", frozenset({"bb"}), path=p)
    assert injection_ledger.read_rendered("s1", path=p) == frozenset({"aa", "bb"})


def test_a_foreign_session_ledger_reads_empty(tmp_path: Path) -> None:
    """A session_id change is an epoch boundary too — it covers the case where
    the SessionStart fire was missed entirely."""
    p = tmp_path / "l.json"
    injection_ledger.begin_epoch("s1", frozenset({"aa"}), path=p)
    assert injection_ledger.read_rendered("s2", path=p) == frozenset()


@pytest.mark.parametrize(
    "payload",
    ["not json at all", "[]", '{"session_id": "s1"}', '{"session_id":"s1","rendered":3}'],
)
def test_a_malformed_ledger_fails_soft_to_verbatim(tmp_path: Path, payload: str) -> None:
    """Fail-soft direction: a broken ledger costs redundant tokens, it never
    suppresses a belief the model has not been shown."""
    p = tmp_path / "l.json"
    p.write_text(payload, encoding="utf-8")
    assert injection_ledger.read_rendered("s1", path=p) == frozenset()


def test_non_utf8_bytes_fail_soft(tmp_path: Path) -> None:
    """`read_text` raises UnicodeDecodeError, a sibling of JSONDecodeError
    under ValueError rather than of OSError — catching OSError alone would let
    it escape."""
    p = tmp_path / "l.json"
    p.write_bytes(b"\xff\xfe not utf-8")
    assert injection_ledger.read_rendered("s1", path=p) == frozenset()


def test_the_ledger_is_byte_stable_for_the_same_content(tmp_path: Path) -> None:
    p = tmp_path / "l.json"
    injection_ledger.begin_epoch("s1", frozenset({"bb", "aa"}), path=p)
    first = p.read_bytes()
    injection_ledger.begin_epoch("s1", frozenset({"aa", "bb"}), path=p)
    assert p.read_bytes() == first
    assert json.loads(first)["rendered"] == ["aa", "bb"]


def test_an_empty_session_id_is_a_no_op(tmp_path: Path) -> None:
    p = tmp_path / "l.json"
    injection_ledger.begin_epoch(None, frozenset({"aa"}), path=p)
    injection_ledger.record_rendered("", frozenset({"aa"}), path=p)
    assert not p.exists()


# --- the flag -------------------------------------------------------------


def test_the_flag_defaults_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default-OFF, ratified 2026-08-19 (was ON in the 2026-08-11 ruling).

    Flipped because the argument for ON — that the failure direction is
    one-way — was falsified, and because the wrapper cost makes the block
    larger for a small block of short beliefs.
    """
    monkeypatch.delenv(injection_ledger.TURN_DIFFERENTIAL_ENV_VAR, raising=False)
    assert injection_ledger.is_turn_differential_enabled() is False


@pytest.mark.parametrize("raw", ["0", "false", "NO", " off "])
def test_the_env_var_can_disable_it(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    """`explicit=True` is what makes this assertion mean anything.

    Since the default flipped to off (2026-08-19), `is_turn_differential_enabled()`
    returns False whether the falsy value was parsed or ignored entirely —
    deleting the whole `_FALSE` branch of `_env_override` left every test in
    both #1382 files green. Passing `explicit=True` puts the resolver in a
    state where only a *parsed* falsy env value can produce False, so the
    off-switch is distinguishable from the default again.
    """
    monkeypatch.setenv(injection_ledger.TURN_DIFFERENTIAL_ENV_VAR, raw)
    assert injection_ledger.is_turn_differential_enabled() is False
    assert injection_ledger.is_turn_differential_enabled(explicit=True) is False, (
        f"{raw!r} did not override an explicit True — the env var's falsy "
        "branch is not being parsed"
    )


def test_unset_env_does_not_override_an_explicit_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tristate: returning False for "unset" is what makes a resolver
    silently un-overridable by its caller."""
    monkeypatch.delenv(injection_ledger.TURN_DIFFERENTIAL_ENV_VAR, raising=False)
    assert injection_ledger.is_turn_differential_enabled(explicit=False) is False
    monkeypatch.setenv(injection_ledger.TURN_DIFFERENTIAL_ENV_VAR, "1")
    assert injection_ledger.is_turn_differential_enabled(explicit=False) is True
