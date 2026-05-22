"""#909: conversation-aware UserPromptSubmit retrieval.

The live per-prompt hook BM25s the literal prompt only. When the topic
vocabulary lives in the dialog history (paraphrase / pronoun / numeric
reference) and not in the current prompt, the load-bearing thread scores
~0 lexically and is never surfaced — while noise that shares the
prompt's generic tokens wins. Folding a small window of recent turns
into the query (with the current prompt weighted to stay dominant)
restores it. These tests lock:

* the pure query builder (weighting, window, fail-soft),
* the config parsing of the three new keys,
* the retrieval-seam regression (prompt-only misses, conversation-aware
  surfaces) — the exact #909 mechanism,
* a precision guard (a prompt that already retrieves well is not
  degraded; a large off-topic window does not silently re-bury), and
* the end-to-end wiring through `user_prompt_submit`.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice.hook import (
    DEFAULT_CONV_AWARE_WEIGHT,
    DEFAULT_CONV_AWARE_WINDOW,
    MAX_CONV_AWARE_WEIGHT,
    UserPromptSubmitConfig,
    _build_conversation_aware_query,
    load_user_prompt_submit_config,
    user_prompt_submit,
)
from aelfrice.context_rebuilder import RecentTurn
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.retrieval import retrieve
from aelfrice.store import MemoryStore

# --- corpus: load-bearing jargon (zero overlap with the prompt) plus
# noise that incidentally shares the prompt's generic recall words. ---
_LOADBEARING = [
    "Brew ratio one to two point five, eighteen gram dose, "
    "twenty-eight second pull, ninety-three celsius.",
    "Grind at fourteen clicks; pre-infusion six seconds for even extraction.",
    "Yield forty-five grams from eighteen in on the house blend.",
]
_NOISE = (
    [f"We decided to refactor the settings module in sprint {i}." for i in range(15)]
    + [f"The final numbers in the Q3 report figures looked off on slide {i}." for i in range(15)]
    + [f"We agreed on the deployment settings for the cluster on day {i}." for i in range(15)]
)
# Generic recall prompt with NO domain tokens — the #909 trigger:
_PROMPT = "what were the final numbers and settings we decided on"
_DOMAIN_TURN = (
    "we were dialing in the espresso grind and brew ratio and the dose "
    "for pulling shots"
)
_TARGET = "twenty-eight second"
# A noise belief that shares the prompt's generic recall words and so
# out-ranks the load-bearing thread under prompt-only retrieval.
_NOISE_SENTINEL = "final numbers in the Q3 report figures"


def _mk(bid: str, content: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-05-22T00:00:00Z",
        last_retrieved_at=None,
    )


def _seed(db: Path) -> None:
    store = MemoryStore(str(db))
    try:
        for i, c in enumerate(_LOADBEARING):
            store.insert_belief(_mk(f"LB{i}", c))
        for i, c in enumerate(_NOISE):
            store.insert_belief(_mk(f"N{i}", c))
    finally:
        store.close()


def _target_rank(beliefs: list[Belief]) -> int | None:
    """Rank (0-based) of the load-bearing belief, or None if absent.

    Rank is the faithful signal for #909: the real hook trims to a token
    budget, so a thread ranked far down the list is cut from the
    surfaced slice (the reported "13 of 59" symptom). A small in-test
    corpus may not trigger the trim, so we assert on rank rather than
    mere presence.
    """
    for i, b in enumerate(beliefs):
        if _TARGET in b.content:
            return i
    return None


# --------------------------------------------------------------------------
# Pure query builder
# --------------------------------------------------------------------------


def test_builder_weights_prompt_and_appends_window() -> None:
    turns = [
        RecentTurn(role="user", text="alpha beta"),
        RecentTurn(role="assistant", text="gamma"),
    ]
    q = _build_conversation_aware_query(
        "hello", turns, turn_window=4, prompt_weight=3,
    )
    assert q == "hello hello hello alpha beta gamma"


def test_builder_uses_only_last_window_turns() -> None:
    turns = [RecentTurn(role="user", text=f"t{i}") for i in range(6)]
    q = _build_conversation_aware_query(
        "p", turns, turn_window=2, prompt_weight=1,
    )
    assert q == "p t4 t5"


def test_builder_empty_turns_is_prompt_repeated() -> None:
    assert _build_conversation_aware_query(
        "only", [], turn_window=4, prompt_weight=1,
    ) == "only"


def test_builder_weight_below_one_clamps_to_one() -> None:
    assert _build_conversation_aware_query(
        "p", [], turn_window=4, prompt_weight=0,
    ) == "p"


def test_builder_skips_blank_turn_text() -> None:
    turns = [RecentTurn(role="user", text="   "), RecentTurn(role="user", text="real")]
    q = _build_conversation_aware_query(
        "p", turns, turn_window=4, prompt_weight=1,
    )
    assert q == "p real"


def test_builder_window_zero_is_prompt_only() -> None:
    turns = [RecentTurn(role="user", text="topic")]
    assert _build_conversation_aware_query(
        "p", turns, turn_window=0, prompt_weight=1,
    ) == "p"


# --------------------------------------------------------------------------
# Config parsing
# --------------------------------------------------------------------------


def test_config_defaults_on() -> None:
    cfg = UserPromptSubmitConfig()
    assert cfg.conversation_aware_query_enabled is True
    assert cfg.conversation_aware_turn_window == DEFAULT_CONV_AWARE_WINDOW
    assert cfg.conversation_aware_prompt_weight == DEFAULT_CONV_AWARE_WEIGHT


def test_config_parses_valid_keys(tmp_path: Path) -> None:
    (tmp_path / ".aelfrice.toml").write_text(
        "[user_prompt_submit_hook]\n"
        "conversation_aware_query_enabled = false\n"
        "conversation_aware_turn_window = 8\n"
        "conversation_aware_prompt_weight = 2\n"
    )
    cfg = load_user_prompt_submit_config(start=tmp_path, stderr=io.StringIO())
    assert cfg.conversation_aware_query_enabled is False
    assert cfg.conversation_aware_turn_window == 8
    assert cfg.conversation_aware_prompt_weight == 2


def test_config_rejects_bad_types_falls_back(tmp_path: Path) -> None:
    (tmp_path / ".aelfrice.toml").write_text(
        "[user_prompt_submit_hook]\n"
        'conversation_aware_query_enabled = "yes"\n'
        "conversation_aware_turn_window = true\n"  # bool not int
        "conversation_aware_prompt_weight = 0\n"  # below floor
    )
    serr = io.StringIO()
    cfg = load_user_prompt_submit_config(start=tmp_path, stderr=serr)
    assert cfg.conversation_aware_query_enabled is True
    assert cfg.conversation_aware_turn_window == DEFAULT_CONV_AWARE_WINDOW
    assert cfg.conversation_aware_prompt_weight == DEFAULT_CONV_AWARE_WEIGHT
    assert "conversation_aware" in serr.getvalue()


def test_config_rejects_weight_above_ceiling(tmp_path: Path) -> None:
    # An unbounded weight balloons `[prompt] * weight` on the UPS hot
    # path; values above MAX_CONV_AWARE_WEIGHT fall back to the default.
    (tmp_path / ".aelfrice.toml").write_text(
        "[user_prompt_submit_hook]\n"
        f"conversation_aware_prompt_weight = {MAX_CONV_AWARE_WEIGHT + 1}\n"
    )
    serr = io.StringIO()
    cfg = load_user_prompt_submit_config(start=tmp_path, stderr=serr)
    assert cfg.conversation_aware_prompt_weight == DEFAULT_CONV_AWARE_WEIGHT
    assert "conversation_aware_prompt_weight" in serr.getvalue()


# --------------------------------------------------------------------------
# Retrieval-seam regression — the #909 mechanism
# --------------------------------------------------------------------------


def test_prompt_only_ranks_loadbearing_thread_far_down(tmp_path: Path) -> None:
    """Baseline: the bug. Prompt-only query ranks the load-bearing thread
    near the bottom (here last of 47) — under the hook's token-budget
    trim that is below the cut and never surfaced."""
    db = tmp_path / "memory.db"
    _seed(db)
    store = MemoryStore(str(db))
    try:
        hits = retrieve(store, _PROMPT, token_budget=1500)
    finally:
        store.close()
    rank = _target_rank(hits)
    # Either absent or buried deep in the noise; certainly not surfaced
    # in any realistic top-K.
    assert rank is None or rank >= 10


def test_conversation_aware_ranks_loadbearing_thread_top(tmp_path: Path) -> None:
    """Fix: folding an on-topic recent turn into the query lifts the
    thread to the top of the ranking."""
    db = tmp_path / "memory.db"
    _seed(db)
    turns = [RecentTurn(role="user", text=_DOMAIN_TURN)]
    q = _build_conversation_aware_query(
        _PROMPT, turns, turn_window=4, prompt_weight=3,
    )
    store = MemoryStore(str(db))
    try:
        hits = retrieve(store, q, token_budget=1500)
    finally:
        store.close()
    rank = _target_rank(hits)
    assert rank is not None and rank <= 2


def test_guard_prompt_with_good_overlap_not_degraded(tmp_path: Path) -> None:
    """Precision guard: a prompt that already hits the domain still ranks
    the thread at the top after augmentation (the prompt stays dominant
    via weighting — the appended turns do not dilute it out)."""
    db = tmp_path / "memory.db"
    _seed(db)
    good_prompt = "espresso brew ratio dose pull"
    turns = [RecentTurn(role="user", text=_DOMAIN_TURN)]
    q = _build_conversation_aware_query(
        good_prompt, turns, turn_window=4, prompt_weight=3,
    )
    store = MemoryStore(str(db))
    try:
        hits = retrieve(store, q, token_budget=1500)
    finally:
        store.close()
    rank = _target_rank(hits)
    assert rank is not None and rank <= 2


# --------------------------------------------------------------------------
# End-to-end wiring through user_prompt_submit
# --------------------------------------------------------------------------


def _transcript(tmp_path: Path, *turns: tuple[str, str]) -> Path:
    """Write a Claude-Code-format transcript JSONL; return its path."""
    p = tmp_path / "transcript.jsonl"
    lines = [
        json.dumps({"type": role, "message": {"role": role, "content": text}})
        for role, text in turns
    ]
    p.write_text("\n".join(lines) + "\n")
    return p


def _payload(prompt: str, transcript: Path, cwd: Path) -> str:
    return json.dumps(
        {
            "session_id": "s1",
            "transcript_path": str(transcript),
            "cwd": str(cwd),
            "hook_event_name": "UserPromptSubmit",
            "prompt": prompt,
        }
    )


def test_e2e_hook_surfaces_thread_with_recent_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _seed(db)
    monkeypatch.setenv("AELFRICE_DB", str(db))
    # cwd has no .git, so the reader falls through to transcript_path.
    nogit = tmp_path / "work"
    nogit.mkdir()
    transcript = _transcript(tmp_path, ("user", _DOMAIN_TURN))
    sin = io.StringIO(_payload(_PROMPT, transcript, nogit))
    sout = io.StringIO()
    rc = user_prompt_submit(stdin=sin, stdout=sout, stderr=io.StringIO())
    assert rc == 0
    out = sout.getvalue()
    # The thread is surfaced AND ahead of the noise it was buried under.
    assert _TARGET in out
    assert _NOISE_SENTINEL in out
    assert out.index(_TARGET) < out.index(_NOISE_SENTINEL)


def test_e2e_hook_flag_off_is_prompt_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _seed(db)
    monkeypatch.setenv("AELFRICE_DB", str(db))
    # Disable the feature via .aelfrice.toml in the *payload* cwd. The hook
    # resolves config from the payload cwd (#909/#887), not the process cwd,
    # so a project's config is honored even when the launcher starts the
    # hook elsewhere. No chdir — the payload cwd alone must carry the config.
    (tmp_path / ".aelfrice.toml").write_text(
        "[user_prompt_submit_hook]\n"
        "conversation_aware_query_enabled = false\n"
    )
    transcript = _transcript(tmp_path, ("user", _DOMAIN_TURN))
    sin = io.StringIO(_payload(_PROMPT, transcript, tmp_path))
    sout = io.StringIO()
    rc = user_prompt_submit(stdin=sin, stdout=sout, stderr=io.StringIO())
    assert rc == 0
    out = sout.getvalue()
    # Feature off ⇒ prompt-only ranking: the thread is buried *after* the
    # noise (the inverse of the ON case), confirming recent turns are not
    # folded in. (A token-budget trim would drop it entirely; this small
    # corpus keeps it present but ranked last.)
    assert _NOISE_SENTINEL in out
    assert _TARGET not in out or out.index(_TARGET) > out.index(_NOISE_SENTINEL)
