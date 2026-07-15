"""UserPromptSubmit belief-category rerank-on-trigger lane (#1126).

The lane does NOT emit a separate block (that double-injects what
retrieval already returns — see the #1126 R&D). Instead a fired category
lifts its members to the TOP of the single retrieval output, surfaces a
bounded set of members retrieval missed, and prepends a <category-focus>
label. These tests cover reranking, no-duplication, bounded surfacing,
the label, default-off, determinism, and fail-soft.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice import hook as hookmod
from aelfrice.category import CategoryTrigger
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_USER_STATED,
    Belief,
)
from aelfrice.store import MemoryStore


@pytest.fixture(autouse=True)
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "hook-cat.db"
    monkeypatch.setenv("AELFRICE_DB", str(p))
    monkeypatch.delenv("AELFRICE_BELIEF_CATEGORIES", raising=False)
    return p


def _belief(bid: str, content: str, *, locked: bool = False) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash="h_" + bid,
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_USER if locked else LOCK_NONE,
        locked_at="2026-07-15T00:00:00+00:00" if locked else None,
        created_at="2026-07-15T00:00:00+00:00",
        last_retrieved_at=None,
        origin=ORIGIN_USER_STATED if locked else ORIGIN_AGENT_INFERRED,
    )


def _seed(db: Path, name: str, *, keywords: tuple[str, ...] = (),
          always_on: bool = False) -> MemoryStore:
    s = MemoryStore(str(db))
    s.upsert_category(
        name=name, always_on=always_on,
        trigger_json=CategoryTrigger(keywords=keywords).to_json(),
        default_lock="locked",
    )
    return s


# --- _apply_category_boost (unit) --------------------------------------


def test_boost_disabled_by_default(db: Path) -> None:
    s = _seed(db, "git", keywords=("push",))
    m = _belief("aaaa", "push rule")
    s.insert_belief(m)
    s.assign_belief_to_category("aaaa", "git")
    s.close()
    hits = [_belief("bbbb", "other"), m]
    out, focus = hookmod._apply_category_boost(hits, "please push", None, io.StringIO())
    # default-off: unchanged order, no focus
    assert out == hits and focus == []


def test_boost_promotes_member_to_top(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AELFRICE_BELIEF_CATEGORIES", "1")
    s = _seed(db, "git", keywords=("push",))
    m = _belief("aaaa", "sign every commit")
    s.insert_belief(m)
    s.assign_belief_to_category("aaaa", "git")
    s.close()
    # member is LAST in retrieval order; boost must move it first
    hits = [_belief("x1", "distractor 1"), _belief("x2", "distractor 2"), m]
    out, focus = hookmod._apply_category_boost(hits, "please push", None, io.StringIO())
    assert out[0].id == "aaaa"
    assert [h.id for h in out] == ["aaaa", "x1", "x2"]
    assert focus == ["git"]


def test_boost_no_duplication(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELFRICE_BELIEF_CATEGORIES", "1")
    s = _seed(db, "git", keywords=("push",))
    m = _belief("aaaa", "the rule")
    s.insert_belief(m)
    s.assign_belief_to_category("aaaa", "git")
    s.close()
    hits = [_belief("x1", "d1"), m]
    out, _ = hookmod._apply_category_boost(hits, "push", None, io.StringIO())
    # member appears exactly once (reordered, not added)
    assert [h.id for h in out].count("aaaa") == 1
    assert len(out) == len(hits)


def test_boost_surfaces_missed_member_bounded(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AELFRICE_BELIEF_CATEGORIES", "1")
    s = _seed(db, "git", keywords=("push",))
    # member NOT in the hits list (retrieval missed it)
    m = _belief("miss", "a rule retrieval did not return")
    s.insert_belief(m)
    s.assign_belief_to_category("miss", "git")
    s.close()
    hits = [_belief("x1", "d1")]
    out, focus = hookmod._apply_category_boost(hits, "push", None, io.StringIO())
    assert out[0].id == "miss"  # surfaced at top
    assert len(out) == 2
    assert focus == ["git"]


def test_boost_extra_cap(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELFRICE_BELIEF_CATEGORIES", "1")
    s = _seed(db, "git", keywords=("push",))
    for i in range(hookmod.CATEGORY_BOOST_MAX_EXTRA + 5):
        bid = f"m{i:02d}"
        s.insert_belief(_belief(bid, f"missed rule {i}"))
        s.assign_belief_to_category(bid, "git")
    s.close()
    out, _ = hookmod._apply_category_boost([], "push", None, io.StringIO())
    # only the cap's worth of retrieval-missed members are surfaced
    assert len(out) == hookmod.CATEGORY_BOOST_MAX_EXTRA


def test_boost_deterministic_and_no_fire(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AELFRICE_BELIEF_CATEGORIES", "1")
    s = _seed(db, "git", keywords=("push",))
    m = _belief("aaaa", "r")
    s.insert_belief(m)
    s.assign_belief_to_category("aaaa", "git")
    s.close()
    hits = [_belief("x1", "d"), m]
    a, fa = hookmod._apply_category_boost(hits, "push", None, io.StringIO())
    b, fb = hookmod._apply_category_boost(hits, "push", None, io.StringIO())
    assert [h.id for h in a] == [h.id for h in b] and fa == fb
    # no keyword in prompt -> no fire, unchanged
    c, fc = hookmod._apply_category_boost(hits, "unrelated text", None, io.StringIO())
    assert c == hits and fc == []


# --- end-to-end hook ----------------------------------------------------


def _hook(db: Path, prompt: str) -> str:
    payload = json.dumps({"prompt": prompt, "cwd": str(db.parent)})
    sout = io.StringIO()
    rc = hookmod.user_prompt_submit(
        stdin=io.StringIO(payload), stdout=sout, stderr=io.StringIO()
    )
    assert rc == 0
    return sout.getvalue()


def test_e2e_label_and_single_occurrence(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AELFRICE_BELIEF_CATEGORIES", "1")
    s = _seed(db, "git-workflow", keywords=("commit and push",))
    m = _belief("aaaabbbbccccdddd", "sign every commit", locked=True)
    s.insert_belief(m)
    s.assign_belief_to_category(m.id, "git-workflow")
    s.close()
    body = _hook(db, "now commit and push please")
    assert "<category-focus>" in body
    assert "git-workflow" in body
    # rule injected exactly once (no separate block duplicating L0)
    assert body.count("sign every commit") == 1
    # no legacy separate block
    assert "belief-category-rules" not in body


def test_e2e_silent_when_disabled(db: Path) -> None:
    s = _seed(db, "git", keywords=("push",))
    m = _belief("aaaabbbbccccdddd", "a rule", locked=True)
    s.insert_belief(m)
    s.assign_belief_to_category(m.id, "git")
    s.close()
    body = _hook(db, "push please")
    assert "category-focus" not in body


def test_e2e_no_label_when_no_category_fires(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AELFRICE_BELIEF_CATEGORIES", "1")
    s = _seed(db, "git", keywords=("push",))
    m = _belief("aaaabbbbccccdddd", "a rule", locked=True)
    s.insert_belief(m)
    s.assign_belief_to_category(m.id, "git")
    s.close()
    body = _hook(db, "unrelated question about the weather")
    assert "category-focus" not in body
