"""UserPromptSubmit belief-category injection lane (#1126).

Verifies the block builder and the end-to-end hook wiring: default-off,
keyword + always-on firing, determinism, budget truncation, and fail-soft.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice import hook as hookmod
from aelfrice.category import CategoryTrigger
from aelfrice.store import MemoryStore


@pytest.fixture(autouse=True)
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "hook-cat.db"
    monkeypatch.setenv("AELFRICE_DB", str(p))
    # deterministic: no ambient enable unless a test sets it
    monkeypatch.delenv("AELFRICE_BELIEF_CATEGORIES", raising=False)
    return p


def _seed_category(
    db: Path,
    name: str,
    *,
    always_on: bool = False,
    keywords: tuple[str, ...] = (),
    rules: tuple[str, ...] = (),
) -> None:
    from aelfrice.derivation import DerivationInput, derive
    from aelfrice.models import INGEST_SOURCE_CLI_REMEMBER

    s = MemoryStore(str(db))
    try:
        s.upsert_category(
            name=name,
            always_on=always_on,
            trigger_json=CategoryTrigger(keywords=keywords).to_json(),
            default_lock="locked",
        )
        for i, text in enumerate(rules):
            d = derive(
                DerivationInput(
                    raw_text=text,
                    source_kind=INGEST_SOURCE_CLI_REMEMBER,
                    ts=f"2026-07-15T00:00:0{i}+00:00",
                    session_id="s",
                )
            )
            assert d.belief is not None
            s.insert_belief(d.belief)
            s.assign_belief_to_category(d.belief.id, name)
    finally:
        s.close()


# --- block builder directly --------------------------------------------


def test_block_empty_when_disabled(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_category(db, "git-workflow", keywords=("push",), rules=("branch first",))
    # disabled by default
    out = hookmod._maybe_category_injection_block("please push", None, io.StringIO())
    assert out == ""


def test_block_keyword_fires_when_enabled(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AELFRICE_BELIEF_CATEGORIES", "1")
    _seed_category(
        db, "git-workflow", keywords=("push",),
        rules=("run the formatter before every commit",),
    )
    out = hookmod._maybe_category_injection_block("please push it", None, io.StringIO())
    assert "<belief-category-rules>" in out
    assert "[git-workflow]" in out
    assert "run the formatter before every commit" in out


def test_block_no_fire_without_keyword(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AELFRICE_BELIEF_CATEGORIES", "1")
    _seed_category(db, "git-workflow", keywords=("push",), rules=("a rule",))
    out = hookmod._maybe_category_injection_block("unrelated text", None, io.StringIO())
    assert out == ""


def test_block_always_on_fires_regardless(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AELFRICE_BELIEF_CATEGORIES", "1")
    _seed_category(db, "repo-rules", always_on=True, rules=("use uv for python",))
    out = hookmod._maybe_category_injection_block("anything at all", None, io.StringIO())
    assert "[repo-rules]" in out
    assert "use uv for python" in out


def test_block_dedups_across_categories(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AELFRICE_BELIEF_CATEGORIES", "1")
    # one belief in two fired categories: appears once
    from aelfrice.derivation import DerivationInput, derive
    from aelfrice.models import INGEST_SOURCE_CLI_REMEMBER

    s = MemoryStore(str(db))
    try:
        for n in ("a-cat", "b-cat"):
            s.upsert_category(
                name=n, always_on=True, trigger_json="{}", default_lock="locked"
            )
        d = derive(
            DerivationInput(
                raw_text="shared rule text",
                source_kind=INGEST_SOURCE_CLI_REMEMBER,
                ts="2026-07-15T00:00:00+00:00",
                session_id="s",
            )
        )
        assert d.belief is not None
        s.insert_belief(d.belief)
        s.assign_belief_to_category(d.belief.id, "a-cat")
        s.assign_belief_to_category(d.belief.id, "b-cat")
    finally:
        s.close()
    out = hookmod._maybe_category_injection_block("go", None, io.StringIO())
    assert out.count("shared rule text") == 1


def test_block_deterministic(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELFRICE_BELIEF_CATEGORIES", "1")
    _seed_category(db, "zeta", always_on=True, rules=("z rule",))
    _seed_category(db, "alpha", always_on=True, rules=("a rule",))
    a = hookmod._maybe_category_injection_block("x", None, io.StringIO())
    b = hookmod._maybe_category_injection_block("x", None, io.StringIO())
    assert a == b
    # name-ASC ordering: alpha header precedes zeta header
    assert a.index("[alpha]") < a.index("[zeta]")


def test_block_truncates_to_budget(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AELFRICE_BELIEF_CATEGORIES", "1")
    big = tuple(f"rule number {i} " + "x" * 120 for i in range(40))
    _seed_category(db, "repo-rules", always_on=True, rules=big)
    out = hookmod._maybe_category_injection_block("go", None, io.StringIO())
    assert "truncated" in out
    assert len(out) < hookmod.CATEGORY_BLOCK_CHAR_BUDGET + 600


# --- end-to-end hook ----------------------------------------------------


def test_user_prompt_submit_emits_block(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AELFRICE_BELIEF_CATEGORIES", "1")
    _seed_category(
        db, "git-workflow", keywords=("commit and push",),
        rules=("sign every commit",),
    )
    payload = json.dumps({"prompt": "now commit and push please", "cwd": str(db.parent)})
    sout = io.StringIO()
    rc = hookmod.user_prompt_submit(
        stdin=io.StringIO(payload), stdout=sout, stderr=io.StringIO()
    )
    assert rc == 0
    body = sout.getvalue()
    assert "<belief-category-rules>" in body
    assert "sign every commit" in body


def test_user_prompt_submit_silent_when_disabled(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_category(db, "git-workflow", keywords=("push",), rules=("a rule",))
    payload = json.dumps({"prompt": "push please", "cwd": str(db.parent)})
    sout = io.StringIO()
    rc = hookmod.user_prompt_submit(
        stdin=io.StringIO(payload), stdout=sout, stderr=io.StringIO()
    )
    assert rc == 0
    assert "belief-category-rules" not in sout.getvalue()
