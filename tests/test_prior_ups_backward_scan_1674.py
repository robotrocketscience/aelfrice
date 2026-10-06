"""#1674: the sentiment lane's prior-turn lookup reads the audit log
backward and stops at the first match, instead of reading it whole.

The answer must equal the whole-file scan's (the last matching row
wins), including across the rotation boundary and chunk boundaries.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from aelfrice import hook
from aelfrice.hook_audit import AUDIT_ROTATED_SUFFIX, _audit_path_for_db


def _whole_file(audit: Path, session: str) -> list[str]:
    """The pre-#1674 algorithm, over the unchanged `read_hook_audit`."""
    rotated = audit.with_name(audit.name + AUDIT_ROTATED_SUFFIX)
    last: list[str] = []
    try:
        for p in [x for x in (rotated, audit) if x.exists()]:
            for r in hook.read_hook_audit(p):
                if r.get("hook") != "user_prompt_submit" or r.get("session_id") != session:
                    continue
                b = r.get("beliefs")
                if not isinstance(b, list):
                    continue
                last = [
                    x["id"] for x in b
                    if isinstance(x, dict) and isinstance(x.get("id"), str) and x["id"]
                ]
    except ValueError:
        return []
    return last


@pytest.fixture
def audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    p = _audit_path_for_db(db)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _row(rng: random.Random, session: str) -> str:
    kind = rng.random()
    beliefs: object
    if kind < 0.1:
        beliefs = "not-a-list"
    elif kind < 0.2:
        beliefs = []
    else:
        beliefs = [
            {"id": f"b{rng.randrange(10_000)}"} if rng.random() > 0.1 else {"id": ""}
            for _ in range(rng.randrange(1, 5))
        ]
    hook_name = "user_prompt_submit" if rng.random() > 0.15 else "search_tool"
    return json.dumps({"hook": hook_name, "session_id": session, "beliefs": beliefs})


def _lines(rng: random.Random, n: int) -> list[str]:
    out: list[str] = []
    for _ in range(n):
        r = rng.random()
        if r < 0.03:
            out.append("")  # blank line
        elif r < 0.05:
            out.append("[1, 2]")  # JSON, not an object
        else:
            out.append(_row(rng, f"s{rng.randrange(6)}"))
    return out


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("chunk", [7, 64, 65_536])
def test_matches_the_whole_file_scan(
    audit: Path, monkeypatch: pytest.MonkeyPatch, seed: int, chunk: int,
) -> None:
    monkeypatch.setattr(hook, "_AUDIT_BACKWARD_CHUNK", chunk)
    rng = random.Random(seed)
    rotated = audit.with_name(audit.name + AUDIT_ROTATED_SUFFIX)
    if rng.random() < 0.5:
        rotated.write_text("\n".join(_lines(rng, 30)) + "\n", encoding="utf-8")
    body = "\n".join(_lines(rng, 30))
    if rng.random() < 0.5:
        body += "\n"  # with and without a trailing newline
    audit.write_text(body, encoding="utf-8")
    for s in [f"s{i}" for i in range(7)]:  # s6 never appears
        assert hook._load_prior_ups_belief_ids(s) == _whole_file(audit, s), s  # pyright: ignore[reportPrivateUsage]


def test_the_rotated_file_is_read_when_the_current_one_has_no_match(audit: Path) -> None:
    rotated = audit.with_name(audit.name + AUDIT_ROTATED_SUFFIX)
    rotated.write_text(
        json.dumps({"hook": "user_prompt_submit", "session_id": "S", "beliefs": [{"id": "old"}]}) + "\n",
        encoding="utf-8",
    )
    audit.write_text(
        json.dumps({"hook": "user_prompt_submit", "session_id": "T", "beliefs": [{"id": "x"}]}) + "\n",
        encoding="utf-8",
    )
    assert hook._load_prior_ups_belief_ids("S") == ["old"]  # pyright: ignore[reportPrivateUsage]


def test_the_current_file_wins_over_the_rotated_one(audit: Path) -> None:
    rotated = audit.with_name(audit.name + AUDIT_ROTATED_SUFFIX)
    rotated.write_text(
        json.dumps({"hook": "user_prompt_submit", "session_id": "S", "beliefs": [{"id": "old"}]}) + "\n",
        encoding="utf-8",
    )
    audit.write_text(
        json.dumps({"hook": "user_prompt_submit", "session_id": "S", "beliefs": [{"id": "new"}]}) + "\n",
        encoding="utf-8",
    )
    assert hook._load_prior_ups_belief_ids("S") == ["new"]  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("bad", ["{not json", b"\xff\xfe".decode("latin-1")])
def test_a_corrupt_line_newer_than_the_match_still_gives_nothing(audit: Path, bad: str) -> None:
    good = json.dumps({"hook": "user_prompt_submit", "session_id": "S", "beliefs": [{"id": "a"}]})
    audit.write_bytes((good + "\n").encode() + bad.encode("latin-1") + b"\n")
    assert hook._load_prior_ups_belief_ids("S") == []  # pyright: ignore[reportPrivateUsage]


def test_a_corrupt_line_older_than_the_match_is_never_reached(audit: Path) -> None:
    """The documented difference from a whole-file read."""
    good = json.dumps({"hook": "user_prompt_submit", "session_id": "S", "beliefs": [{"id": "a"}]})
    audit.write_text("{not json\n" + good + "\n", encoding="utf-8")
    assert hook._load_prior_ups_belief_ids("S") == ["a"]  # pyright: ignore[reportPrivateUsage]


def test_the_scan_stops_at_the_first_match(audit: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The point of #1674: older lines are never parsed."""
    good = json.dumps({"hook": "user_prompt_submit", "session_id": "S", "beliefs": [{"id": "a"}]})
    audit.write_text("\n".join([good] * 1000 + [good]) + "\n", encoding="utf-8")
    parsed: list[bytes] = []
    real = hook._parse_audit_line  # pyright: ignore[reportPrivateUsage]

    def counting(raw: bytes, path: Path) -> dict[str, object] | None:
        parsed.append(raw)
        return real(raw, path)

    monkeypatch.setattr(hook, "_parse_audit_line", counting)
    assert hook._load_prior_ups_belief_ids("S") == ["a"]  # pyright: ignore[reportPrivateUsage]
    assert len([p for p in parsed if p.strip()]) == 1
