"""#1371 §10: onboard and transcript ingest must agree about code fences.

`extraction.extract_sentences` has stripped fenced blocks since the
first release. `scanner._split_paragraphs` did not — it split on blank
lines and kept anything at or over `_MIN_PARAGRAPH_CHARS`, with no fence
handling at all and no fence category in `is_noise`. So the *same
document* produced different beliefs depending on which door it came
through: the onboard door stored raw XML samples, shell transcripts and
JSON payloads as prose, indexed them into FTS5 and the entity index, and
spent retrieval budget on markup.

The asymmetry was the bug, so the fix is one shared implementation
(`noise_filter.strip_code_fences`) called by both paths. These tests
ingest one document both ways and compare.
"""
from __future__ import annotations

import json
from pathlib import Path

from aelfrice.ingest import ingest_jsonl
from aelfrice.scanner import scan_repo
from aelfrice.store import MemoryStore

# A marker that appears only inside the fence. If it reaches a belief on
# either path, that path stored the fence body.
_FENCE_MARKER = "zzfencebodyzz"

_DOCUMENT = f"""The retrieval budget is fifty beliefs per turn.

```xml
<aelfrice-rebuild session_id="20260427T154010Z-3f8a">
  <recent-turns>{_FENCE_MARKER}</recent-turns>
</aelfrice-rebuild>
```

Locked beliefs are injected ahead of the ranked lane every turn.
"""


def _onboard_contents(tmp_path: Path) -> list[str]:
    root = tmp_path / "onboard"
    root.mkdir()
    (root / "doc.md").write_text(_DOCUMENT, encoding="utf-8")
    store = MemoryStore(":memory:")
    try:
        scan_repo(store, root, now="2026-08-05T00:00:00Z")
        rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT content FROM beliefs"
        ).fetchall()
        return [str(r["content"]) for r in rows]
    finally:
        store.close()


def _transcript_contents(tmp_path: Path) -> list[str]:
    p = tmp_path / "turns.jsonl"
    p.write_text(
        json.dumps({
            "schema_version": 1, "ts": "2026-08-05T00:00:00Z",
            "role": "user", "text": _DOCUMENT,
            "session_id": "S1", "turn_id": "t1",
        }) + "\n",
        encoding="utf-8",
    )
    store = MemoryStore(":memory:")
    try:
        ingest_jsonl(store, p)
        rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT content FROM beliefs"
        ).fetchall()
        return [str(r["content"]) for r in rows]
    finally:
        store.close()


def test_neither_path_stores_the_fence_body(tmp_path: Path) -> None:
    onboard = _onboard_contents(tmp_path)
    transcript = _transcript_contents(tmp_path)
    assert onboard, "onboard produced no beliefs — fixture is inert"
    assert transcript, "transcript produced no beliefs — fixture is inert"
    for label, contents in (("onboard", onboard), ("transcript", transcript)):
        assert not [c for c in contents if _FENCE_MARKER in c], (label, contents)
        assert not [c for c in contents if "```" in c], (label, contents)


def test_both_paths_keep_the_prose_around_the_fence(tmp_path: Path) -> None:
    """The strip must remove the fence, not the document.

    Without this assertion a fix that dropped every paragraph would pass
    the test above.
    """
    onboard = _onboard_contents(tmp_path)
    transcript = _transcript_contents(tmp_path)
    for label, contents in (("onboard", onboard), ("transcript", transcript)):
        joined = "\n".join(contents)
        assert "retrieval budget is fifty beliefs" in joined, (label, contents)
        assert "Locked beliefs are injected" in joined, (label, contents)


def test_fence_derived_belief_sets_agree(tmp_path: Path) -> None:
    """The parity assertion itself: the fence contributes to neither set.

    Full set equality is not the contract — the onboard path is
    paragraph-granular and the transcript path is sentence-granular, so
    the two sets legitimately differ in shape. What must agree is the
    fence-derived subset, which is now empty on both sides.
    """
    onboard = {c for c in _onboard_contents(tmp_path) if _FENCE_MARKER in c or "```" in c}
    transcript = {
        c for c in _transcript_contents(tmp_path) if _FENCE_MARKER in c or "```" in c
    }
    assert onboard == transcript == set()
