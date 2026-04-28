"""Context-rebuilder: PreCompact-driven retrieval-curated context block.

When Claude Code's context window approaches its compaction threshold,
the harness fires a PreCompact hook. This module is the script-side
half of that hook: it reads the most recent N turns from a transcript
log, runs aelfrice retrieval against those turns to surface load-
bearing beliefs, and emits a single XML-tag-delimited context block
that Claude Code injects above the next prompt.

v1.1.0-alpha is a vertical slice. Several quality dimensions in the
design spec are deferred until later releases:

  * Session-scoped retrieval. The schema's session_id field is not
    yet populated at write-time; retrieval falls back to global L0
    locked + L1 FTS5 BM25.
  * Triple-extractor query construction. The MVP concatenates recent
    turn text into a flat query string. A triple-extracted query
    will replace this once the extractor ships.
  * Posterior-weighted ranking. The MVP uses the public BM25 ranker.
    The Bayesian-weighted ranker is a v1.3.x candidate.
  * Suppress-mode coordination with the harness. The MVP runs in
    augment mode only -- the rebuild block is emitted, but the
    harness's default summarization still runs. Suppress mode lands
    once eval-harness fidelity calibration justifies it.

The pure rebuild() function is decoupled from the I/O paths so the
eval harness can drive it directly with a pre-loaded list of recent
turns. Two adapters convert different on-disk transcript formats
into that list: read_recent_turns_aelfrice() for the canonical
turns.jsonl format described in docs/specs/transcript_ingest.md, and
read_recent_turns_claude_transcript() for Claude Code's internal
transcript format.

Output schema:

    <aelfrice-rebuild>
      <recent-turns>
        <turn role="user">...</turn>
        <turn role="assistant">...</turn>
      </recent-turns>
      <retrieved-beliefs budget_used="N/M">
        <belief id="..." locked="true">...</belief>
        <belief id="..." locked="false">...</belief>
      </retrieved-beliefs>
      <continue/>
    </aelfrice-rebuild>

The <continue/> marker is the stable signal the model interprets as
"resume the prior task using the above context, do not greet, do not
summarize."
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from aelfrice.models import LOCK_NONE, LOCK_USER, Belief
from aelfrice.store import MemoryStore

DEFAULT_N_RECENT_TURNS: Final[int] = 10
DEFAULT_TOKEN_BUDGET: Final[int] = 2000
DEFAULT_PER_TOKEN_LIMIT: Final[int] = 20
MIN_QUERY_TOKEN_LENGTH: Final[int] = 4
"""Drop tokens shorter than this from the rebuild query.

Single-character and short stopword-class tokens ("a", "the", "is")
are noise on the FTS5 side. The rebuilder uses per-token union
retrieval to work around the public store's AND-only FTS5 semantics;
filtering short tokens keeps the union cheap and high-signal.
"""
_CHARS_PER_TOKEN: Final[float] = 4.0
MAX_TURN_TEXT_CHARS: Final[int] = 500
"""Per-turn text truncation in the rebuild block.

Recent turns are decoration, not load-bearing -- the retrieved beliefs
carry the durable session state. Cap each turn's quoted text so the
block size stays bounded even when one turn is enormous (e.g., a
pasted log).
"""

OPEN_TAG: Final[str] = "<aelfrice-rebuild>"
CLOSE_TAG: Final[str] = "</aelfrice-rebuild>"

_AELFRICE_LOG_RELPATH: Final[Path] = Path(".git") / "aelfrice" / "transcripts" / "turns.jsonl"


@dataclass(frozen=True)
class RecentTurn:
    """One normalized turn fed to rebuild().

    Adapters convert wire-format transcript records into this shape.
    """
    role: str  # "user" or "assistant"
    text: str


def rebuild(
    recent_turns: list[RecentTurn],
    store: MemoryStore,
    *,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> str:
    """Build the rebuild context block. Pure function.

    Given a list of recent turns and an open MemoryStore, retrieve the
    most relevant beliefs and emit the formatted XML block.
    Deterministic given the same inputs and store contents -- the
    eval harness depends on this.

    Empty recent_turns: retrieval returns L0 locked only, the block is
    still well-formed (no <recent-turns> section).

    Retrieval semantics: L0 locked beliefs are always first. L1 hits
    come from a per-token union over the recent-turn text -- workaround
    for the public store's AND-only FTS5 semantics, see
    _retrieve_for_rebuild() docstring.
    """
    hits = _retrieve_for_rebuild(
        store, recent_turns, token_budget=token_budget
    )
    return _format_block(recent_turns, hits, token_budget=token_budget)


def _retrieve_for_rebuild(
    store: MemoryStore,
    recent_turns: list[RecentTurn],
    *,
    token_budget: int,
) -> list[Belief]:
    """Per-token union retrieval over recent-turn text.

    The public retrieve() ANDs all tokens of its query expression
    (because store._escape_fts5_query AND-joins per FTS5 default).
    For rebuild input -- a concatenation of conversational turns --
    a single AND query rarely matches any belief because no single
    document contains all the conversation's words. We instead issue
    one search per significant query token, union the results,
    deduplicate by belief id, and trim to the token budget the same
    way retrieve() does.

    Future replacement: when posterior-weighted ranking ships
    (v1.3+), this function should call into that ranker directly --
    it will produce a single ranked list without needing the
    per-token workaround.

    L0 locked beliefs always come first and are never trimmed. L1 hits
    are ordered by first appearance across the per-token searches.
    """
    locked: list[Belief] = store.list_locked_beliefs()
    locked_ids: set[str] = {b.id for b in locked}

    tokens = _query_tokens(recent_turns)
    l1: list[Belief] = []
    seen_ids: set[str] = set(locked_ids)
    for tok in tokens:
        for b in store.search_beliefs(tok, limit=DEFAULT_PER_TOKEN_LIMIT):
            if b.id in seen_ids:
                continue
            seen_ids.add(b.id)
            l1.append(b)

    used: int = sum(_estimate_belief_tokens(b) for b in locked)
    out: list[Belief] = list(locked)
    for b in l1:
        cost: int = _estimate_belief_tokens(b)
        if used + cost > token_budget:
            break
        out.append(b)
        used += cost
    return out


def _query_tokens(recent_turns: list[RecentTurn]) -> list[str]:
    """Whitespace-split each turn's text, keep tokens >= MIN_QUERY_TOKEN_LENGTH.

    Order is preserved (earliest mention first). Duplicate tokens are
    dropped on first appearance.
    """
    seen: set[str] = set()
    out: list[str] = []
    for t in recent_turns:
        if not t.text:
            continue
        for raw in t.text.split():
            tok = raw.strip()
            if len(tok) < MIN_QUERY_TOKEN_LENGTH:
                continue
            key = tok.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(tok)
    return out


def _estimate_belief_tokens(b: Belief) -> int:
    if not b.content:
        return 0
    return int((len(b.content) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)


def _format_block(
    recent_turns: list[RecentTurn],
    hits: list[Belief],
    *,
    token_budget: int,
) -> str:
    lines: list[str] = [OPEN_TAG]
    if recent_turns:
        lines.append("  <recent-turns>")
        for t in recent_turns:
            text = _normalize_turn_text(t.text)
            role = _xml_attr_value(t.role)
            lines.append(f'    <turn role="{role}">{_xml_escape(text)}</turn>')
        lines.append("  </recent-turns>")
    if hits:
        used_chars = sum(len(b.content) for b in hits)
        lines.append(
            f'  <retrieved-beliefs budget_used="{used_chars}/{token_budget * 4}">'
        )
        for b in hits:
            locked = "true" if b.lock_level == LOCK_USER else "false"
            content = _normalize_turn_text(b.content)
            lines.append(
                f'    <belief id="{_xml_attr_value(b.id)}" '
                f'locked="{locked}">{_xml_escape(content)}</belief>'
            )
        lines.append("  </retrieved-beliefs>")
    lines.append("  <continue/>")
    lines.append(CLOSE_TAG)
    lines.append("")
    return "\n".join(lines)


def _normalize_turn_text(text: str) -> str:
    """Collapse whitespace and truncate to MAX_TURN_TEXT_CHARS."""
    collapsed = " ".join(text.split())
    if len(collapsed) > MAX_TURN_TEXT_CHARS:
        return collapsed[:MAX_TURN_TEXT_CHARS] + "..."
    return collapsed


def _xml_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _xml_attr_value(s: str) -> str:
    return _xml_escape(s).replace('"', "&quot;")


def read_recent_turns_aelfrice(path: Path, n: int) -> list[RecentTurn]:
    """Read tail of an aelfrice turns.jsonl into RecentTurn records.

    Schema per docs/specs/transcript_ingest.md: each line is a JSON
    object with at least {"role": str, "text": str}. Other fields
    (session_id, turn_id, ts, context) are ignored at MVP -- they
    will be used by later releases for session-scoped retrieval.

    Robust: malformed lines are skipped, missing files return empty.
    Reads the entire file (a turns.jsonl is bounded by PreCompact
    rotation; large enough to matter only between rotations).
    """
    if not path.exists():
        return []
    out: list[RecentTurn] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(r, dict):
            continue
        rd = cast(dict[str, object], r)
        role = rd.get("role")
        body = rd.get("text")
        if not isinstance(role, str) or role not in ("user", "assistant"):
            continue
        if not isinstance(body, str) or not body.strip():
            continue
        out.append(RecentTurn(role=role, text=body))
    return out[-n:] if n > 0 else []


def read_recent_turns_claude_transcript(
    path: Path, n: int
) -> list[RecentTurn]:
    """Adapter for Claude Code's internal transcript format.

    MVP-only fallback used when the canonical aelfrice turns.jsonl
    does not exist (typical pre-transcript_ingest setup). Reads
    Claude Code's per-session JSONL at ~/.claude/projects/<hash>/
    <session>.jsonl. Schema is internal-harness JSON and subject to
    change between Claude Code releases; this adapter is best-effort
    and fails closed (returns [] on shape mismatch).

    Records of interest:
      {"type": "user", "message": {"role": "user", "content": "..."}}
      {"type": "assistant", "message": {"role": "assistant",
        "content": [{"type": "text", "text": "..."}, ...]}}

    Tool-call records, sub-agent records, and meta records are
    skipped.
    """
    if not path.exists():
        return []
    out: list[RecentTurn] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(r, dict):
            continue
        rd = cast(dict[str, object], r)
        record_type = rd.get("type")
        if record_type not in ("user", "assistant"):
            continue
        msg = rd.get("message")
        if not isinstance(msg, dict):
            continue
        msg_typed = cast(dict[str, object], msg)
        role = msg_typed.get("role")
        if not isinstance(role, str) or role not in ("user", "assistant"):
            continue
        content = msg_typed.get("content")
        body = _extract_text_from_claude_content(content)
        if not body:
            continue
        out.append(RecentTurn(role=role, text=body))
    return out[-n:] if n > 0 else []


def _extract_text_from_claude_content(content: object) -> str:
    """Pull the human-readable text out of a Claude message content field.

    Claude's content is either a bare string (user prompts) or a list
    of content blocks (assistant turns and structured user messages).
    We concatenate every "text"-typed block; non-text blocks (tool
    use, tool result, images) are ignored.
    """
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in cast(list[object], content):
        if not isinstance(block, dict):
            continue
        bd = cast(dict[str, object], block)
        if bd.get("type") != "text":
            continue
        t = bd.get("text")
        if isinstance(t, str) and t.strip():
            parts.append(t.strip())
    return "\n".join(parts).strip()


def find_aelfrice_log(cwd: Path) -> Path | None:
    """Walk upward from cwd to find a .git/ root, return its turns.jsonl path.

    Returns the path even if the file does not exist -- callers use
    Path.exists() to decide. Returns None if no .git/ is found in any
    ancestor (cwd is outside a git repo).
    """
    cur = cwd.resolve()
    while True:
        if (cur / ".git").exists():
            return cur / _AELFRICE_LOG_RELPATH
        if cur.parent == cur:
            return None
        cur = cur.parent
