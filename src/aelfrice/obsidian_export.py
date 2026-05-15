"""Obsidian vault exporter for the belief graph (#630).

Emits one Markdown note per belief into ``<vault>/aelfrice/``. Each note
carries the typed edges as YAML front-matter lists (queryable via the
Dataview plugin) plus a wikilink section so Obsidian's graph view shows
the link topology. **One-way: DB -> vault.** No reverse sync; SQLite
remains the source of truth.

Two structural limits are accepted, not papered over (per the proposal):

1. **Obsidian's built-in graph view does not scale.** Force-directed
   layout chokes around a few thousand nodes; every zoom or hover
   re-runs layout. Use ``--scope query`` / ``--max-notes`` to bound the
   export, or use ``aelf graph`` (#629) for query-anchored visualisation.
2. **Obsidian's graph view is untyped.** Wikilinks are wikilinks; the
   aelfrice edge types are not distinguishable in the graph. Edge types
   are preserved in YAML front-matter and queryable via Dataview, but
   graph-view will not show them.

Determinism contract
--------------------
Same store state + same flags -> byte-identical note bodies and
identical file set under ``<vault>/aelfrice/``. Edge lists are sorted by
destination id; wikilinks are sorted by edge type then destination id.

Wipe-and-emit semantics
-----------------------
``write_vault`` removes the entire ``<vault>/aelfrice/`` directory
before writing the new note set (ratified Q2). This is one-way export;
hand-edited adjacent notes outside that directory are not touched.
Diff-and-update is intentionally out of scope for v1.
"""
from __future__ import annotations

import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from aelfrice.bfs_multihop import expand_bfs
from aelfrice.models import (
    EDGE_POTENTIALLY_STALE,
    EDGE_TYPES,
    LOCK_USER,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore

# Hard ceiling on the ``--scope all`` export (ratified Q3). Beyond this
# ``--force`` is required; the help-text refusal cites this constant by
# value, so changes here propagate to the operator-facing message.
HARD_MAX_NOTES: Final[int] = 5000

# Default cap on the export when no explicit ``--max-notes`` is given
# (proposal default, preserved by the operator ratification).
DEFAULT_MAX_NOTES: Final[int] = 500

# Short-id prefix length used in note filenames. Keep stable -- altering
# this regenerates filenames on the next export and invalidates any
# vault-side wikilinks the user has hand-authored.
SHORT_ID_LEN: Final[int] = 12

# Cap on the slug-portion of a filename. Belief content can be long;
# the slug just exists for human readability. Keep below typical
# filesystem name limits with headroom for the ``<short_id>-`` prefix
# and ``.md`` suffix.
SLUG_MAX_LEN: Final[int] = 60

# Subdirectory under the operator-supplied vault path that we own.
# Wipe-on-export only touches this directory.
VAULT_SUBDIR: Final[str] = "aelfrice"

# YAML keys for the edge lists. We emit every structural edge type in
# ``EDGE_TYPES`` plus the ``POTENTIALLY_STALE`` marker (which is not in
# EDGE_TYPES but carries Dataview-queryable signal). Keys are
# lowercased edge-type names; future edge types added to the model
# surface here automatically, which is the intended invariant -- the
# proposal's enumeration of 7 types was outdated against the live model
# (which now has 10 structural edges plus the marker).
_EDGE_YAML_KEYS: Final[tuple[str, ...]] = tuple(
    sorted({t.lower() for t in EDGE_TYPES} | {EDGE_POTENTIALLY_STALE.lower()})
)

# Verbatim disclaimer strings. Both must appear in CLI help text AND in
# the README section -- the proposal locks this as an acceptance criterion
# ("if the disclaimers feel embarrassing, the feature isn't ready").
PERF_DISCLAIMER: Final[str] = (
    "Obsidian's built-in graph view does not scale. Force-directed "
    "layout chokes around a few thousand nodes; every zoom or hover "
    "re-runs layout. Do not export your full belief store and expect "
    "the graph view to be usable. Use --scope query or --max-notes to "
    "bound the export, or use `aelf graph` (sister issue) for "
    "query-anchored visualization that works at any store size."
)
EDGE_TYPE_DISCLAIMER: Final[str] = (
    "Obsidian's graph view is untyped. Wikilinks are wikilinks; the "
    "aelfrice edge types are not distinguishable in the graph. Edge "
    "types are preserved in YAML front-matter and queryable via the "
    "Dataview plugin (https://blacksmithgu.github.io/obsidian-dataview/), "
    "but graph view will not show them. This is an Obsidian limitation, "
    "not an export bug."
)


@dataclass(frozen=True)
class ExportResult:
    """Summary of a write_vault run for CLI surface and tests."""
    notes_written: int
    vault_dir: Path
    truncated_at_cap: bool


_SLUG_KILL_RE = re.compile(r"[^a-z0-9]+")
_SLUG_STRIP_RE = re.compile(r"^-+|-+$")


def slugify(text: str, max_len: int = SLUG_MAX_LEN) -> str:
    """Return a kebab-case ascii slug from ``text``.

    Deterministic, no external deps. Empty / all-punctuation input
    collapses to ``"belief"`` (filenames must have a slug body so the
    ``<short_id>-<slug>.md`` shape stays consistent).
    """
    lowered = text.lower()
    slug = _SLUG_KILL_RE.sub("-", lowered)
    slug = _SLUG_STRIP_RE.sub("", slug)
    if not slug:
        return "belief"
    if len(slug) > max_len:
        slug = slug[:max_len]
        slug = _SLUG_STRIP_RE.sub("", slug) or "belief"
    return slug


def note_filename(belief: Belief) -> str:
    """``<short_id>-<slug>.md`` -- deterministic per belief."""
    short = belief.id[:SHORT_ID_LEN]
    return f"{short}-{slugify(belief.content or '')}.md"


def _edges_by_type(edges: list[Edge]) -> dict[str, list[str]]:
    """Group destination ids by edge-type, sorted for determinism.

    Edge types unknown to the YAML key set are dropped silently --
    those are structural-edge oddities that shouldn't surface in the
    user-facing front-matter. The audit path for such edges is
    `aelf doctor`, not this exporter.
    """
    grouped: dict[str, list[str]] = defaultdict(list)
    valid_keys = set(_EDGE_YAML_KEYS)
    for e in edges:
        key = e.type.lower()
        if key not in valid_keys:
            continue
        grouped[key].append(e.dst)
    return {k: sorted(set(v)) for k, v in grouped.items()}


def _yaml_quote(s: str) -> str:
    """Quote a YAML scalar value. Conservative -- always doubles.

    Used for short-prose values (origin, lock state). Belief content
    is not YAML-emitted; it lives in the body.
    """
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _posterior_mean(b: Belief) -> float:
    denom = b.alpha + b.beta
    if denom <= 0:
        return 0.5
    return b.alpha / denom


def render_yaml_frontmatter(
    belief: Belief,
    edges_out: list[Edge],
) -> str:
    """Render the YAML front-matter block including trailing ``---``.

    Always emits every structural edge key (empty list when absent) so
    Dataview queries can rely on field existence. Posterior mean is
    rounded to 4 dp for stability across float-rounding drift.
    """
    grouped = _edges_by_type(edges_out)
    lines: list[str] = ["---"]
    lines.append(f"belief_id: {belief.id}")
    lines.append(f"origin: {_yaml_quote(belief.origin)}")
    lines.append(f"lock_level: {_yaml_quote(belief.lock_level)}")
    lines.append(f"posterior_mean: {round(_posterior_mean(belief), 4)}")
    lines.append(f"alpha: {belief.alpha}")
    lines.append(f"beta: {belief.beta}")
    if belief.type:
        lines.append(f"type: {_yaml_quote(belief.type)}")
    if belief.retention_class:
        lines.append(f"retention_class: {_yaml_quote(belief.retention_class)}")
    for key in _EDGE_YAML_KEYS:
        dsts = grouped.get(key, [])
        if not dsts:
            lines.append(f"{key}: []")
            continue
        lines.append(f"{key}:")
        for dst in dsts:
            short = dst[:SHORT_ID_LEN]
            # Wikilink target is the same short-id prefix used in
            # filenames; Obsidian resolves by exact-or-prefix match
            # depending on vault settings, so emitting the short-id
            # (rather than the full path) keeps things robust.
            lines.append(f'  - "[[{short}]]"')
    lines.append("---")
    return "\n".join(lines) + "\n"


def render_body(
    belief: Belief,
    edges_out: list[Edge],
) -> str:
    """Render the Markdown body: content + provenance + wikilink section.

    The wikilink section gives Obsidian's graph view something to draw
    (it scrapes wikilinks from the body, not the front-matter). Edge
    type is included as plain text so even without Dataview the user
    sees what kind of link it is.
    """
    grouped = _edges_by_type(edges_out)
    parts: list[str] = []
    parts.append(belief.content or "")
    parts.append("")
    parts.append("## Provenance")
    parts.append("")
    parts.append(f"- **Belief id:** `{belief.id}`")
    parts.append(f"- **Origin:** `{belief.origin}`")
    parts.append(f"- **Lock level:** `{belief.lock_level}`")
    parts.append(
        f"- **Posterior:** α={belief.alpha}, β={belief.beta} "
        f"(mean={round(_posterior_mean(belief), 4)})"
    )
    if belief.lock_level == LOCK_USER and belief.locked_at:
        parts.append(f"- **Locked at:** `{belief.locked_at}`")
    parts.append(f"- **Created at:** `{belief.created_at}`")
    parts.append("")
    parts.append("## Connections")
    parts.append("")
    any_edge = False
    for key in _EDGE_YAML_KEYS:
        dsts = grouped.get(key)
        if not dsts:
            continue
        any_edge = True
        parts.append(f"### {key}")
        parts.append("")
        for dst in dsts:
            short = dst[:SHORT_ID_LEN]
            parts.append(f"- [[{short}]]")
        parts.append("")
    if not any_edge:
        parts.append("_(no outbound edges)_")
        parts.append("")
    return "\n".join(parts)


def render_note(belief: Belief, edges_out: list[Edge]) -> str:
    """Build the full note text (front-matter + body)."""
    return render_yaml_frontmatter(belief, edges_out) + "\n" + render_body(
        belief, edges_out
    )


def _select_beliefs_all(
    store: MemoryStore, limit: int
) -> list[Belief]:
    """All active beliefs, deterministic id-ASC, capped at ``limit``."""
    return store.list_active_beliefs(limit=limit, order="id_asc")


def _select_beliefs_recent(
    store: MemoryStore, limit: int
) -> list[Belief]:
    """Most-recently-created active beliefs, capped at ``limit``."""
    return store.list_active_beliefs(limit=limit, order="recent")


def _select_beliefs_query(
    store: MemoryStore,
    query: str,
    *,
    neighborhood_hops: int,
    limit: int,
    k_seeds: int,
) -> list[Belief]:
    """BM25 seeds + N-hop neighbourhood, capped at ``limit``.

    Local seeds only (peer federation is out of scope for this v1; the
    seed-scope plumbing in `_seeds_with_scopes` is BFS-only and the
    export is local-vault by definition). Result is unioned and sorted
    by id ASC for byte-identical output across repeated runs.
    """
    seeds = store.search_beliefs(query, limit=k_seeds)
    if not seeds:
        return []
    if neighborhood_hops < 1:
        return sorted(seeds, key=lambda b: b.id)[:limit]
    hops = expand_bfs(
        seeds,
        store,
        max_depth=neighborhood_hops,
        nodes_per_hop=8,
        total_budget=max(limit, 32),
    )
    by_id: dict[str, Belief] = {b.id: b for b in seeds}
    for hop in hops:
        by_id.setdefault(hop.belief.id, hop.belief)
    ordered = sorted(by_id.values(), key=lambda b: b.id)
    return ordered[:limit]


def select_beliefs(
    store: MemoryStore,
    *,
    scope: str,
    query: str | None,
    max_notes: int,
    neighborhood_hops: int,
    k_seeds: int,
) -> list[Belief]:
    """Resolve ``--scope`` and return the export set.

    Scopes:
      * ``all`` -- every active belief, id-ASC, capped at ``max_notes``.
      * ``recent`` -- newest-first, capped at ``max_notes``.
      * ``query`` -- BM25 seeds + ``neighborhood_hops`` BFS expansion.
    """
    if scope == "all":
        return _select_beliefs_all(store, max_notes)
    if scope == "recent":
        return _select_beliefs_recent(store, max_notes)
    if scope == "query":
        if not query:
            raise ValueError("--scope query requires --query <text>")
        return _select_beliefs_query(
            store,
            query,
            neighborhood_hops=neighborhood_hops,
            limit=max_notes,
            k_seeds=k_seeds,
        )
    raise ValueError(f"unknown scope: {scope!r}")


def write_vault(
    beliefs: list[Belief],
    store: MemoryStore,
    vault_root: Path,
) -> ExportResult:
    """Wipe ``<vault_root>/aelfrice/`` and re-emit one note per belief.

    Pure-ish: reads ``store.edges_from`` once per belief, writes one
    file per belief. Returns the count of notes written. Does not
    touch any path outside ``<vault_root>/aelfrice/``.
    """
    target = vault_root / VAULT_SUBDIR
    if target.exists():
        if not target.is_dir():
            raise NotADirectoryError(
                f"{target} exists and is not a directory; refusing to wipe"
            )
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=False)

    # Sort by id ASC so write order is deterministic; matters for
    # filesystems that surface inode creation order via readdir().
    for belief in sorted(beliefs, key=lambda b: b.id):
        edges_out = store.edges_from(belief.id)
        note = render_note(belief, edges_out)
        (target / note_filename(belief)).write_text(note, encoding="utf-8")

    return ExportResult(
        notes_written=len(beliefs),
        vault_dir=target,
        truncated_at_cap=False,
    )
