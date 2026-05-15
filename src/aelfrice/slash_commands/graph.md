---
name: aelf:graph
description: Emit a query-anchored subgraph (DOT or JSON) around a belief, expanded N hops via BFS. Deterministic; pipe DOT to `dot -Tsvg` to render.
argument-hint: <belief-id-or-keyword> [--hops N] [--edge-types T1,T2,...] [--format dot|json] [--preview-chars N] [--out PATH]
allowed-tools:
  - Bash
---
<objective>
Surface the small subgraph rooted at a belief, expanded N hops via the
BFS substrate. Query-anchored by design — the full graph is rarely the
user's question; "what's connected to X within N hops" almost always is.
Static snapshot; deterministic across runs. Default format is Graphviz
DOT (`dot`, `neato`, `sfdp` consumable); `--format json` emits a
renderer-agnostic `{nodes, edges}` dict.
</objective>

<process>
Run: `uv run aelf graph $ARGUMENTS`
Display the output verbatim. Do not add commentary.
</process>
