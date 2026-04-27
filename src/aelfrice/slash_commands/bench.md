---
name: aelf:bench
description: Run the aelfrice benchmark harness and print a reproducible JSON score report.
allowed-tools:
  - Bash
---
<objective>
Produce a publishable score for the current aelfrice build by
running the v0.9.0-rc benchmark harness: 16 hand-authored beliefs
across 4 topics, 16 queries with one known correct answer each,
scored by hit@1 / hit@3 / hit@5 / MRR plus p50 and p99 retrieval
latency.
</objective>

<process>
Run: `uv run aelf bench`

The default invocation seeds an in-memory SQLite store with the
synthetic corpus, runs every query, and prints a single JSON
document. Pass `--db PATH` to seed a real on-disk store instead
(useful for inspecting the post-run state). Pass `--top-k N` to
override the retrieval depth used for hit@k accounting.

Display the JSON output verbatim. Do not add commentary.
</process>
