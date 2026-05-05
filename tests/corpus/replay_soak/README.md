# `tests/corpus/replay_soak/` — public v0.1 ingest fixture (#403 A)

Hand-authored ingest-log rows that exercise every `INGEST_SOURCE_KIND`
(except `legacy_unknown`). The scheduled `replay-soak` workflow loads
this fixture, derives a belief per row via `aelfrice.derivation.derive`,
appends an `ingest_log` row via `MemoryStore.record_ingest`, inserts the
belief, then asserts `replay_full_equality(store).has_drift is False`.

## Why public

Per the operator ratification on #403 (2026-05-04T19:54:28Z), the v0.1
soak gate runs end-to-end on the public corpus alone — no lab corpus
required. This makes the gate reproducible from any clone, and the
`.replay-soak-status.json` history is part of the public audit trail.

A lab-side accumulating store remains a stretch artifact for richer
drift signal but is **not** the v0.1 gate.

## Authoring discipline (locked boundary)

Per the directory-of-origin rule, no row may originate from `~/.claude/`
(memory files, handoff docs, session transcripts) in any form —
paraphrasing, abstracting, and "synthetic-after-reading" all count.

Allowed sources for row text:

- **Product code / docs** committed under `~/projects/aelfrice` (README,
  CHANGELOG, source comments, public docs).
- **Public-domain text** authored fresh for the corpus (new sentences
  written specifically for this fixture, not derived from any private
  source).

Each row's `provenance` field declares the origin so future re-labellers
can audit. Synthetic-fresh rows use `provenance: "public-domain-v0.1"`;
product-code-derived rows use `provenance: "product:<file>:<line>"` or
similar pointer.

## Layout

```
tests/corpus/replay_soak/
├── README.md                                  (this file)
└── v0.1/
    ├── filesystem_v0_1.jsonl                  10 rows
    ├── git_v0_1.jsonl                         10 rows
    ├── python_ast_v0_1.jsonl                  10 rows
    ├── mcp_remember_v0_1.jsonl                10 rows
    ├── cli_remember_v0_1.jsonl                10 rows
    └── feedback_loop_synthesis_v0_1.jsonl     10 rows
```

v0.1 target: 60 rows total, 10 per source kind. Ratification floor was
60–120; v0.1 starts at the floor and v0.2 expansion can extend per
kind without breaking the runner.

## Per-row shape

Every line is a JSON object. Fields:

| Field | Type | Description |
|---|---|---|
| `id` | string | Stable unique row id within the file. Convention: `<source_kind>-<NNN>`. |
| `source_kind` | string | One of the 6 non-`legacy_unknown` `INGEST_SOURCE_KINDS`. Must equal the file's kind. |
| `source_path` | string \| null | Conventional shape per kind (e.g., `doc:README.md` for `filesystem`). May be null. |
| `raw_text` | string | Non-empty, non-question text that `aelfrice.derivation.derive` will persist. The classifier's `persist=False` paths (empty, question form) are explicitly out of scope here. |
| `raw_meta` | object \| null | Optional per-source-kind metadata that the runner will JSON-encode into the log row. May be null. |
| `provenance` | string | Origin declaration — `public-domain-v0.1` for fresh-authored, `product:<file>` for product-code-derived. Required non-empty. |
| `note` | string | One-line authoring note: which classification path or replay invariant this row exercises. Required non-empty. |

## Validation

`tests/test_replay_soak_corpus.py` enforces:

1. Every `*.jsonl` file under `tests/corpus/replay_soak/v0.1/` parses
   line-by-line.
2. Every row has the fields above with the correct shapes.
3. `source_kind` matches the filename prefix.
4. `id` values are unique within each file.
5. The `replay_full_equality` invariant — for every row, derive the
   belief, write the log + belief, then assert
   `mismatched + derived_orphan == 0` over the full fixture.

Test (4) is the bench-gate equivalent for the soak; the workflow runs
the same code path on `main` daily and appends to
`.replay-soak-status.json`.
