# v2.0 evaluation corpus — schema (#307)

Six bench-gated v2.0 modules ship/no-ship on positive impact against a labeled
corpus. This directory holds that corpus. The harness in #288 reads it; the
modules in #193, #197, #199, #201, #228, #229 are evaluated against it.

## Layout

```
tests/corpus/v2_0/
├── README.md                          (this file)
├── dedup/                             #197
│   └── *.jsonl
├── enforcement/                       #199
│   └── *.jsonl
├── contradiction/                     #201
│   └── *.jsonl
├── wonder_consolidation/              #228
│   └── *.jsonl
├── promotion_trigger/                 #229
│   └── *.jsonl
└── sentiment/                         #193
    └── *.jsonl
```

One JSONL file per logical batch so each bench gate has independent fixtures
and `git diff` stays readable. Filenames are advisory; the harness globs
`*.jsonl` per module directory.

## Per-line shape

Every line is a JSON object. Field set by module. Common envelope fields are
required for **all** modules:

| Field | Type | Description |
|---|---|---|
| `id` | string | Stable unique id within the module file. Conventional prefix `<module>-<short>-<NNN>`. |
| `provenance` | string | Where the example came from (transcript hash, doc ref, "synthetic-v0.X", etc.). Free text; required to be non-empty. |
| `labeller_note` | string | One-line rationale for the label. Required to be non-empty. Used for audit + future re-labelling. |
| `label` | string | Module-specific. Allowed values listed below. |
| `seed` | bool (optional) | `true` for examples committed only to anchor the schema. **Seed rows do NOT count toward the v0.1 ≥50/module target.** Default `false` if omitted. |

### Module-specific fields

| Module | Extra required fields | Allowed `label` values |
|---|---|---|
| `dedup` | `belief_a` (string), `belief_b` (string) | `duplicate`, `near-duplicate`, `distinct` |
| `enforcement` | `user_directive` (string), `agent_output` (string) | `compliant`, `violated`, `n/a` |
| `contradiction` | `belief_a` (string), `belief_b` (string) | `contradicts`, `compatible`, `unrelated` |
| `wonder_consolidation` | `seed_belief` (string), `retrieved_neighbors` (list[string]) | `1`, `2`, `3`, `4`, `5` (phantom-quality rating) |
| `promotion_trigger` | `belief_sequence` (list[string]) | `should_promote`, `should_not` |
| `sentiment` | `user_message` (string) | `positive`, `negative`, `neutral` |

## v0.1 acceptance (per #307)

- ≥ 50 non-seed entries per module file (300 total).
- Real examples only — no synthetic generation in v0.1. Synthetic scaffolding
  is permitted in v0.2 after #228 settles consolidation strategy.
- Schema validation runs in CI (see `tests/test_corpus_schema.py`).

## Validation

`tests/test_corpus_schema.py` enforces:

1. Every `*.jsonl` file under `tests/corpus/v2_0/<module>/` parses line-by-line.
2. Every row has the common envelope fields, non-empty `provenance` and
   `labeller_note`, and a module-allowed `label` value.
3. Every row has the module-specific extra fields with the right shape.
4. `id` values are unique per module.

The test does **not** enforce the ≥50 threshold yet — that flips on once v0.1
labelling lands.
