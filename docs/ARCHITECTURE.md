# ARCHITECTURE

Module map, data flow, and design decisions for aelfrice.

> Pre-`v1.0.0`. Architecture stabilises at `v1.0.0`; the `v1.x` line
> recovers `v2.0` features incrementally with evidence justifying each
> addition.

---

## Design principles

1. **Stdlib + SQLite only.** No vector DB, no embeddings, no cloud
   sync, no LLM in the hot path. Pure Python 3.12 stdlib plus
   `sqlite3` (with WAL + FTS5). The `[mcp]` extra is the only
   non-stdlib dependency, and it's optional.
2. **Bayesian, not "vibes".** Confidence is a Beta-Bernoulli
   posterior (`α / (α + β)`). Updates are mathematically grounded; no
   ad-hoc score arithmetic.
3. **`apply_feedback` is the central endpoint.** Not an admin
   command, not a hook, not an undocumented path. Read+write of
   `demotion_pressure` is required.
4. **Clean by construction, not clean by audit.** Filtering is a
   tripwire, not a gate.
5. **Locks are user-asserted ground truth.** A `user`-locked belief
   short-circuits decay (lock floor) and bypasses the L1 token
   budget on retrieval. Contradicting feedback against a lock
   accumulates `demotion_pressure`; cross a threshold and the lock
   auto-demotes.

---

## Module map

All source lives under `src/aelfrice/`. Each file is single-purpose;
cross-module imports are one-directional (lower in the list imports
from higher).

| Module | Responsibility |
|---|---|
| `models.py` | `Belief`, `Edge`, `FeedbackEvent`, `OnboardSession` dataclasses; module-level constants for belief / edge / lock / onboard-state types. No I/O, no SQL. |
| `scoring.py` | `posterior_mean`, `decay`, `relevance_combiner`. Type-specific half-lives (factual 14d, requirement 30d, preference 12w, correction 24w). Lock-floor short-circuit. Decay target is the Jeffreys prior `(0.5, 0.5)`. |
| `store.py` | SQLite schema, WAL setup, FTS5 mirror of `beliefs.content`, CRUD for beliefs / edges / feedback / onboard sessions, `propagate_valence` (broker-confidence-attenuated). |
| `retrieval.py` | `retrieve(store, query, token_budget)`. Two-layer: L0 = locked beliefs auto-loaded regardless of query; L1 = FTS5 BM25 keyword hits. L0 is never trimmed by the budget; L1 is trimmed from the tail. Char-based token estimate (4 chars/token). |
| `feedback.py` | `apply_feedback(store, belief_id, valence, source)` — the single Bayesian-update path at runtime. Writes a row to `feedback_history` for every successful update. Drives demotion-pressure increment and auto-demote at threshold. |
| `correction.py` | No-LLM heuristic correction detector (text-pattern based). Producer of `correction`-typed beliefs during onboarding and feedback. |
| `classification.py` | `TYPE_PRIORS` plus a regex fallback that maps onboarding candidates onto belief types. Polymorphic onboard state machine. |
| `scanner.py` | `scan_repo(store, path)` orchestrator combining three extractors (filesystem walk, git log, AST) with classification and the store. Idempotent against a previously onboarded path. |
| `health.py` | Regime classifier (`insufficient_data` / `supersede` / `ignore` / `balanced`) backed by aggregate confidence, lock density, edge density. |
| `cli.py` | `argparse`-driven 10-subcommand CLI. Resolves DB path from `$AELFRICE_DB` or `~/.aelfrice/memory.db`. Entry point exposed as `aelf` in `[project.scripts]`. |
| `mcp_server.py` | FastMCP server with 8 tools mirroring the CLI. `[mcp]` optional dependency. |
| `setup.py` | Idempotent install / uninstall of a Claude Code `UserPromptSubmit` hook in `settings.json`. Atomic on-disk write via sibling tempfile + `os.replace`. |
| `hook.py` | `aelfrice.hook:main` — process Claude Code spawns when the hook fires. Reads JSON payload from stdin, runs retrieval, emits an `<aelfrice-memory>...</aelfrice-memory>` block on stdout. Non-blocking by contract. Entry point exposed as `aelf-hook`. |
| `slash_commands/` | One `<cmd>.md` per CLI subcommand. Shipped as package data so future tooling can copy them into `~/.claude/commands/aelf/`. A test enforces 1:1 correspondence with CLI subparsers. |

Removed from earlier rounds: `config.py` (folded into `cli.py` at v0.6.0).

---

## Data model

### `Belief`

| Field | Type | Purpose |
|---|---|---|
| `id` | `str` | Stable identifier (16-char hex for user locks; content-hash-derived elsewhere). |
| `content` | `str` | The belief text. Mirrored into the FTS5 virtual table. |
| `content_hash` | `str` | SHA-256 of `content`. Used to deduplicate against re-ingest. |
| `alpha`, `beta` | `float` | Beta-Bernoulli posterior parameters. Confidence is `α / (α + β)`. |
| `type` | `str` | One of `factual` / `correction` / `preference` / `requirement`. Drives decay half-life. |
| `lock_level` | `str` | One of `none` / `user`. `user` short-circuits decay. |
| `locked_at` | `str?` | ISO-8601 UTC timestamp of the lock event, or `None`. |
| `demotion_pressure` | `int` | Accumulator of contradicting-feedback events against a lock; threshold-triggered auto-demote. |
| `created_at` | `str` | ISO-8601 UTC creation time. |
| `last_retrieved_at` | `str?` | ISO-8601 UTC of most recent retrieval, or `None`. |

### `Edge`

Five edge types: `SUPPORTS`, `CITES`, `CONTRADICTS`, `SUPERSEDES`,
`RELATES_TO`. `propagate_valence` only walks `SUPPORTS` and
`CONTRADICTS` paths; the others are informational and load-bearing
for retrieval ranking.

### `FeedbackEvent`

Audit row written by every successful `apply_feedback` call. Records
`belief_id`, `valence`, `prior_alpha`, `prior_beta`, `new_alpha`,
`new_beta`, `source`, `created_at`. Enables post-hoc reconstruction
of the brain's update history.

### `OnboardSession`

State machine row driving the polymorphic onboarding flow. States
include `pending` and `completed`; intermediate states track which
extractor stage is in flight.

---

## Bayesian update path

`apply_feedback(store, belief_id, valence, source)`:

1. Load the belief.
2. **If `valence > 0`:** `α ← α + 1`. Confidence rises.
3. **If `valence < 0`:** `β ← β + 1`. Confidence falls. Plus, if the
   belief is locked, walk all `CONTRADICTS` edges and increment
   `demotion_pressure` for the lock anchor; if pressure crosses
   `DEMOTION_THRESHOLD` (default 5), auto-demote the lock.
4. Write the new `α`, `β`, `demotion_pressure`, `lock_level` back
   atomically.
5. Append a `FeedbackEvent` row. Always.

The write path goes through `store.update_belief` which is a single
SQLite UPDATE wrapped in the connection's default autocommit
transaction. No two-phase write; correctness comes from each
mutation being a single SQL statement.

---

## Retrieval flow

`retrieve(store, query, token_budget=2000, l1_limit=50)`:

```
┌──────────────────────────────┐
│ L0: store.list_locked()      │  always loaded; never trimmed
│   sorted by locked_at desc   │
└─────────────┬────────────────┘
              │
┌─────────────▼────────────────┐
│ L1: FTS5 BM25 keyword search │  query escaped; ranked by relevance
│   limit l1_limit             │
└─────────────┬────────────────┘
              │
┌─────────────▼────────────────┐
│ Dedupe: drop L1 hits whose   │
│   id appears in L0           │
└─────────────┬────────────────┘
              │
┌─────────────▼────────────────┐
│ Trim L1 from the tail until  │
│   sum(estimated_tokens) ≤    │
│   token_budget. L0 is never  │
│   trimmed.                   │
└──────────────────────────────┘
```

Token estimate: `(len(content) + 3) // 4`. Conservative.

Empty query: returns L0 only (FTS5 has nothing to match).

Locked-set overflow: if L0 alone exceeds `token_budget`, the full L0
set is still returned and L1 is empty. Locks always win.

---

## Onboarding flow

`scan_repo(store, project_path)` runs three extractors against a
project directory, classifies the candidates, and inserts non-duplicate
beliefs:

1. **Filesystem walk** — non-binary text files matched against
   patterns. Produces `factual` / `requirement` candidates.
2. **Git log** — commit messages and authorship metadata.
   Produces `factual` candidates and edges between commits and
   touched-file beliefs.
3. **AST extractor** — Python AST (initially) for function and class
   names with their docstrings. Produces `factual` candidates.

The classification stage routes each candidate to a belief type via
`TYPE_PRIORS` plus a regex fallback. Idempotent: re-running
`scan_repo` against the same path skips beliefs whose `content_hash`
already exists.

---

## Claude Code integration

```
                     ┌────────────────────────────────────┐
                     │ ~/.claude/settings.json            │
                     │   hooks.UserPromptSubmit: [        │
                     │     {hooks:[{type:"command",       │
                     │              command:"aelf-hook"}]}│
                     │   ]                                │
                     └────────────────┬───────────────────┘
                                      │  written by aelf setup
                                      │
                          ┌───────────▼────────────┐
                          │  Claude Code (LLM)     │
                          └───────────┬────────────┘
                                      │ user submits prompt
                                      │ JSON payload on stdin →
                          ┌───────────▼────────────┐
                          │  aelf-hook subprocess  │
                          │  (aelfrice.hook:main)  │
                          └───────────┬────────────┘
                                      │ retrieve(store, prompt)
                          ┌───────────▼────────────┐
                          │  ~/.aelfrice/memory.db │
                          └───────────┬────────────┘
                                      │ <aelfrice-memory> block on stdout
                          ┌───────────▼────────────┐
                          │  Claude Code injects   │
                          │  block as added context│
                          └────────────────────────┘
```

Non-blocking contract: `aelf-hook` always exits 0. Empty stdin,
malformed JSON, missing `prompt` field, retrieval exceptions — all
emit nothing on stdout and exit 0. Internal exceptions are written
to stderr (Claude Code captures and surfaces these in the hook log)
but never propagate.

---

## Test layers

| Layer | Marker | What it covers |
|---|---|---|
| Unit | (default) | One property per test. Pyright strict. ~5s suite-wide timeout. |
| Property | (default) | Pre-registered invariants: Bayesian inertia, decay-required, lock-floor sharpness, token-budget invariant, broker-attenuation. |
| Regression | `@pytest.mark.regression` | Cumulative integration scenarios. One per shipped milestone with cross-module behavior: retrieval round-trip, feedback loop, onboarding flow, Claude Code setup→hook→unsetup. |

Run all: `uv run pytest`. Regression only:
`uv run pytest -m regression`.

---

## What's out of scope through `v1.0.0`

These features were in `v2.0` but are deliberately deferred until the
`v1.x` line because the rebuild aims at a small evidence-backed core
first:

- HRR / holographic reduced representation retrieval.
- BFS multi-hop graph retrieval.
- Entity index / NER.
- LLM in the hot path (classification, correction, retrieval).
- Sentence-transformer embeddings.
- Multi-source provenance tagging.
- Bitemporal `event_time` (separate from `created_at`).
- Rigor-tier metadata.

Each of these will land with a benchmark showing it improved a
measurable end-to-end metric over the v1.0 baseline.
