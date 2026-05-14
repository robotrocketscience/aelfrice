# Feature spec: `aelf core` CLI (#439)

**Status:** implementation spec
**Issue:** #439
**Sibling commands:** `aelf locked` (`src/aelfrice/cli.py:_cmd_locked`), `aelf unlock` (`src/aelfrice/cli.py:_cmd_unlock`)
**ROADMAP slot:** v2.0.0 — *"`core` / `unlock` / `delete` / `confirm` (CLI surface)"*

---

## Purpose

Surface the **load-bearing subset** of the belief store — the beliefs that
anchor retrieval and that a human reviewer should look at first when deciding
whether the store is healthy. `aelf locked` already lists L0 ground truth;
`aelf core` widens the lens to include independently corroborated beliefs and
beliefs with strong positive posteriors, so the operator sees the full
foundation, not just the user-asserted slice.

---

## Definition of "core"

A belief is **core** if **any** of the following hold:

1. **Locked** — `lock_level != 'none'` (any L0 lock tier).
2. **Corroborated** — `corroboration_count >= MIN_CORROBORATION` (default 2).
3. **High posterior** — `posterior_mean >= MIN_POSTERIOR` (default 2/3) **AND**
   `alpha + beta >= MIN_ALPHA_BETA` (default 4).

Where `posterior_mean = alpha / (alpha + beta)`.

### Why these three signals

| Signal | What it captures | Why it's load-bearing |
|---|---|---|
| Lock | User-asserted ground truth | Operator has explicitly declared this canonical (#178). |
| Corroboration ≥ 2 | Same content-hash re-ingested from ≥2 distinct sources (`belief_corroborations`, #190) | Independent confirmation — not just self-reinforcement from one source. |
| Posterior ≥ 2/3 with α+β ≥ 4 | Multi-event majority-positive Beta-Bernoulli posterior | Distinguishes "got used a lot, kept being right" from one-shot `confirm` events. |

### Why these defaults

- **`MIN_CORROBORATION = 2`** — the prior is `corroboration_count = 0` (no
  re-ingests beyond the first). `1` would mean "seen a second time from the
  same source," `2` is the first independent re-confirmation.
- **`MIN_POSTERIOR = 2/3`** — the Beta(1,1) prior has mean 0.5; one positive
  event Beta(2,1) is exactly 2/3. Pairing with `MIN_ALPHA_BETA >= 4` rules out
  the single-event case so the threshold genuinely reflects sustained
  positive feedback.
- **`MIN_ALPHA_BETA = 4`** — the prior has α+β = 2; ≥4 means at least two net
  feedback events have landed. With the 2/3 posterior gate this requires at
  minimum Beta(3,1), i.e., two positive events and zero harmful, or any other
  multi-event majority-positive shape.

All three thresholds are tunable via flags so the operator can dial the lens
wider or tighter without rebuilding.

### What `core` is **not**

- **Not "everything"** — uses thresholds; non-load-bearing beliefs are
  excluded.
- **Not duplicate of `aelf locked`** — `locked` is L0 only; `core` is L0 plus
  the corroboration / posterior subsets.
- **Not the same as `aelf wonder`** — `wonder` surfaces *consolidation
  candidates* (beliefs that look like they should merge); `core` surfaces
  *currently load-bearing beliefs* (beliefs that already anchor retrieval).

---

## Contract

```
aelf core [--json]
          [--limit N]
          [--min-corroboration N]
          [--min-posterior FLOAT]
          [--min-alpha-beta N]
          [--locked-only]
          [--no-locked]
```

| Flag | Default | Notes |
|---|---|---|
| `--json` | off | Switch text → JSON list output. |
| `--limit N` | none | Cap result count after filtering / sort. |
| `--min-corroboration N` | `2` | Threshold for the corroboration signal. `0` disables. |
| `--min-posterior FLOAT` | `0.6666...` | Threshold for posterior-mean signal. `0.0` disables. |
| `--min-alpha-beta N` | `4` | Co-gate on posterior signal — rules out single-event case. |
| `--locked-only` | off | Equivalent to setting both other signals to "disabled". |
| `--no-locked` | off | Suppress the locked subset (debug — surface only the corroboration / posterior subsets). |

`--locked-only` and `--no-locked` are mutually exclusive; passing both is a CLI
error (exit 2 — argparse default).

Exit codes:

- `0` — success (including empty result; "store empty" message goes to stdout
  per `aelf locked` precedent).
- `1` — store I/O error (matches `aelf locked` behaviour).

---

## Output — text

One line per belief, sorted as: locked first (DESC by `locked_at`, ASC by `id`
tiebreak — same as `aelf locked`), then non-locked by `posterior_mean` DESC,
`id` ASC tiebreak.

```
<belief-id> [LOCK,CORR=3,α=4.0,β=1.0,μ=0.800]: <content one-line>
```

Tag block fields, comma-separated, omitted when not signalling:

- `LOCK` — present if `lock_level != 'none'`.
- `CORR=N` — `corroboration_count` value, if `>= MIN_CORROBORATION`.
- `α=F.F`, `β=F.F`, `μ=0.NNN` — present if the posterior signal triggered (so
  the operator can see *why* a non-locked belief is in the list).

Empty store / no matches:

```
no core beliefs
```

(parallels `aelf locked`'s `no locked beliefs`).

## Output — `--json`

```json
[
  {
    "id": "...",
    "content": "...",
    "lock_level": "user",
    "alpha": 9.0,
    "beta": 0.5,
    "posterior_mean": 0.947,
    "corroboration_count": 0,
    "signals": ["lock"]
  },
  ...
]
```

`signals` is a sorted list drawn from `{"lock", "corroboration", "posterior"}`
— exactly the signals that put this belief in the result. Always non-empty
for any returned row.

---

## Implementation

`_cmd_core` in `src/aelfrice/cli.py`, registered alongside `_cmd_locked`. No
new store method needed — composition over existing API:

```python
def _cmd_core(args, out):
    store = _open_store()
    try:
        locked = [] if args.no_locked else store.list_locked_beliefs()
        candidates = []
        if not args.locked_only:
            # Walk all beliefs once and filter in Python — fine at v2.0
            # store sizes (~10^4 rows). If this becomes a hot path the
            # store can grow a `list_core_candidates(min_corr, min_post,
            # min_ab)` method, but defer until measurement justifies it.
            for bid in store.list_belief_ids():
                b = store.get_belief(bid)
                if b is None or b.lock_level != "none":
                    continue  # locked already handled above
                if _qualifies(b, args):
                    candidates.append(b)
    finally:
        store.close()
    _emit(locked, candidates, args, out)
```

Where `_qualifies` checks the corroboration / posterior gates, and `_emit`
dedupes-by-id (locked subset takes precedence over candidates), applies
`--limit` after sort, and routes text vs `--json`.

### Why no new store method (v2.0)

`list_locked_beliefs` is one query; `list_belief_ids` + `get_belief` per id is
N+1, but at v2.0 store sizes (target ~10^4 beliefs per project per #437
benchmarks) the round-trip cost is in the low milliseconds and the simplicity
is worth more than the optimisation. If `aelf core` becomes a per-turn hot
path (it isn't — it's a research-line / operator-introspection verb), promote
to a single SQL query.

### Mutual exclusion in argparse

`--locked-only` and `--no-locked` registered with
`subparser.add_mutually_exclusive_group()`; argparse raises exit 2 on conflict
with a clear stderr message. No custom validation.

### Slash command

`src/aelfrice/slash_commands/core.md` mirrors `unlock.md` and `confirm.md`
(`$ARGS` passthrough). Add to `EXPECTED_COMMANDS` in
`tests/test_slash_commands.py`.

### Doc update

Append a row to `docs/user/COMMANDS.md` between `locked` (line 20) and `unlock`
(line 21):

```
| `core [--json] [--limit N] [--min-corroboration N] [--min-posterior FLOAT] [--min-alpha-beta N] [--locked-only] [--no-locked]` | (v2.0+, #439) Surface load-bearing beliefs: locked ∪ {corroboration ≥ 2} ∪ {posterior ≥ 2/3 with α+β ≥ 4}. Read-only. |
```

---

## Test plan (sketch — implementation PR)

Unit tests in `tests/test_cli_core.py` against a fixture store with five
beliefs hand-built to exercise each branch:

| Fixture id | Lock | α, β | corr | Expected `core` |
|---|---|---|---|---|
| `b-locked` | user | 1, 1 | 0 | yes (LOCK) |
| `b-corr` | none | 1, 1 | 3 | yes (CORR) |
| `b-posterior` | none | 4, 1 | 0 | yes (μ=0.8, α+β=5) |
| `b-thin-posterior` | none | 2, 1 | 0 | **no** — μ=0.667 but α+β=3 |
| `b-prior` | none | 1, 1 | 0 | no |

Tests cover:

1. Default output text — all four selection signals produce expected rows.
2. `--json` round-trips through `json.loads`; signals list is correct.
3. `--locked-only` filters to `b-locked` only.
4. `--no-locked` suppresses `b-locked`.
5. `--locked-only --no-locked` exits 2.
6. `--limit 1` after default sort returns `b-locked` (locked-first).
7. Empty store prints `no core beliefs` and exits 0.
8. Threshold flags — `--min-corroboration 4` drops `b-corr`; `--min-posterior
   0.0 --min-alpha-beta 0` includes everything that isn't the prior.

Slash-command registration test (`tests/test_slash_commands.py`) extended with
`core` in `EXPECTED_COMMANDS`.

---

## Out of scope (deferred)

- **PageRank weighting.** The original issue body mentions "high-PageRank" as
  a candidate signal. PageRank infrastructure is not in the store as of v2.0;
  the closest existing primitive is `graph_spectral.py` (signed-Laplacian /
  heat kernel for retrieval). Adding a graph-centrality signal to `aelf core`
  is a follow-up once a PageRank pass is computed and persisted; tracked as
  future work, not blocking this CLI.
- **Hibernation interaction.** Hibernated beliefs (`hibernation_score`
  populated) are still candidates if they meet the lock / corroboration /
  posterior gates; the spec does not introduce a hibernation-aware filter.
  If the operator wants hibernation-respecting output, that's a flag to add
  later.
- **`aelf core --explain <id>`.** Useful future verb to print the tag block
  with the threshold values that would change membership. Out of scope for
  the v2.0 ship.

---

## Provenance refs

- #439 — this issue.
- #178 — locks (`lock_level`, `aelf lock` / `aelf locked`).
- #190 — `belief_corroborations` table.
- #441 / #390 — `aelf confirm` (sibling user-affirmation verb).
- ROADMAP `docs/concepts/ROADMAP.md` v2.0.0 row — "core / unlock / delete / confirm
  (CLI surface)".
