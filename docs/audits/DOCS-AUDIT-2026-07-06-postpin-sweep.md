# Docs audit — 2026-07-06 (post-pin coverage sweep)

> **Coverage-closure pass** for the #1075 v4.0.0 documentation-audit gate. The
> four accuracy tranches + the post-implementation scoring pass verified the
> documentation surface against an inventory **pinned at `fe3f29c4`** (552 `.py`).
> Between that pin and HEAD `f0f284fd`, **20 new `.py` files** landed (HEAD now
> 572 `.py`) that never appeared in any disposition ledger. Per the #1075 AC
> ("100% coverage at HEAD — absence from the ledger means the audit is
> incomplete, not that the file was fine"), this pass sweeps exactly those 20
> post-pin files' **in-code documentation** (module docstrings, class/function
> docstrings, `--help` text, inline behavioral comments) against code at HEAD
> `f0f284fd`, and ledgers them.
>
> Scope is the **added** files only. Pre-existing files modified after the pin
> were already in the inventory and audited at their then-state; re-auditing
> their post-pin diffs would be the full re-audit the operator explicitly
> de-scoped. This pass re-pins the in-code inventory to HEAD `f0f284fd`.

## Method

Each file read in full (src modules) or docstring-scanned with a
behavior spot-check (tests/benchmarks, per the `DOCS-AUDIT-2026-07-04-tests-bench`
lens). Every documented, checkable claim — flag/env names, defaults, thresholds,
API symbols, short-circuit order, run commands, referenced CLI verbs — matched
against code at HEAD. Findings classified CRITICAL / HIGH / MEDIUM / LOW; the one
finding was independently re-verified before the fix.

## Tally

**1 MEDIUM (fixed in place), 19 verified-clean.** 0 CRITICAL, 0 HIGH, 0 LOW,
0 code-side drift.

### Finding — MEDIUM (fixed)

- **`aelf reconcile-claude-memory` user-facing but absent from `docs/user/COMMANDS.md`.**
  The command (registered `cli.py`, visible `--help`, shipped post-pin under
  #1089 axis-1 / PR #1094) had no row in the canonical command reference, while
  its sibling `audit-claude-memory` (#935) is documented there. Re-verified the
  command is visible (not `argparse.SUPPRESS`) and its help/behavior before
  writing the fix. **Fixed:** added a `reconcile-claude-memory` row to
  `COMMANDS.md` from the verified help text (sweep semantics, `--project` /
  `--force`, idempotence, one-way non-authoritative contract).

## Disposition ledger — 20 post-pin `.py` files @ `f0f284fd`

### `src/aelfrice/` (3)

| File | Disposition |
|---|---|
| `belief_context.py` | **verified-clean.** Module + `ContextResult` + `recover_context` + `_iter_turn_records` docstrings match behavior (best-effort join + `DERIVED_FROM` anchor recovery, `DEFAULT_MAX_TURN_MATCHES=5`, read-only, deterministic sorted walk). ~36% / ~28% coverage figures are lab measurements (internal-consistency only, externally sourced). Consumed by hidden `aelf context`. |
| `claude_memory_reconcile.py` | **verified-clean.** `ingest_memory_text` frontmatter→origin mapping (user/feedback → `user_validated` frozen as `route_overrides`; project/reference/absent → `None`; never locks), `reconcile_claude_memory` non-recursive `*.md` scan skipping `MEMORY.md`, `maybe_reconcile_claude_memory` sentinel short-circuit order + "never break `aelf setup`" all match. |
| `introspect.py` | **verified-clean.** Grounding threshold (`0.5`), `_status` edge logic (incoming `RESOLVES`→decided, `POTENTIALLY_STALE`→stale?, outgoing `RESOLVES`→decides, else floated), `build_report` signature/filters/determinism all match. |

### `benchmarks/` (4)

| File | Disposition |
|---|---|
| `entity_persist_ablation.py` | **verified-clean.** #1096 AUC ablation (posterior-only vs S1 vs posterior+log(S1)); synthetic corpus, real `entity_persistence_scores` API, correct run command; no live-store content. |
| `origin_tiebreak_ablation.py` | **verified-clean.** #1089 axis-2 offline ablation docstring matches. |
| `temporal_spine_latency.py` | **verified-clean.** #1064 G3 latency bench docstring matches. |
| `temporal_spine_shadow.py` | **verified-clean.** #1064 G2 shadow eval docstring matches. |

### `tests/` (13)

| File | Disposition |
|---|---|
| `bench_gate/_entity_persist_mixed_store.py` | **verified-clean.** #1096 G2 synthetic mixed-corpus store helper. |
| `bench_gate/test_entity_persist_g2_mixed_corpus.py` | **verified-clean.** #1096 G2 evidence eval. |
| `test_belief_context.py` | **verified-clean.** Docstring `aelf context` reference is accurate (command is registered, hidden). |
| `test_claude_memory_reconcile.py` | **verified-clean.** #1089 shared ingest core. |
| `test_cli_reconcile_claude_memory.py` | **verified-clean.** #1089 CLI integration. |
| `test_cli_retire_restore.py` | **verified-clean.** #1081 reversible soft-delete CLI. |
| `test_entity_persist_demote.py` | **verified-clean.** #1096 demotion lane. |
| `test_introspect.py` | **verified-clean.** #1081 honest-signal view. |
| `test_origin_tiebreak.py` | **verified-clean.** #1089 axis-2 tie-break. |
| `test_temporal_spine_ablation.py` | **verified-clean.** #1064 G2 top-rank invariance. |
| `test_temporal_spine_latency.py` | **verified-clean.** #1064 G3 latency bench units. |
| `test_temporal_spine_repro.py` | **verified-clean.** #1064 G5 determinism/repro. |
| `test_temporal_spine_shadow.py` | **verified-clean.** #1064 G2 shadow-eval harness. |

## Coverage statement

With this ledger, every `.py` file at HEAD `f0f284fd` (572) appears in a #1075
disposition ledger: 552 in the four accuracy tranches (pinned `fe3f29c4`), 20
here. The in-code documentation inventory is **re-pinned to `f0f284fd`**. This
closes the #1075 documentation-audit coverage AC ("100% coverage at HEAD, no
sampling") against literal HEAD, not merely the earlier pin.

## Verification limit

None beyond the standard lab-measurement caveat (the `belief_context.py`
coverage percentages are externally sourced and checked for internal
consistency only).
