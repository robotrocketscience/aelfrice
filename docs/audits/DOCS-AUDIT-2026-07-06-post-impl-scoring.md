# Docs audit — 2026-07-06 (post-implementation scoring pass)

> The **post-implementation documentation pass** sequenced by the #1075 operator
> directive (2026-07-05): *"once #1081 + #1086 land, a targeted docs update must
> capture the new belief-scoring behavior … across `docs/concepts/`, `docs/user/`,
> slash-command docs and CHANGELOG, ledgered into the audit record. Without it the
> audit no longer reflects HEAD."* Run against HEAD `748a225a` (v3.6.0 +
> post-release main, after #1081 merged as PR #1101 and #1086 closed on its
> junk-sink objective). This is a **forward-documentation** pass, not an accuracy
> re-audit: the four earlier #1075 tranches (doc-file, in-code src, in-code
> tests/bench, doc-file tranche-2) already verified the pre-existing surface; this
> pass adds documentation for behavior that shipped *after* those tranches.
>
> **Important:** the directive's original feature list predates the #1086
> measurement outcomes. The docs were written to reflect **what actually shipped on
> HEAD**, not the directive verbatim. Two corrections were load-bearing:
> - the organic sink is the **entity-persistence demotion lane (default-off, #1096)**,
>   *not* decay-to-hibernation — a temporal/cold-decay sink was measured empirically
>   inert (the junk is *hot*, not stale);
> - the **decided-vs-floated / `EDGE_RESOLVES` producer (fix #4) did NOT ship** — it
>   was measured non-retrofittable and de-scoped (#1100, closed). `aelf introspect`
>   reports a floated-vs-decided *status* off existing edges, but no ingest producer
>   was added. Docs document the shipped read-only status, not a lifecycle producer.

## Behavior documented (all on `main` at `748a225a`)

| Shipped behavior | PR(s) | Key artefact |
|---|---|---|
| Retrieval exposure is **audit-only by default** — a surfacing no longer moves α/β (`AELFRICE_EXPOSURE_UPDATES_POSTERIOR`, default off) | #1091 | `hook_search.record_retrieval`, `feedback.apply_feedback(update_posterior=…)` |
| **Recurrence is a separate axis** (`corroboration_count`), never conflated into the posterior | #1091 | regression-locked: `test_hook_fire_records_exposure_without_moving_posterior`, `test_original_alpha_beta_unchanged_on_hit` |
| **Entity-persistence demotion lane** — the organic sink, default-off (`use_entity_persist_demote` / `AELFRICE_ENTITY_PERSIST_DEMOTE`) | #1096, #1099 | `retrieval.py` L1 rerank; `MemoryStore.entity_persistence_scores` |
| **Origin-priority tie-break** — within-tier, default-off (`use_origin_tiebreak` / `AELFRICE_ORIGIN_TIEBREAK`) | #1089 | ranked-tier tie-break |
| **`aelf introspect`** — read-only honest-signal view | #1081 (PR #1101) | `aelfrice.introspect.build_report` + `introspect.md` slash |
| **`aelf retire` / `aelf restore`** — reversible soft-delete curation | #1081 (PR #1101) | `store.restore_belief`; `retire.md` / `restore.md` slashes |

## Edits by file

| File | Change |
|---|---|
| `docs/concepts/PHILOSOPHY.md` | New paragraph under *Bayesian, not vector*: "exposure is not endorsement" + recurrence-as-separate-axis. |
| `docs/concepts/ARCHITECTURE.md` | `hook_search.py` row annotated audit-only (#1086); retrieval-lane section gains the two default-off rerank modifiers (entity-persistence demotion, origin tie-break). |
| `docs/concepts/HARNESS_INTEGRATION.md` | Comparison table "Apply feedback" cell corrected: explicit signals move posteriors, exposure is audit-only. |
| `docs/concepts/ROADMAP.md` | Deferred-feedback-sweeper shipped-log entry annotated as superseded-as-default by #1086. |
| `docs/user/CONFIG.md` | `[retrieval]` knob list + two new `###` flag subsections: `use_entity_persist_demote`, `use_origin_tiebreak`. |
| `docs/user/COMMANDS.md` | Memory-operations table gains `retire` / `restore` / `introspect`; `sweep-feedback` entry reframed (no automatic consumer; exposure audit-only by default). |
| `docs/user/SLASH_COMMANDS.md` | Count 26→29; reference table gains `/aelf:introspect` / `/aelf:retire` / `/aelf:restore`; intro milestone note. |
| `docs/user/LIMITATIONS.md` | *Sharp edges* framing on exposure corrected; three new edges: organic-sink demotion lane (default-off), origin tie-break (default-off), decided-vs-floated not auto-tracked. |
| `docs/user/PRIVACY.md` | Bayesian-update-writer bullet clarified: automatic exposure is audit-only. |
| `CHANGELOG/v3.md` | `### Documentation` note under `[Unreleased]`. |

## What was deliberately **not** changed
- `docs/audits/DOCS-AUDIT-2026-07-04*.md` — frozen snapshots; not edited (per the audits-README freeze convention).
- `aelf sweep-feedback` mechanics and the `[implicit_feedback]` / deferred-feedback module docs: the **manual** sweep still applies `+ε` (default 0.05) α when run, so those descriptions remain accurate — only the *automatic/hook* exposure default changed. Over-correcting them would have introduced new inaccuracy.
- `RESOLVES` edge-type documentation (`ARCHITECTURE.md`, `ROADMAP.md`): `RESOLVES` is a real shipped wonder-lifecycle edge type and its docs are accurate; it is distinct from the de-scoped decided-vs-floated *producer* (fix #4).
- `doctor --promote-retention`: reclassifies `retention_class` (not the posterior) from retrieval/corroboration recurrence signals — not in tension with exposure-not-endorsement; left as-is.

## Verification
- All ten edited files render; no dangling conflict markers.
- Flag names, env vars, defaults, PR numbers, and API symbols cross-checked against `748a225a` code (`hook_search.py`, `deferred_feedback.py`, `feedback.py`, `introspect.py`, `store.py`, `cli.py`, the three shipped slash `.md` files).
- The two regression-lock test names cited in PHILOSOPHY/COMMANDS confirmed present on `main`.
