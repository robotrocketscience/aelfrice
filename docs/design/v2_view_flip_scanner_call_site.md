# Scanner call-site migration for the view-flip (#265 PR-B)

**Status:** memo — written before PR-B implementation per PR-A's body.
**Scope:** `src/aelfrice/scanner.py` only. The other four bypass call sites
named in PR-A (`wonder/simulator.py`, `benchmark.py`, `migrate.py`, the
`store.insert_beliefs()` bulk wrapper) are mechanical migrations and do not
require a memo.

## Why scanner is the hard one

The regex path was already migrated to the worker in #264 part 2
(scanner.py:312-335). The remaining bypass is the **LLM-classify path**
(scanner.py:262-310). The inline comment at scanner.py:230-232 names the
blocker:

> The LLM path remains direct-write — its alpha/beta/origin/audit_source
> are not yet representable in `raw_meta` for the worker to reconstruct.

The router emits per-candidate `LLMRoute(belief_type, origin, persist, alpha,
beta, audit_source)` (scanner.py:128-142). `derive()` produces these fields
from deterministic classification — it does not know about the router's
output. Routing through the worker today would silently drop the router's
decisions on the floor.

## What must flow from scanner → ingest_log → worker

Five fields the deterministic `derive()` cannot reconstruct:

| Field          | Source                        | Where it lands today                |
|----------------|-------------------------------|-------------------------------------|
| `belief_type`  | LLM classifier                | `Belief.type`                       |
| `origin`       | LLM classifier                | `Belief.origin`                     |
| `alpha`        | LLM classifier (confidence)   | `Belief.alpha`                      |
| `beta`         | LLM classifier (confidence)   | `Belief.beta`                       |
| `audit_source` | LLM classifier (fallback tag) | `feedback_history` row, post-insert |

`persist=False` is consumed before any state change — it does not need to
flow through the log. Skip the `record_ingest` call entirely.

## Decision: extend `raw_meta` with a `route_overrides` sub-dict

Two existing patterns live in `raw_meta`: `call_site` (string sentinel) and
`override_belief_type` (string sentinel, read by
`derivation_worker._derivation_input_from_row`). Adding more
top-level sentinel keys (`override_origin`, `override_alpha`, …) bloats the
flat namespace and makes audit logs noisy. Group them.

```python
raw_meta = {
    "call_site": CORROBORATION_SOURCE_FILESYSTEM_INGEST,
    "route_overrides": {
        "belief_type": "factual",
        "origin": "agent_inferred",
        "alpha": 1.4,
        "beta": 0.6,
        "audit_source": "llm_router_v1",  # optional
    },
}
```

When `route_overrides` is absent (the default — regex path, transcript
ingest, etc.), the worker's behavior is byte-identical to today.

## Worker contract change

`_derivation_input_from_row` already forwards `override_belief_type` from
`raw_meta` to `DerivationInput` (derivation_worker.py:140-168). Extend it to
also forward `route_overrides` as a typed sub-field on `DerivationInput`,
e.g. `DerivationInput.route_overrides: RouteOverrides | None`.

`derive()` then applies the overrides on its output `Belief` when present.
The override semantics are **post-derivation splice**, not a different
codepath: `derive()` runs as today, then if `route_overrides` is set, the
output belief's `(type, origin, alpha, beta)` are replaced. This keeps the
deterministic classifier's edge-emission and other side-channel logic
intact — only the fields the LLM owns are overridden.

`audit_source` is not a `Belief` field. The worker emits a
`feedback_events` row (using the existing `insert_feedback_event` API)
for the canonical belief_id when `route_overrides.audit_source` is set
and the belief was newly inserted (not corroborated). This collapses the
scanner.py:304-310 logic into the worker.

## Scanner shape after migration

```python
for idx, candidate in enumerate(filtered):
    created_at = candidate.commit_date or timestamp
    raw_meta: dict[str, Any] = {
        "call_site": CORROBORATION_SOURCE_FILESYSTEM_INGEST,
    }
    if routes is not None:
        route = routes[idx]
        if not route.persist:
            skipped_non_persisting += 1
            continue
        raw_meta["route_overrides"] = {
            "belief_type": route.belief_type,
            "origin": route.origin,
            "alpha": route.alpha,
            "beta": route.beta,
        }
        if route.audit_source is not None:
            raw_meta["route_overrides"]["audit_source"] = route.audit_source
    log_id = store.record_ingest(
        source_kind=INGEST_SOURCE_FILESYSTEM,
        source_path=candidate.source,
        raw_text=candidate.text,
        session_id=sid,
        ts=created_at,
        raw_meta=raw_meta,
    )
    log_ids.append(log_id)

if log_ids:
    run_worker(store)
    # back-fill ScanResult counters from stamped rows (existing logic)
```

The two-path `routes is not None` / regex split collapses into one loop.
The LLM-route-vs-regex distinction lives in `raw_meta` only.

## `aelf rebuild` interaction (PR-C concern, flagged here)

`route_overrides` are **frozen at ingest time** and re-applied on every
rebuild. Rationale: the LLM router's decision was the ground-truth ingest
event for that row — replaying the log under a new rule-set should not
re-roll it (we have no LLM available at rebuild time, and even if we did,
non-determinism would break replay equality). Document in PR-C:
`route_overrides` survives `aelf rebuild`; only fields `derive()`
reconstructs deterministically participate in rule-set replay.

## Out of scope for PR-B

- Changing `derive()` semantics for non-overridden fields.
- Eliminating the `LLMRoute` dataclass — kept as the in-memory shape
  the router emits before scanner serializes it into `raw_meta`.
- Migrating the four simpler call sites
  (`wonder/simulator.py`, `benchmark.py`, `migrate.py`, bulk `insert_beliefs`)
  — they don't need the route-override machinery; they get a private
  `_insert_belief_unchecked()` path. Detail in PR-B itself.

## Acceptance for the scanner slice

1. `RouteOverrides` typed dataclass added to `aelfrice.derivation` (or a
   new `route_overrides` module).
2. `_derivation_input_from_row` reads `raw_meta["route_overrides"]`.
3. `derive()` (or a post-derive splice) applies overrides to the output
   belief's `(type, origin, alpha, beta)` when set.
4. Worker emits the `feedback_events` audit row when
   `route_overrides.audit_source` is set and the belief was newly
   inserted.
5. Scanner LLM-classify path uses `record_ingest` + `run_worker` only;
   no `insert_or_corroborate` or `insert_belief` call survives outside
   the worker.
6. Tests:
   - regex-path scan unchanged (byte-identical canonical output flag-on
     and flag-off).
   - LLM-route scan with flag-off: same belief content as today.
   - LLM-route scan with flag-on: same belief content as flag-off,
     `derived_belief_ids` populated on log rows, `feedback_events` audit
     row written when `audit_source` is set.
   - `replay_full_equality` (#262) returns zero drift after an LLM-route
     scan with flag-on.
