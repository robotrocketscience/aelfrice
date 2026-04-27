# Design Memo: CRDT Primitives for v3 Cross-Project Federation

**Status:** v3 architectural direction. Pre-implementation; identifies the shape of the work and the one schema change worth landing forward-compatibly in v1.x to avoid losing user data later.

---

## Frame

aelfrice's v3 plan is cross-project federation: scopes (per-project memory stores) that share a subset of beliefs while preserving project-local isolation for the rest. The existing `SUPERSEDES` / `CONTRADICTS` / lock mechanics already do most of the work conflict-free systems need to do — they are reinventing CRDT primitives one timestamp at a time. Under federated writes (independent scopes updating shared beliefs concurrently), the current "bare timestamp + serial number" approach silently drops conflicting writes.

This memo maps each primitive onto the right CRDT and identifies the schema deltas that would close the gap. The argument is not "rebuild now"; it is "the gap is small if addressed before federation ships, and large if federation ships first."

---

## Primitive map

| aelfrice mechanism | Current encoding | CRDT shape | What changes under federation |
|---|---|---|---|
| `SUPERSEDES` edge | Bare wall-clock timestamp + serial; total order assumed | Lamport / version-vector ordering | Detect concurrent supersedes from different scopes; route to policy instead of silent timestamp tiebreak |
| `CONTRADICTS` edge | Relationship between two resolved beliefs | MV-Register: contradiction is the value, not the relation | Surface the conflict as a multi-valued state until policy resolves |
| Locked belief | `lock_level = user`, demotion-pressure threshold | 2P-Set with provenance | Remove only via explicit unlock with provenance; cross-scope locks compose |
| Confidence (`α, β`) | Sum of feedback events | Two independent G-Counters → PN-Counter | Helpful / unhelpful counts converge under arbitrary delivery order |
| Tombstone / GC | None today | Causal-stable garbage collection | Required if `α, β` counters or supersession metadata are not to grow without bound |

The order matters: get supersession ordering wrong and you silently lose user corrections; get the others "merely sub-optimal" and you waste storage or surface false contradictions.

---

## 1. SUPERSEDES → version vectors

Today, a `SUPERSEDES` edge is `(src_belief, dst_belief, type=SUPERSEDES, weight=0)` where `dst` is older. The "older" claim is enforced at write time by comparing wall-clock timestamps or serials.

**Failure mode under federation.** Two scopes both have a copy of belief X. Scope A's user issues a correction; scope A creates `Y_A SUPERSEDES X`. Scope B, independently, on a partition or earlier sync state, creates `Y_B SUPERSEDES X`. Both `Y_A` and `Y_B` have valid timestamps. Naive merge picks the later timestamp and drops the earlier correction. The user who issued the discarded correction will never know it was lost — there is no surface for the conflict.

**Fix.** Tag every belief and every edge with a **version vector** keyed by scope id:

```
version_vector = { scope_id: monotonic_counter }
```

- Local event (write in scope S): `vv[S] += 1`.
- Receive-and-merge from scope T: `vv[k] = max(vv_local[k], vv_received[k])` for all k, then `vv[S] += 1`.
- Compare two vectors `a, b`:
  - if `a ≤ b` componentwise: `a` causally precedes `b`. Keep `b`.
  - if `b ≤ a`: keep `a`.
  - else: **concurrent**. Surface the conflict.

**Schema change.** A `version_vector JSON` column on `beliefs` and `edges`, or — more SQLite-idiomatic — a `belief_versions(belief_id, scope_id, counter)` table indexed on `belief_id`.

**v1.x forward-compat.** This is the one CRDT change worth landing in v1.x as a forward-compatible additive column. Existing rows get `version_vector = {local_scope: 1}` at install time. Federation is not yet active, so the vector compares trivially. When federation ships, the metadata is already accumulated and no painful migration is needed.

---

## 2. CONTRADICTS → MV-Register

A `CONTRADICTS` edge expresses "two concurrent values, neither subsumes." That is exactly an MV-Register (Multi-Value Register). The cleaner shape is to make the contradiction a property of the *belief*, not a relationship between two resolved beliefs:

```
Belief.value: { v1: { content, version_vector }, v2: { content, version_vector }, ... }
```

Concurrent writes accumulate as values. A causally-later write that subsumes both collapses the register to a single value. The application layer (or a deterministic precedence rule, like the v1.0.1 `user_stated > user_corrected > document_recent > agent_inferred` ordering) decides which value wins for retrieval purposes; the register holds the unresolved state.

**Pragmatic compromise.** A belief's `content_hash` is currently a primary identity element. Multi-valuing the content means rethinking what a belief's identity is. Keep the current encoding (CONTRADICTS edge), and add a derived `unresolved_contradictions` flag on `beliefs` that retrieval surfaces. Treat the edge as the implementation; MV-Register as the model.

**Schema change.** A derived `unresolved_contradictions BOOLEAN` (or a query at retrieval time). No structural change to `edges`.

---

## 3. Locked beliefs → 2P-Set with provenance

A 2P-Set (two-phase set) has separate add and remove operations; once an element is removed it cannot be re-added. The intuition for locks: "once explicitly unlocked, the lock cannot reappear without a new explicit lock." The current code mostly does this — auto-demote zeros `lock_level` and `locked_at`; `aelf lock` is the only path to re-add. Under federation, two scopes could independently lock and unlock the same belief and end up in inconsistent states.

The 2P-Set encoding tags locks and unlocks with provenance:

```
locks    : append-only (belief_id, scope_id, locked_at,    locked_by,    version_vector)
unlocks  : append-only (belief_id, scope_id, unlocked_at,  unlocked_by,  version_vector)

is_locked(belief_id) :=
    there exists a lock not subsumed by a causally-later unlock
```

**Federation behaviour.** If anyone locked it, it's locked. Unlocking requires an explicit unlock from a scope with authority, or all scopes' locks expiring. More conservative than the current single-flag model; matches user intuition.

**Schema change.** Replace `lock_level + locked_at` on `beliefs` with `belief_locks` and `belief_unlocks` tables. Derived current-state computed from the diff. Backward-compat migration synthesizes one lock row per existing locked belief.

---

## 4. Confidence counts → PN-Counter

Two independent G-Counters (grow-only, summed across replicas) form a PN-Counter. Each scope maintains its own `α_scope`, `β_scope`; the federated value is the sum across scopes. Convergence is automatic — addition is commutative, associative, idempotent under exactly-once delivery.

This is the lowest-risk, highest-payoff CRDT change. It also has the least visible benefit at v1.x scope (single user, no federation). v3 territory.

**Schema change.** Replace `alpha REAL, beta REAL` on `beliefs` with `belief_counters(belief_id, scope_id, alpha, beta)`. Aggregate at retrieval time. Or — to avoid hot-path joins — keep aggregate columns on `beliefs` and a per-scope shadow table for federation reconciliation.

---

## 5. Tombstones and causal-stable GC

State-based CRDTs grow without bound unless garbage-collected. Tombstones (markers for "this was deleted") accumulate and need eventual removal. Removing them too early causes the CRDT to "resurrect" deleted items when they sync from a slow replica.

**Standard solution.** Causal stability: an operation O is causally stable when every replica has seen O's causal predecessors (its version vector is `≥` O's vector). Once stable, O can be GC'd without resurrection risk.

**aelfrice's lever.** Federation membership is small (handful of scopes per user, possibly per team). Causal stability is reachable in practice. The GC mechanism: periodically compute the minimum version vector across all known scopes; any tombstone with a vector below that minimum is safe to remove.

This is operational complexity, not data-model complexity. v3.

---

## What this memo proposes for v1.x

Only the SUPERSEDES → version-vector change costs user data if missed before federation ships. Land that forward-compatibly in v1.x as an additive column or shadow table; existing rows get a trivial vector; federation activates the comparison logic later.

The other CRDT framings are documentation moves — they let aelfrice describe what the existing primitives are doing in terminology consistent with the wider distributed-systems literature. That alone improves how the project communicates with reviewers and contributors who know CRDTs.

Causal stability and GC stay in v3. They are not paid for until federation creates the bills.

---

## What this memo does not propose

- Implementing federation in v1.x. v3 territory.
- A general-purpose CRDT library. Each primitive can be implemented as a few hundred lines of Python over the existing SQLite store.
- Operational federation (replication transport, conflict-resolution UI, scope-discovery protocol). Those are downstream of getting the data model right.

---

## References

- Shapiro, Preguiça, Baquero, Zawirski (2011), *Conflict-Free Replicated Data Types* — foundational CRDT paper.
- Almeida, Shoker, Baquero (2018), *Delta state replicated data types* — delta CRDTs and dotted version vectors.
- Lamport (1978), *Time, clocks, and the ordering of events in a distributed system* — Lamport timestamps.
- Mattern (1989), *Virtual time and global states of distributed systems* — version vectors.
- Cross-reference: [write-log-as-truth.md](write-log-as-truth.md) — the ingest log is the natural unit of inter-scope replication; CRDT primitives apply to derived state, log is canonical.
