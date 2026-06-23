# HRR vocabulary-bridge expansion lane (#981)

Status: **implemented, default-OFF; ablation landed.** Default-flip is out of
scope (reverses locked #605; routed to a re-opened #897).

## What this is

A deterministic, embeddings-free retrieval lane that probes the HRR
structural index for **single-hop semantic neighbours of the FTS5 seeds** and
merges them into the candidate set before scoring. It restores the
`use_hrr=True` L3 step the predecessor's docstring labelled "HRR
vocabulary-bridge expansion" — the lane the v3.0.1 calibration notes name as
the cause of the LoCoMo delta (66.1% → 40.88%) — re-expressed on the current
`HRRStructIndex` substrate rather than the removed predecessor `HRRGraph`.

It is **not**:

- the `use_hrr_structural` lane (#152) — that is a marker-routed query path
  (`<KIND>:<target_id>`) that *replaces* the textual lane; this lane is an
  *additive candidate source* that runs alongside L1/BFS;
- the `vocab_bridge.py` query-rewrite cascade (#433/#536, removed, `TypeError`
  by design);
- the `(subject,predicate)`-collision SUPERSEDES prototype (#895, wontfix).

## Mechanics

The struct index encodes each belief's outgoing edges as a superposition of
bound role–filler pairs: `struct[b] = Σ bind(role[e.kind], id[e.dst])`. From a
seed belief the lane recovers both edge directions:

- **forward** out-neighbours: `unbind(role[kind], struct[seed])` recovers the
  superposed `id` vectors of `kind`-typed destinations; a cleanup matvec
  against the id-vector matrix ranks candidate beliefs;
- **reverse** in-neighbours: the existing `HRRStructIndex.probe(kind, seed)`
  ranks beliefs whose outgoing structure contains `bind(role[kind], id[seed])`.

Probed edge kinds are the predecessor's semantic set ∩ the live
`models.EDGE_TYPES`: `SUPERSEDES, CONTRADICTS, SUPPORTS, CITES, TESTS,
IMPLEMENTS` (`CALLS` is not in the current schema). Co-occurrence/structural
kinds (`RELATES_TO`, `DERIVED_FROM`, `TEMPORAL_NEXT`, `RESOLVES`) are excluded
as noise.

`precompute_expand_neighbors` materialises a byte-stable
`hrr_expand_neighbors` SQLite table (forward + reverse, per active belief);
`expand_seeds` reads it at query time with a live-probe fallback. The lane's
results also seed BFS, matching the predecessor.

### Similarity floor

Single-hop typed edges are **bimodal** under this algebra: a present edge
recovers its bound term exactly (`unbind(role_k, bind(role_k, id_t)) · id_t ==
1.0` forward; the matching `bind` self-dot `≈ 1.0` reverse), independent of
the belief's degree — only additive crosstalk from the belief's *other*
bindings varies, at `~1/sqrt(dim)` (~0.044 at dim=512) per term. True matches
cluster at ~1.0 and absent edges at ~noise, so the floor (`0.5`) sits squarely
in the gap. Empirically a spurious hit on a no-such-edge belief scores
~0.12–0.17 at dim=512 — well below 0.5.

## Determinism (#605, #437)

Every operation is a numpy FFT / matvec over the deterministically-seeded
struct matrix. There is no `random` / `betavariate` / Thompson sampling
anywhere in the lane (asserted by an AST scan in `tests/test_hrr_expand.py`).
The neighbour table is byte-equal across two runs over the same store
(tie-break: similarity DESC, neighbor_id ASC). Default-OFF keeps `retrieve_v2`
output byte-identical to the pre-#981 path.

## Flag

`use_hrr_expand`, resolved env (`AELFRICE_HRR_EXPAND`) > kwarg > TOML
(`[retrieval] use_hrr_expand`) > **default False**, via
`retrieval.is_hrr_expand_enabled`. Passing it never raises (AC1).

## Ablation (`benchmarks/hrr_expand_ablation.py`)

The #977-sweep arm for this lane. LoCoMo under a 2×4 matrix — edge substrate
{`none`, `contradicts-988`} × flag arms {`baseline`, `+hrr-expand`, `+bfs`,
`all-on`} — scored with the LoCoMo adapter's deterministic retrieval-overlap
F1 (the scorer the issue's honest-uplift caveat references; no LLM reader, so
the arm comparison is fully reproducible). Absolute F1 is a retrieval-recall
proxy, not reader accuracy — the **relative arm deltas** are the deliverable.

**Scope note.** Only LoCoMo is run: it is the corpus the 66.1%→40.88% delta
is measured on, and the full LongMemEval / MAB / StructMemEval corpora are not
present locally (only micro smoke fixtures), with several structurally
unscorable per the project's bench notes. They are deferred.

### Headline finding

The HRR-expand and BFS lanes **traverse semantic edges**, and vanilla LoCoMo
ingest writes **zero** edges (1240 beliefs, 0 edges per conversation). So:

- **without an edge substrate** (`substrate=none`) every arm is identical and
  `expand_hits=0` — the lane is **inert**. This is the #977 "BFS inert / 0
  edges" finding, generalised to HRR-expand.
- **with the #988 substrate** (`substrate=contradicts-988`,
  `write_semantic_edges` minting 1394 CONTRADICTS edges across LoCoMo10,
  ~139/conversation) the lane **activates** — it merges 446 beliefs across the
  corpus — but the CONTRADICTS-only bridge yields **no measurable
  retrieval-overlap F1 uplift** (flat-to-marginally-negative).

This empirically confirms the issue's "do not pre-bank the 25.2 pp" caveat and
the #988-reframes-#981/#977 framing: HRR-expand is a *consumer* of a semantic-
edge substrate. Its uplift is gated on that substrate existing and being
dense/typed enough to bridge answer-bearing beliefs — which the current
default-off, CONTRADICTS-only #988 writer does not yet provide on LoCoMo.

### Full LoCoMo10 results

Full LoCoMo10 (1947 QA, retrieval-overlap F1, budget 2000):

| substrate | arm | overall | Δ vs base | lane merges | c1 multi-hop | c2 temporal | c3 open | c4 single-hop | c5 adversarial |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| none (0 edges) | baseline | 2.11% | — | 0 | 10.46% | 0.75% | 0.80% | 1.10% | 0.0% |
| none | +hrr-expand | 2.11% | +0.00 | 0 | 10.46% | 0.75% | 0.80% | 1.10% | 0.0% |
| none | +bfs | 2.11% | +0.00 | 0 | 10.46% | 0.75% | 0.80% | 1.10% | 0.0% |
| none | all-on | 2.11% | +0.00 | 0 | 10.46% | 0.75% | 0.80% | 1.10% | 0.0% |
| contradicts-988 (1394 edges) | baseline | 2.11% | — | 0 | 10.46% | 0.75% | 0.80% | 1.10% | 0.0% |
| contradicts-988 | **+hrr-expand** | 2.11% | **−0.00** | **446** | 10.46% | 0.74% | 0.79% | 1.09% | 0.0% |
| contradicts-988 | +bfs | 2.11% | +0.00 | 0 | 10.46% | 0.75% | 0.80% | 1.10% | 0.0% |
| contradicts-988 | all-on | 2.11% | −0.00 | 446 | 10.46% | 0.74% | 0.79% | 1.09% | 0.0% |

Reading: with no edge substrate every arm is byte-identical (lane inert). With #988's 1394 CONTRADICTS edges the HRR-expand lane **activates** — it merges 446 beliefs across the corpus — yet overall F1 is unchanged and the per-category effect is a marginal **negative** (temporal/open/single-hop each down ~0.0001). `+bfs` stays byte-identical to baseline even with edges (BFS inert, per #977). Absolute F1 is low because this is retrieval-token-overlap against the concatenated context, not reader accuracy; the arm *deltas* are the deliverable, and they are flat-to-slightly-negative.

## Decision gate

Per #897 (reaffirm #605, no bench-number floor re-triggers reconsideration):
this lands the lane + ablation **default-OFF**. A default flip reverses #605
and requires re-opening #897 with isolated bench evidence and an explicit
philosophy amendment — bench uplift alone is insufficient. The ablation above
shows no uplift to bank regardless.
