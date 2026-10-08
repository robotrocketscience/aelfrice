# Historical design memos

Design memos for features that have shipped and stabilized. Kept here as the rationale trail for the original architectural decisions — not as current implementation spec.

The general `docs/design/README.md` disclaimer ("specs may be out of date relative to shipped code; treat the source as the truth and the spec as the historical intent") applies double here: every memo in this directory's underlying issue is **CLOSED**, and the corresponding code is **in tree** — with two exceptions. `lru_query_cache.md`, whose subject (`RetrievalCache`) was deleted in [#1418](https://github.com/robotrocketscience/aelfrice/issues/1418), is now the record of a removal rather than of shipped code. `hibernation_lifecycle.md` describes a lifecycle that was never built; its storage columns were dropped in [#1658](https://github.com/robotrocketscience/aelfrice/issues/1658). If you're trying to understand a feature's current behaviour, read the source first; if you're trying to understand *why* it was designed that way, read the memo here.

## Inventory

| Memo | Shipping issue | Module(s) at HEAD |
|---|---|---|
| `substrate_decision.md` | [#196](https://github.com/robotrocketscience/aelfrice/issues/196) | `src/aelfrice/scoring.py` (Beta-Bernoulli Option B) |
| `v2_view_flip.md` | [#265](https://github.com/robotrocketscience/aelfrice/issues/265) | `src/aelfrice/store.py` (`AELFRICE_WRITE_LOG_AUTHORITATIVE`) |
| `v2_derivation_worker.md` | [#264](https://github.com/robotrocketscience/aelfrice/issues/264) | `src/aelfrice/derivation_worker.py` |
| `v2_replay.md` | [#262](https://github.com/robotrocketscience/aelfrice/issues/262) | `src/aelfrice/replay.py` (`replay_full_equality`) |
| `v2_relationship_detector.md` | [#201](https://github.com/robotrocketscience/aelfrice/issues/201) | `src/aelfrice/relationship_detector.py` |
| `v2_phantom_promotion_trigger.md` | [#229](https://github.com/robotrocketscience/aelfrice/issues/229) | `src/aelfrice/promotion.py` |
| `rebuild_silent_vs_always.md` | [#289](https://github.com/robotrocketscience/aelfrice/issues/289) | `src/aelfrice/context_rebuilder.py` (floor constants) |
| `relevance_floor.md` | [#289](https://github.com/robotrocketscience/aelfrice/issues/289) | `src/aelfrice/context_rebuilder.py` |
| `query_understanding.md` | [#291](https://github.com/robotrocketscience/aelfrice/issues/291) | `context_rebuilder.py` (`_query_for_recent_turns()`) |
| `belief_retention_class.md` | [#290](https://github.com/robotrocketscience/aelfrice/issues/290) | `src/aelfrice/models.py` (`retention_class_for_source()`) |
| `rebuild_eval_harness.md` | [#288](https://github.com/robotrocketscience/aelfrice/issues/288) | `src/aelfrice/context_rebuilder.py` (rebuild log surface) |
| `lru_query_cache.md` | [#69](https://github.com/robotrocketscience/aelfrice/issues/69) (shipped v1.0.3) | **none** — `RetrievalCache` deleted in [#1418](https://github.com/robotrocketscience/aelfrice/issues/1418); `src/aelfrice/retrieval.py` and `src/aelfrice/graph_spectral.py` point here for the registry it left behind |
| `hibernation_lifecycle.md` | [#196](https://github.com/robotrocketscience/aelfrice/issues/196) (never built) | **none** — the `hibernation_score` and `activation_condition` columns were dropped in [#1658](https://github.com/robotrocketscience/aelfrice/issues/1658) |

Relocated 2026-05-26 per docs audit DOCS-AUDIT-2026-05-26.md.
