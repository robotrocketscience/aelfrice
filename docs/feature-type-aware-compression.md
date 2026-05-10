# Feature spec: Type-aware compression (#434)

**Status:** implemented behind default-OFF flag; pack-loop budget rewrite landed; lab bench A2 / A4 pending
**Issue:** #434
**Recovery-inventory line:** [`docs/ROADMAP.md`](ROADMAP.md) — *"Type-aware compression | v2.0.0"*
**Substrate prereqs:** #290 (retention class column + per-source defaults, shipped v1.6.0), #141 (context rebuilder, shipped v1.4.0)

---

## Purpose

At a fixed retrieval `token_budget`, today's path either includes a belief verbatim or trims it from the tail. Per [`docs/belief_retention_class.md`](belief_retention_class.md), beliefs vary in expected lifetime: `fact` rows describe stable codebase state; `snapshot` rows are research-thought; `transient` rows are PR-window scratch. The retrieval pack treats them identically. Type-aware compression spends fewer tokens per `snapshot` and `transient` belief so more total beliefs fit, without sacrificing fidelity on the `fact` rows that carry load.

The compressor is **structural**: deterministic, byte-stable, no LLM. The contract is "same belief input → same compressed output, byte-for-byte" — issue #434 acceptance #3 makes this explicit, and is the single line that distinguishes this from a summariser.

---

## Contract

```python
from aelfrice.compression import compress_for_retrieval, CompressedBelief

cb: CompressedBelief = compress_for_retrieval(belief, *, locked: bool)
```

`CompressedBelief` is a small wrapper around the original `Belief` plus its compressed form:

```python
@dataclass(frozen=True)
class CompressedBelief:
    belief: Belief
    rendered: str          # the post-compression text the pack emits
    rendered_tokens: int   # _estimate_tokens(rendered)
    strategy: str          # "verbatim" | "headline" | "stub"
```

Inputs:

- `belief: Belief` — the belief as returned by the lane fan-out.
- `locked: bool` — `True` when `belief.lock_state == LOCK_USER` or the caller has otherwise classified the belief as L0. Locks always render `"verbatim"` regardless of `retention_class`.

Output: a single `CompressedBelief` whose `rendered_tokens ≤ _belief_tokens(belief)` (i.e. compression is monotone-non-increasing in cost). The compressor is total: every `(belief, locked)` pair returns a `CompressedBelief`.

The function is pure and deterministic. No store reads, no clock reads, no env reads, no random.

---

## Strategy table

| Retention class | `locked=True` | `locked=False` | Notes |
|---|---|---|---|
| `fact` | verbatim | verbatim | Always full-text. The class is "stable codebase state"; truncation here is a fidelity loss. |
| `snapshot` | verbatim | **headline** | Headline = first sentence (split on `. ` / `.\n`), with a trailing `…` when truncated. |
| `transient` | verbatim | **stub** | Stub = belief id + retention class only, no content. Compressed-out content is recoverable via `store.get_belief(id)` if the consumer needs it. |
| `unknown` | verbatim | verbatim | Migration-safety. Don't compress beliefs whose retention class hasn't been classified yet — that is a separate audit (#290 §6) before type-aware compression can fire on them. |

The `locked → verbatim` rule mirrors the existing rule that L0 beliefs are never trimmed (`retrieval.py:950`: *"L0 beliefs are never trimmed."*) — locks override every retention-class decision the same way they override hibernation (#196) and the relevance floor (#289).

### Headline strategy details

- Split on the first occurrence of `. ` or `.\n` outside a balanced code-fence span (`` ``` ``). Code fences are preserved as a unit so we don't truncate mid-fence and produce un-renderable Markdown.
- If no sentence boundary is found within the first `MAX_HEADLINE_CHARS = 240` characters, hard-truncate at the last whitespace boundary ≤ 240 and append `…`.
- A belief whose `len(content) ≤ MAX_HEADLINE_CHARS` and contains no internal sentence boundary renders verbatim — the headline strategy never makes the rendered text *longer* than the source.

### Stub strategy details

- `rendered = "[stub: belief={id} class=transient]"` — fixed format, no payload.
- Tokens: `_estimate_tokens` over the literal string above. At `_CHARS_PER_TOKEN = 4` and a typical `belief.id` length (~12 chars), that is ~10 tokens regardless of belief content size.
- The stub is **deliberately recognizable** so a downstream consumer that wants the full content can detect the marker and call `store.get_belief(id)` against the original. This is the "compressed-out but not deleted" semantic.

---

## Where compression sits

The compressor runs **after** lane fan-out and ranking, **before** the budget pack. The current pack loop at `retrieval.py:1048-1085` (and the parallel block at `:1197-1232`) reads:

```python
used: int = locked_used + sum(_belief_tokens(b) for b in l25)
for b in candidates:
    cost: int = _belief_tokens(b)
    if used + cost > token_budget:
        break
    out.append(b)
    used += cost
```

The compressed-pack rewrites this to:

```python
used: int = locked_used + sum(cb.rendered_tokens for cb in compressed_l25)
for cb in compressed_candidates:
    if used + cb.rendered_tokens > token_budget:
        break
    out.append(cb)              # CompressedBelief, not raw Belief
    used += cb.rendered_tokens
```

`compressed_candidates` is the post-compression sequence produced by mapping `compress_for_retrieval(b, locked=...)` over the ranked candidate list. The same applies to `compressed_l25`.

### Configuration

A new `use_type_aware_compression` flag follows the established convention at `retrieval.py:118-131`:

1. `retrieve(..., use_type_aware_compression=True)` kwarg (highest precedence).
2. `AELFRICE_TYPE_AWARE_COMPRESSION=1` env var.
3. `[retrieval] use_type_aware_compression = true` in `.aelfrice.toml`.
4. Default OFF at v2.0.0 until bench-gate clears (acceptance #2).

When OFF, the compressor short-circuits to `verbatim` for every belief, making `rendered_tokens == _belief_tokens(belief)` and `out` element-equivalent to today's `Belief` list (a `CompressedBelief` with `strategy="verbatim"` is rendered identically to its `belief.content`). This preserves the v1.x retrieval contract under the default.

### `RetrievalResult` shape

`RetrievalResult.beliefs` today is `list[Belief]`. The minimum viable change is to add a parallel `list[CompressedBelief]` field (e.g. `compressed_beliefs`) and leave `beliefs` populated for back-compat. A consumer that wants the compressed render reads `compressed_beliefs[i].rendered`; a consumer that wants the raw `Belief` keeps reading `beliefs[i]`. Both lists are the same length and ordered the same way.

The implementation PR may instead replace `Belief` with `CompressedBelief` everywhere downstream and add a `cb.belief` accessor; that is a wider rename and is left to the impl PR's decision based on call-site count.

---

## Storage

None. The compressor is a pure function over `(Belief, locked)`. No new schema, no new table, no persistence, no cache.

If a future revision needs to memoise compressed renders (e.g. for very large beliefs), the cache is a per-process `dict[belief_id, CompressedBelief]` keyed off `(belief.id, belief.content_hash)` — invalidated naturally by content-hash change. That is **out of scope at v2.0.0**.

---

## Reconciliation

### vs. the existing tail-trim

The current pack drops beliefs from the tail when `token_budget` is hit (`retrieval.py:744`, `:1052`, `:1202`). Type-aware compression does not replace this; it **runs ahead of it**. Compression reduces per-belief cost; tail-trim still fires on the post-compression cost list when it must. The two compose: a `transient` belief that costs ~10 tokens after stub-compression rarely gets tail-trimmed; a verbatim `fact` belief still gets trimmed if the budget is pathologically tight.

### vs. retention-class soft-downweight

Per `docs/belief_retention_class.md` § "Recommendation summary": *"Soft down-weight in ranking, not hard expiry. Beliefs don't expire; their `final_score` gets a class-and-age multiplier before the floor (#289) is applied."* That is a **ranking-stage** mechanism — it changes what enters the pack. Type-aware compression is a **packing-stage** mechanism — it changes how many of the entered beliefs fit. They compose: ranking decides who is in, compression decides how many get the full-text treatment.

### vs. context rebuilder (#141, v1.4)

The rebuilder consumes retrieval output and writes a continuation-fidelity-scored block at `<git-common-dir>/aelfrice/rebuild_logs/`. With compression on, the rebuilder gets more total beliefs in the same `[rebuilder] token_budget` (default in `context_rebuilder.py:111`). Acceptance #4 makes this measurable: continuation-fidelity uplift at fixed budget.

The rebuilder does not need a separate config knob; it inherits the retrieval-side flag. If `use_type_aware_compression` is on at retrieval call-time, the rebuilder reads compressed output transparently via the `RetrievalResult` shape change above.

### vs. type / source-tier axes

Belief carries three orthogonal axes: `belief.belief_type` (factual / correction / preference / requirement, `models.py:13-22`), `belief.source_kind` (the ingest path), and `belief.retention_class` (this spec's input). The compressor reads only `retention_class`. Other axes are out of scope:

- `belief.belief_type` does not predict expected-lifetime — a `factual` belief can be `transient` (#281 example in `docs/belief_retention_class.md`), and a `preference` belief can be `fact`-class.
- `belief.source_kind` is upstream of `retention_class` (the per-source defaults in `models.py:_RETENTION_DEFAULT_BY_SOURCE`); the compressor consumes the materialised retention class, not the source.

---

## Acceptance

### A1 — corpus

A labeled `compression_uplift` corpus lives under `tests/corpus/v2_0/compression_uplift/`, mirroring the v2.0 corpus scaffold. Each row encodes a (query, ground-truth-set, mixed-class belief population) triple. The `bench_gated` autouse marker keeps public CI green when `AELFRICE_CORPUS_ROOT` is unset; labeled content lives in the lab repo only, per the published corpus policy.

### A2 — token-budget recovery

At a fixed `token_budget` (default `2400` per `retrieval.py:97 DEFAULT_TOKEN_BUDGET`):

```
recall@k(use_type_aware_compression=ON)  >  recall@k(use_type_aware_compression=OFF)
```

on the `compression_uplift` fixture. Threshold: **strictly positive uplift** (issue acceptance #2 phrases this as "measurably more recovered facts"). The threshold is positive, not a fixed magnitude — the bench gates ship-or-defer, not headline-number-matches.

### A3 — determinism

`compress_for_retrieval(b, locked=L)` is byte-stable across processes. A property test asserts:

```python
@hypothesis.given(belief_strategy(), st.booleans())
def test_compress_deterministic(b, locked):
    assert compress_for_retrieval(b, locked=locked).rendered \
        == compress_for_retrieval(b, locked=locked).rendered
```

A second test compares against a fixture of `(belief_id, expected_rendered)` pairs checked into the public test suite, so a refactor that perturbs the rendering breaks the test.

### A4 — rebuilder fidelity

The continuation-fidelity scorer (#141 v1.4 deliverable) is run on the rebuild_logs corpus with `use_type_aware_compression={OFF, ON}`. Bench-gate: ON ≥ OFF on continuation-fidelity score at the same `[rebuilder] token_budget`. Tolerance band: `≥ baseline − 0.005` (a half-point of the fidelity-score noise floor, mirroring the BM25F bench-gate band at #154).

### A5 — composition tracker

The #154 composition tracker doc gains a row for `use_type_aware_compression`: input shape, output shape, where it sits, bench verdict. This is **not a lane** — it is a packing-stage transform. The tracker row is present for operator clarity.

---

## Bench-gate / ship-or-defer policy

`needs-spec` → `bench-gated` once this spec lands. Implementation is the next gate. **The implementation PR ships only on positive bench evidence per A2 and A4.** A mechanically-correct implementation that fails either ships merged-but-default-OFF or gets reverted; it does not ship default-ON without a benchmark cut.

---

## Out of scope at v2.0.0

- **LLM-based compression / summarisation.** Determinism (A3) is an explicit acceptance criterion. Any path that calls a model is excluded.
- **Compression of `fact`-class beliefs.** They render verbatim. If `fact` is the dominant retention class in a real-world store, type-aware compression has nothing to do — that is the correct behaviour for a fact-heavy corpus, not a bug.
- **Compression of locked beliefs.** Locks override retention class. This mirrors the L0-never-trimmed rule.
- **Per-belief-type strategies.** `belief.belief_type` is not consulted. If a future revision wants type-axis compression (e.g. always-verbatim-for-`requirement`), that is a separate spec; this one closes the retention-class axis only.
- **Online / streaming compression.** The compressor is called over the full ranked candidate list. Streaming compress-as-rank is out of scope.
- **`MAX_HEADLINE_CHARS` tuning.** 240 is the spec default. Per-store tuning via `[retrieval] headline_max_chars` is implementable but **deferred** — ship one default first; tune on bench evidence.

---

## Implementation prereqs

- `src/aelfrice/models.py` — `RETENTION_*` constants, `Belief.retention_class` field. Shipped v1.6.0 (#290).
- `src/aelfrice/retrieval.py:215` — `_belief_tokens(b)` and `_estimate_tokens()`. The compressor reuses the same estimator on `rendered`.
- `src/aelfrice/retrieval.py:118-131` — flag-resolution convention.
- `src/aelfrice/retrieval.py:1048-1085, :1197-1232` — pack loops to rewrite.
- `src/aelfrice/context_rebuilder.py` — consumer of compressed output for A4.
- `tests/corpus/v2_0/` — corpus scaffold + autouse `bench_gated` marker. Shipped v1.6.0 (#307 / #311).
- `tests/bench_gate/` — harness. Shipped v1.6.0 (#319 / #320).

All substrate is on `main` as of `68dafc0`. No new dependencies. No schema changes.

---

## Open questions for review

1. **Stub format.** `"[stub: belief={id} class=transient]"` is one option. Alternatives: empty string (zero-cost but indistinguishable from a deleted belief); JSON-shaped marker; emoji prefix. The spec defaults to the bracketed form because it is grep-friendly and survives prose mixing. Pick at impl-PR review.
2. **`CompressedBelief` vs. parallel-list shape.** Two options under "RetrievalResult shape" above. The wider rename is correct long-term; the parallel field is faster to land. The impl PR should pick based on consumer count (a grep for `RetrievalResult.beliefs` will tell).
3. **Headline-strategy on code-fence-only content.** A belief whose entire content is one ``` ```python ... ``` `` block has no sentence boundaries by the §"Headline strategy details" rule. The fallback (hard-truncate at 240 chars on whitespace) preserves Markdown well-formedness only by accident. Consider a "content-is-code-only → verbatim" override; settle in impl PR.
4. **Promotion path interaction.** `aelf doctor --promote-retention` (`doctor.py:1138`) flips a belief from `snapshot` to `fact`. A belief mid-promotion shifts compression strategy from `headline` to `verbatim`. This is correct behaviour (the belief just earned full-text rendering), but it makes deterministic regression tests across promotion-state-changes brittle. The A3 fixture should pin retention class explicitly per row, not rely on whatever `retention_class_for_source` chose at fixture-build time.
