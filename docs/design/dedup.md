# Dedup — `aelf doctor --dedup`

Audit-only near-duplicate detection over the belief store, shipped at v2.0 per [#197](https://github.com/robotrocketscience/aelfrice/issues/197).

The detector pairs two cheap deterministic signals:

- **Jaccard** over lowercase Unicode-word tokens — fast prefilter, no allocation per character pair.
- **Levenshtein ratio** (`1 - edit_distance / max(len_a, len_b)`) — second-stage confirmation; guards against shared-vocabulary false positives that Jaccard alone would accept.

Both thresholds must clear for a pair to count as a near-duplicate.

## Usage

```bash
aelf doctor --dedup
```

Read-only: walks every belief pair, runs the prefilter, emits a clustered report. No edges are inserted, no beliefs are mutated.

### Per-run flags

```bash
aelf doctor --dedup \
  --dedup-jaccard 0.7 \
  --dedup-levenshtein 0.9 \
  --dedup-max-pairs 1000
```

Each `--dedup-*` flag overrides one knob for the current run only.

### Project defaults via `.aelfrice.toml`

```toml
[dedup]
jaccard_min = 0.8
levenshtein_min = 0.85
max_candidate_pairs = 5000
```

Walk-up resolution: the loader walks from cwd up through ancestor directories looking for `.aelfrice.toml`. Malformed values fall back to the module defaults with a stderr trace; the loader never raises.

## Defaults

| knob | default | source |
| --- | --- | --- |
| `jaccard_min` | 0.8 | research-line ratification |
| `levenshtein_min` | 0.85 | research-line ratification |
| `max_candidate_pairs` | 5000 | research-line ratification |

The 0.8 Jaccard floor is intentionally strict — token-set divergence between e.g. "don't" / "do not" pushes that pair below the floor even though Levenshtein would clear. Lower the threshold via `--dedup-jaccard` for paraphrase-style detection at the cost of more false positives.

## Output shape

```
aelf doctor dedup
========================================
Beliefs scanned         : 1483
Candidate pairs visited : 1099303
Duplicate pairs         : 12
Duplicate clusters      : 4

Clusters:
  belief-abc-123  (3 members)
    * belief-abc-123
      belief-abc-456
      belief-abc-789
  ...

Top duplicate pairs (jaccard, levenshtein):
  belief-abc-123  ~  belief-abc-456  (j=0.923, l=0.971)
  ...
```

`Candidate pairs visited` is the raw O(n²) count. `Duplicate pairs` is the post-Jaccard, post-Levenshtein survivor count. Clusters are the union-find collapse of those pairs into connected components; the `*` marks each cluster's deterministic representative (the lexicographically smallest member id).

## What's not in this command

The audit is **read-only by design**. The write-path SUPERSEDES hook — collapsing duplicates by inserting `SUPERSEDES` edges from older to newer at every `ingest_turn` / `onboard` / `apply_feedback` write — is the bench-gated R2 deferred behind the v2.0 corpus benchmark per #197 ratification. Until that lands, use this command to inspect candidate clusters and review them by hand.

## Related

- Spec memo: [`v2_dedup.md`](v2_dedup.md).
- Issue: [#197](https://github.com/robotrocketscience/aelfrice/issues/197).
- Scope cut: dedup is one of six bench-gated v2.0 modules; corpus contract at [#307](https://github.com/robotrocketscience/aelfrice/issues/307), bench-gate harness at [#319](https://github.com/robotrocketscience/aelfrice/issues/319).
