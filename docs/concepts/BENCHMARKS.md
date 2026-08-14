# Benchmarks

aelfrice contains two benchmark surfaces. The two surfaces have different
purposes. The two surfaces also have different cadences.

| Surface | Location | Purpose | Runtime | Cost | Cadence |
|---|---|---|---|---|---|
| Synthetic regression | `src/aelfrice/benchmark.py` | Catch retrieval/scoring regressions | <1s | $0 | Every PR (CI) |
| Academic suite | `benchmarks/` | Reproduce published numbers against external benchmarks | minutes–hours | LLM API spend | Nightly (bench-canonical cron) + per-PR smoke (bench-smoke) + manual dispatch |

The synthetic harness is a measurement instrument. It is **not** a proof of the
central feedback claim. Through v1.2 the posterior did not drive ranking. v1.3
added the partial Bayesian re-rank. v1.6 added the eval harness and the
heat-kernel composition wiring. v1.7 made BM25F default-on. v2.1 made
use_heat_kernel and use_hrr_structural default-on. See
[LIMITATIONS](../user/LIMITATIONS.md).

The academic suite is the reproducibility deliverable. It has five adapters:
MAB, LoCoMo, LongMemEval, StructMemEval, and AMA-Bench. At v1.0 all five
adapters were inert scaffolds. All five have run end-to-end since v2.0.
`aelf bench all` is the canonical entry point. For the status of each adapter, see
[`benchmarks/README.md`](../../benchmarks/README.md).

> **`retrieve_v2` and the production hook path have converged (v4.0).** The
> academic-suite adapters and the lane ablations call **`retrieve_v2()`**. As of
> v4.0 the production `UserPromptSubmit` hook path (`retrieve()`) is a thin
> adapter over `retrieve_v2`. [#1107](https://github.com/robotrocketscience/aelfrice/issues/1107)
> made that cutover. The staged lanes have graduated onto the production path one
> at a time, each one behind its own gate. **These lanes are now live on
> production retrieval:**
>
> - temporal spine ([#1064](https://github.com/robotrocketscience/aelfrice/issues/1064))
> - entity-persistence demotion ([#1096](https://github.com/robotrocketscience/aelfrice/issues/1096))
> - intentional clustering ([#436](https://github.com/robotrocketscience/aelfrice/issues/436))
> - HRR-structural ([#152](https://github.com/robotrocketscience/aelfrice/issues/152)), where HRR is holographic reduced representation
>
> Uplift attributable to these lanes therefore reflects production behaviour,
> not just the eval surface. The temporal spine's +14.6pp LoCoMo coverage is one
> such number. Two staged lanes are **exposed on `retrieve_v2` but held OFF by
> default**. These two lanes do **not** affect production, so their numbers are
> ablation-only:
>
> - origin tie-break ([#1089](https://github.com/robotrocketscience/aelfrice/issues/1089)), refuted on LoCoMo ([#1013](https://github.com/robotrocketscience/aelfrice/issues/1013))
> - HRR-expand ([#981](https://github.com/robotrocketscience/aelfrice/issues/981)), measured recall-neutral ([#1001](https://github.com/robotrocketscience/aelfrice/issues/1001))
>
> The always-on lanes also reflect production behaviour. The always-on lanes are
> BM25F, entity-index L2.5, breadth-first search (BFS), and type-aware
> compression. Three lanes previously listed here as always-on are not always-on.
> The γ posterior rerank and the ζ posterior rerank both resolve default-off.
> [#1162](https://github.com/robotrocketscience/aelfrice/issues/1162) returned
> the heat-kernel lane to default-off. The heat-kernel lane had been flagged on
> since v2.1.0 while it remained unreachable, since no production caller builds
> the eigenbasis it needs. No benchmark number on this page was ever produced
> with heat propagation active.

## Run the synthetic harness

```bash
aelf bench                   # default top-k=5
aelf bench --top-k 3
```

The output is a single JSON `BenchmarkReport`:

```json
{"benchmark_name": "aelfrice-bench-v1", "...": "...", "hit_at_1": 0.875, "hit_at_3": 1.0, "hit_at_5": 1.0,
 "mrr": 0.92, "p50_latency_ms": 0.4, "p99_latency_ms": 1.1}
```

The harness is deterministic against fresh in-memory stores. The corpus is 16
beliefs × 16 queries.

## What the academic suite measures

aelfrice's task is **known-item search over behavioural directives**. The agent
has corrected the user, locked a rule, or recorded a decision. On the next
relevant prompt we want that specific item retrieved. The natural headline
metric is **mean reciprocal rank (MRR)**.

We also report the metrics that each external benchmark *defines*. These metrics
make our results comparable with prior published systems:

- token-F1 on LoCoMo
- substring exact match on MAB
- GPT-4o judge on LongMemEval
- LLM-judge accuracy on StructMemEval and AMA-Bench

Those metrics frame topical relevance ("does the document cover the topic").
They do not frame behavioural relevance ("does this directive apply to what the
agent is about to do"). We report both numbers. The headline positioning is on
MRR.

### Read the canonical numbers with the reader in mind

**`aelf bench all` runs no reader.** The command ingests, retrieves, and scores.
There is no generation pass in `benchmarks/run.py`. The judge-based figures
above come from the two-pass protocol below. We run that protocol by hand. Those
figures are not what the canonical JSON contains. Everything in
`benchmarks/results/*.json` falls into one of two families. The distinction
decides how much a movement is worth (#1160):

- **Reader-dependent** — `f1`, `substring_exact_match`, LoCoMo's `overall_f1` and `category_f1`. These metrics hand the joined retrieval context to a scorer. That scorer was written for a model's *answer*. Token-F1 between about 2000 tokens of context and a three-token gold answer has precision around 3/2000. The number therefore moves with the **token budget** as much as with the ranking. If you halve the budget, the reported F1 roughly doubles, and retrieval returns strictly less. Treat a gain here as a lead to investigate, not as a result.
- **Reader-independent** — `retrieval_quality.mrr` and `retrieval_quality.recall_at_k`. Every adapter that scores a blob reports these two metrics. They read the *ordering* of the retrieved list. Retrieval fills the budget in rank order. A smaller budget truncates the tail, so a smaller budget can only lower these two metrics. To move them, you must rank a relevant belief higher. That is the thing the benchmark exists to measure.

Some metrics belong to neither family. The harness cannot compute those metrics
at all. These metrics report the string **`n/a`** rather than `0.0`. The harness
records the reason under the `_not_applicable` key of the same object:

| Metric | Why `n/a` |
| --- | --- |
| `exact_match` (MAB, LongMemEval, AMA-Bench) | This metric compares the prediction to the gold answer as whole normalised strings. The prediction is the retrieved context. The retrieved context is never one short answer. |
| `locomo.category_f1.5` | The adversarial category scores a refusal. Nothing in this path can refuse. The harness also excludes this category from `overall_f1`. `overall_f1` is now the mean over the categories that were scored. `scored_qa` reports how many questions were scored, beside the unchanged `total_qa`. |

`benchmarks/tolerance.py` records an `n/a` leaf as `NOT_APPLICABLE`. It tallies
the leaf and prints it. It excludes the leaf from the rollup. Unlike a `pass`, a
`NOT_APPLICABLE` leaf does not count as evidence that anything was measured. A
run where every metric is `n/a` therefore fails as `NO_DATA`. Reporting these
metrics as `0.0` was worse than useless. `0.0` is the *worst possible score*, so
the canonical file read as a total retrieval failure. A tolerance band around
`0.0` also turned any genuine fix into a band excursion.

That distinction is also why the LongMemEval multi-session aggregation gap
(see [LIMITATIONS](../user/LIMITATIONS.md#out-of-scope)) shows up as a low
number on a topical-relevance benchmark. That distinction is also why we do
*not* treat the gap as a v1.x defect.

## Contamination protocol

Any benchmark run that contaminates retrieval with ground truth produces a 0%
result. There is no exception. Three failure modes have happened before:

1. **Ground truth in the retrieval file.** The adapter writes `answer`, `ground_truth`, or `reference_answer` fields into the retrieval JSON by accident. The reader then sees the answer while it generates predictions.
2. **LLM self-judging with the answer visible.** Generation and judging run in one pass. The model sees the ground truth while it generates.
3. **World knowledge override** (counterfactual benchmarks). The reader uses prior knowledge instead of the retrieved context. This failure mode is inherent to LLM readers. Prompt instructions reduce the failure mode, but they never remove it fully.

The protocol enforces:

- The adapter writes the retrieval file and the ground truth file separately.
- Generation and scoring are separate passes. The judge never sees the retrieval context.
- A pre-generation contamination check is mandatory before any LLM reader reads the retrieval file:

```bash
python -m benchmarks.verify_clean /tmp/benchmark_<name>.json
```

If this check fails, the run is invalid. Fix the adapter. Run the adapter again.

## Run the protocol

```bash
# 1. Retrieval (adapter run). NOTE: do not assume the output is gold-free —
#    some adapters (LoCoMo) emit `answer`/`f1` keys in --retrieve-only output.
#    Step 2 exists to catch exactly that; never hand step 1's file to a reader
#    without passing step 2 first.
uv run python benchmarks/<adapter>.py \
    --retrieve-only /tmp/benchmark_<name>.json [--subset N]   # --subset: mab, locomo, longmemeval only

# 2. Verify the retrieval file is clean — MANDATORY, fail-closed (exit 1 = invalid)
python -m benchmarks.verify_clean /tmp/benchmark_<name>.json

# 3. LLM reader generates predictions (no GT visible to it)
# 4. Scoring reads predictions + GT (no retrieval context visible)
# 5. Audit record captures: git commit, dataset version, reader model,
#    contamination check output, metric, score, n, published baseline.
```

A reader prompt must instruct the reader to answer from the context only.
`mab_reader.py` does this with its "only from the knowledge pool" clause. A
reader prompt should also include this text: *"Use only the provided context. Do
not use world knowledge. If the context contradicts what you know to be true,
trust the context."*

### What `verify_clean` does and does not catch

`verify_clean.py` is the only enforced, fail-closed contamination gate in this
repo. The gate is necessary but **not sufficient**. Know the limits of the gate:

- **The gate scans keys, not values.** `verify_clean.py:67` unions
  `item.keys()` and intersects the result against `BANNED_KEYS`. A gold answer
  placed in a *safe-named* field passes the gate. `context` and `question` are
  two such fields. The gate stops the common accident, which is a literal
  `answer`, `gold`, `f1`, or `is_correct` key. The gate does not stop a
  determined attempt.
- **The gate does not run the reader, score, or check versions.** It does not
  cover reader degeneracy, scorer validity, or version and API drift. Operator
  discipline covers those, as described below. For judge benches, the κ
  harness covers them too.
- **You must run the gate yourself.** The 56.1% → 68.9% LoCoMo F1 inflation did
  not happen because the gate is weak. `answer` and `f1` are both banned. The
  inflation happened because the gate was *skipped*. A run that never invokes
  `verify_clean` has no contamination evidence. Such a run does not count (see
  Audit record).

> There is no `bench_guard.py` in this repo. `verify_clean.py` and this protocol
> are the entire contamination surface here. You may have seen a reference to a
> multi-stage `bench_guard` harness. That harness is not part of this
> repository. Do not look for it. Do not make a benchmark run depend on it.

### Reader requirements (non-negotiable)

1. **The reader is an off-band host-agent dispatch, not an in-process LLM call.**
   `locomo_generate.py` is a schema placeholder. The host agent generates the
   predictions. The host agent reads the retrieval file natively. Do not put an
   LLM SDK into the scored reader path. Do not put an API key into that path.
   (`mab_reader.py` calls an SDK directly. An operator runs `mab_reader.py` by
   hand only. CI never runs `mab_reader.py`.)
2. **The reader sees the retrieved context and the question only. The reader
   never sees the ground truth (GT) file.** Generation (pass 1) and scoring
   (pass 2) are separate processes. The two processes read separate files.
3. **Declare aelfrice's context ordering in the prompt.** The adapter joins the
   retrieved beliefs in rank order, where **first-in-context = most recent /
   most relevant**. A reader that assumes last-is-most-relevant misreads
   conflicting facts. Such a reader scores ~0% on state-tracking and
   conflict-resolution tasks. For a conflict task, give the reader an explicit
   rule. For example, the mab prompt gives this rule: "the newer fact has the
   larger serial number; resolve conflicts by taking the newest".

### Version drift

An adapter that runs against the wrong API surface produces numbers that look
real. Those numbers measure nothing:

- **Use `retrieve_v2`.** All in-tree adapters import
  `from aelfrice.retrieval import retrieve_v2 as retrieve`
  (`locomo_adapter.py:28`, `longmemeval_adapter.py:27`,
  `structmemeval_adapter.py:28`) with `use_bfs=True` and the default-on
  `use_hrr_structural` lane. #536 retired the legacy `use_hrr` alias.
  A lab-side 2.0.0-era adapter that calls bare `retrieve()` does *run*, but it
  measures a no-op substrate (`avg_beliefs=0`).
- **Pin and record the provenance.** The same adapter gives different numbers
  across substrate versions. LoCoMo moved from 66.1% to 40.88% with
  byte-identical scoring once HRR became a no-op post-#605. The `git_commit`
  field of the audit record, the dataset version, and the active flags are what
  make a number interpretable. `aelf bench all` is the single canonical entry
  point, so every adapter shares one substrate version.

## Per-benchmark specifics

| Benchmark | Dataset | Metric | Notes |
|---|---|---|---|
| MAB FactConsolidation | `ai-hyz/MemoryAgentBench` Conflict_Resolution | substring exact match (paper's normalisation) | The adapter uses 4,096-token chunks. It splits the text with NLTK `sent_tokenize`. The prompt must state the rule for conflict resolution by serial number. |
| LoCoMo | `locomo10.json` | token-F1 with Porter stemming | The adapter preserves the session boundaries on ingest. Category 5 is forced-choice. |
| LongMemEval | `xiaowu0162/longmemeval-cleaned` oracle | GPT-4o binary judge (paper) | The adapter passes question_date to retrieval for temporal grounding. |
| StructMemEval | yandex-research/StructMemEval | LLM judge binary | The adapter uses synthetic timestamps and temporal_sort. You must disclose both. |
| AMA-Bench | `AMA-bench/AMA-bench` test | LLM judge accuracy (paper: Qwen3-32B) | You must disclose an alternative judge. |

## Eval-judge calibration

LLM-judge benchmarks collapse to noise if the judge is not reproducible. The
LLM-judge benchmarks are LongMemEval, StructMemEval, AMA-Bench, and the
context-rebuilder eval harness at `benchmarks/context-rebuilder/`. The aelfrice
project is locked on a deterministic narrow surface
([PHILOSOPHY](PHILOSOPHY.md), #605). A single-run judge verdict does not
establish that the *eval* itself is deterministic enough to gate a release on.

The calibration target is **Cohen's κ inter-judge agreement** across N≥3
independent judge invocations. Those invocations run over the same
`(expected, actual)` pairs.

### Two κ measures

| Measure | What it captures | Threshold |
|---|---|---|
| **Inter-judge κ** (run vs run, pairwise) | Judge reproducibility — same pair, same verdict across independent calls | **≥ 0.70** (gate) |
| **Judge-vs-baseline κ** (judge run vs `score_substring_exact_match`) | How much semantic lift the judge adds over the zero-LLM baseline | reported, not gated |

Inter-judge κ ≥ 0.70 is "substantial agreement" on the Landis-Koch scale.
Below 0.70, the judge's per-run verdicts are within noise of disagreement.
Below 0.70, the headline score in the run's audit record is not a defensible
release gate.

We report judge-vs-baseline κ for posterity, but we do **not** gate on it. A
high value means the judge is not earning its API cost, because the substring
baseline would have done the job. A low value is expected. A low value is what
the judge exists to provide. A threshold here either rejects a useful judge or
accepts a lazy one.

### Run protocol

```bash
# 1. Generate the (expected, actual) pair set via the host-agent
#    eval-replay flow (#600): run the harness with a run dir —
#    it writes per-cell replay_requests.jsonl (carrying `expected`);
#    dispatch those off-band and write replay_responses.jsonl
#    (carrying `actual`); re-run the same command so the responses
#    are joined by turn_idx and rows failing the substring check are
#    tagged reason=needs_llm_judge.
uv run python benchmarks/context-rebuilder/eval_harness.py \
    --mode threshold-sweep --corpus <corpus> \
    --run-dir benchmarks/results/687_run/run_${i} \
    --out benchmarks/results/687_run/sweep_${i}.json

# 2. For each of N>=3 independent judge runs: emit judge requests from
#    the needs_llm_judge rows via judges.llm_judge.write_judge_requests()
#    (operator-invoked, default-off — see benchmarks/context-rebuilder/
#    README.md § Operator flow), producing judge_requests.jsonl;
#    dispatch each request off-band with the host CLI using
#    JUDGE_PROMPT_TEMPLATE (fresh session per run, no shared cache);
#    write judge_responses.jsonl, then fold verdicts with
#    read_judge_responses() / apply_judge_verdicts().

# 3. Compute pairwise inter-judge κ and judge-vs-baseline κ.
uv run python -m benchmarks.context_rebuilder.kappa \
    --runs benchmarks/results/687_run/judge_{1,2,3}.json \
    --baseline benchmarks/results/687_run/substring_baseline.json \
    --out benchmarks/results/687_run/judge_kappa.json
```

### Zero-LLM baseline

The deterministic baseline is `score_substring_exact_match(prediction,
ground_truth) > 0` (see `benchmarks/qa_scoring.py:53`). Binarized verdicts
from this baseline form the comparison vector for judge-vs-baseline κ.

If the eval task is `subject-match + load-bearing-claim`, substring
systematically misses semantic-match cases. That task is the context-rebuilder
hot-start interpretation. The miss is expected. The miss is why this gate is
report-only.

### Judge-kappa artifact

Every multi-judge run produces `benchmarks/results/<run-id>/judge_kappa.json`:

```json
{
  "run_id": "687_run",
  "n_runs": 3,
  "n_pairs": 18,
  "judge_model": "<judge-model>",
  "baseline": "score_substring_exact_match",
  "inter_judge_kappa": {
    "run_1_vs_run_2": 0.78,
    "run_1_vs_run_3": 0.72,
    "run_2_vs_run_3": 0.74,
    "mean": 0.75,
    "min": 0.72
  },
  "judge_vs_baseline_kappa": {"run_1": 0.29, "run_2": 0.33, "run_3": 0.31, "mean": 0.31},
  "failure_reasons": [],
  "per_run_hot_start_fidelity": [1.0, 0.94, 1.0],
  "hot_start_fidelity_mean": 0.98,
  "calibrated": true
}
```

`calibrated: true` requires:

- `inter_judge_kappa.min ≥ 0.70`. This value is the **min** across all
  run-pairs, not the mean. The min avoids averaging out a single noisy
  run-pair.
- `hot_start_fidelity_mean ≥ 0.80`. This bar comes from the #592 acceptance
  criterion (AC). It is ratified under the measurement scope below (see "Bench
  measurement scope").
- N≥3 runs

### Bench measurement scope

`hot_start_fidelity` measures **rebuilder-pack quality at the limit of a
cooperative, instructed reader**. It is *not* a measurement of typical
post-clear recall by an arbitrary host CLI.

`benchmarks/context-rebuilder/eval_harness.py::_assemble_post_clear_prompt`
assembles the replay prompt. That prompt prepends `POST_CLEAR_INSTRUCTION`.
`POST_CLEAR_INSTRUCTION` is a conservative cooperative-reader instruction. It
asks the dispatched child task to:

1. Answer the specific question the user_turn asks (no question
   conflation across adjacent atoms in the rebuild block).
2. Cite the specific fact from the rebuild block that the user_turn
   asks about (no dropped facts when the answer is present in pack).
3. Stay scoped to the user_turn's atom (no cross-contamination from
   adjacent atoms even when they are present in the pack).

The instruction is deliberately conservative. It does **not** ask the
reader to recapitulate named atoms verbatim. #797 rejected the
verbatim-recap wording, because that wording would let any pack that
contains the named atoms score 1.0, whatever the quality of the pack.
Such a score would defeat the bench as a regression-detection signal.

**The 0.80 AC bar is ratified against this instructed-reader
measurement.** A fall in this number means one of two things:
rebuilder-pack quality has regressed, *or* the reader instruction has
decohered. Both are signals worth catching. Meeting the 0.80 bar does
not mean that a typical un-instructed host CLI recalls 0.80 of the same
atoms after a clear. That is a separate question. If that question needs
an answer, it belongs in a separate AC and separate instrumentation.

Refs: #592 (original AC), #687 / PR #727 (κ commit-3), #777 (κ Run 2),
issue #797 (P3 ratification — instructed-reader scope + AC at 0.80).

### Sample-size caveat

At N=3 runs over ~18 deduplicated pairs, the 95% confidence interval on a
κ point estimate of 0.70 spans roughly 0.45–0.90. Those pairs are the
#592 hot-start corpus. The gate is therefore noisy at the standard N=3.
There are two ways to tighten the gate:

1. **Bump to N=5 runs.** This costs ~1.67× the judge API cost (5/3 of
   N=3), but it cuts the confidence interval roughly in half. We
   recommend N=5 for any pre-release ratification run.
2. **Accept the noisy gate.** Document the confidence-interval range in
   the run's audit record. This option is suitable for routine bench
   regression checks. It is not suitable for release-gate decisions.

A failed κ-gate is **not** a code bug. It means one of two things: the
judge prompt needs tightening, or the underlying corpus contains
ambiguous pairs that no judge can classify reproducibly. A tighter
judge prompt has more explicit refusal-on-ambiguity rules. To find
which pairs disagreed, inspect the per-judge input JSONL files. Those
files are `run_<i>.jsonl`, and
`read_judge_responses` reads their `{turn_idx, matched, rationale}` rows.
kappa.json itself only carries per-pair agreement scores, not per-row
verdicts.

## Scorer recalibration

An adapter can misreport scores without warning. This happens when the adapter
regresses, not when the product regresses.  `verify_clean.py` guards against ground-truth contamination
of retrieval files. `benchmarks/recalibrate.py` guards against scorer-logic
drift. It checks each scorer against a pinned oracle fixture.

### Problem

A subtly broken scorer reports a low number. CI cannot distinguish that
number from a real product regression. Examples of such a break are a wrong
field read, bag-of-words scoring where stemmed F1 was intended, and a stale
category dispatch.  A broken scorer was the root cause of several "low
benchmark number" investigations. Those investigations found harness
artifacts rather than product defects.

### Mechanism

Each scored adapter that has deterministic (stdlib-only) scoring logic has
a pinned **oracle fixture** under `benchmarks/oracle_fixtures/`. The fixture
contains
`(prediction, ground_truth[, category], expected_lower, expected_upper)`
tuples.  The recalibration check:

1. Loads each fixture JSON.
2. Calls the scorer function directly on each tuple (no retrieval, no LLM).
3. Asserts the returned score falls within `[expected_lower, expected_upper]`.
4. Exits non-zero if any tuple is outside its band.

A band is an explicit author-specified range, not a relative-to-canonical
computation.  A narrow band, for example `[1.0, 1.0]`, exercises exact
correctness. A wider band, for example `[0.65, 0.68]`, tolerates minor
floating-point differences. The wider band still catches the class of bugs
that shift scores significantly: a field swap, a wrong branch, or a wrong
formula.

### Run the check

```bash
uv run python -m benchmarks.recalibrate          # all adapters
uv run python -m benchmarks.recalibrate locomo   # one adapter
uv run python -m benchmarks.recalibrate --list   # show registered adapters
```

The check also runs automatically in the `bench-smoke` CI job on every PR
that touches `benchmarks/`.

### Covered adapters

| Adapter | Oracle fixture | Scorer checked | Notes |
|---|---|---|---|
| LoCoMo | `oracle_fixtures/locomo_scorer.json` | `locomo_adapter.score_qa` | All 5 categories (1=multi-hop, 2=temporal, 3=open-ended, 4=single-hop, 5=adversarial) |
| MAB | `oracle_fixtures/mab_scorer.json` | `qa_scoring.*` | `score_exact_match`, `score_substring_exact_match`, `score_f1`, `score_multi_answer` |

### Adapters not oracle-checked

| Adapter | Reason |
|---|---|
| LongMemEval | An LLM judge produces the final "score" as a binary verdict. The verdict-aggregation logic (`longmemeval_score.py`) is a trivial counter. `tests/test_longmemeval_scoring.py` already covers that counter.  Oracle-checking the judge call itself requires a live model. |
| StructMemEval | The same reason applies. The judge is per-task. `test_structmemeval_*` tests the aggregation logic. |
| AMA-Bench | There is no end-to-end oracle check. The MAB fixture covers the shared `qa_scoring.score_multi_answer` helper only. |

### Adding an oracle fixture for a new adapter

1. Create `benchmarks/oracle_fixtures/<adapter>_scorer.json`. Follow the
   schema of the existing fixtures:
   ```json
   {
     "adapter": "<adapter>",
     "scorer": "<dotted.module.path.to.scorer_fn>",
     "description": "...",
     "oracles": [
       {
         "label": "descriptive name",
         "<scorer-specific input fields>": "...",
         "expected_lower": 0.0,
         "expected_upper": 1.0
       }
     ]
   }
   ```
2. Add a runner function `_run_<adapter>_oracle` in `benchmarks/recalibrate.py`.
   The function calls the scorer. It feeds each tuple through
   `_check_oracle_tuple`.
3. Register the adapter in `ADAPTER_FIXTURES` and `_RUNNERS`.
4. Add tests in `tests/test_recalibrate.py`:
   - A `test_<adapter>_oracle_all_pass` that verifies the real scorer passes.
   - A broken-scorer variant to prove the mechanism fires.
5. Run `uv run python -m benchmarks.recalibrate <adapter>` to confirm. Then
   run `uv run pytest tests/test_recalibrate.py -v` to confirm.

An oracle fixture must be **synthetic**. Do not derive a fixture from an
upstream dataset. A fixture should cover all branches and categories of the
scorer logic.

## Audit record

Every academic run produces:

```json
{
  "benchmark": "...",
  "git_commit": "...",
  "adapter": "benchmarks/...",
  "reader_model": "...",
  "contamination_check": "CLEAN",
  "metric": "...",
  "score": 0.XX,
  "n": 100,
  "published_baseline": "..."
}
```

The audit record is required for the run to count. A run without an audit
record does not enter `benchmarks/results/`.

## See also

- [`src/aelfrice/benchmark.py`](../../src/aelfrice/benchmark.py) — synthetic harness source.
- [`benchmarks/README.md`](../../benchmarks/README.md) — per-adapter activation status.
- [ROADMAP § v2.0.0](ROADMAP.md) — the shipped milestone at which the academic suite reproduces every headline number.
