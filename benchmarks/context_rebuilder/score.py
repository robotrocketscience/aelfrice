"""Continuation-fidelity scorer for the context-rebuilder eval harness.

Implements the v1.4.0 answer-match metric (#138) on top of the
scaffolding that shipped with #136. Sits between `replay.run()` (which
walks a fixture turn-by-turn) and the headline JSON output (which now
carries `continuation_fidelity` alongside `token_budget_delta` and
`hook_latency_ms`).

Per `docs/context_rebuilder.md` § Validation:

    Continuation fidelity: does the agent answer the same subsequent
    questions correctly? Scored binary by the agent's responses
    against the ground-truth original session.

For each post-clear *assistant* turn in a replayed transcript, the
scorer compares the agent's answer (what the rebuilder + agent
produced after the clear) to the original session's answer at the
same turn (the ground truth). Per-turn scores aggregate to a single
`continuation_fidelity` ∈ [0, 1].

## Method choice (v1.4.0): `exact`

`exact` is the only method shipped at v1.4.0:

* Deterministic: same fixture + same answers -> identical score.
* Reproducible across machines: no model, no embeddings, no clock.
* No network: zero outbound calls. Eval-time outbound is acceptable
  per spec but not the v1.4.0 default; see `--score-method` toggle.
* Cheap: O(n) on turn count.

Comparison is done on a normalized form of the answer text:

  1. Unicode NFC normalization (collapses combining sequences).
  2. Lowercase.
  3. Whitespace collapse (any run of whitespace -> single space).
  4. Strip leading + trailing whitespace.

The normalization is intentionally conservative -- it absorbs only
the differences a literal `==` would otherwise reject for trivial
reasons (case, whitespace), without leaning into semantic
equivalence.

### Known false-positive modes (`exact` says "match" when fidelity is actually low)

* **Regression-shaped restatement.** Original answer was "v1.2.0
  shipped the alpha rebuilder"; post-clear answer is the same
  string -- but the rebuilder dropped working-state context the
  agent silently re-derived from prior turns. `exact` reports 1.0;
  the spec's "working-state loss" failure mode is invisible.
* **Quoted-from-rebuild.** The rebuilder's `<aelfrice-rebuild>`
  block already contains a verbatim quote of the original answer;
  the agent regurgitates it without genuinely continuing. `exact`
  scores this as a match; the failure (the agent didn't reason)
  goes uncaught.

### Known false-negative modes (`exact` says "miss" when fidelity is actually high)

* **Paraphrase / re-ordering.** Post-clear answer is semantically
  identical but uses different words ("v1.2.0 was the first
  release with the rebuilder" vs. the original "v1.2.0 shipped the
  alpha rebuilder"). `exact` rejects; an embedding or LLM-judge
  scorer would accept. This is the primary reason the spec parks
  LLM-judge / embedding scoring as v1.5.x.
* **Trailing differences.** Post-clear answer adds a clarifying
  sentence: original "Yes." vs. post-clear "Yes -- both paths
  fall back via AELFRICE_TRANSCRIPTS_DIR." Strict equality
  rejects; a substring or embedding scorer would accept.
* **Numeric formatting.** "32 tokens" vs. "thirty-two tokens" --
  same fact, different surface form, `exact` rejects.

These modes are documented (not hidden) so v1.4.x calibration runs
can correlate the per-task-type fidelity numbers with the failure
modes the spec calls out (`docs/context_rebuilder.md § Failure
modes`).

## Method toggle

`score_continuation_fidelity(..., method='exact')` is the only
enum value accepted in v1.4.0. The signature accepts the
`'embedding'` and `'llm-judge'` literal-string values for
forward-compatibility -- both raise `NotImplementedError` with a
pointer to the spec issue. The CLI exposes `--score-method` with
`exact` as default (see `__main__.py`).

LLM-judge scoring is parked deliberately: it introduces an
eval-time outbound call (acceptable per spec § Open) but is not
the v1.4.0 default and would couple CI green to network state.
Embedding similarity is parked because it adds a heavyweight
optional dep (sentence-transformers or similar) and the LLM-judge
path subsumes its accuracy.

## Empty / vacuous cases

* **Zero post-clear assistant turns.** The fixture cleared at the
  last turn (or after it). There are no post-clear answers to
  score, so the metric is vacuously perfect: returns `1.0`. We
  pick `1.0` over `NaN` so that aggregate dashboards stay
  numeric; `n_post_clear_assistant_turns` is recorded alongside
  the score for callers that need to gate on "did we even
  measure anything?".
* **No clear was injected.** `replay_result.clear_injected_at is
  None`. There is no notion of "post-clear" without a clear; the
  scorer returns `1.0` and the harness records 0 scored turns.
  This is the perfect-replay baseline (full transcript, no
  rebuild, agent's answers are the original answers by
  construction).

## Reproducibility contract

Same fixture + same `post_clear_answers` -> identical score, byte
for byte, on every machine. The scorer is a pure function of its
inputs; no `time.monotonic()`, no RNG, no I/O. The
`tests/test_continuation_fidelity_scorer.py` suite includes a
double-run determinism test that asserts this.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Final, Literal

from benchmarks.context_rebuilder.replay import ReplayResult, ReplayTurnResult

#: The set of method strings the scorer accepts.
#:
#: Only `exact` is implemented at v1.4.0. The other two are listed
#: so type-checking catches typos and so a future PR (#139 onward)
#: can wire them in without changing the public signature.
ScoreMethod = Literal["exact", "embedding", "llm-judge"]

#: Ship-default method for v1.4.0. Mirrors the docs and the CLI
#: default in `__main__.py`. Anchored as a constant so a PR that
#: changes the default also has to change this name (visible in diff).
DEFAULT_SCORE_METHOD: Final[ScoreMethod] = "exact"


@dataclass(frozen=True)
class FidelityScore:
    """Result of one continuation-fidelity scoring run.

    Three fields:

    * `score`: aggregate fidelity ∈ [0.0, 1.0]. `1.0` means every
      post-clear assistant turn matched ground truth; `0.0` means
      none did. Vacuous cases (no post-clear assistant turns) are
      `1.0` by documented convention.
    * `method`: which method produced the score (echoes the
      caller's choice). Carried into the JSON so downstream
      consumers can tell `exact` runs from future LLM-judge runs
      without inferring from context.
    * `n_post_clear_assistant_turns`: count of turns that *could*
      have been scored. Together with `score`, lets a caller
      distinguish "1.0 from a perfect replay of 8 answers" from
      "1.0 from the vacuous case with 0 answers".
    * `per_turn`: optional per-turn match flags (1 / 0). Empty
      list when there were no post-clear assistant turns.
      Useful for debugging which specific turns regressed.
    """

    score: float
    method: ScoreMethod
    n_post_clear_assistant_turns: int
    per_turn: list[int] = field(default_factory=lambda: [])  # typed-empty for pyright

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable view, embedded in the harness output."""
        return {
            "score": self.score,
            "method": self.method,
            "n_post_clear_assistant_turns": self.n_post_clear_assistant_turns,
            "per_turn": list(self.per_turn),
        }


def _normalize_for_exact(text: str) -> str:
    """Conservative text normalization for the `exact` method.

    Steps (in order):
      1. Unicode NFC normalization.
      2. Lowercase.
      3. Whitespace collapse.
      4. Strip.

    The normalization is documented in the module docstring and
    intentionally narrow. Any change to this function is a behaviour
    change to the headline metric.
    """
    nfc = unicodedata.normalize("NFC", text)
    lowered = nfc.lower()
    collapsed = " ".join(lowered.split())
    return collapsed


def _post_clear_assistant_turns(
    replay_result: ReplayResult,
) -> list[ReplayTurnResult]:
    """Subset of replay turns that are scoreable.

    A turn is scoreable iff:
      * A clear was injected (`clear_injected_at is not None`).
      * The turn comes strictly after the clear turn.
      * The turn is an assistant turn (we score *answers*, not the
        user prompts that elicited them).

    Returned in the order they appear in `replay_result.turns`.
    """
    clear_at = replay_result.clear_injected_at
    if clear_at is None:
        return []
    out: list[ReplayTurnResult] = []
    for t in replay_result.turns:
        if t.turn_index <= clear_at:
            continue
        if t.role != "assistant":
            continue
        out.append(t)
    return out


def _fixture_assistant_texts_after_clear(
    replay_result: ReplayResult,
    fixture_turn_texts: list[str],
) -> list[str]:
    """Pick out the original assistant texts that line up with post-clear turns.

    `fixture_turn_texts` is the list of `text` fields in the
    fixture's `ReplayTurn` records, indexed by `turn_index`. We
    walk the post-clear assistant subset of `replay_result.turns`
    and return the matching texts in the same order.
    """
    out: list[str] = []
    for t in _post_clear_assistant_turns(replay_result):
        if 0 <= t.turn_index < len(fixture_turn_texts):
            out.append(fixture_turn_texts[t.turn_index])
        else:
            # Defensive: if we can't index back, treat the slot as
            # an empty string. Matches the "missing data is a miss"
            # interpretation -- the alternative (raise) would couple
            # the scorer to fixture-loader bugs in noisy ways.
            out.append("")
    return out


def score_continuation_fidelity(
    replay_result: ReplayResult,
    *,
    fixture_turn_texts: list[str],
    post_clear_answers: list[str] | None = None,
    method: ScoreMethod = DEFAULT_SCORE_METHOD,
) -> FidelityScore:
    """Aggregate continuation-fidelity ∈ [0, 1] for one replay.

    Parameters
    ----------
    replay_result:
        The output of `replay.run(fixture, inject=...)`. Provides
        the per-turn record list and the clear index.
    fixture_turn_texts:
        Texts of the original fixture turns, indexed by
        `turn_index` (i.e. `[turn0_text, turn1_text, ...]`). These
        are the ground-truth answers post-clear.
    post_clear_answers:
        Optional. The agent's answers post-clear, in the same
        order as `_post_clear_assistant_turns(replay_result)`. If
        `None` (the v1.4.0 scaffolding default, used until the
        rebuilder hook lands in #139), the original session's
        ground-truth answers are used as both sides -- the
        perfect-replay baseline. Score is `1.0` by construction.
    method:
        Which scoring method to use. Only `exact` is implemented
        at v1.4.0; `embedding` and `llm-judge` raise
        `NotImplementedError`.

    Returns
    -------
    FidelityScore:
        Aggregate score, method echo, and per-turn match flags.

    Raises
    ------
    ValueError:
        If `post_clear_answers` is provided but its length does
        not match the number of post-clear assistant turns.
        Length mismatch is a programming error (the caller is
        feeding a misaligned list); we surface it loudly rather
        than silently scoring against shifted indices.
    NotImplementedError:
        If `method` is `embedding` or `llm-judge`. The error
        message points at the spec issue (#138) for the parking
        rationale.

    Notes
    -----
    Vacuous cases (no clear, or no post-clear assistant turns)
    return `score=1.0` with `n_post_clear_assistant_turns=0`. See
    module docstring for the documented choice of 1.0 over NaN.
    """
    if method == "embedding":
        raise NotImplementedError(
            "embedding-based scoring is parked for v1.5.x; see #138 "
            "and docs/context_rebuilder.md § Validation. v1.4.0 ships "
            "method='exact' only."
        )
    if method == "llm-judge":
        raise NotImplementedError(
            "LLM-judge scoring is parked for v1.5.x; see #138 and "
            "docs/context_rebuilder.md § Validation. v1.4.0 ships "
            "method='exact' only (no eval-time outbound calls)."
        )
    if method != "exact":
        # Defensive: Literal narrows the type but a runtime caller
        # could still hand us a bad string. Keep the error specific.
        raise ValueError(
            f"unsupported score method: {method!r} "
            f"(supported at v1.4.0: 'exact')"
        )

    # Build the ground-truth post-clear assistant texts.
    truth_texts = _fixture_assistant_texts_after_clear(
        replay_result, fixture_turn_texts
    )
    n = len(truth_texts)

    # Decide which "agent" texts we're scoring.
    if post_clear_answers is None:
        # Scaffolding default: score the fixture against itself,
        # which is the perfect-replay baseline. Documented so a
        # caller doesn't mistake 1.0 for a passing rebuilder.
        agent_texts = list(truth_texts)
    else:
        if len(post_clear_answers) != n:
            raise ValueError(
                f"post_clear_answers length {len(post_clear_answers)} "
                f"does not match the {n} post-clear assistant turn(s) "
                f"in the replay result"
            )
        agent_texts = list(post_clear_answers)

    # Vacuous case: no scoreable turns. By documented convention,
    # this is 1.0 (vacuously perfect), with `n=0` recorded so
    # callers can detect it.
    if n == 0:
        return FidelityScore(
            score=1.0,
            method="exact",
            n_post_clear_assistant_turns=0,
            per_turn=[],
        )

    # `exact` per-turn comparison.
    per_turn: list[int] = []
    for truth, agent in zip(truth_texts, agent_texts):
        if _normalize_for_exact(truth) == _normalize_for_exact(agent):
            per_turn.append(1)
        else:
            per_turn.append(0)

    aggregate = sum(per_turn) / float(n)
    return FidelityScore(
        score=aggregate,
        method="exact",
        n_post_clear_assistant_turns=n,
        per_turn=per_turn,
    )
