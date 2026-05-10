# v3.0 spec: sentiment-feedback hook production wire-up (#606)

Spec for issue [#606](https://github.com/robotrocketscience/aelfrice/issues/606).
Production wire-up of `aelfrice.sentiment_feedback` (shipped at v2.0 per #193) into a
live `UserPromptSubmit` hook lane. Module is pure today; nothing calls it on a
real hook fire.

## What's being decided

1. Hook lane.
2. Which retrieval window receives the sentiment-driven α/β bump.
3. Audit-log surface for sentiment fires.
4. Opt-in flag scope (one flag or two).
5. Privacy posture re: existing transcript-ingest opt-out.

## Decisions

### 1. Hook lane: `UserPromptSubmit`

The corrective prompt ("no, that's wrong", "fix it") arrives in user prompt N
*after* the assistant has acted on the retrieval block from prompt N-1. The
right time to apply the correction is at UPS for prompt N, before the next
retrieval fires — that way the bumped posteriors are already reflected in the
hits this prompt returns.

`Stop` would fire after the assistant's turn, before the user has had a chance
to react. Wrong lane.

### 2. Retrieval window: the previous UPS audit record for the same `session_id`

The `hook_audit.jsonl` already records every UPS retrieval with structured
`beliefs[*].id`. The sentiment lane reads that JSONL, filters by
`hook == "user_prompt_submit"` AND `session_id == <current>`, takes the
most-recent prior row, and extracts the belief ids.

Boundary cases:

- **No prior UPS row for session.** First prompt; nothing to bump. No-op.
- **Prior UPS row has empty `beliefs`.** Retrieval returned nothing; no-op.
- **Audit disabled by config.** No prior beliefs visible to the sentiment
  hook either; no-op. Surfaced in `aelf health` so the operator can see why
  the lane is silent.

Single-session only. Cross-session sentiment propagation is out of scope per
#606.

### 3. Audit surface: new `hook_audit` tag `"sentiment_feedback"`

One JSONL row per sentiment fire. Fields:

- `hook = "sentiment_feedback"`
- `session_id`
- `prompt_prefix` (≤ `AUDIT_PROMPT_PREFIX_CAP`)
- `pattern` (named match, e.g. `"wrong"`, `"i_told_you"`)
- `matched_text` (literal substring; bounded by regex shape, typically ≤ 30
  chars)
- `sentiment` (`"positive"` | `"negative"`)
- `valence` (signed float passed to `apply_feedback`)
- `escalated` (bool — set when `detect_correction_frequency` fires)
- `belief_ids` (list of ids actually bumped)
- `n_beliefs` (count)

The existing `feedback_history` table still gets one row per affected belief
via `apply_feedback`, with `source = sentiment_inferred` (module-level
constant — unchanged from v2.0). The hook-audit row is the *event-level*
record; `feedback_history` is the *belief-level* record. Both exist; they
serve different queries.

### 4. Opt-in: single flag

`[feedback] sentiment_from_prose = true` in `.aelfrice.toml` (or
`AELFRICE_FEEDBACK_SENTIMENT_FROM_PROSE=1`). The existing
`sentiment_feedback.is_enabled()` already resolves this. The hook calls
`is_enabled(config)` and short-circuits when false. No new config key; the
module flag is the hook flag.

Programmatic use of `detect_sentiment` / `apply_sentiment_to_pending` from
non-hook code paths is unchanged — those functions are pure and have no
config-flag dependency.

### 5. Privacy posture

The UPS hook already reads every user prompt to do retrieval. The sentiment
lane does *more processing* on data already in the hook's hands — it does not
add a new data surface.

What's stored that wasn't stored before:

- One hook-audit row tagged `sentiment_feedback` per matched prompt
  (`pattern` + `matched_text` ≤ ~30 chars, no raw prompt body).
- One `feedback_history` row per affected belief id.

What's NOT stored:

- Raw prompt content (cap at `AUDIT_PROMPT_PREFIX_CAP` already enforced for
  UPS audit; same cap reused here).
- Anything that leaves the machine. No outbound calls. Stdlib regex only.

**Transcript-ingest opt-out semantics.** The acceptance criterion in #606 says
the hook must respect the transcript-ingest opt-out. Interpretation: the
sentiment lane is gated by the same `[feedback] sentiment_from_prose` flag,
which is opt-in (default false). A user who has chosen not to install
transcript-ingest hooks has already implicitly opted out of all prose-side
analysis surfaces; the new lane requires explicit opt-in via the existing
config key, which they will not have set. No additional plumbing needed.

`aelf health` surfaces enabled/disabled state via the existing
`_sentiment_from_prose_state` helper. No change to that surface for v3.0.

## Determinism

- Pure stdlib regex against the prompt string.
- Length guard at 200 chars rejects long pastes (unchanged from module).
- Same prompt + same prior-UPS audit row → byte-identical
  `apply_feedback` calls + audit row.
- The PHILOSOPHY locked memory (`Avoid embeddings + non-determinism in
  retrieval`) is preserved: the hot path is regex; no embedding lookup,
  no LLM, no learned classifier.

## Acceptance mapping (#606)

| #606 AC | Where satisfied |
|---|---|
| 1. Spec memo + hook lane + decay policy | this memo |
| 2. Determinism property | §Determinism above; existing module + regex |
| 3. Privacy property | §5 Privacy posture above |
| 4. Two-session bench fixture | `tests/test_hook_sentiment_feedback.py::test_correction_lowers_subsequent_ranking` |
| 5. Audit row per fire | §3 above; new tag `sentiment_feedback` |

## Out of scope

- Cross-session sentiment propagation (sentiment in session A bumping
  posteriors that surface for session B).
- Multi-language sentiment (English-only regex bank, unchanged from
  v2.0 module).
- Ranked distribution (equal-weighted distribution preserved per the
  v2.0 ratification: "matches the research-line behavior; ranked
  distribution adds a knob without an evidence-gate").
- Per-rank decay of the bump (every pending id receives the full
  signal; if the prior turn returned 5 hits, all 5 get bumped equally).
- LLM-judged sentiment (deferred to the #605 PHILOSOPHY-determinism
  decision; this lane stays regex-only regardless of that outcome).

## Refs

- `src/aelfrice/sentiment_feedback.py` — pure module (v2.0 shipped)
- `src/aelfrice/hook.py` — UPS hook entry-point (`user_prompt_submit`)
- `docs/v2_sentiment_feedback.md` — original v2.0 evaluation memo
- #193 — v2.0 evaluation gate (CLOSED COMPLETED 2026-05-03)
- #606 — this issue
