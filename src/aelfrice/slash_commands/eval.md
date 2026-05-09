---
name: aelf:eval
description: Run the relevance-calibration harness (P@K / ROC-AUC / Spearman ρ) on a synthetic corpus.
argument-hint: (optional) --corpus PATH --k N --seed N --json
allowed-tools:
  - Bash
---
<objective>
Score retrieval relevance against a labeled corpus of
`(query, known_belief, noise_beliefs)` rows and print the calibration
metrics ratified at #365 (P@K, ROC-AUC, Spearman ρ).

With no arguments, runs against the bundled public synthetic corpus —
deterministic at fixed seed, so two consecutive runs produce
bytes-identical output.
</objective>

<process>
Run: `uv run aelf eval $ARGUMENTS`
Display the output verbatim. Do not add commentary.
</process>
