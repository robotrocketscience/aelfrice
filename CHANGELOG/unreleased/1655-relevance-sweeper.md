### Removed

- **The relevance sweeper and the two knobs it fed ([#1655](https://github.com/robotrocketscience/aelfrice/issues/1655)).** The detector was meant to mark which injected beliefs an answer used. It had marked 0 of 1,684 scored injections, because it needed the whole belief to appear verbatim in the reply. Looser word-overlap matchers scored only 1.3 to 1.4 times a random-session null, which measures topicality, not use. Nothing consumed the signal, and the sweeper scanned the transcript on every prompt, which took 40 to 50 ms. Removed:
  - the sweeper, `relevance_detection`, the two store methods only it called, and the `aelf doctor` close-the-loop section;
  - the #760 expansion-gate token-threshold knob (`AELFRICE_META_BELIEF_EXPANSION_GATE_TOKEN_THRESHOLD`), so the gate uses its fixed threshold of 80, and `should_run_expansion` no longer takes `store` or `now_ts`;
  - the #758 posterior-temperature knob (`AELFRICE_META_BELIEF_POSTERIOR_TEMPERATURE`), so the γ rerank uses `T = 1.0`.

  Neither knob was ever enabled, and both learned only from the relevance signal. Retrieval output is byte-identical to before on 84 queries across 5 entry points, with the γ rerank on and off. `injection_events` rows are still recorded, because exploration reads them, now with an empty consumer list. The #756 half-life and #757 anchor-weight knobs keep their own evidence paths.
