### Added

- **A disabled complaint now records the beliefs it would have demoted ([#1677](https://github.com/robotrocketscience/aelfrice/issues/1677)).**
  - **Before:** with `[feedback] sentiment_negative` off, which is the default, a negative match wrote a `sentiment_feedback` audit row marked `negative_disabled` before the hook loaded the prior turn's beliefs. A re-measurement could count the fires but couldn't grade them against their targets.
  - **Now:** the row also lists `target_ids`, the prior turn's beliefs the complaint would have demoted. `belief_ids` stays the list that actually moved, which is empty. Nothing moves, as before.
  - **Re-test producer:** `benchmarks/sentiment_negative_retest_1677.py` counts fresh disabled fires, draws a seeded grading sheet, and scores two graders' labels. For each grader it reports precision with a Wilson lower bound, plus Cohen's kappa and the verdict against the 70% bar. Run it with `uv run python -m benchmarks.sentiment_negative_retest_1677`.
