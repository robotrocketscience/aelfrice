### Added

- **A disabled complaint now records the beliefs it would have demoted ([#1677](https://github.com/robotrocketscience/aelfrice/issues/1677)).**
  - **Before:** with `[feedback] sentiment_negative` off, which is the default, a negative match wrote a `sentiment_feedback` audit row marked `negative_disabled` before the hook loaded the prior turn's beliefs. A re-measurement could count the fires but couldn't grade them against their targets.
  - **Now:** the row also lists `target_ids`. These are the prior turn's live, unlocked beliefs, which are the ones the enabled lane would move: it skips deleted beliefs, and the lock floor refuses locked ones. `belief_ids` stays the list that actually moved, which is empty, and nothing moves, as before. With the hook audit off, no row is written and the targets aren't looked up.
  - **Re-test producer:** `benchmarks/sentiment_negative_retest_1677.py` does four things:
    - It counts fresh disabled fires in the hook audit.
    - With `--transcripts`, it also counts the prompts the detector scores negative in the host's session transcripts, the population #1647 used.
    - It draws a seeded grading sheet.
    - It scores two graders' labels. For each grader it reports precision with a Wilson lower bound, plus Cohen's kappa and the verdict against the 70% bar.

    The verdict stays `insufficient` until each grader has graded at least 30 fires. Run the producer with `uv run python -m benchmarks.sentiment_negative_retest_1677`.
