### Changed

- **Praise now raises the confidence of the beliefs behind the answer by default; complaints still need an opt-in ([#1647](https://github.com/robotrocketscience/aelfrice/issues/1647)).** Sentiment-from-prose feedback was off by default. Short replies were measured on 5,398 logged prompts (2,999 of them not harness records), with a random sample of 200 labelled by two blind graders (they agreed on 191 of 200, Cohen's kappa 0.82):
  - Positive matches ("perfect", "looks good") were 95–98% precise.
  - Negative matches were 58–64% precise. A bare "no", "fix it", and "try again" mostly open a new instruction rather than judge the answer.

  **Positive half:** `[feedback] sentiment_from_prose` now defaults to `true`, so praise applies with no setup. Setting it to `false`, or `AELFRICE_FEEDBACK_SENTIMENT_FROM_PROSE=0`, still turns the whole lane off.

  **Negative half:** complaints need a new key, `[feedback] sentiment_negative = true`, or `AELFRICE_FEEDBACK_SENTIMENT_NEGATIVE=1`. The negative patterns were tightened:
  - "no" counts only when it opens the prompt;
  - "wrong" counts only as a verdict;
  - "still wrong", "still broken", and similar are matched;
  - "fix it" and "try again" are gone.

  On 34 graded negative matches the tuning never saw (35 drawn; one was unclear to both graders), the tightened patterns scored 73.5% and 61.8%. The pre-registered bar was 70% on both graders, and one fell short. That's why negatives stay opt-in.

  The narrowing costs recall: the tightened patterns keep 57 of the 116 negative matches the old ones made, and some lost ones were real verdicts, for example "no" after an image reference ("[Image #1] no, that's still the old version"). Four prompts that used to match a negative pattern now match a positive one instead. Positive precision was measured under the old pattern set; it was not measured again. A negative match with the key off moves nothing, but still lands in the hook audit marked `negative_disabled`, so the lane can be measured again on fresh prompts.

  **Harness records:** the detector no longer scores them, such as a background task's notice, as your reaction.

  **CI audit:** `benchmarks/posterior_channel_audit.py` now asserts the new defaults: the positive lane on, the negative lane off.

  If you had opted in with `sentiment_from_prose = true` and want complaints to keep counting, also set `sentiment_negative = true`.
