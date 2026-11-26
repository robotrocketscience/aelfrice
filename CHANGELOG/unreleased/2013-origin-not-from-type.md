### Removed

- **The origin backfill and `aelf migrate` no longer set origin `user_corrected` from the belief type ([#2013](https://github.com/robotrocketscience/aelfrice/issues/2013)).** The one-shot origin backfill, which runs on the first writable open of a store that has no `origin_backfill_complete` marker, gave every belief of type `correction` whose origin was `unknown` the origin `user_corrected`. `aelf migrate` did the same when it copied beliefs. Nothing showed that the user made those corrections, so they took the trust of a user correction. Both writers now leave such a belief at origin `unknown`. A locked belief whose origin is `unknown` still becomes `user_stated` on both paths, and a belief that already has origin `user_corrected` keeps it. The backfill still writes its marker. The opt-in Stop-hook auto-lock (`AELF_AUTOLOCK_CORRECTIONS=1`) still picks beliefs of type `correction` and sets their origin to `user_stated`. [#2022](https://github.com/robotrocketscience/aelfrice/issues/2022) tracks it.

  To upgrade, you don't need to do anything. A store that a release since v4.2.0 has opened for writing already has the marker, so its backfill doesn't run again and its origins don't change.

  - On by default: yes.
  - Reached from the hook path: yes. A hook opens the store for writing, so on a store that has no marker, the hook's open runs the backfill once.
  - Measured on: nothing measured.
