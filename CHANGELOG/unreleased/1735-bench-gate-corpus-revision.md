### CI

- **The bench-gate script says which corpus revision it reads ([#1735](https://github.com/robotrocketscience/aelfrice/issues/1735)).**
  - **The cause:** `scripts/run_bench_gate.sh` defaults the corpus root to the primary lab checkout, which reads whatever branch that checkout has checked out. At the 5.1.0 cut it was on a working branch with a superseded corpus. The tier failed, and nothing in the output showed that the corpus was stale.
  - **Now:** before the tests run, the script prints the corpus root's checkout, branch, commit, and whether the corpus has uncommitted changes. Files under the root that no commit tracks, including ignored ones, count as uncommitted. When the root isn't inside a git checkout, the script says so.
  - **Warning:** it warns when the branch isn't `main`, when the corpus has uncommitted changes, or when the root isn't in a checkout with at least one commit. An inherited `GIT_DIR` or `GIT_WORK_TREE` doesn't change the report.
  - **`--dry-run`:** prints those lines and the quoted pytest command, then exits without running the tier.
  - **Docs:** `RELEASING.md` step 7 now says a release run must read the lab corpus at a clean `main`. It shows how to point `AELFRICE_CORPUS_ROOT` at a worktree of `main`, reusing one that already exists.
