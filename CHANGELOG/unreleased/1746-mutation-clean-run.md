### CI

- **The per-PR mutation job runs its mutants again, and says when a run tested nothing ([#1746](https://github.com/robotrocketscience/aelfrice/issues/1746)).**
  - **The cause:** mutmut first runs the suite unmodified inside `mutants/` and stops at the first failing test, leaving every mutant `not checked`. Seven tests failed there, and one more failed only because of test order. The Report step exits 0 by design, so the PR showed a green tick over runs that executed nothing.
  - **Copied files:** `SECURITY.md` and `pyright_baseline.json` are now in `[tool.mutmut] also_copy`. The guard that derives that list from the suite now also recognizes a bare `REPO` constant, which is why it missed both.
  - **Excluded tests:** four test functions are left out of the mutation run only, through `--deselect` in `[tool.mutmut] pytest_add_cli_args`, each with its reason stated there:
    - two scans that list tracked files with `git ls-files`, which finds none in the untracked `mutants/` copy;
    - a source scan that reads mutmut's mutated copies;
    - a producer subprocess that inherits mutmut's stats mode.
  - **Test order:** `tests/test_budget_aa_replicate.py` failed when `tests/test_budget_census.py` loaded the census first in the same process, as mutmut's collection pass does. Its loader now drops the cached census.
  - **Warning:** when every mutant is `not checked`, or mutmut lists none, the Report step now emits a `::warning::` and puts a warning at the top of the step summary. The job still exits 0.
  - **Measured locally:** with mutmut 3.8.0 scoped to `src/aelfrice/db_paths.py`, the run went from 275 `not checked` to 211 killed and 64 survived.
