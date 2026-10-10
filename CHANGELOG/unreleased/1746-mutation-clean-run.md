### CI

- **The per-PR mutation job runs its mutants again, and says when a run tested nothing ([#1746](https://github.com/robotrocketscience/aelfrice/issues/1746)).**
  - **The cause:** mutmut first runs the suite unmodified inside `mutants/`. If any test fails there, mutmut stops before running a mutant, and every mutant stays `not checked`. Seven tests failed there, and one more failed only because of test order. The Report step exits 0 by design, so the PR showed a green tick over runs that executed nothing.
  - **Copied files:** `SECURITY.md` and `pyright_baseline.json` are now in `[tool.mutmut] also_copy`. The guard that derives that list from the suite missed both, because it didn't recognize a bare `REPO` constant. It does now.
  - **Excluded tests:** four test functions are left out of the mutation run only, through `--deselect` in `[tool.mutmut] pytest_add_cli_args`, each with its reason stated there:
    - two scans that list tracked files with `git ls-files`, which finds none in the untracked `mutants/` copy;
    - a source scan that reads mutmut's mutated copies;
    - a producer subprocess that inherits mutmut's stats mode.
  - **Test order:** `tests/test_budget_aa_replicate.py` failed when `tests/test_budget_census.py` loaded the census first in the same process, as mutmut's collection pass does. Its loader now drops the cached census.
  - **Warning:** when every mutant is `not checked`, or mutmut lists none, the Report step now emits a `::warning::` and puts a warning at the top of the step summary. A failing clean run is one cause; mutmut also stops early when it finds no test for any mutant, or when its forced-fail check fails. The job still exits 0.
  - **Measured locally:** with mutmut 3.8.0 scoped to `src/aelfrice/db_paths.py`, all 275 mutants were `not checked` before the fix. After it, on the current `db_paths.py`, 242 of 316 mutants were killed and 74 survived, with none left `not checked`.
  - **Pinned list:** a test now asserts the exact `pytest_add_cli_args` list, so dropping a `--deselect` or changing its flag fails the suite instead of the next mutation run.
