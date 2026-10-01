### Fixed

- **Running the test suite no longer writes fixture beliefs into your real repo store ([#1678](https://github.com/robotrocketscience/aelfrice/issues/1678)).** `tests/test_transcript_round_trip.py` drives `PreCompact`, which spawns a detached `aelf ingest-transcript`. That child resolves its store with `db_path()`, and unless `AELFRICE_DB` is set, `db_path()` uses the git common dir of its working directory. Under pytest, that's your checkout, so each run added the fixture's beliefs and `ingest_log` rows to `<repo>/.git/aelfrice/memory.db`, the store that injects context into your real sessions. One store held five such beliefs and 35 `ingest_log` rows under the session id `round-trip-session`. The suite-wide `_sandbox_real_home` fixture in `tests/conftest.py` now sets `AELFRICE_DB` to a store in the session sandbox, so every child process inherits it. A test that exercises git-dir resolution deletes the variable with its own `monkeypatch.delenv`. Some tests still set or pop `os.environ["AELFRICE_DB"]` directly, which would change the pin for every later test, so a function-scoped fixture sets it again before and after each test. A new guard, `tests/test_suite_store_sandbox_1678.py`, starts a child the way the spawn does and fails if it resolves a store under the repo's git dir, and fails if a bare `os.environ` change in one test reaches the next. `tests/test_search_tool_hook_bash_telemetry.py` also stopped writing `./telemetry/` into your checkout: it set `AELFRICE_DB` to `:memory:`, whose parent is the working directory.

  This change doesn't touch rows that earlier runs already wrote. To remove them from your store, run these commands from inside your checkout or any of its worktrees:

  1. Find the store and back it up:

     ```sh
     DB="${AELFRICE_DB:-$(git rev-parse --path-format=absolute --git-common-dir)/aelfrice/memory.db}"
     cp "$DB" "$DB.bak"
     ```

  2. Delete the fixture beliefs: `aelf introspect --session round-trip-session --limit 0 --json | jq -r '.groups[].beliefs[].id' | xargs -n1 aelf delete --yes`.
  3. Delete the fixture's `ingest_log` rows, which no `aelf` command removes: `sqlite3 "$DB" "DELETE FROM log_versions WHERE log_id IN (SELECT id FROM ingest_log WHERE session_id = 'round-trip-session'); DELETE FROM ingest_log WHERE session_id = 'round-trip-session';"`.

  Delete the `ingest_log` rows in the same pass as the beliefs. A log row whose belief is gone counts as a `derived_orphan` in the replay check, and that bucket counts as drift.
