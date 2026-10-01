### Internal

- **The lock-loss figures in [#1620](https://github.com/robotrocketscience/aelfrice/issues/1620) now have a producer.** Run `benchmarks/lock_loss_census_1620.py --store <path>` to recount the issue's measurements from a store. The census reports:

  - The `lock_level` distribution over active beliefs and over every row, separately. A time-boxed lock whose expiry is at or before `--now` counts as `user_expired_unswept`, because the product drops it the next time it opens the store. The census compares the two as instants, not as strings, so a `--now` in another UTC offset gives the same answer. `--now` must be an ISO-8601 instant with a UTC offset; the census rejects any other value and reports the instant it used in UTC.
  - Beliefs whose `lock_expires_at` isn't an ISO-8601 instant with a UTC offset, as `lock_expiry_unparseable`. The census doesn't count such a lock as expired.
  - Beliefs whose content is an aelfrice command: `/aelf:<name>`, or `aelf <subcommand>` with a subcommand the CLI registers.
  - Beliefs that hold a rendered `<belief id=` block, and active speculative-origin beliefs.
  - With `--pattern`, one instruction's family, with each member's posterior mean, corroboration count, and `lock:unlock` and `lock:expire` history.

  Stores are per repository, so you can pass `--store` more than once, and the cross-store view shows a lock that landed in another repository's store. The census opens each store through a `mode=ro` URI with `query_only` set, so it never writes the database file. On a WAL store that has no `-wal` or `-shm` file yet, SQLite creates both when the census opens it. The census truncates belief content in its output. `tests/test_lock_loss_census_1620.py` covers each figure on synthetic stores.
