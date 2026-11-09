# Real-store fire census

The fire census checks, for each feature on `main`, whether the feature has
fired or written anything in your stores. It holds one read-only probe per
feature. A probe counts the rows or log lines that the feature writes when
it runs, and it reports one of these statuses for each store:

- `fired`: the probe found the feature's trace, and the count is above 0.
  Read the note too: a few probes infer the count from indirect rows.
- `silent`: the probe can read where the trace would be, and the count is 0.
  A store that has no table or column for the feature also reports `silent`,
  because the feature never ran there.
- `untraceable`: the store can't show whether the feature ran. The feature
  writes nothing durable, another writer leaves the same trace, or the store
  lacks the input the probe needs. On a store whose schema predates
  aelfrice, some probes report `untraceable` instead of `silent`; the note
  says which applies.
- `error`: the probe raised an exception, or it found a trace that breaks the
  feature's own contract, such as a due lock that was never swept. The note
  holds the exception's class name or a fixed description. A probe that reads
  only the global log reports `NoGlobalContext` when you don't pass
  `--global-dir`.

The census doesn't judge quality. A `fired` status means the feature ran,
not that it helped. Issue
[#1887](https://github.com/robotrocketscience/aelfrice/issues/1887) holds the
method, the pre-registered predictions, and the results of the first run.

## Before you begin

- Run the census only on copies of your stores. The runner refuses a live
  store: any path that resolves, through symlinks and in any letter case,
  into a `.git` folder, under `~/.aelfrice`, to the store `$AELFRICE_DB` or
  `$AELFRICE_USER_DB` names, or to a `[store]` marker store.
- Keep the copies, the output, and the labels file outside this repository.
  The labels file maps the opaque store labels back to your folder paths.
- Make sure you have enough free disk space for the copies. A copy is the
  size of the store folder, sidecars included.

## Copy your stores

For each store that you want to include, copy its folder with the helper:

```bash
uv run python -m benchmarks.real_store_fire_census.copy_store \
    path/to/repo/.git/aelfrice /tmp/census-copies/example
```

The helper never writes under the source folder, and it refuses a source
that holds a symlink. It reads every file as plain bytes, including
`memory.db-wal`, and checks that no `memory.db` or `-wal` changed while it
read them. When a writer keeps changing them, it copies each db through
sqlite's backup API instead, from a read-only connection. Then it folds the
WAL into the copy, checks the copy with `PRAGMA quick_check`, and prints
counts only. It leaves out `-shm` files, which sqlite rebuilds. To see how
many files and bytes a copy takes without copying anything, add
`--dry-run`.

If you copy by hand instead, wait until the store is idle, copy the whole
folder, and then fold the WAL into the copy so that the runner can read it:

```bash
cp -Rp path/to/repo/.git/aelfrice /tmp/census-copies/example
sqlite3 /tmp/census-copies/example/memory.db 'PRAGMA wal_checkpoint(TRUNCATE);'
```

The runner refuses a copy whose `memory.db-wal` still holds frames, because
it reads with `immutable=1`, which ignores the WAL.

To include the global store, the user store, and the global logs, copy
`~/.aelfrice` with the helper too. The copy also holds the legacy stores
under `projects/`.

## Run the census

1.  List the probes and the store labels without opening any store:

    ```bash
    uv run python -m benchmarks.real_store_fire_census \
        /tmp/census-copies/example /tmp/census-copies/other --dry-run
    ```

    A dry run writes no labels file, even when you pass `--labels-out`.

1.  Run every probe and write the results:

    ```bash
    uv run python -m benchmarks.real_store_fire_census \
        /tmp/census-copies/example /tmp/census-copies/other \
        --global-dir /tmp/census-copies/dotdir \
        --out /tmp/census-out/results.json \
        --labels-out /tmp/census-out/labels.json
    ```

    To run only some probes, add `--only A012,A175`.

Positional arguments are copies of repository stores. To add a copy of a
`git:<name>` store, use `--named`. To add a copy of a legacy per-project
store from `~/.aelfrice/projects/`, use `--legacy`. A few probes apply to one
kind of store only, and these flags tell the runner which kind each copy is.

## Read the output

The output is JSON that holds integers and fixed labels only. It never holds
belief text, prompt text, transcript text, or paths. Stores appear as
`store01`, `named01`, `legacy01`, `global`, and `user`, numbered in the
sorted order of their paths. Each probe entry has these fields:

- `outcome`: `fired` when any store fired, else `silent` when any store was
  silent, else `error` or `untraceable`.
- `stores`: the status, count, and integer detail for each store. A probe
  that reads only the global log runs on the `global` store. An untraceable
  probe has one entry, `_`.

The labels file is written only when you pass `--labels-out` without
`--dry-run`, with mode `0600`. Don't commit the labels file or paste it into
an issue.

## How the runner keeps stores unchanged

The runner opens each `memory.db` through a sqlite URI with `mode=ro` and
`immutable=1`, and sets `PRAGMA query_only`. With `mode=ro` alone, sqlite
creates `-wal` and `-shm` files beside a WAL-mode database and rewrites the
`-shm` index. With `immutable=1`, it opens neither file and takes no locks.
The runner reads sidecars with plain file reads. It refuses an `--out` or
`--labels-out` path inside a store that it reads, in a live store location,
or in this repository.

A marker store that a `.aelfrice.toml` names by an absolute path is
recognized only when the store sits below that file or when you run the
census from below it.

## Probe limits

These probes have no test in which they report `fired`:

- `A082`: Fires only on audit rows from before 5.1.0 or a block over a bound
  the hook enforces.
- `A120`: The recap line prints only; reports untraceable with a replay in
  detail.
- `A134`: PreCompact, the Stop flush, and doctor write the same rows;
  untraceable.
- `A143`: Store-marker routing leaves no trace of the rule that chose the
  store.
- `A161`: The filter records no drop; untraceable once scan rows exist.
- `A169`: Edges left by the reverted writers can't be told from other edges.
- `A199`: A continuation and a killed background child leave the same rows.
- `A211`: The suffix goes to stderr only; untraceable when a belief
  qualifies.
- `A236`: No production path writes the struct index.
- `A240`: The lane persists nothing; untraceable once its edges exist.
- `A254`: The rerank persists nothing; untraceable once its edges exist.
- `A280`: Doctor only reads the hooks' telemetry rows.

Every other traceable probe has a test that makes it report `fired`. A028's
writer was removed in 5.1.0, so its test plants the row with SQL.

These probes have no test in which they report `silent`:

- `A107`: A search-hook fire whose hits were already in the ring leaves no
  id.
- `A120`: The recap line prints only.
- `A143`: Store-marker routing leaves no trace.
- `A171`: The speaker gate writes nothing; fired is inferred from dropped
  lines.
- `A236`: No production path writes the struct index.
- `A280`: Doctor only reads the hooks' telemetry rows.
- `A292`: The migration guard deletes its marker on success.

`registry.NO_FIRED_TEST` and `registry.NO_SILENT_RESULT` hold the same lists.

## Add or change a probe

Each subsystem has a module, `probes_<subsystem>.py`, that exports `PROBES`.
A probe is a function from `core.Ctx` to `core.Result`. `core.Result`
rejects any count or detail value that isn't an integer and any note that
contains a slash. Each probe module has a test module under `tests/` that
builds stores with aelfrice's own code. For a new probe, add a test that
makes it report `fired` and one that makes it report `silent`, or add it to
the lists in `registry.py` and in this section with the reason.
