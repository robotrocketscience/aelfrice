# Real-store fire census

The fire census checks, for each feature on `main`, whether the feature has
fired or written anything in your stores. It holds one read-only probe per
feature. A probe counts the rows or log lines that the feature writes when
it runs, and it reports one of these statuses for each store:

- `fired`: the count is above 0.
- `silent`: the probe can see the feature's trace, and the count is 0.
- `untraceable`: the feature writes nothing that a later read can count, or
  the store's schema predates the feature.
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

- Run the census only on byte copies of your stores. Never point it at a
  live store: the runner refuses any path inside a `.git` folder or under
  `~/.aelfrice`.
- Keep the copies, the output, and the labels file outside this repository.
  The labels file maps the opaque store labels back to your folder paths.
- Make sure you have enough free disk space for the copies. A copy is the
  size of the store folder, sidecars included.

## Copy your stores

For each store that you want to include, copy its folder with the helper.
The helper copies every file byte for byte, including `memory.db-wal` and
`memory.db-shm`, and then folds the WAL into the copy and checks it with
`PRAGMA quick_check`:

```bash
uv run python -m benchmarks.real_store_fire_census.copy_store \
    path/to/repo/.git/aelfrice /tmp/census-copies/example
```

To see how many files and bytes a copy takes without copying anything, add
`--dry-run`.

If you copy by hand instead, copy the whole folder, then fold the WAL into
the copy so that the runner can read it:

```bash
cp -Rp path/to/repo/.git/aelfrice /tmp/census-copies/example
sqlite3 /tmp/census-copies/example/memory.db 'PRAGMA wal_checkpoint(TRUNCATE);'
```

The runner refuses a copy whose `memory.db-wal` still holds frames, because
it reads with `immutable=1`, which ignores the WAL.

To include the global store and the user store, copy `~/.aelfrice` the same
way.

## Run the census

1.  List the probes and the store labels without opening any store:

    ```bash
    uv run python -m benchmarks.real_store_fire_census \
        /tmp/census-copies/example /tmp/census-copies/other --dry-run
    ```

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

The labels file is written only when you pass `--labels-out`, with mode
`0600`. The runner refuses a labels path inside this repository. Don't
commit the labels file or paste it into an issue.

## How the runner keeps stores unchanged

The runner opens each `memory.db` through a sqlite URI with `mode=ro` and
`immutable=1`, and sets `PRAGMA query_only`. With `mode=ro` alone, sqlite
creates `-wal` and `-shm` files beside a WAL-mode database and rewrites the
`-shm` index. With `immutable=1`, it opens neither file and takes no locks.
The runner reads sidecars with plain file reads, and it refuses an `--out`
or `--labels-out` path inside a store that it reads.

## Add or change a probe

Each subsystem has a module, `probes_<subsystem>.py`, that exports `PROBES`.
A probe is a function from `core.Ctx` to `core.Result`. `core.Result`
rejects any count or detail value that isn't an integer and any note that
contains a slash. Each probe module has a test module under `tests/` that
builds a store with aelfrice's own code, makes the feature fire, and checks
that the probe reports `fired`, and that it reports `silent` on an empty
store.
