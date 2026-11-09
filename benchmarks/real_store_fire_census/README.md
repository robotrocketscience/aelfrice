# Real-store fire census

The fire census checks, for each feature on `main`, whether the feature has
fired or written anything in your stores. It holds one read-only probe per
feature. A probe counts the rows or log lines that the feature writes when
it runs, and it reports one of these statuses for each store:

- `fired`: the probe found a trace that only this feature leaves, and the
  count is above 0. If any other code path can write the same rows, the
  probe reports `ambiguous` instead. When the note says the count is
  inferred from indirect rows, the count is a lower bound.
- `ambiguous`: the probe found a trace, and the count is above 0, but another
  writer leaves the same trace, and the store can't show which of them did.
  The note names the other writer. Count an `ambiguous` result as neither
  `fired` nor `silent`.
- `silent`: the store would show a trace if the feature had run there, and it
  shows none. The count is 0, or null when the store has no `memory.db`. A
  store that has no table or column for the feature's own trace is `silent`,
  because the feature never wrote there.
- `untraceable`: the store can't show whether the feature ran, and the count
  is null. Either the feature writes nothing durable, a run can leave no
  trace so a count of 0 proves nothing, or the store lacks the input the
  feature acts on or the table or log that the probe reads. A probe whose
  kind is `untraceable` reports this on every store.
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

- Run the census only on copies that the copy helper made. The runner reads
  a store folder only when a copy manifest, `.fire-census-copy`, sits at or
  above it, and `--global-dir` only when the manifest sits in that folder.
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

The destination must not exist yet. The helper creates it, and it refuses a
destination inside the source, inside this repository, or in a live store
location, comparing folders by device and inode rather than by path.

The helper never opens the source with sqlite and never writes under it. It
opens every file with `O_NOFOLLOW`, and it refuses a source that holds a
symlink. It reads every file except `-shm` files as plain bytes, then reads
each `memory.db` and `memory.db-wal` a second time and compares the bytes,
size, mtime, and inode. When anything differs, a writer was active, so it
throws that attempt away and tries again, up to 20 times. If every attempt
meets a writer, it exits with status 1 and asks you to close the sessions
that use the store. Any other failure ends the run at once, and the helper
removes only the folders that it created.

In its temp folder, the helper folds the WAL into each db, checks it with
`PRAGMA quick_check`, and removes the empty `-wal` and the `-shm` that the
check left. Last, it writes the copy manifest, which lists every file it
copied, and moves the copy into the destination. It refuses to write a
manifest into a folder that holds a file it didn't copy or that is a live
store location. It prints counts only. To see how
many files and bytes a copy takes without copying anything, add `--dry-run`.

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

- `outcome`: `fired` when any store fired, else `ambiguous` when any store
  was ambiguous, else `silent` when any store was silent, else `error` or
  `untraceable`.
- `stores`: the status, count, and integer detail for each store. A probe
  that reads only the global log runs on the `global` store. An untraceable
  probe has one entry, `_`.

The labels file is written only when you pass `--labels-out` without
`--dry-run`, with mode `0600`. Don't commit the labels file or paste it into
an issue.

## How the runner keeps stores unchanged

The checks below prevent accidents: pointing the census at a live store, a
stale path, or a symlink, or reading a copy that changed after it was made.
They don't defend against a user who writes a manifest by hand.

### The manifest rule

The copy manifest records the device and inode of the copy folder, the
copy's folders, and, for every file the helper copied, its path inside the
copy, device, inode, link count, size, and mtime. It holds no source path.

Every file the runner reads goes through the manifest: the db, its `-wal`
and `-shm`, every sidecar and log, and every file in the global copy. A file
must be listed and must still match its entry. The runner opens it with
`O_NOFOLLOW` and checks the open file's inode again with `fstat`. Listing a
folder must give exactly the entries that the manifest lists. A file that
isn't listed, a listed file that changed or is missing, a symlink in place
of a listed file, and a manifest moved to another folder all make the run
exit with status 2. The message names no path.

### Reading a db

The runner opens each db through the descriptor it checked, by having
sqlite open `/dev/fd/<n>`, so a db swapped after the check isn't read. This
behavior is tested on macOS only; the test skips on other systems. Where
`/dev/fd` doesn't exist, the runner opens the path and checks the inode
again after the open.

The sqlite URI sets `mode=ro` and `immutable=1`, and the connection sets
`PRAGMA query_only`. With `mode=ro` alone, sqlite creates `-wal` and `-shm`
files beside a WAL-mode database and rewrites the `-shm` index. With
`immutable=1`, it opens neither file and takes no locks, but it also ignores
the WAL, so the runner refuses a db whose `memory.db-wal` still holds frames.

### Live locations

The runner also refuses a path that is, or sits below, a git directory,
`~/.aelfrice`, the folder of the store that `$AELFRICE_DB` or
`$AELFRICE_USER_DB` names, or a `[store]` marker store. It compares these by
device and inode, so a symlink, a firmlink, another letter case, or a `..`
path doesn't hide one. A marker store that a `.aelfrice.toml` names by an
absolute path is recognized only when the store sits below that file or when
you run the census from below it.

### Outputs

The runner writes `--out` and `--labels-out` through a descriptor of the
output folder, to a new temporary file that it creates exclusively, and then
renames that file into place. It exits with status 2 when an output path is
an existing symlink or becomes one during the run, when the output folder is
a symlink, holds a `memory.db`, or sits inside any copy at any depth, when
the path is in a live store location or this repository, and when `--out`
and `--labels-out` name the same file. Two existing files are the same when
`os.path.samefile` says so. Otherwise two paths are the same when their names
match ignoring case and their folders are the same folder by device and
inode, so a symlinked folder or a case-only difference doesn't hide a match.
The runner makes every check before it creates any output folder, so a
refused run leaves no folder behind.

## Probe limits

These probes have no test in which they report `fired` or `ambiguous`:

- `A082`: Fires only on audit rows from before 5.1.0 or a block over a bound
  the hook enforces.
- `A120`: The recap line prints only; reports untraceable with a replay in
  detail.
- `A143`: Store-marker routing leaves no trace of the rule that chose the
  store.
- `A161`: The filter records no drop; untraceable once scan rows exist.
- `A211`: The suffix goes to stderr only; untraceable when a belief
  qualifies.
- `A236`: No production path writes the struct index.
- `A240`: The lane persists nothing; untraceable once its edges exist.
- `A254`: The rerank persists nothing; untraceable once its edges exist.
- `A280`: Doctor only reads the hooks' telemetry rows.

Every other traceable probe has a test that makes it report `fired` or
`ambiguous`. A028's writer was removed in 5.1.0, so its test plants the row
with SQL.

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
rejects any count or detail value that isn't an integer, any note that
contains a slash, and a count that doesn't fit the status: above 0 for
`fired` and `ambiguous`, 0 or null for `silent`, and null for
`untraceable`. Each probe module has a test module under `tests/` that
builds stores with aelfrice's own code. For a new probe, add a test that
makes it report `fired` or `ambiguous` and one that makes it report
`silent`, or add it to the lists in `registry.py` and in this section with
the reason.
