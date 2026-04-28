# Workflow-agnostic project identity

**Status:** spec.
**Target milestone:** v1.5.0 (TBD).
**Tracking issue:** TBD.
**Dependencies:** stdlib only (subprocess, hashlib, pathlib, tempfile,
re). Calls `git rev-list --max-parents=0 HEAD`,
`git rev-parse --show-toplevel`, `git rev-parse --git-common-dir`,
`git rev-parse --is-shallow-repository`.
**Risk:** medium. Replaces three live resolvers (`cli.db_path`,
`project_warm._project_id`, `aelfrice-db-path.resolve_db_path`) with
one chain. Includes a one-time data migration. The migration is
byte-preserving and idempotent; rollback is by restoring the legacy
DB file.

## Summary

Aelfrice's existing project-identity scheme uses three independent
code paths to compute "which project is this?", each derived from
absolute filesystem paths. They all produce different values for the
same logical project, and they all break when you `git clone` /
`mv` / set up on a fresh machine. The DB silently forks and beliefs
become unreachable.

This spec replaces them with one identity chain, in priority order:

1. `$AELFRICE_PROJECT_ID` env override.
2. Tracked `<show-toplevel>/.aelfrice/project-id` file.
3. `sha256[:12]` of the lex-smallest output of
   `git rev-list --max-parents=0 HEAD`.
4. `sha256[:12]` of the absolute working-directory path.

DB lives at `~/.aelfrice/projects/<id>/memory.db`. Migration from
the legacy location (`<git-common-dir>/aelfrice/memory.db`) is a
tempfile-fsync-rename copy with a `.migrated-to` sentinel.

The design is content-derived (layer 3), with explicit user-facing
opt-outs (layers 1 and 2) and a non-git fallback (layer 4). It is
invariant under `git clone`, `git worktree add`, `mv`, mirror+reclone,
remote-URL rewrite, and forking — eight distinct workflows verified
in the design's R&D campaign.

The known exceptions and their mitigations are documented in
[§ "Drift hazards"](#drift-hazards) below.

## Goals

- Identity does not fork on the workflows the user-stated principle
  enumerates: branching (where branches share a root), worktree-add,
  clone, mv, fresh-machine setup, switching between bare and non-bare
  checkouts.
- A user can opt to share identity across forks (default) or to
  pin a separate identity per project (via the tracked file).
- Identity is computable in isolation (no network, no central
  registry).
- Migration from the existing scheme is idempotent and
  interrupt-safe; data is byte-preserved across the move.
- Per-resolve cost is comparable to the existing resolvers
  (~110 ms p50, dominated by subprocess overhead).

## Non-goals

- Cross-user identity-sharing across machines. Each user has their
  own `~/.aelfrice/projects/<id>/`. Two users with the same upstream
  repo each get a private DB at the same id; no automatic
  synchronization.
- Cryptographic identity uniqueness. The 48-bit slug is fit for
  per-user scale (collision math + measurement; see
  [§ "Identity collision"](#identity-collision)).
- Identity stability across history rewrites. By design, content-
  derived identity changes when the content (commit graph reachability
  from HEAD) changes. Mitigation: layer 2.

## Resolution chain

### Layer 1 — `$AELFRICE_PROJECT_ID`

If the env var is set and passes input validation, return it
verbatim. Otherwise fall through.

```python
override = env.get("AELFRICE_PROJECT_ID", "")
validated = _validate_layer_input(override)
if validated is not None:
    return validated
```

### Layer 2 — tracked `.aelfrice/project-id` file

If `<git rev-parse --show-toplevel>/.aelfrice/project-id` exists,
read its contents stripped, validate, and return. Otherwise fall
through.

```python
top = _show_toplevel(cwd)              # may be None for non-git cwds
if top is not None:
    p = top / ".aelfrice" / "project-id"
    if p.is_file():
        validated = _validate_layer_input(p.read_text().strip())
        if validated is not None:
            return validated
```

The use of `--show-toplevel` (not `--show-superproject-working-tree`)
is deliberate. Inside a submodule, this reads the submodule's tracked
file, honoring the "submodule wins from inside" rule.

The file is honored regardless of git tracking state (committed,
staged, or merely on disk). The cross-collaborator share property
applies **only** when the file is committed; uncommitted files are
local overrides.

### Layer 3 — root-commit OID

If `cwd` is in a git repo and the repo is **not shallow**, compute
`sha256[:12]` of the lex-smallest output of
`git rev-list --max-parents=0 HEAD`.

```python
if top is not None and not _is_shallow_repository(cwd):
    oids = _root_commit_oids(cwd)
    if oids:
        return _sha256_12(min(oids))
elif top is not None:
    # Shallow repo: layer 3 is unsafe. Warn and fall through.
    sys.stderr.write(
        "aelf: shallow clone detected; identity falls back to cwd-hash. "
        "Run `git fetch --unshallow` for full-clone identity, or set "
        "$AELFRICE_PROJECT_ID / commit .aelfrice/project-id.\n"
    )
```

The shallow-detect check is mandatory. `git rev-list --max-parents=0
HEAD` returns the shallow-tip OID in shallow repos (because
`.git/shallow` marks parent links as severed), which is a different
commit OID than the original root. Without the check, a CI pipeline
that shallow-clones would silently fork the developer's DB.

The lex-smallest pick across all returned root OIDs handles repos
with multiple roots (subtree merges, `--allow-unrelated-histories`,
`git replace --graft`, orphan branches). The pick is deterministic
across all derivative states (clone, worktree, mv) of one repo.

### Layer 4 — cwd-path hash

For non-git directories, shallow repos, and any case where layers
1-3 are unavailable, return `sha256[:12]` of `cwd.resolve()`.

```python
return _sha256_12(str(cwd.resolve()))
```

This is intentionally per-cwd. Two non-git directories (or two
shallow CI checkouts at different paths) get different ids.

## Input validation

Layer 1 and layer 2 read user-controlled values. Both must validate
before use:

```python
import re
_VALID_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{8,128}$")

def _validate_layer_input(value: str) -> str | None:
    if not value:
        return None
    stripped = value.strip()
    if not stripped:                     # whitespace-only treated as absent
        return None
    if not _VALID_ID_RE.fullmatch(stripped):
        sys.stderr.write(
            f"aelf: invalid project-id `{stripped[:20]}...`; falling through.\n"
        )
        return None
    return stripped
```

Plus defense-in-depth at path construction time:

```python
constructed = (aelfrice_home / "projects" / id_value / "memory.db").resolve()
projects_root = (aelfrice_home / "projects").resolve()
if not constructed.is_relative_to(projects_root):
    raise ValueError(
        f"aelf: refusing to open DB outside projects root: {constructed}"
    )
```

The regex rejects path-traversal characters (`/`, `..`), whitespace,
newlines, BOM, and other non-printable input that survived the
ENV/file read but would corrupt or escape path construction. It
allows the 12-hex resolver output (which is 12 chars long, in
`[0-9a-f]`) and standard UUID forms (36 chars, `[0-9a-fA-F-]`).

## DB location

`~/.aelfrice/projects/<id>/memory.db`, where `~` is `Path.home()` or
the user's `XDG_DATA_HOME`-equivalent (TBD per CONFIG.md).

The DB no longer lives under `<git-common-dir>/aelfrice/`. The change
puts the DB out of `.git/` so that `git gc` / `git prune` cannot
trip over it, and so that bare repos (no working tree) can still
host an aelfrice DB.

## Migration from the legacy location

```python
def migrate(cwd, aelfrice_home, *, interrupt_after_copy=False):
    common = git_common_dir(cwd)
    legacy = common / "aelfrice" / "memory.db"
    new_id = resolve_id(cwd)
    new_path = aelfrice_home / "projects" / new_id / "memory.db"
    sentinel = common / "aelfrice" / ".migrated-to"

    if sentinel.is_file():
        return "noop_already_migrated"
    if not legacy.is_file():
        return "noop_no_legacy"
    if new_path.exists():
        if sha256_file(new_path) == sha256_file(legacy):
            sentinel.write_text(new_id)
            return "recovered"
        return "conflict"

    new_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=new_path.parent, prefix=".tmp-migrate-", suffix=".db",
    )
    try:
        with os.fdopen(fd, "wb") as out, legacy.open("rb") as inp:
            shutil.copyfileobj(inp, out, length=65536)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, str(new_path))
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise

    if interrupt_after_copy:
        return "migrated"
    sentinel.write_text(new_id)
    return "migrated"
```

State transitions:

- `noop_already_migrated` — sentinel present; legacy known-handled.
- `noop_no_legacy` — nothing to migrate.
- `migrated` — fresh copy + sentinel write.
- `recovered` — destination already populated by a prior interrupted
  attempt; sha matches legacy; write sentinel and return.
- `conflict` — destination exists with different bytes; refuse to
  overwrite. Operator must decide.

The `tempfile.mkstemp` + `os.fsync` + `os.replace` mechanism gives
two POSIX guarantees:

1. The rename is atomic on the same filesystem; either `memory.db`
   has the old contents or the new tempfile's contents, never partial
   bytes.
2. `os.fsync` before the rename forces the kernel to commit the
   tempfile's bytes to disk; a power loss after rename retains the
   new contents.

An interrupt between rename and sentinel write leaves the dest with
correct bytes and the legacy untouched. The recovery path detects
the sha-match on retry and stamps the sentinel without re-copying.

## Concurrency

End-state under concurrent `migrate()` calls is **guaranteed
consistent**: the legacy DB is byte-preserved, the dest DB has its
bytes, the sentinel is written.

Status codes returned by individual threads are **not mutually
exclusive**. Three threads racing past the precondition check before
any completes the rename will each return `"migrated"`. The on-disk
state is correct (POSIX rename atomic; identical bytes from the same
legacy source; identical sentinel content); only the return-code
accounting is loose.

Callers that need exactly-once accounting wrap `migrate()` in a file
lock:

```python
import fcntl
lock_path = common / "aelfrice" / ".lock"
lock_path.parent.mkdir(parents=True, exist_ok=True)
with open(lock_path, "w") as lock:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    result = migrate(cwd, aelfrice_home)
```

The default `migrate()` omits the lock for portability and minimal
dependency surface. Data is always safe; only metrics need this.

## Drift hazards

By design, layer-3 identity is content-derived from the set of root
commits reachable from HEAD. Operations that change this set change
identity. The complete enumerated list of confirmed triggers:

| operation | identity effect |
| --- | --- |
| `git rebase --root` | rewrites root → new sha → new id |
| `git replace --graft` | adds parent to root → root no longer a root → new id |
| `git commit --amend` on a 1-commit (root) repo | new root sha → new id |
| `git merge --allow-unrelated-histories` | adds new root; flips id if added root sorts lex-smaller |
| `git switch --orphan` + commit | new branch with new root; HEAD on orphan reaches only orphan root → flips id on `git checkout orphan` |
| `git merge` of an orphan branch | merged HEAD reaches both roots; lex-smaller wins; id may equal orphan_pre, main_pre, or neither |
| `git filter-branch --subdirectory-filter` | rewrites every commit; new root |
| `git clone --depth=N` | shallow tip is the visible "root"; flips id (caught by layer-3.5 shallow-detect) |

**Universal mitigation: write a tracked `.aelfrice/project-id` file**.
Layer 2 short-circuits before layer 3, so the file pins identity
through any commit-graph mutation. This is the recommended workflow
for users with `gh-pages` orphan branches, subtree-merge consumers,
and anyone running history rewrites.

For users who don't anticipate drift but get hit by it, the migration
tool detects identity flips:

- On startup, compute new-id for current `cwd`.
- If `~/.aelfrice/projects/<new-id>/memory.db` is missing, scan
  `~/.aelfrice/projects/` for any directory whose record matches the
  current project (via `project.txt` or equivalent).
- If found, surface to the user: *"Project identity may have
  changed. Previous DB at <old>/. Move? Skip?"*

## Memo (optional, scoped)

Per-resolve cost is ~110 ms p50, dominated by `git rev-parse`
subprocess invocation. Cost is **flat** with respect to commit count
(measured at 50, 200, 1000, 3000 synthetic commits: max delta 8 ms).

For long-running processes (the MCP server, daemons), a process-local
memo is worthwhile:

```python
class IdentityCache:
    def __init__(self):
        self._cache = {}

    def resolve(self, cwd):
        key = str(cwd.resolve())            # M2 — cwd-keyed
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        rid = resolve_id(cwd)
        self._cache[key] = rid
        return rid
```

Hit cost: ~38 microseconds (4 orders of magnitude faster than cold).

For one-shot CLI invocations and Claude Code hooks, the memo offers
nothing — each invocation is a fresh process with an empty cache. Do
not bother.

**Do not key on `git_common_dir`**. Looking up that key requires
another `git rev-parse` subprocess (~50 ms), which defeats the memo.
The cwd-keyed variant trades worktree-cache-sharing for sub-millisecond
hits; in any single-process workload, this is the right tradeoff.

A persistent sidecar cache at `~/.aelfrice/projects/<id>/.identity-
cache.json` is **deferred**. The ~110 ms cold cost fits the hook
latency budget (500 ms p95) with 4× headroom. The sidecar's
invalidation complexity (root commit changes → identity changes →
cache stale) is not warranted by current measurements.

## Identity collision

The 48-bit slug (`sha256[:12]`) is fit for per-user scale. Birthday-
paradox math:

- N=10k: E[collision pairs] ≈ 10⁸ / 5.6 × 10¹⁴ = 0
- N=1M: E[collision pairs] ≈ 1.8
- N=16.77M (= √(2⁴⁸)): E[collision pairs] ≈ 0.5; P(any collision) ≈ 50%
- N=100k (plausible per-user upper bound): E[collision pairs] ≈ 2 × 10⁻⁵

Empirical verification at all three N matched the math (Poisson
distribution on observed counts). Local-corpus scan at N=38 had zero
collisions at any prefix length ≥ 6 hex chars.

The 48-bit width is the smallest that gives sub-microsecond
collision risk at user scale while keeping ids short enough to be
ergonomic in directory names.

## Performance

| scale | cold p50 | cold p95 | warm (M2) p50 |
| --- | --- | --- | --- |
| 50 commits | 108 ms | 134 ms | 0.038 ms |
| 200 commits | 107 ms | 125 ms | 0.038 ms |
| 1000 commits | 111 ms | 185 ms | 0.040 ms |
| 3000 commits | 116 ms | 135 ms | 0.040 ms |

Cost is essentially flat across scale. Subprocess overhead dominates
(~95 ms minimum for any `git` invocation regardless of work). The
marginal walk cost is ~3 ms per 1000 commits.

Hook latency budget: 500 ms p95 (per the search-tool hook header).
Cold p95 at 3000 commits is 135 ms — comfortable. Even at projected
50k commits, p50 extrapolates to ~115 ms.

## Removals

Three resolvers are removed in the same release:

- `cli.db_path()` — replaced by `resolve_id()` + the new path
  construction.
- `project_warm._project_id()` — replaced by the same.
  `_git_resolve()` is retained (still useful for the worktree-root
  computation), but the project-id derivation routes through the
  new chain.
- `~/.claude/hooks/aelfrice-db-path.py:resolve_db_path` — replaced by
  a thin shell wrapper that calls `aelf db-path` (or equivalent CLI
  subcommand).

The `aelf` binary gains a `db-path [<cwd>]` subcommand that exposes
the resolver to shell scripts (the hooks). Output is the absolute
path to the DB the resolver chose.

## Out of scope (future work)

- **Sidecar cache** for cross-process amortization (deferred).
- **Multi-platform NFC/NFD invariance** verification (Linux + macOS
  matrix). Layer 4 cwd-hash sees whatever bytes the OS returns from
  `Path.resolve()`; modern macOS APFS preserves NFC, but legacy HFS+
  forces NFD. A user moving a project across these filesystems may
  see layer-4 ids differ.
- **Cross-user submodule embed**. A downstream project that embeds
  aelfrice itself (or any shared library) as a submodule has cwd-
  inside-the-submodule resolve to the embedded library's root-commit
  OID, which is a globally-shared identity. Per-machine private DBs
  preserve user data isolation, but the convention is fragile if
  ever extended to shared filesystems.
- **Shallow-clone alternative handling**. Currently shallow falls
  through to layer 4 with a warning. An alternative is to walk
  `.git/shallow` and re-derive root from the original parents; not
  implemented because the layer-2 opt-out is simpler and covers the
  CI workflow case adequately.

## Implementation notes

- Subprocess timeout of 5 seconds on every `git` invocation. Empty
  stdout (e.g. unborn HEAD) is a fall-through condition, not an
  error.
- All `git` invocations should pass `--path-format=absolute` where
  applicable (already used by the existing `_git_common_dir`).
- The resolver should be a pure function: `(cwd, env) → id`. State
  belongs in the optional memo, not in module globals.
