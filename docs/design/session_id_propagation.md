# session_id propagation contract

Implements [#192](https://github.com/robotrocketscience/aelfrice/issues/192)
(phantom-prereqs T3). Every ingest entry point that produces a row
in `beliefs` or `ingest_log` is responsible for stamping a
`session_id` so downstream consumers can distinguish *N sessions
asserted X once each* from *one session asserted X N times*.

## What `session_id` means

A `session_id` is a string that identifies the conversational or
operational context that produced an ingest event. Different
surfaces produce structurally different session strings; downstream
code should treat the value as opaque.

| surface | session_id source | shape |
|---|---|---|
| `ingest_turn()` (library) | explicit kwarg → `$AELF_SESSION_ID` → NULL+warn | caller-defined |
| `aelf-transcript-logger` JSONL → replay | `sessionId` field on the JSONL turn | caller-defined |
| `hook_commit_ingest` | `sha256(branch + commit_hash)[:16]` | 16-hex |
| `scanner.scan_repo` (filesystem onboard) | `sha256(scan:<root>:<ts>)[:16]` | 16-hex |
| `classification.accept_classifications` (onboard accept) | active `OnboardSession.id` | UUID-ish |
| `aelf lock` CLI | `--id` arg → `$AELF_SESSION_ID` → NULL+warn | caller-defined |
| MCP `aelf_lock` tool | MCP `session_id` arg → `$AELF_SESSION_ID` → NULL+warn | caller-defined |

A NULL `session_id` means "no session attribution available";
session-coherent retrieval skips NULL rows so legacy data and
unattributed writes don't cause false-positive cross-session
co-occurrence signals.

## Inference contract (#192 R0 ratification, 2026-05-03)

Three surfaces share a single inference contract via
`aelfrice.session_resolution.resolve_session_id(explicit, surface_name=...)`:

1. `ingest.ingest_turn` (Q1.a)
2. `cli._cmd_lock` — `aelf lock` (Q3.a)
3. `mcp_server.tool_lock` — MCP `aelf_lock` (Q4.a)

Resolution order:

1. **Explicit value** — if the caller passes `session_id`, that
   wins unconditionally.
2. **Environment** — read `$AELF_SESSION_ID`. The variable is
   intended to be set by a hook or harness wrapping a long-running
   ingest call site.
3. **NULL + warn** — if neither is set, the call proceeds with
   `session_id=NULL` and emits a one-shot stderr warn keyed on the
   `surface_name`. Subsequent calls from the same surface in the
   same process are silent (one warn per surface per process).

## Surfaces with structural session sources

These do **not** call the helper because they have a deterministic
session source already:

* **`hook_commit_ingest`** — the commit hash + branch is the session.
* **`scanner.scan_repo`** — `_derive_scan_session_id(root, ts)`
  produces a synthetic per-scan id; reproducible from the same root
  and timestamp.
* **`classification.accept_classifications`** — `OnboardSession.id`
  IS the session; no inference needed.
* **`transcript_logger`** — writes the upstream JSONL with the
  `sessionId` field intact; replay through `ingest_turn()` is what
  propagates it onto beliefs.

## Operator notes

* To set the env var for a long ingest run:
  `export AELF_SESSION_ID="$(uuidgen)"`. The value is opaque; any
  non-empty string is accepted.
* Warns go to stderr only. Capture via `2>session-warns.log` if
  needed. Once-per-surface dedup means a long batch produces at
  most a few lines, not thousands.
* `aelf lock` accepts `--id <session-id>` as an explicit override;
  see `aelf lock --help`.
* The MCP `aelf_lock` tool accepts a `session_id` field in its
  request envelope; the MCP layer fills it from the active session
  context.

## What is NOT in scope for #192

* `aelf_correct` MCP tool (Q4 ratified out-of-scope; future track).
* Synthetic per-process UUIDs as a fallback (Q1.c rejected — env
  var is the single inference surface).
* Backfilling NULL `session_id` on legacy beliefs.
