# LLM-Haiku onboard classifier (opt-in)

Design memo for the v1.3.0 milestone. Tracking issue: [#145](https://github.com/robotrocketscience/aelfrice/issues/145).

Cross-references: [ROADMAP.md § v1.3.0](ROADMAP.md#v130--retrieval-wave),
[PRIVACY.md § Optional outbound calls](PRIVACY.md#optional-outbound-calls),
[`src/aelfrice/classification.py`](../src/aelfrice/classification.py),
[`src/aelfrice/scanner.py`](../src/aelfrice/scanner.py),
[CONFIG.md](CONFIG.md), [COMMANDS.md](COMMANDS.md#memory-operations).

Status: shipped opt-in at v1.3.0; default-on at v1.5.0 via host-driven
classification through the `/aelf:onboard` slash command (no API key
required). The original v1.3.0 design below is unchanged — the
direct-API path (`aelf onboard --llm-classify`) remains the
API-key-user fallback. The v1.5.0 default-on flow is layered on top
via two additive CLI flags (`--emit-candidates` /
`--accept-classifications`) that wrap the existing polymorphic
handshake in [`classification.py`](../src/aelfrice/classification.py)
without changing it. Implementation tracking issue:
[#238](https://github.com/robotrocketscience/aelfrice/issues/238).

### v1.5.0 host-driven classification (slash-command flow)

The typical aelfrice user already runs an LLM host. Demanding a
separate `ANTHROPIC_API_KEY` to opt into LLM classification kept the
higher-quality path out of reach for that population. v1.5.0 makes
`/aelf:onboard` orchestrate the four-class classifier through the
host's own Task dispatch against the cheapest model in its stack:

1. `uv run aelf onboard <path> --emit-candidates` — persists a
   PENDING `onboard_sessions` row, prints
   `{session_id, n_already_present, sentences[]}` as JSON. No
   network.
2. The host batches the `sentences[]` (≤ 50 per batch) and dispatches
   one Task per batch with the four-class classification template.
   Each batch returns
   `[{index, belief_type, persist}, ...]`.
3. `uv run aelf onboard --accept-classifications --session-id <id>
   --classifications-file -` — reads the aggregated classifications
   from stdin, applies them via `accept_classifications`, prints an
   `AcceptOnboardResult` JSON summary. No network.

`--no-subagents` (or absence of the host's Task tool) falls through
to the deterministic regex classifier. The four-gate boundary policy
in § 4 below applies only to the direct-API path; the host-driven
path makes zero direct calls to `https://api.anthropic.com/` from
the aelfrice CLI itself.

### v1.3.0 design memo (unchanged)


## 1. Motivation

The v1.0/v1.2 onboard classifier in
[`classification.py`](../src/aelfrice/classification.py) is a regex
keyword matcher over four belief types (`factual`, `correction`,
`preference`, `requirement`). It is fast, deterministic, offline, and
has known recall ceilings on prose-heavy content:

- A correction phrased without the keyword set (`detect_correction`
  patterns) is silently classified `factual` and inherits the wrong
  Beta prior.
- A requirement phrased descriptively ("the publish workflow blocks on
  green CI") rather than imperatively ("CI must be green before
  publish") will not trip `_REQUIREMENT_KEYWORDS` and lands as
  `factual`.
- Provenance origin (`document_recent` / `agent_inferred` /
  `user_corrected` / `user_stated`, see
  [`models.py:55-75`](../src/aelfrice/models.py)) is set unconditionally
  to `agent_inferred` by `scan_repo`, with no signal from the source
  text. The LLM-classify path is the first place we get to differentiate
  origin from text content (e.g., a CHANGELOG entry → `document_recent`;
  a `## Decisions` block in a planning doc → `agent_inferred`).

Haiku is a cheap, fast classifier with substantially higher recall on
the four-type taxonomy. Trading cost (and one outbound call at onboard
time) for higher recall is the v1.3.0 deliverable.

The polymorphic onboard handshake
([`classification.py:start_onboard_session` / `accept_classifications`](../src/aelfrice/classification.py))
already exists for the case where a host LLM (Claude Code, an MCP
client) is available and willing to classify in its own context. The
LLM-classify path is the **non-host case**: `aelf onboard` invoked
from a plain shell, no host context, classification done by aelfrice
calling Haiku directly.

## 2. Scope

### In scope

- New optional dep group `[onboard-llm]` adding the `anthropic` SDK.
- New CLI flag `aelf onboard --llm-classify` for one-off opt-in.
- New `[onboard.llm]` block in `.aelfrice.toml` for sticky preference.
- One-time first-use confirmation prompt before any outbound traffic.
- A Haiku-backed classifier function with the same return shape as
  `classify_sentence` (`belief_type`, `alpha`, `beta`, `persist`,
  `pending_classification`).
- Origin assignment from the LLM (`document_recent` /
  `agent_inferred` — `user_*` origins are never assigned by the
  classifier; those come from explicit user actions).
- Per-onboard-run cost telemetry surfaced on stdout.
- Per-run abort threshold (token cap) with a clear message.
- Regex-fallback path on Haiku unreachable / timeout / rate-limit /
  malformed response, with an audit-row note.
- Activation of `benchmarks/mab_llm_entity_adapter.py` (currently
  not present, per [benchmarks/README.md:41-44](../benchmarks/README.md)).

### Out of scope

- Use of Haiku or any other LLM in the **retrieval** path. Retrieval
  stays SQLite + FTS5 + BM25 at every milestone (see
  [ROADMAP.md § Non-goals](ROADMAP.md#non-goals)).
- Streaming or partial classification — onboard is a batch operation.
- Caching Haiku responses across `aelf onboard` runs. The deterministic
  belief id (`sha256(source\x00text)[:16]`) already deduplicates against
  the local store, so re-onboard does not re-call Haiku for any
  candidate already present. No cross-machine cache.
- Replacing the regex classifier. `--llm-classify` is opt-in. Default
  remains regex.

## 3. Opt-in surface

**Decision: both flag and config block. Both default OFF.**

Rationale: the flag covers one-off "I want LLM classification on this
one project right now" without editing the project config. The config
block covers "I always want LLM classification on this project, every
time I or CI re-onboard." Either path requires a deliberate user
action. The flag wins when both are present (least-surprise: the flag
on the command line is more recent intent than a stale config).

### CLI flag

```bash
aelf onboard <path> --llm-classify
```

- Default OFF.
- Implies `[onboard-llm]` extra is installed; otherwise `aelf onboard`
  exits 1 with a one-line install hint and does not contact the
  network.
- `aelf onboard <path> --llm-classify=false` forces regex even if the
  config block has it on.

### Config block

```toml
# .aelfrice.toml
[onboard.llm]
# Opt in to LLM-Haiku classification at onboard time.
# Default: false. When true, aelf onboard contacts the
# Anthropic API. Requires the [onboard-llm] extra and
# the ANTHROPIC_API_KEY env var.
enabled = false

# Hard cap on total input+output tokens per onboard run.
# Default: 200_000. Run aborts mid-stream if exceeded;
# already-classified candidates are kept (idempotent re-run
# resumes where the cap hit). 0 disables the cap.
max_tokens = 200_000

# Model id. Pinned by default to keep classification stable
# across releases. Override only if you have a reason.
model = "claude-haiku-4-5-20251001"
```

Forward compatibility: the existing `[noise]` block parser ignores
unknown tables (CONFIG.md § Schema). Adding `[onboard.llm]` is
additive; old configs without it default OFF.

### Resolution order

1. `--llm-classify=false` on the command line → off.
2. `--llm-classify` (or `--llm-classify=true`) on the command line → on.
3. `[onboard.llm].enabled = true` in `.aelfrice.toml` → on.
4. Default → off.

## 4. Network boundary policy

**This is the first outbound network call in the default install path
that carries user content.** The update notifier in `lifecycle.py`
makes a TTL-gated GET to PyPI but never transmits anything (PRIVACY.md
§ Your data never leaves your machine). The LLM-classify path does
transmit content from the user's repo to Anthropic.

The boundary policy is therefore non-negotiable.

### Provider

- **Anthropic Haiku.** Model id `claude-haiku-4-5-20251001` (current
  Haiku at this codebase's date, 2026-04-27). Pinned in
  `[onboard.llm].model`. Override allowed but discouraged.
- **No other provider.** No abstraction layer. No "bring your own
  endpoint." A second provider is a separate spec.

### Auth

- **Single env var: `ANTHROPIC_API_KEY`.** Read at call time. Never
  read from `.aelfrice.toml`, never from a config file, never from a
  keyring, never prompted interactively. If the env var is unset and
  `--llm-classify` is on, `aelf onboard` exits 1 with the message
  "ANTHROPIC_API_KEY not set; --llm-classify requires it. Either
  unset --llm-classify or export the key."

### Dependency

- **New optional extra `[onboard-llm]`** in `pyproject.toml`:
  ```toml
  onboard-llm = ["anthropic>=0.40"]
  ```
- The default `pip install aelfrice` install does **not** pull
  `anthropic`. Only `pip install aelfrice[onboard-llm]` does.
- The `aelfrice` package never imports `anthropic` at top level. The
  import is local to the LLM-classify call site so a default install
  has no `anthropic` symbol resolvable. Mirrors the `[mcp]` extra
  pattern (`fastmcp` is similarly local-imported).

### One-time confirmation prompt

The first time `--llm-classify` is invoked on a given machine, before
any outbound traffic, `aelf onboard` prints a confirmation prompt to
stderr and reads y/N from stdin:

```
aelf onboard --llm-classify

This will send sentences extracted from the files under
<resolved-path> (paragraphs from .md/.rst/.txt/.adoc files,
git commit subjects, Python docstrings) to Anthropic's
API for classification. The content of those files will
leave your machine.

aelfrice will not send file paths, env vars, secrets it
finds, or anything outside the extracted candidate text.
You can audit what would be sent with:

  aelf onboard <path> --dry-run

Continue? [y/N]:
```

- A `y` writes a sentinel file at `~/.aelfrice/llm-classify-consented`
  recording the timestamp, model id, and aelfrice version. Subsequent
  runs see the sentinel and skip the prompt.
- `N`, EOF, any other input, or stdin not a TTY → exit 1, no network
  call. (Non-interactive environments: CI, scripts, hooks. The user
  must opt in interactively once on a workstation; CI runs that need
  it must explicitly create the sentinel ahead of time.)
- A new model id, or a new aelfrice **major** version, invalidates the
  sentinel and re-prompts. Patch and minor bumps do not.
- `aelf onboard --llm-classify --revoke-consent` clears the sentinel.
- The sentinel lives in `~/.aelfrice/`, not the per-project DB
  directory: consent is per-user-per-machine, not per-project.

The sentinel is a UX optimization, not a security boundary. The real
boundary is the explicit `--llm-classify` flag (or the explicit
`[onboard.llm].enabled = true` in a checked-in config). Removing the
sentinel is fine; the user just sees the prompt again.

### Dry-run

`aelf onboard <path> --llm-classify --dry-run` runs the full extractor
+ noise-filter + dedup pipeline and prints the candidates that would
be sent (one per line, prefixed with the source string), token-counted,
**without** contacting the network. Lets the user audit the wire before
consenting. No sentinel side-effect.

### What is sent, what is not

**Sent:**
- The candidate sentence/paragraph text exactly as extracted by
  `scanner.extract_filesystem` / `extract_git_log` / `extract_ast`.
- The candidate's `source` string (e.g., `doc:README.md:p3`,
  `git:commit:abcdef0`, `ast:src/foo.py:func:bar`).
- A small system prompt (templated, no user data).

**Not sent:**
- File contents beyond the extracted candidate. The classifier sees a
  paragraph, not the whole document.
- Env vars, including `ANTHROPIC_API_KEY` (used only as bearer token).
- Working directory path, hostname, username, machine id.
- Git remotes, git config, git author email.
- Files matching `INEDIBLE` (already excluded upstream by
  `scanner._iter_doc_files` / `_iter_py_files`).
- Files in `_SKIP_DIRS` (already excluded upstream).
- Files filtered by the noise filter (already excluded upstream).

The opt-out boundary is the same one that already exists for local
ingest: anything `is_inedible(path)` rejects is invisible to the LLM
path too. The check happens before classification, as documented in
PRIVACY.md.

## 5. Prompt template

The system prompt locks the classifier to the four-type taxonomy + the
two non-user origin tiers (`document_recent`, `agent_inferred`).
`user_*` origin tiers are reserved for explicit user actions (lock,
correction-detector hit on a user-stated sentence) and the LLM is not
permitted to assign them.

### System prompt

```
You are classifying short text candidates extracted from a software
project's documentation, git history, and Python docstrings. Each
candidate becomes a unit of memory in a Bayesian belief store.

For each candidate, return a JSON object with three fields:
  belief_type: one of "factual", "correction", "preference",
               "requirement"
  origin:      one of "document_recent", "agent_inferred"
  persist:     true if the candidate should become a stored belief,
               false if it should be dropped (questions, headings,
               table-of-contents lines, navigational text,
               meta-commentary, anything ephemeral).

Definitions:
  factual      A statement of fact, decision, or analysis. Default.
  correction   A statement that overrides or corrects a previous
               claim ("not X but Y", "actually Z", "the earlier
               version was wrong because ...").
  preference   A stated preference, taste, or convention ("we prefer
               composition", "always use uv", "I want explicit
               types").
  requirement  A hard rule, constraint, must-do, or invariant ("CI
               must be green", "no global state", "Python 3.12+").

  document_recent   The candidate reads as committed prose from the
                    project's own documentation or commit history —
                    something a human wrote down deliberately.
                    Default for paragraphs from .md/.rst files and
                    for git commit subjects.
  agent_inferred    The candidate reads as machine-extracted or
                    incidental — a docstring fragment, a templated
                    line, anything where the underlying assertion
                    was not necessarily reviewed by a human.

Return one JSON object per candidate, in input order, as a JSON
array. No prose before or after the array. No markdown fences.
```

### Few-shot examples (one block, in-prompt)

```
Input:
  [{"index": 0, "source": "doc:README.md:p4",
    "text": "aelfrice ships with no telemetry. The capability does not exist in the package."},
   {"index": 1, "source": "git:commit:abc1234",
    "text": "fix: handle empty store in retrieval BM25 path"},
   {"index": 2, "source": "ast:src/foo.py:func:bar",
    "text": "Returns the user's home directory."},
   {"index": 3, "source": "doc:CONTRIBUTING.md:p2",
    "text": "What is the expected response time on PRs?"},
   {"index": 4, "source": "doc:CHANGELOG.md:p7",
    "text": "Earlier versions documented bullets-per-line; that was wrong, the actual unit is bullets-per-paragraph."}]

Output:
  [{"belief_type": "requirement", "origin": "document_recent", "persist": true},
   {"belief_type": "factual",     "origin": "document_recent", "persist": true},
   {"belief_type": "factual",     "origin": "agent_inferred",  "persist": true},
   {"belief_type": "factual",     "origin": "agent_inferred",  "persist": false},
   {"belief_type": "correction",  "origin": "document_recent", "persist": true}]
```

The few-shot covers each of the four `belief_type`s plus a
`persist=false` example (the question), plus both origins. The
implementation PR may add one more block if the test corpus shows
recall gaps; the load-bearing decision is the single-array,
index-aligned, no-prose response format, since that is what the
parser depends on.

### Request shape

- One Haiku request per onboard run, batching all candidates into a
  single user message. (Anthropic supports ~200k context tokens at
  Haiku 4.5; the entire candidate set for a typical project fits.)
- For very large projects the implementation may shard into N requests
  of ≤K candidates each. Sharding boundary is implementation choice;
  acceptance only requires that total tokens-consumed telemetry is
  exact.
- Temperature 0. Stop sequences default. `max_tokens` on the response
  side capped at the input candidate count × ~30 tokens (every output
  object is small).

### Parsing

- Strict JSON parse on the response.
- Per-candidate validation: `belief_type ∈ BELIEF_TYPES`,
  `origin ∈ {document_recent, agent_inferred}`, `persist ∈ {true,
  false}`. Invalid → drop that candidate, count it under
  `skipped_invalid_classification`, do not fall back to regex on a
  per-candidate basis (the response was structured but the model
  hallucinated a field — the right move is to keep the rest of the
  batch).
- Mismatched array length → fall back to regex for the whole batch
  (the response is fundamentally broken; treat as a Haiku failure,
  see § 6).

## 6. Cost model

### Token budget per belief

Estimated from the prompt template + few-shot block: ~800 input tokens
amortized across the system prompt and few-shot, plus ~30 input tokens
per candidate (source + text), plus ~30 output tokens per candidate.

For a project with N candidates: input ≈ 800 + 30·N, output ≈ 30·N.
Total ≈ 800 + 60·N.

A typical aelfrice-shaped project (hundreds of candidates after the
noise filter) lands around 30k–60k tokens per onboard run. At Haiku
4.5 pricing this is sub-cent. The implementation PR will add a
calibration test that asserts on a fixture corpus and tracks drift.

### Per-run abort threshold

`[onboard.llm].max_tokens` (default 200_000) is the hard cap. The
classifier tracks running tokens consumed across shards and aborts
mid-stream when the cap is exceeded.

- Already-classified candidates are kept (idempotent: their beliefs
  land in the store). The deterministic belief id ensures a re-run
  resumes from where the cap hit.
- Aborted runs exit 1 with the message "onboard aborted: token cap
  reached at <consumed>/<cap>. Re-run to resume, or raise
  `[onboard.llm].max_tokens` in `.aelfrice.toml`."
- `max_tokens = 0` disables the cap. Documented as power-user.

### Telemetry surface

`aelf onboard --llm-classify` prints, on completion, a one-line
summary appended to the existing `ScanResult` summary:

```
onboard: 247 inserted, 12 skipped (existing), 38 skipped (noise),
  4 skipped (non-persisting), 312 candidates total
onboard.llm: model=claude-haiku-4-5-20251001 input_tokens=8124
  output_tokens=7410 total_tokens=15534 requests=1 fallbacks=0
```

Fields:

- `model` — the model id actually called (so override visible).
- `input_tokens`, `output_tokens`, `total_tokens` — Anthropic's billed
  counts, summed across shards.
- `requests` — number of HTTP requests the implementation made.
- `fallbacks` — number of candidates that fell back to regex (see
  § 7).

Telemetry is stdout-only, never network-emitted, never written
anywhere outside the user's terminal. (PRIVACY.md § No telemetry
remains true: aelfrice still does not phone-home about anything,
including its own LLM usage.)

The implementation may also write a per-run JSON line to
`<git-common-dir>/aelfrice/onboard-llm.log` as audit material; that
file lives under `.git/` (already not git-tracked, per PRIVACY.md).

## 7. Failure modes

**Recommendation: fall back to regex with an audit-row note.**

Rationale: onboarding is best-effort. A partial belief store from a
regex-fallback is more useful than a failed run that leaves the user
with nothing. Onboard is the bootstrapping step; if Haiku is
unreachable, the user should still get *some* memory populated, with a
clear marker that LLM classification didn't run.

### Trigger conditions

- Connection refused / DNS failure / TLS error.
- 5xx from Anthropic.
- 429 (rate limit) — no retry inside the `aelf onboard` run; fall back
  immediately. Retry-after handling is implementation choice but must
  not exceed 30s of total wait.
- Request timeout (default 30s per request).
- 401/403 (auth) — **do not** fall back. The user explicitly opted in
  with a key that is broken or revoked; surfacing that is more
  important than papering over it. Exit 1 with the auth error from
  Anthropic, no candidates classified, no beliefs inserted.
- Malformed response that fails the strict JSON parse or fails
  array-length validation.

### Fallback behaviour

When the trigger fires:

1. Every remaining (and the failed) candidate is reclassified
   synchronously by the existing regex `classify_sentence`.
2. The `scan_repo` insert path proceeds normally.
3. Each fallback insertion writes a row to `feedback_history` with
   `source = "onboard.llm.fallback:<reason>"` and
   `feedback_type = "audit"` (no posterior change). This makes the
   fallback observable via `aelf stats` and the audit log.
4. The `fallbacks` field in the telemetry line reflects the count.
5. Exit code 0 (the run succeeded; LLM was unavailable, regex covered).

### Auth-failure behaviour

`401`/`403` is a deliberate exception to the fallback policy. The
user took the explicit step of setting `ANTHROPIC_API_KEY` and turning
on `--llm-classify`; if the key is bad we want them to see that, not
silently degrade to a regex run that they didn't ask for.

### Token-cap behaviour

Distinct from the above: the per-run `max_tokens` cap aborts the run
(exit 1, partial insertion, idempotent resume) rather than falling
back to regex. The user explicitly set the cap; honouring it precisely
is the right call.

## 8. Activation in `benchmarks/`

Per the issue: `mab_llm_entity_adapter`. This adapter is currently
**not present** in the public repo
([benchmarks/README.md:41-44](../benchmarks/README.md)) — it is
listed alongside `mab_triple_adapter.py` and
`mab_entity_index_adapter.py` as deferred until the upstream features
land.

The v1.3.0 implementation PR ports `mab_llm_entity_adapter.py` from
the lab. Activation means:

1. The adapter file lives under `benchmarks/` and runs against the
   public retrieval surface (`store.retrieve(...)`).
2. The adapter is invoked via `[benchmarks]` extras (mirrors
   `mab_adapter.py` pattern).
3. A retrieve-only baseline is captured in `benchmarks/results/` at
   v1.3.0 tag-time (`benchmarks/results/v1.3.0.json` or named
   adjacent to the existing `v1.2.0-pre.json` artifact).
4. The adapter does not itself call Haiku; it operates on a store
   that was populated by `aelf onboard --llm-classify`. The
   benchmark measures retrieval quality on an LLM-classified corpus,
   not the classifier itself. (Classifier quality is a separate
   fixture-based test in `tests/`.)

## 9. Acceptance criteria for the implementation PR

The implementation PR may not merge unless:

1. **Default-off verified.** A test asserts that `aelf onboard <path>`
   with no flag and no config block makes zero outbound HTTP calls.
   Mocked `httpx`/`anthropic` client + `assert not called`.
2. **Opt-in path tested.** A test (with mocked Anthropic client)
   asserts `aelf onboard --llm-classify` calls Haiku once, parses the
   response, and inserts beliefs with the LLM-assigned types and
   origins (not `agent_inferred` for everything).
3. **Config-block path tested.** Equivalent test with
   `[onboard.llm].enabled = true` and no flag.
4. **Confirmation-prompt tested.** A test asserts the first run
   without the sentinel emits the prompt to stderr and exits 1 on
   stdin = "n" before any outbound call. A second test asserts the
   prompt is suppressed when the sentinel exists.
5. **Network-call test for `--dry-run`.** Mocked client + assert
   not called when `--llm-classify --dry-run` is set.
6. **Fallback path tested.** Mocked client raises a connection error
   on N>0 candidates; assert regex `classify_sentence` is invoked for
   the affected candidates, beliefs are inserted, telemetry shows
   `fallbacks=N`, exit code 0.
7. **Auth-failure path tested.** Mocked client raises 401; assert no
   beliefs inserted, exit 1, error message mentions Anthropic auth.
8. **Token-cap tested.** Mocked client returns usage that crosses
   `[onboard.llm].max_tokens` mid-shard; assert run exits 1, beliefs
   inserted up to the cap remain, re-run resumes correctly.
9. **Cost telemetry tested.** End-to-end test asserts the
   `onboard.llm:` summary line is emitted and contains `model`,
   `input_tokens`, `output_tokens`, `total_tokens`, `requests`,
   `fallbacks`.
10. **Optional-import contract.** A test in the existing import-shape
    suite asserts `aelfrice` (default install) does not import
    `anthropic` at any module load. Mirrors the existing test for
    `fastmcp`.
11. **PRIVACY.md updated.** The "Optional outbound calls" section
    lands in the same PR series.
12. **CONFIG.md updated.** The `[onboard.llm]` block schema and
    worked example land alongside the existing `[noise]` schema.
13. **CHANGELOG entry.** v1.3.0 line item naming the new extra and
    the new flag.
14. **`uv run pytest -x -q` green.**

## 10. Dependencies

- Internal: none. The polymorphic onboard handshake
  (`start_onboard_session` / `accept_classifications`) already exists
  but the LLM-classify path does not need to use it — `scan_repo` can
  call the new classifier inline, in the same loop where it currently
  calls `classify_sentence`. The polymorphic handshake remains for
  the host-LLM-in-context use case.
- External: `anthropic>=0.40` (in the `[onboard-llm]` extra only).
- Activates: `benchmarks/mab_llm_entity_adapter.py` (port from lab).

## 11. Open questions deferred to implementation

These are implementation-detail choices that do not affect the
boundary policy or the user-visible contract. The implementation PR
may decide either way:

- Sharding threshold (single-request vs. shard-at-K-candidates).
  Acceptance only requires correct telemetry; a single-request
  implementation is fine for v1.3.0.
- Exact retry policy for 429s (with or without Retry-After header
  parsing). Capped at 30s total per § 7.
- Whether to write the per-run JSON audit line (§ 6 last paragraph).
  Useful for debugging but not required for acceptance.
- Whether `--revoke-consent` is a flag on `onboard` or its own
  subcommand. Either works.

## 12. Rollback

If the v1.3.0 release ships the implementation and a problem surfaces
in the field:

- Users disable per-project: remove `[onboard.llm].enabled = true`
  from `.aelfrice.toml`, or stop passing `--llm-classify`. The default
  regex path resumes at the next onboard.
- Users disable system-wide: `pip uninstall anthropic` (the
  `[onboard-llm]` extra). `--llm-classify` then exits 1 on the
  install-hint check before any network attempt.
- Maintainers disable the feature: a v1.3.x patch can hard-gate
  `--llm-classify` to exit 1 with a deprecation message, leaving the
  rest of v1.3.0 (entity index, BFS, posterior-weighted ranking)
  intact. The opt-in surface means no user is silently affected by
  the gate.

The feature is non-load-bearing for the v1.3.0 milestone. Retrieval
improvements (entity index, BFS) ship independently.
