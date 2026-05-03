# Privacy and Security

Verifiable properties of the codebase, not marketing claims. Each can be confirmed by reading the source.

<p align="center"><img src="assets/08-setr.png" width="60%" alt="A single figure split down its midline — a blue-robed teacher on the left, a red-black dragon on the right — flanked by a spiral glass tower and a basalt column"></p>

## Your data never leaves your machine

The store, retrieval, scoring, scanner, and feedback paths run locally against SQLite. No network code in `store.py`, `retrieval.py`, `scoring.py`, `feedback.py`, `scanner.py`, or `cli.py`. Confirm:

```bash
grep -rE "requests|httpx|urllib|aiohttp|socket\.|http\." src/aelfrice/
```

The optional `[mcp]` extra (`fastmcp`) speaks MCP over stdio to the host on the same machine. No remote sockets.

The single exception: the update notifier (`lifecycle.py`) makes one TTL-gated GET to `https://pypi.org/pypi/aelfrice/json` to check for new releases. Disable with `export AELF_NO_UPDATE_CHECK=1`. The notifier never transmits anything; it only reads.

## No telemetry

No usage tracking. No analytics. No phone-home. **Not opt-out — the capability does not exist in the shipped package.** No conditional import, no commented-out endpoint, no env-var toggle to enable it. Confirm by reading `pyproject.toml`: only `[mcp]` and `[onboard-llm]` add anything beyond stdlib, and neither is installed by `pip install aelfrice` alone.

## Onboard-time outbound call

aelfrice runs LLM-quality classification once per project, at onboard time. The runtime stays local; no day-to-day operation makes any outbound call.

**The default flow at v1.5.0+ is host-driven and makes zero direct calls from the aelfrice CLI itself.** When the user runs `/aelf:onboard <path>` from a host that exposes a Task tool (Claude Code and similar), the slash command body in [`src/aelfrice/slash_commands/onboard.md`](../src/aelfrice/slash_commands/onboard.md) drives the classifier through the host's own model dispatch against the cheapest model in its stack. The host already has whatever credentials and billing it needs; aelfrice does not require an API key, and the aelfrice package never imports `anthropic` on this path. The user's data already goes to their host LLM — aelfrice just reuses the cheapest model in that stack to do classification once.

**The direct-API fallback (`aelf onboard --llm-classify`) is explicit-opt-in and unchanged from v1.3.0.** It is the path for users who don't run a host with a Task tool and instead want aelfrice to call the Anthropic API directly. Four gates guard it; if any of these is missing, no outbound call is made:

1. The user installed the optional extra: `pip install aelfrice[onboard-llm]`.
2. The user set `ANTHROPIC_API_KEY` in their environment.
3. The user passed `--llm-classify` on the `aelf onboard` command line, **or** added `[onboard.llm].enabled = true` to `.aelfrice.toml`.
4. The user accepted the one-time, per-machine, interactive confirmation prompt (or pre-created the consent sentinel for CI use).

The v1.5.0 default-on flip in `[onboard.llm].enabled` is non-destructive: users without the `[onboard-llm]` extra or without `ANTHROPIC_API_KEY` who haven't passed `--llm-classify` get the v1.0/v1.2 regex behaviour silently (soft-fall in `check_gates`). Confirm by reading `pyproject.toml` (`[onboard-llm]` is opt-in extras), `src/aelfrice/scanner.py` (regex classifier is the default classifier), and `docs/llm_classifier.md` § 4 (full boundary policy on the direct-API path).

**What is sent when opted in:** the candidate sentences/paragraphs the onboard scanner already extracts (markdown paragraphs, git commit subjects, Python docstrings), plus their `source` strings (e.g., `doc:README.md:p3`), plus a templated system prompt that contains no user data.

**What is never sent:** file contents beyond the extracted candidate, the `ANTHROPIC_API_KEY` itself (used only as bearer token), working directory paths, hostnames, usernames, machine ids, git remotes, git config, git author email, or anything in a file matching the `INEDIBLE` marker or in a `_SKIP_DIRS` directory. The opt-out surface is the same one that already governs local ingest.

**Telemetry remains zero.** aelfrice does not phone-home about its own LLM usage. Tokens consumed are reported on stdout to the user only, never written to any network endpoint or logging service. On the direct-API path, `aelf onboard --llm-classify` makes one or more requests to `https://api.anthropic.com/`; nothing else. On the host-driven path, the aelfrice CLI makes zero direct outbound calls — the host LLM handles its own network IO under its own credentials.

**Update notifier remains the only outbound call from aelfrice itself by default.** The TTL-gated GET to `https://pypi.org/pypi/aelfrice/json` (covered above) is read-only and unconditional; the direct-API LLM-classify path is opt-in and conditional. Those are the only two outbound calls the shipped aelfrice package can make, and one of them transmits no data.

**Confirm at the source:**

```bash
# anthropic SDK is not installed by default
pip show anthropic 2>/dev/null || echo "not installed (expected default)"

# aelfrice never imports anthropic at module load
grep -rn "import anthropic\|from anthropic" src/aelfrice/

# the only opt-in surface is --llm-classify or [onboard.llm].enabled
grep -rn "llm-classify\|onboard\.llm\|llm_classify" src/aelfrice/
```

## No accounts

No sign-in, no API key, no sync server. Everything is one local SQLite file. Back up by copying the file. aelfrice ships no mechanism for sharing memory contents between users, machines, or projects — see [LIMITATIONS](LIMITATIONS.md).

## Per-project isolation

Each project gets its own DB at `<repo>/.git/aelfrice/memory.db`. Beliefs from project A cannot leak into project B — they live in different `.git/` directories. Worktrees of one repo share a DB through `--git-common-dir`, by design.

`.git/` is not git-tracked. The brain graph never crosses the git boundary.

Resolution order:

1. `$AELFRICE_DB` if set (override; `:memory:` is honoured).
2. `<git-common-dir>/aelfrice/memory.db` inside any git work-tree.
3. `~/.aelfrice/memory.db` outside git (legacy fallback).

## You control all writes

- New beliefs from `onboard` and ingest hooks are inserted unlocked. Only an explicit `aelf lock` (or MCP `aelf:lock`) marks something permanent.
- Lock prior is `(α, β) = (9.0, 0.5)` — durable but not unkillable. Five contradicting feedback events auto-demote the lock.
- `aelf demote` removes a lock immediately. The belief itself remains; you can also delete it via the store API.
- Every Bayesian update goes through `apply_feedback` and writes one `feedback_history` audit row. Provenance is queryable.

## Optional inbound prose inspection: `sentiment_from_prose` (v2.0+)

When `[feedback] sentiment_from_prose = true` is set in `.aelfrice.toml` (or `AELFRICE_FEEDBACK_SENTIMENT_FROM_PROSE=1` is set in the environment), aelfrice runs each user prompt the host hook surfaces through a 24-pattern regex bank ([`src/aelfrice/sentiment_feedback.py`](../src/aelfrice/sentiment_feedback.py)) and writes one `feedback_history` row per matched pattern, distributed across the previous turn's retrieved beliefs.

**Default off.** Existing users see no behavior change.

This is an *inbound* prose-inspection surface — aelfrice already received the prompt via the host hook to do retrieval. The new behavior is regex matching plus implicit Bayesian updates, not new data access.

**What is read:** every user prompt the hook receives, capped at 200 characters (longer prompts skip detection on the assumption that they carry task content rather than feedback signal).

**What is stored:** one `feedback_history` audit row per matched pattern, recording `(belief_id, valence, source="sentiment_inferred", created_at)`. The audit row carries the matched pattern id and substring; the full prompt text is **never** stored.

**What leaves the machine:** nothing. Stdlib regex; no outbound calls. Same determinism contract as the rest of the runtime — same prompt produces same matches and same updates.

**`aelf health` surfaces the state.** `aelf health` prints `sentiment-from-prose feedback: enabled (<N> matches)` when the feature is on, or `disabled` otherwise, so its effect is visible at a glance.

To turn it off after enabling, remove the config line (or set `[feedback] sentiment_from_prose = false`). Already-applied feedback rows remain in `feedback_history` as audit history; deleting them requires direct store access.

## What aelfrice does not control

The cloud LLM at the other end of your prompt sees whatever aelfrice injects. That's inherent to using a cloud LLM. Mitigations:

- **Per-query token budget** (default 2,000). The full memory is never injected.
- **L0/L1 ordering** surfaces locks plus query-relevant matches, not a memory dump.
- **Per-project isolation** means cross-project context cannot bleed in.

If a fact must never leave your machine, do not store it.

## Batch ingest of historical sessions

`aelf ingest-transcript --batch ~/.claude/projects/` pulls existing Claude Code session JSONLs into the local belief graph. Those JSONLs may contain pasted secrets, customer data, or anything you typed in chat. There is no PII scrubber on the v1.2 ingest path. Review before backfilling. Use `--since` to scope to recent sessions if older logs predate your secret-handling discipline.

## Per-file opt-out: `INEDIBLE` marker (v1.3+)

Any file whose basename contains the literal string `INEDIBLE` (case-sensitive, all caps, anywhere in the basename) is unconditionally skipped by every aelfrice ingest path:

- `aelf onboard` filesystem walk and AST walk.
- `aelf ingest-transcript` (single-file invocation).
- `aelf ingest-transcript --batch DIR` recursive scan.

Examples that match: `INEDIBLE.md`, `INEDIBLE_secrets.txt`, `notes_INEDIBLE.txt`, `partINEDIBLEpart.py`. Examples that do not: `inedible.md`, `Inedible.md`. Case sensitivity is intentional — the marker should be unmistakable in directory listings.

Directory basenames also count: a directory named `INEDIBLE/` (or `INEDIBLE_drafts/`) is not descended at all, so its contents never reach a classifier or an extractor regardless of file names underneath.

The check is on the basename, not the content. When `is_inedible(path)` returns True, aelfrice does not open, read, or hash the file. The check happens before any classification, before any tokenization, before any noise filter — earlier in the pipeline than any other exclusion in the codebase.

Mechanism: see [`src/aelfrice/inedible.py`](../src/aelfrice/inedible.py). The predicate is the only opt-out aelfrice respects deterministically across every ingest path; reproduce with `python3 -c "from aelfrice.inedible import is_inedible; print(is_inedible('your/path.md'))"`.

## Reproducible from source

All `onboard` beliefs come from files you already have: code, docs, git history. After `rm <resolved-db-path>`, re-running `aelf onboard .` is deterministic up to the classifier. The state of the world is your codebase, not the memory.

## SQLite only

No external database, no vector DB, no cloud storage. WAL journaling for crash safety. Fully rebuildable from your source files plus your lock list.

## Reporting

See [SECURITY.md](../SECURITY.md). Privacy issues are treated as security issues.
