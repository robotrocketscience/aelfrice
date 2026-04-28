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

## Optional outbound calls

aelfrice ships one optional path that, when explicitly opted into, sends content from your codebase to a third party: the v1.3.0 LLM-Haiku onboard classifier. The boundary policy is non-negotiable; the design lives in [llm_classifier.md](llm_classifier.md).

**Default off.** A baseline `pip install aelfrice` does not pull the `anthropic` SDK and does not have any code path that contacts Anthropic. To use the LLM-classify path, all four of these must hold:

1. The user installed the optional extra: `pip install aelfrice[onboard-llm]`.
2. The user set `ANTHROPIC_API_KEY` in their environment.
3. The user passed `--llm-classify` on the `aelf onboard` command line, **or** added `[onboard.llm].enabled = true` to `.aelfrice.toml`.
4. The user accepted the one-time, per-machine, interactive confirmation prompt (or pre-created the consent sentinel for CI use).

If any of those four are missing, no outbound call is made. Confirm by reading `pyproject.toml` (`[onboard-llm]` is opt-in extras), `src/aelfrice/scanner.py` (regex classifier is the default classifier), and `docs/llm_classifier.md` § 4 (full boundary policy).

**What is sent when opted in:** the candidate sentences/paragraphs the onboard scanner already extracts (markdown paragraphs, git commit subjects, Python docstrings), plus their `source` strings (e.g., `doc:README.md:p3`), plus a templated system prompt that contains no user data.

**What is never sent:** file contents beyond the extracted candidate, the `ANTHROPIC_API_KEY` itself (used only as bearer token), working directory paths, hostnames, usernames, machine ids, git remotes, git config, git author email, or anything in a file matching the `INEDIBLE` marker or in a `_SKIP_DIRS` directory. The opt-out surface is the same one that already governs local ingest.

**Telemetry remains zero.** aelfrice does not phone-home about its own LLM usage. Tokens consumed are reported on stdout to the user only, never written to any network endpoint or logging service. `aelf onboard --llm-classify` makes one or more requests to `https://api.anthropic.com/`; nothing else.

**Update notifier remains the only outbound call by default.** The TTL-gated GET to `https://pypi.org/pypi/aelfrice/json` (covered above) is read-only and unconditional; the LLM-classify path is opt-in and conditional. Together these are the only two outbound calls the shipped package can make, and one of them transmits no data.

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
