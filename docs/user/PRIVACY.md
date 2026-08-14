# Privacy and Security

This page lists verifiable properties of the codebase. These are not marketing claims. You can confirm each property when you read the source.

<p align="center"><img src="../assets/08-setr.png" width="60%" alt="A single figure split down its midline — a blue-robed teacher on the left, a red-black dragon on the right — flanked by a spiral glass tower and a basalt column"></p>

## Your data never leaves your machine

The store path, the retrieval path, the scoring path, the scanner path and the feedback path run locally against SQLite. These files contain no network code: `store.py`, `retrieval.py`, `scoring.py`, `feedback.py`, `scanner.py` and `cli.py`. Confirm this with the following command:

```bash
grep -rE "requests|httpx|urllib|aiohttp|socket\.|http\." src/aelfrice/
```

**That grep is necessary but not sufficient. This paragraph gives what the grep misses.** The grep matches in-process HTTP calls and socket calls only. The grep cannot match a network call that the code makes when it starts another program. `subprocess.run(["gh", ...])` contains none of those tokens. One shipped hook does exactly that — see the pre-issue guard below. Use this second command to cover that class as well:

```bash
grep -rE "subprocess|Popen|os\.system" src/aelfrice/
```

Then read the argv of each hit. `git` is local. `gh` is not local.


Two exceptions are on by default:

- **The update notifier** (`lifecycle.py`) makes one GET request to `https://pypi.org/pypi/aelfrice/json`, gated by a time-to-live (TTL). The request checks for new releases. To disable the notifier, use `export AELF_NO_UPDATE_CHECK=1`. The notifier transmits nothing. The notifier only reads.
- **The pre-issue duplicate guard** (`pre_issue_create_hook.py`, installed by default since v3.4.0) runs `gh issue list --search <tokens>` before a `gh issue create` tool call. The guard warns about duplicates. **The guard transmits data.** The tokens come from the issue title you typed. The guard fires only on `gh issue create`. The guard never fires on ordinary retrieval, on ingest or on any other command. To disable the guard, use `export AELFRICE_NO_PRE_ISSUE_GUARD=1`. To bypass a single call, use `ALLOW_DUP_ISSUE=1`. To never install the guard, use `aelf setup --no-pre-issue-guard`.

## No telemetry

aelfrice does not report back to any server. **The shipped package contains no network telemetry.** The package holds no conditional import for telemetry, no commented-out endpoint and no environment-variable switch to enable telemetry. A local-only writer of session statistics exists (`aelf session-delta`, [`src/aelfrice/telemetry.py`](../../src/aelfrice/telemetry.py)). That writer stays inert until you wire a SessionEnd hook yourself. The writer appends counts to `~/.aelfrice/telemetry.jsonl` and makes no network calls. Confirm this when you read `pyproject.toml`. The base install adds only `numpy`, `scipy` and `snowballstemmer`. Those three packages do the local retrieval mathematics and contain no network code. The `[onboard-llm]`, `[archive]` and `[benchmarks]` extras are opt-in. `pip install aelfrice` alone installs none of the three extras.

## Onboard-time outbound call

aelfrice runs LLM-quality classification one time for each project, at onboard time. The runtime stays local. No day-to-day operation makes an outbound call.

**The default flow at v1.5.0+ is host-driven and makes zero direct calls from the aelfrice CLI itself.** When the user runs `/aelf:onboard <path>` from a host that exposes a Task tool (Claude Code and similar), the slash command body in [`src/aelfrice/slash_commands/onboard.md`](../../src/aelfrice/slash_commands/onboard.md) drives the classifier through the host's own model dispatch against the cheapest model in its stack. The host already has whatever credentials and billing it needs; aelfrice does not require an API key, and the aelfrice package never imports `anthropic` on this path. The user's data already goes to their host LLM — aelfrice just reuses the cheapest model in that stack to do classification once.

**The direct-API fallback (`aelf onboard --llm-classify`) requires an explicit opt-in on three of its four gates.** This path is for the users who do not run a host with a Task tool. Those users want aelfrice to call the vendor API directly. Four gates guard the path. If one of these gates is missing, aelfrice makes no outbound call:

1. The user installed the optional extra with `pip install aelfrice[onboard-llm]`.
2. The user set `ANTHROPIC_API_KEY` in their environment.
3. `[onboard.llm].enabled` resolves true. It resolves true **by default since v1.5.0**. To hold this gate closed, set `enabled = false` in `.aelfrice.toml`, or pass `--llm-classify=false`. Gates 1, 2 and 4 are the explicit-opt-in gates: the install of the extra, the API key, and the consent prompt or the consent sentinel.
4. The user accepted the interactive confirmation prompt. That prompt appears one time on each machine. As an alternative, the user created the consent sentinel in advance for use in continuous integration (CI).

The v1.5.0 default-on flip in `[onboard.llm].enabled` is non-destructive. A user who did not pass `--llm-classify` gets the v1.0/v1.2 regex behaviour silently, if that user has no `[onboard-llm]` extra or no `ANTHROPIC_API_KEY`. `check_gates` performs this soft-fall. Confirm the behaviour when you read these three sources:

- `pyproject.toml` — `[onboard-llm]` is an opt-in extra.
- `src/aelfrice/scanner.py` — the regex classifier is the default classifier.
- `docs/design/llm_classifier.md` § 4 — the full boundary policy on the direct-API path.

**What aelfrice sends when you opt in:** the candidate sentences and paragraphs that the onboard scanner already extracts. Those candidates are markdown paragraphs, git commit subjects and Python docstrings. aelfrice also sends their `source` strings, for example `doc:README.md:p3`. aelfrice also sends a templated system prompt that contains no user data.

**What aelfrice never sends:**

- file contents beyond the extracted candidate,
- the `ANTHROPIC_API_KEY` itself, which serves only as the bearer token,
- working directory paths,
- hostnames,
- usernames,
- machine ids,
- git remotes,
- the git config,
- the git author email,
- anything in a file that matches the `INEDIBLE` marker,
- anything in a `_SKIP_DIRS` directory.

The opt-out surface is the same surface that already governs the local ingest.

### `aelf doctor --classify-orphans` — the one path that sends *stored* content

Every other outbound path sends text that aelfrice just read out of your project files. This path is different, and this page describes it separately for that reason. The path asks the vendor model to assign a type to the beliefs still marked `unknown`. To do that, the path sends **the content of those stored beliefs**. That content can include statements you typed that aelfrice captured from conversation transcripts.

The boundary of this path is *not* identical to the boundary of onboard. This command passes `enabled=True` internally. Therefore `[onboard.llm].enabled = false` does **not** close a gate here. The run of the command is itself the explicit opt-in. These conditions guard the command:

- you installed the `[onboard-llm]` extra,
- you set `ANTHROPIC_API_KEY`,
- you invoked `aelf doctor --classify-orphans` explicitly, and no default path and no background path reaches that command,
- and a consent sentinel records the `stored_beliefs` scope.

These points apply to that last condition:

- The consent sentinel records **which data classes** aelfrice showed you when you accepted. The `scopes` key holds that record. An `aelf onboard` sentinel grants `onboard_candidates` only.
- `--classify-orphans` requires the `stored_beliefs` scope. `--classify-orphans` prompts separately for that scope. The prompt carries a disclosure that names the stored belief content explicitly. Acceptance of that scope does **not** widen what `aelf onboard` sends. Acceptance enables no recurring call and no background call.
- A sentinel written before this scoping existed has no `scopes` key. aelfrice reads such a sentinel as onboard-only. That sentinel does not authorise this path, so aelfrice prompts you one time.
- `aelf doctor --classify-orphans --dry-run` counts the candidate set and prints it, with **no gate check and no network call**. Therefore you can audit exactly what aelfrice would send before you consent.
- `aelf doctor revoke-llm-consent` removes the sentinel, and therefore removes both scopes.

Before v4.2.0 this command hardcoded its gate to open, and the command never read the sentinel. With the extra installed and `ANTHROPIC_API_KEY` exported, the command transmitted belief content with no prompt (#1172). If you ran the command on an affected version, the belief content went to the vendor API. `revoke-llm-consent` could not have prevented that transmission, because the command consulted no sentinel.

**Telemetry remains zero.** aelfrice does not phone-home about its own LLM usage. Tokens consumed are reported on stdout to the user only, never written to any network endpoint or logging service. On the direct-API path, `aelf onboard --llm-classify` makes one or more requests to `https://api.anthropic.com/`; nothing else. On the host-driven path, the aelfrice CLI makes zero direct outbound calls — the host LLM handles its own network IO under its own credentials.

**The shipped aelfrice package has six outbound-capable paths. Two of these paths are on by default.** This page counts "paths", not "calls", because either LLM path may issue several batched requests in one run.

| path | default | transmits |
|---|---|---|
| Update notifier — a TTL-gated GET to `https://pypi.org/pypi/aelfrice/json`. Disable it with `AELF_NO_UPDATE_CHECK=1`. | **on** | nothing; the path only reads |
| Pre-issue duplicate guard — `gh issue list --search <tokens>` before `gh issue create`. Disable it with `AELFRICE_NO_PRE_ISSUE_GUARD=1`. Bypass it one time with `ALLOW_DUP_ISSUE=1`. To never install it, use `aelf setup --no-pre-issue-guard`. | **on** since v3.4.0 | **yes** — the tokens come from the issue title you typed |
| `aelf onboard --llm-classify` | opt-in, consent-gated | extracted candidate sentences |
| `aelf doctor --classify-orphans` | opt-in, consent-gated | content of stored beliefs |
| `aelf gate list` — `gh issue list` and `gh issue view` against the repo that aelfrice detects from your git remote (`gate_list.py`) | off; an explicit command, and hidden from `--help` | the repo identity and the label filters; no belief content |
| `aelf upgrade` and the one-shot uv-tool migration — `uv tool install aelfrice` (`lifecycle.py`) | off; an explicit command | nothing beyond the package request itself |

Of the two default-on paths, only the notifier transmits nothing. The pre-issue guard does transmit. An audit is most likely to miss the pre-issue guard, because the guard reaches the network through another program and not through a socket. See the note on the verification grep above.

The last three rows are all of that kind: each of them runs another program. Those three rows are the reason that the count in this section has been wrong before. Only the first three rows appear under the in-process grep. The other three rows appear only under the `subprocess` grep. If you add an outbound path, add its row here. Then check which of the two greps would have found that path.

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

aelfrice has no sign-in, no API key and no sync server. Everything is one local SQLite file. To make a backup, copy that file. aelfrice ships no mechanism to sync or to distribute the memory contents between users or machines. **v3.0 ships read-only cross-project federation** (#650 / #655 / #688). A project may declare peer DB paths in a local `knowledge_deps.json`. The project then surfaces the `global` and `shared:<name>` beliefs of those peers in full-text search version 5 (FTS5) and breadth-first search (BFS). But this operation uses the local filesystem only. It makes no network call, and it sends no telemetry. The local DB is the sole writer for its own rows. A mutation against a foreign belief ID raises `ForeignBeliefError` at the API surface. See [LIMITATIONS § Sharing, sync, or distributed-write federation](LIMITATIONS.md#sharing-sync-or-distributed-write-federation).

## Per-project isolation

Each project gets its own DB at `<repo>/.git/aelfrice/memory.db`. The beliefs of project A do not leak into project B, because the two projects live in different `.git/` directories. Those beliefs reach project B only if you declare peer DBs explicitly in `knowledge_deps.json`. That mechanism is the v3.0 read-only federation, see "No accounts" above. The worktrees of one repo share one DB through `--git-common-dir`. This is by design.

git does not track `.git/`. The brain graph never crosses the git boundary.

Resolution order:

1. `$AELFRICE_DB`, if it is set. This value is an override, and aelfrice honours `:memory:`.
2. `<git-common-dir>/aelfrice/memory.db` inside any git work-tree.
3. `~/.aelfrice/memory.db` outside git. This is the legacy fallback.

## You control all writes

- aelfrice inserts the new beliefs from `onboard` and from the ingest hooks unlocked. Only an explicit `aelf lock` or `/aelf:lock` marks a belief permanent. The exception is `AELF_AUTOLOCK_CORRECTIONS=1`, which you must opt into. That option lets the Stop hook auto-lock the session corrections at the end of a turn.
- The lock prior is `(α, β) = (9.0, 0.5)`, which is durable. Passive feedback does not move a lock. aelfrice audits the event and holds the posterior ([#1168](https://github.com/robotrocketscience/aelfrice/issues/1168)). `aelf confirm` is an explicit affirmation, and it is exempt from that rule. The same behaviour holds at v3.x, because [#814](https://github.com/robotrocketscience/aelfrice/issues/814) removed the v2.x auto-demote. To change a lock, use `aelf unlock`, `aelf delete` or `aelf demote`.
- `aelf demote` removes a lock immediately. The belief itself remains. You can also delete that belief through the store API.
- Every Bayesian update writes one `feedback_history` audit row. The explicit signals write that row through `apply_feedback`. The manual sweep of deferred retrieval exposure writes that row through its own atomic update and insert. Automatic retrieval *exposure* is audit-only by default since #1086. That exposure writes a `feedback_history` row for the recurrence record, but it does not move the posterior — see [LIMITATIONS](LIMITATIONS.md). You can query the provenance in both cases.

## Optional inbound prose inspection: `sentiment_from_prose` (v2.0 module, v3.0 hook wire-up)

The module with the regex sentiment detector shipped at v2.0. No live hook reached that module until v3.0 #606. You can set `[feedback] sentiment_from_prose = true` in `.aelfrice.toml`, or set `AELFRICE_FEEDBACK_SENTIMENT_FROM_PROSE=1` in the environment. aelfrice then runs each user prompt that the host hook surfaces through a regex bank of 24 patterns ([`src/aelfrice/sentiment_feedback.py`](../../src/aelfrice/sentiment_feedback.py)). aelfrice uses the first pattern that matches, and at most one pattern for each prompt. For that pattern, aelfrice writes one `feedback_history` row for each belief retrieved in the previous turn.

**Default off.** Existing users see no behaviour change.

This is an *inbound* surface for prose inspection. aelfrice already received the prompt through the host hook to do the retrieval. The new behaviour is the regex matching and the implicit Bayesian updates. The new behaviour is not a new data access.

**What aelfrice reads:** every user prompt that the hook receives, with a cap of 200 characters. A longer prompt skips the detection, on the assumption that the prompt carries task content and not a feedback signal.

**What aelfrice stores:** one `feedback_history` row for each updated belief. Each row records only `(belief_id, valence, source="sentiment_inferred", created_at)`. The store holds no pattern id, no matched substring and no prompt text. aelfrice writes three other items to the local hook-audit log `hook_audit.jsonl`: the matched pattern id, the matched substring, and the prompt prefix. The prompt prefix is the first 200 characters. In practice the prefix is the whole prompt, because this lane fires only on prompts of 200 characters or fewer. The log `hook_audit.jsonl` is a sibling of `memory.db`. The log is on by default. To disable the log, use `AELFRICE_HOOK_AUDIT=0` or `[hook_audit] enabled = false`. That log never leaves the machine.

**What leaves the machine:** nothing. The lane uses the regexes of the standard library, and it makes no outbound call. The lane holds the same determinism contract as the rest of the runtime. The same prompt produces the same matches and the same updates.

**`aelf health` surfaces the state.** When the feature is on, `aelf health` prints `sentiment-from-prose feedback: enabled (<N> matches)`. When the feature is off, `aelf health` prints `disabled`. Therefore you can see the effect of the feature quickly.

To turn the feature off after you enabled it, remove the configuration line. As an alternative, set `[feedback] sentiment_from_prose = false`. The feedback rows that aelfrice already applied remain in `feedback_history` as audit history. To delete those rows, you need direct access to the store.

## What aelfrice does not control

The cloud LLM that receives your prompt sees everything that aelfrice injects. That property is inherent in the use of a cloud LLM. aelfrice applies these mitigations:

- **A token budget for each query.** The default is 1,500 for the UserPromptSubmit hook and the SessionStart hook. The default for the library retrieval API is 2,400. aelfrice never injects the full memory.
- **The L0/L1 ordering** surfaces the locks and the matches that are relevant to the query. That ordering does not surface a dump of the memory.
- **The per-project isolation** stops the context of one project from entering another project. That context enters only if you declare peer DBs explicitly in `knowledge_deps.json`.

If a fact must never leave your machine, do not store it.

## Batch ingest of historical sessions

`aelf ingest-transcript --batch ~/.claude/projects/` pulls existing Claude Code session JSONLs into the local belief graph. Those JSONLs may contain pasted secrets, customer data, or anything you typed in chat. There is no PII scrubber on the v1.2 ingest path. Review before backfilling. Use `--since` to scope to recent sessions if older logs predate your secret-handling discipline.

## Per-file opt-out: `INEDIBLE` marker (v1.3+)

Every aelfrice ingest path unconditionally skips a file whose basename contains the literal string `INEDIBLE`. The match is case-sensitive, in all capitals, and at any position in the basename. These are the ingest paths:

- the filesystem walk and the abstract syntax tree (AST) walk of `aelf onboard`,
- `aelf ingest-transcript`, in the single-file invocation,
- the recursive scan of `aelf ingest-transcript --batch DIR`.

These examples match: `INEDIBLE.md`, `INEDIBLE_secrets.txt`, `notes_INEDIBLE.txt` and `partINEDIBLEpart.py`. These examples do not match: `inedible.md` and `Inedible.md`. The case sensitivity is intentional. The marker should be unmistakable in a directory listing.

The basenames of the directories also count, on every ingest path. aelfrice excludes a directory named `INEDIBLE/` or `INEDIBLE_drafts/`, together with everything below it. `aelf onboard` prunes such a directory during its filesystem walk and its AST walk. The `aelf ingest-transcript` paths, both single-file and `--batch`, check the basename of the file **and** the names of its ancestor directories through `is_inedible_path` (#958). Therefore aelfrice never ingests a transcript JSONL file inside a directory with an INEDIBLE name, whatever the name of the file itself.

The check reads the basename, not the content. When `is_inedible(path)` returns True, aelfrice does not open the file, does not read it and does not hash it. The check happens before any classification, before any tokenization and before any noise filter. The check happens before aelfrice reads any file content.

For the mechanism, see [`src/aelfrice/inedible.py`](../../src/aelfrice/inedible.py). This predicate is the only opt-out that aelfrice respects deterministically on every ingest path. To reproduce the result, run `python3 -c "from aelfrice.inedible import is_inedible; print(is_inedible('your/path.md'))"`.

## Reproducible from source

All `onboard` beliefs come from files you already have: the code, the documentation and the git history. After `rm <resolved-db-path>`, a new run of `aelf onboard .` is deterministic up to the classifier. The state of the world is your codebase, not the memory.

## SQLite only

aelfrice uses no external database, no vector DB and no cloud storage. aelfrice uses write-ahead log (WAL) journaling for crash safety. You can rebuild the onboard-derived beliefs fully from your source files and your lock list. The beliefs ingested from transcripts and from commits rebuild only from their original session JSONL files and from the git history.

## Reporting

See [SECURITY.md](../../SECURITY.md). aelfrice treats a privacy issue as a security issue.
