# Privacy and Security

This page lists verifiable properties of the codebase, not marketing claims. You can confirm every one of them by reading the source.

<p align="center"><img src="../assets/08-setr.png" width="60%" alt="A single figure split down its midline — a blue-robed teacher on the left, a red-black dragon on the right — flanked by a spiral glass tower and a basalt column"></p>

## Your data never leaves your machine

The store, retrieval, scoring, scanner, and feedback paths all run locally against SQLite. These files contain no network code: `store.py`, `retrieval.py`, `scoring.py`, `feedback.py`, `scanner.py`, and `cli.py`. To confirm that, run this command:

```bash
grep -rE "requests|httpx|urllib|aiohttp|socket\.|http\." src/aelfrice/
```

**That grep is necessary but not sufficient. Here is what it misses.** It matches in-process HTTP calls and socket calls only. It cannot match a network call that the code makes by starting another program: `subprocess.run(["gh", ...])` contains none of those tokens. One shipped hook does exactly that; see the pre-issue duplicate guard below. To cover that class as well, run a second command:

```bash
grep -rE "subprocess|Popen|os\.system" src/aelfrice/
```

Then read the argv of each hit. `git` is local; `gh` is not.


Two exceptions are on by default:

- **The update notifier** (`lifecycle.py`) makes one GET request to `https://pypi.org/pypi/aelfrice/json` to check for new releases, gated by a time-to-live (TTL). It transmits nothing; it only reads. To disable the notifier, run `export AELF_NO_UPDATE_CHECK=1`.
- **The pre-issue duplicate guard** (`pre_issue_create_hook.py`, installed by default since v3.4.0) runs `gh issue list --search <tokens>` before a `gh issue create` tool call, and warns you about duplicates. **This guard does transmit data:** the tokens come from the issue title you typed. It fires only on `gh issue create`, never on ordinary retrieval, on ingest, or on any other command. To disable the guard, run `export AELFRICE_NO_PRE_ISSUE_GUARD=1`. To bypass a single call, set `ALLOW_DUP_ISSUE=1`. To never install the guard, run `aelf setup --no-pre-issue-guard`.

## No telemetry

aelfrice doesn't report back to any server. **The shipped package contains no network telemetry:** no conditional import for telemetry, no commented-out endpoint, and no environment-variable switch that turns telemetry on. A local-only writer of session statistics does exist (`aelf session-delta`, [`src/aelfrice/telemetry.py`](../../src/aelfrice/telemetry.py)), and it stays inert until you wire a SessionEnd hook yourself. It appends counts to `~/.aelfrice/telemetry.jsonl` and makes no network calls. To confirm that, read `pyproject.toml`: the base install adds only `numpy`, `scipy`, and `snowballstemmer`, and those three packages do the local retrieval math and contain no network code. The `[onboard-llm]`, `[archive]`, and `[benchmarks]` extras are opt-in; `pip install aelfrice` on its own installs none of the three.

## Onboard-time outbound call

aelfrice runs LLM-quality classification once per project, at onboard time. The runtime stays local, and no day-to-day operation makes an outbound call.

**The default flow at v1.5.0+ is host-driven and makes zero direct calls from the aelfrice CLI itself.** When you run `/aelf:onboard <path>` from a host that exposes a Task tool (Claude Code and similar), the slash command body in [`src/aelfrice/slash_commands/onboard.md`](../../src/aelfrice/slash_commands/onboard.md) drives the classifier through the host's own model dispatch, against the cheapest model in its stack. The host already has whatever credentials and billing it needs, aelfrice doesn't require an API key, and the aelfrice package never imports `anthropic` on this path. Your data already goes to your host LLM, so aelfrice reuses the cheapest model in that stack to classify once.

**The direct-API fallback (`aelf onboard --llm-classify`) requires an explicit opt-in on three of its four gates.** This path is for people who don't run a host with a Task tool. Those people want aelfrice to call the vendor API directly. Four gates guard the path, and if any one of them is missing, aelfrice makes no outbound call:

1. You installed the optional extra with `pip install aelfrice[onboard-llm]`.
2. You set `ANTHROPIC_API_KEY` in your environment.
3. `[onboard.llm].enabled` resolves true, which it does **by default since v1.5.0**. To hold this gate closed, set `enabled = false` in `.aelfrice.toml`, or pass `--llm-classify=false`. Gates 1, 2, and 4 are the explicit opt-in gates: the install of the extra, the API key, and the consent prompt or the consent sentinel.
4. You accepted the interactive confirmation prompt, which appears once per machine. Alternatively, you created the consent sentinel in advance for use in continuous integration (CI).

The v1.5.0 default-on flip in `[onboard.llm].enabled` is non-destructive. If you didn't pass `--llm-classify` and you have no `[onboard-llm]` extra or no `ANTHROPIC_API_KEY`, you silently get the v1.0/v1.2 regex behavior. `check_gates` performs this soft-fall. To confirm the behavior, read these three sources:

- `pyproject.toml` — `[onboard-llm]` is an opt-in extra.
- `src/aelfrice/scanner.py` — the regex classifier is the default classifier.
- `docs/design/llm_classifier.md` § 4 — the full boundary policy on the direct-API path.

**What aelfrice sends when you opt in:** the candidate sentences and paragraphs that the onboard scanner already extracts. Those candidates are markdown paragraphs, git commit subjects, and Python docstrings. aelfrice also sends their `source` strings, for example `doc:README.md:p3`, along with a templated system prompt that contains no user data.

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

Every other outbound path sends text that aelfrice has read straight out of your project files. This path is different, and this page describes it separately for that reason. The path asks the vendor model to assign a type to the beliefs still marked `unknown`, and to do that it sends **the content of those stored beliefs**. That content can include statements you typed that aelfrice captured from conversation transcripts.

The boundary of this path is *not* identical to the boundary of onboard. The command passes `enabled=True` internally, so `[onboard.llm].enabled = false` does **not** close a gate here. Running the command is itself the explicit opt-in. These conditions guard the command:

- you installed the `[onboard-llm]` extra,
- you set `ANTHROPIC_API_KEY`,
- you ran `aelf doctor --classify-orphans` explicitly, and no default path and no background path reaches that command,
- and a consent sentinel records the `stored_beliefs` scope.

That last condition works like this:

- The consent sentinel records **which data classes** aelfrice showed you when you accepted, and the `scopes` key holds that record. An `aelf onboard` sentinel grants `onboard_candidates` only.
- `--classify-orphans` requires the `stored_beliefs` scope. When you run `--classify-orphans`, it prompts separately for that scope, and the prompt carries a disclosure that names the stored belief content explicitly. Accepting that scope does **not** widen what `aelf onboard` sends, and it enables no recurring call and no background call.
- A sentinel written before this scoping existed has no `scopes` key. aelfrice reads such a sentinel as onboard-only, so that sentinel doesn't authorize this path and aelfrice prompts you once.
- `aelf doctor --classify-orphans --dry-run` counts the candidate set and prints it, with **no gate check and no network call**, so you can audit exactly what aelfrice would send before you consent.
- `aelf doctor revoke-llm-consent` removes the sentinel, and therefore removes both scopes.

Before v4.2.0, this command hardcoded its gate to open and never read the sentinel. With the extra installed and `ANTHROPIC_API_KEY` exported, the command transmitted belief content with no prompt (#1172). If you ran the command on an affected version, that belief content went to the vendor API, and `revoke-llm-consent` couldn't have prevented the transmission, because the command consulted no sentinel.

**Telemetry remains zero.** aelfrice sends no report about its own LLM usage. It prints the tokens consumed on stdout for you to read, and never writes them to a network endpoint or a logging service. On the direct-API path, `aelf onboard --llm-classify` makes one or more requests to `https://api.anthropic.com/`, and nothing else. On the host-driven path, the aelfrice CLI makes zero direct outbound calls, because the host LLM handles its own network I/O under its own credentials.

**The shipped aelfrice package has six outbound-capable paths. Two of these paths are on by default.** This page counts paths, not calls, because either LLM path can issue several batched requests in one run.

| path | default | transmits |
|---|---|---|
| Update notifier — a TTL-gated GET to `https://pypi.org/pypi/aelfrice/json`. Disable it with `AELF_NO_UPDATE_CHECK=1`. | **on** | nothing; the path only reads |
| Pre-issue duplicate guard — `gh issue list --search <tokens>` before `gh issue create`. Disable it with `AELFRICE_NO_PRE_ISSUE_GUARD=1`. Bypass it one time with `ALLOW_DUP_ISSUE=1`. To never install it, use `aelf setup --no-pre-issue-guard`. | **on** since v3.4.0 | **yes** — the tokens come from the issue title you typed |
| `aelf onboard --llm-classify` | opt-in, consent-gated | extracted candidate sentences |
| `aelf doctor --classify-orphans` | opt-in, consent-gated | content of stored beliefs |
| `aelf gate list` — `gh issue list` and `gh issue view` against the repo that aelfrice detects from your git remote (`gate_list.py`) | off; an explicit command, and hidden from `--help` | the repo identity and the label filters; no belief content |
| `aelf upgrade` and the one-shot uv-tool migration — `uv tool install aelfrice` (`lifecycle.py`) | off; an explicit command | nothing beyond the package request itself |

Of the two default-on paths, only the notifier transmits nothing; the pre-issue guard does transmit. An audit is most likely to miss the pre-issue guard, because the guard reaches the network by running another program rather than through a socket. See the note on the verification grep above.

The last three rows are all of that kind: each of them runs another program. Those three rows are why the count in this section has been wrong before. Only the first three rows show up under the in-process grep; the other three show up only under the `subprocess` grep. When you add an outbound path, add its row here, then check which of the two greps would have found that path.

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

aelfrice has no sign-in, no API key, and no sync server. Everything lives in one local SQLite file, so to make a backup, copy that file. aelfrice ships no mechanism to sync or distribute the memory contents between users or machines. **v3.0 ships read-only cross-project federation** (#650 / #655 / #688): a project can declare peer DB paths in a local `knowledge_deps.json`, and it then surfaces the `global` and `shared:<name>` beliefs of those peers in full-text search version 5 (FTS5) and breadth-first search (BFS). This operation uses the local filesystem only. It makes no network call, and it sends no telemetry. The local DB is the sole writer for its own rows, and a mutation against a foreign belief ID raises `ForeignBeliefError` at the API surface. See [LIMITATIONS § Sharing, sync, or distributed-write federation](LIMITATIONS.md#sharing-sync-or-distributed-write-federation).

## Per-project isolation

Each project gets its own DB at `<repo>/.git/aelfrice/memory.db`. The beliefs of project A do not leak into project B, because the two projects live in different `.git/` directories. Those beliefs reach project B only if you declare peer DBs explicitly in `knowledge_deps.json`, which is the v3.0 read-only federation described in the "No accounts" section above. The worktrees of one repo share one DB through `--git-common-dir`, by design.

git does not track `.git/`. The brain graph never crosses the git boundary.

Resolution order:

1. `$AELFRICE_DB`, if that variable is set. This value is an override, and aelfrice honors `:memory:`.
2. `<git-common-dir>/aelfrice/memory.db` inside any git work-tree.
3. `~/.aelfrice/memory.db` outside git. This is the legacy fallback.

## You control all writes

- aelfrice inserts the new beliefs from `onboard` and from the ingest hooks unlocked. Only an explicit `aelf lock` or `/aelf:lock` marks a belief permanent. The one exception is `AELF_AUTOLOCK_CORRECTIONS=1`, which you have to opt into: it lets the Stop hook auto-lock the session corrections at the end of a turn.
- The lock prior is `(α, β) = (9.0, 0.5)`, which is durable. Passive feedback doesn't move a lock; aelfrice audits the event and holds the posterior ([#1168](https://github.com/robotrocketscience/aelfrice/issues/1168)). `aelf confirm` is an explicit affirmation, and it is exempt from that rule. The same behavior holds at v3.x, because [#814](https://github.com/robotrocketscience/aelfrice/issues/814) removed the v2.x auto-demote. To change a lock, run `aelf unlock`, `aelf delete`, or `aelf demote`.
- `aelf demote` removes a lock immediately. The belief itself remains, and you can also delete that belief through the store API.
- Every Bayesian update writes one `feedback_history` audit row. The explicit signals write that row through `apply_feedback`. The manual sweep of deferred retrieval exposure writes that row through its own atomic update and insert. Automatic retrieval *exposure* is audit-only by default since #1086: it writes a `feedback_history` row for the recurrence record, but it doesn't move the posterior, as [LIMITATIONS](LIMITATIONS.md) describes. You can query the provenance in both cases.

## Optional inbound prose inspection: `sentiment_from_prose` (v2.0 module, v3.0 hook wire-up)

The module with the regex sentiment detector shipped at v2.0, but no live hook reached that module until v3.0 #606. To switch it on, set `[feedback] sentiment_from_prose = true` in `.aelfrice.toml`, or set `AELFRICE_FEEDBACK_SENTIMENT_FROM_PROSE=1` in the environment. aelfrice then runs each user prompt that the host hook surfaces through a regex bank of 24 patterns ([`src/aelfrice/sentiment_feedback.py`](../../src/aelfrice/sentiment_feedback.py)), takes the first pattern that matches, and applies at most one pattern per prompt. For that pattern, it writes one `feedback_history` row for each belief retrieved in the previous turn.

**Default off.** Existing users see no change in behavior.

This is an *inbound* surface for prose inspection. aelfrice already received the prompt through the host hook to do the retrieval. What is new is the regex matching and the implicit Bayesian updates, not a new data access.

**What aelfrice reads:** every user prompt that the hook receives, capped at 200 characters. A longer prompt skips the detection, on the assumption that the prompt carries task content and not a feedback signal.

**What aelfrice stores:** one `feedback_history` row for each updated belief. Each row records only `(belief_id, valence, source="sentiment_inferred", created_at)`, so the store holds no pattern id, no matched substring, and no prompt text. aelfrice writes three other items to the local hook-audit log `hook_audit.jsonl`: the matched pattern id, the matched substring, and the prompt prefix. The prompt prefix is the first 200 characters. In practice the prefix is the whole prompt, because this lane fires only on prompts of 200 characters or fewer. The log `hook_audit.jsonl` sits beside `memory.db`, and it is on by default. To disable the log, set `AELFRICE_HOOK_AUDIT=0` or `[hook_audit] enabled = false`. That log never leaves the machine.

**What leaves the machine:** nothing. The lane uses the regexes of the standard library, makes no outbound call, and holds the same determinism contract as the rest of the runtime: the same prompt produces the same matches and the same updates.

**`aelf health` surfaces the state.** When the feature is on, `aelf health` prints `sentiment-from-prose feedback: enabled (<N> matches)`. When the feature is off, `aelf health` prints `disabled`. That way you can see the effect of the feature quickly.

To turn the feature off after you enabled it, remove the configuration line, or set `[feedback] sentiment_from_prose = false`. The feedback rows that aelfrice already applied remain in `feedback_history` as audit history. To delete those rows, you need direct access to the store.

## What aelfrice does not control

The cloud LLM that receives your prompt sees everything that aelfrice injects. That property is inherent in the use of a cloud LLM. aelfrice applies these mitigations:

- **A token budget for each query.** The default is 1,500 for the UserPromptSubmit hook and the SessionStart hook, and 2,400 for the library retrieval API. aelfrice never injects the full memory.
- **The L0/L1 ordering** surfaces the locks and the matches that are relevant to your query, not a dump of the memory.
- **The per-project isolation** keeps the context of one project out of another project. That context crosses only if you declare peer DBs explicitly in `knowledge_deps.json`.

If a fact must never leave your machine, don't store it.

## Batch ingest of historical sessions

`aelf ingest-transcript --batch ~/.claude/projects/` pulls existing Claude Code session JSONL files into the local belief graph. Those files might contain pasted secrets, customer data, or anything else you typed in chat, and the v1.2 ingest path has no scrubber for personally identifiable information (PII). Review them before you backfill. If older logs predate your secret-handling discipline, use `--since` to scope the run to recent sessions.

## Per-file opt-out: `INEDIBLE` marker (v1.3+)

Every aelfrice ingest path unconditionally skips a file whose basename contains the literal string `INEDIBLE`. The match is case-sensitive, in all capitals, and at any position in the basename. The ingest paths are:

- the filesystem walk and the abstract syntax tree (AST) walk of `aelf onboard`,
- `aelf ingest-transcript`, in the single-file invocation,
- the recursive scan of `aelf ingest-transcript --batch DIR`.

These names match: `INEDIBLE.md`, `INEDIBLE_secrets.txt`, `notes_INEDIBLE.txt`, and `partINEDIBLEpart.py`. These names do not: `inedible.md` and `Inedible.md`. The case sensitivity is intentional: the marker should be unmistakable in a directory listing.

The basenames of the directories count too, on every ingest path. aelfrice excludes a directory named `INEDIBLE/` or `INEDIBLE_drafts/`, together with everything below it. `aelf onboard` prunes such a directory during its filesystem walk and its AST walk. The `aelf ingest-transcript` paths, both single-file and `--batch`, check the basename of the file **and** the names of its ancestor directories through `is_inedible_path` (#958). aelfrice therefore never ingests a transcript JSONL file inside a directory with an INEDIBLE name, whatever the name of the file itself.

The check reads the basename, not the content. When `is_inedible(path)` returns True, aelfrice doesn't open the file, read it, or hash it. That check runs before any classification, tokenization, or noise filter, and before aelfrice reads any file content.

For the mechanism, see [`src/aelfrice/inedible.py`](../../src/aelfrice/inedible.py). This predicate is the only opt-out that aelfrice respects deterministically on every ingest path. To reproduce the result, run `python3 -c "from aelfrice.inedible import is_inedible; print(is_inedible('your/path.md'))"`.

## Reproducible from source

All `onboard` beliefs come from files you already have: your code, your documentation, and your git history. After `rm <resolved-db-path>`, a new run of `aelf onboard .` is deterministic up to the classifier. The state of the world is your codebase, not the memory.

## SQLite only

aelfrice uses no external database, no vector DB, and no cloud storage. It uses write-ahead log (WAL) journaling for crash safety. You can rebuild the onboard-derived beliefs entirely from your source files and your lock list. The beliefs ingested from transcripts and from commits rebuild only from their original session JSONL files and from the git history.

## Reporting

See [SECURITY.md](../../SECURITY.md). aelfrice treats a privacy issue as a security issue.
