# Contributing

Thank you for your interest in a contribution. aelfrice is a project with one author. The bar for a change is this: "is the system better afterward, in a way that's defensible by tests."

## Status

For the current line, read [CHANGELOG.md](CHANGELOG.md). It gives the latest release. Read [ROADMAP](docs/concepts/ROADMAP.md) for the landing record. Issues are welcome. The author evaluates each pull request (PR) on a case-by-case basis. The bar is "moves the system measurably forward, justifies the change with a test."

Best categories of PR:

- A bug fix with a regression test. The test fails before the fix and passes after the fix.
- A documentation fix: a typo, a broken link, or a claim that is stale against the current code.
- A change that closes one of the [known limitations](docs/user/LIMITATIONS.md). Open an issue first, and agree on the approach.

These changes are hard to land without agreement in advance:

- A new CLI subcommand.
- A change to the schema.
- A change that adds a hard dependency.
- The return of an earlier research-line feature. A benchmark or an experiment must show the impact first.

## How to file a useful issue

Write a title of one line. Use lowercase. Do not put a period at the end.

```
search: locked-belief order is unstable when budget is exhausted
```

Include these items in the body:

- **What happened.** Give the exact CLI invocation and the exact output.
- **What you expected.** Give one line.
- **Environment.** Give the operating system, the Python version (`python --version`), the aelfrice version, and the host agent if the host agent is relevant.
- **A minimal reproduction.** Give a directory where `aelf onboard <here>` reproduces the problem. Alternatively, give the smallest sequence of CLI calls that causes the problem.

Do not include your real memory.db file. That file is at `<repo>/.git/aelfrice/memory.db`, or at `~/.aelfrice/memory.db` for a directory that is not a git repository. The file contains your private beliefs. Reproduce the problem on a scratch database (`AELFRICE_DB=/tmp/scratch.db`). Share the scratch database.

## Triage labels

The issue tracker uses a small vocabulary of labels. The in-repo `aelf gate
list` aggregator (`src/aelfrice/gate_list.py`) reads `gate:ratify` /
`gate:prereq` / `bench-gated` / `gate:license` through `gh`. Operator-side
scan tooling reads the remaining labels. This repository does not ship that
scan tooling. The scan tooling decides whether an open issue is ready for a
claim, or whether to hide the issue from the active queue. Apply one or more
of these labels at file-time when an issue is not immediately actionable:

- **`gate:operator`** — an operator decision or operator-side data must
  arrive before the issue can move. Example: a tracker that opens when
  enough telemetry has accumulated to baseline against (#749, #488).
- **`gate:prereq`** — blocked on another tracked work item. Examples are a
  sub-task, an upstream dependency, or a framework that must land first.
- **`gate:lab-corpus`** — blocked on lab-side corpus delivery. The
  public-tree work cannot exercise its acceptance criteria until the corpus
  is committed.
- **`gate:ratify`** — the issue needs ratification of a design decision
  before implementation begins.
- **`gate:umbrella`** — an umbrella issue that coordinates sub-issues. It
  has no implementation surface of its own. It closes when its children
  close.
- **`bench-gated`** — the implementation has shipped. The only outstanding
  work is a benchmark run. The result of the run decides whether to flip a
  default, ship a tuning change, or revert (#769, #697, #491).
- **`attn:decisions-needed`** — the operator must adjudicate something
  before the issue can move. This label differs from `gate:operator`: the
  operator has all the information, but has not yet made the choice.

An issue that carries one of these labels appears in the scanner inventory
output. The scan tooling excludes such an issue from the "next actionable"
list. The right label at file-time prevents a re-evaluation of the issue on
every fresh scan.

## Where to look for work

[ROADMAP](docs/concepts/ROADMAP.md) carries the version-by-version landing record and the current active line. [LIMITATIONS](docs/user/LIMITATIONS.md) lists the known gaps against current HEAD. The issue tracker is the canonical source of the work in flight.

The contributions with the largest effect tend to land in three places:

- **Bench gates.** Several `bench-gated` issues wait on benchmark runs. Each run decides whether to flip a default or to revert (see `gh issue list --label bench-gated`).
- **Triage drift.** Some issues carry the `Blocked` label although their blockers have since closed. Surface the status flip instead of an implementation.
- **Stale-documentation fixes.** Fix anything where the documentation makes a false statement about the code at current HEAD. A documentation audit lives in [docs/audits/](docs/audits/). The latest pass enumerates what is outstanding.

## What the project will not do

- Vector embeddings or approximate nearest-neighbour (ANN) search in retrieval. Both would require a hard dependency on a vector library. Such a dependency defeats the local-stdlib design.
- Cloud sync, accounts, or any data path that is not local.
- A web user interface.
- Integration with chat platforms.

## Development setup

```bash
git clone https://github.com/robotrocketscience/aelfrice.git
cd aelfrice
uv sync --all-groups
uv run pytest tests/ -x -q
uv run pyright src/
```

Conventions:

- Use a conventional-commit prefix: `feat:`, `fix:`, `perf:`, `refactor:`, `test:`, `docs:`, `build:`, `ci:`, `style:`, `revert:`, `exp:`, `chore:`, `release:`, `gate:`, `audit:`.
- Make atomic commits. Each commit moves the tree from one tested green state to the next tested green state.
- Every change of behaviour needs a test.
- `pyright --strict` must pass.

### Your local test budgets are 4 times the CI budgets

The suite multiplies every wall-clock timeout at collection by
`AELF_TEST_TIMEOUT_SCALE`. This variable defaults to **4** on your machine.
Every workflow that runs pytest pins the variable to **1** (#1472). A loaded
laptop was failing tests that had no defect. `aelf-pr-open.sh` runs pytest
with `-x`, so the first such failure ended the run.

Know this consequence: **a timing failure that you cannot reproduce locally
is expected**, because CI runs the same test on a quarter of your budget. To
reproduce CI exactly, run this command:

```bash
AELF_TEST_TIMEOUT_SCALE=1 uv run pytest tests/ -x -q
```

A malformed value falls back to the default and raises no error. A malformed
value is non-numeric, zero, negative, or absurdly large. Zero is the value
that matters. `pytest-timeout` reads a timeout of 0 as *disabled*. A scale of
zero that multiplied through would remove every budget in the suite without a
message, and every test would still pass.

### The bench-gate tier does not run here

A green `pytest tests/` run on this repository does **not** mean that the
quality gates passed. It means that the run skipped them.

The quality gates for retrieval, compression and clustering are in
`tests/bench_gate/`. These tests carry the `bench_gated` marker. They skip
unless `AELFRICE_CORPUS_ROOT` points at a labelled evaluation corpus. That
corpus is private, and this repository is public. The whole tier therefore
skips on every public CI run. The run prints a `bench-gate tier` summary line
with the number of skips, so nobody can mistake the skips for passes.

**This disposition is deliberate (#1420 §3), not an oversight.** The project
considered the self-hosted-runner alternative and rejected it. `ci.yml` is
`on: pull_request` and runs `uv run pytest tests/`, so a fork PR executes its
own test files on whatever host runs them. A self-hosted runner on a public
repository is arbitrary code execution by any fork author.

The mitigations are all real: a one-shot non-privileged container, network
isolation, no repository secrets, and an approving label gate on fork PRs.
Together they still amount to a standing security commitment. The purpose of
that commitment is to move one quality signal earlier. The commitment is not
worth it at the size of this repository.

**The tier runs at the release cut, and nowhere else** (#1477, operator
ruling 2026-08-11). Step 7 of `docs/concepts/RELEASING.md` makes
`scripts/run_bench_gate.sh` a mandatory step. The output of that step goes
into the release PR. There is no cron job and no lab-side CI job. The private
repository has no CI at all, so "runs lab-side" was never true of anything.

A job built in the private repository today would exercise a minority of the
tier, because the corpus covers only some of the scaffolded modules. The
release cut is the moment when the project consumes the verdicts. A checklist
step that blocks the cut is the one schedule that cannot stop without notice.

Read the summary block of the tier, not the pass count. The block separates
three states:

- Tests that **executed** against the corpus.
- Tests that skipped because a named corpus **module** is missing or empty.
- The whole tier skipped for want of a corpus root.

Only the first state is a verdict. A run that reports "N passed" while most
modules skipped is the normal case today. The block states which tests are in
which state.

If you change retrieval ranking, compression, or clustering behaviour, say so
in the PR body. Expect the quality evidence to come from a corpus-bearing run
rather than from a green CI run.

### Changelog entries — one file per entry

**Add a file under `CHANGELOG/unreleased/`. Do not edit the
`[Unreleased]` block of `CHANGELOG/v4.md`.**

```
CHANGELOG/unreleased/<issue>-<slug>.md
```

```markdown
### Fixed

- **One-line title ([#1475](https://github.com/robotrocketscience/aelfrice/issues/1475)).** Body prose.
```

Put exactly one `### <Category>` heading and exactly one top-level `- `
bullet in each file. The category is one of `Added`, `Changed`,
`Deprecated`, `Removed`, `Fixed`, `Security`, `Performance`,
`Documentation`, `Build`, `CI`, `Dependencies`, `Internal`, `Reverted`,
`Notes`. This set is the `CATEGORIES` list in
`scripts/collate_changelog.py`. It is exactly the set that the committed
changelogs already use. Collation preserves indented continuation
paragraphs under the bullet verbatim. `scripts/collate_changelog.py`
refuses a file that breaks either rule, and it does not guess.

The directory is flat, and it holds nothing else. A different suffix
(`.txt`, `.markdown`, an uppercase `.MD`) is an error. An extensionless
file or a subdirectory is an error as well. The error names the path.
Collation, `scripts/check_changelog_dupes.py` and `release-docs-check`
all report this error. All three refuse the file, and none of them skips
it. A file that collation does not collect is a file that the release
omits without any message.

The reason (#1475): entries are single lines of 2,000-4,500 characters.
Thirteen of fourteen open PRs were inserting them into the same
eight-line region. Every merge then forced a hand resolution on every
remaining PR. That resolution works on two 4 KB lines. A line has no
granularity inside it, so the resolution can drop an entry and leave no
trace in the diff. Two branches that add files at distinct paths never
conflict.

**Transition.** The `[Unreleased]` block is still valid. Collation emits
the block first and the files second. A PR that already edits that block
does not need a rebase onto this convention before a merge. Write new
entries as files.

**At release time**, `scripts/collate_changelog.py` folds the block and
the files into the dated section of `CHANGELOG/v<major>.md`. It then
empties the directory — see
[docs/concepts/RELEASING.md](docs/concepts/RELEASING.md).
`release-docs-check` fails a release PR that leaves the block or the
directory undrained. `scripts/check_changelog_dupes.py` compares the
entry files against each other and against the block. It therefore still
catches two PRs that restate the same fix in two files.

### Documentation style

The documentation uses ASD-STE100, which is Simplified Technical English.
Apply these rules when you write or change a documentation file:

- Write one idea in each sentence.
- Write in the active voice, and name the actor.
- Use the imperative for an instruction.
- Use no idiom and no metaphor.
- Use one term for one concept. Use the term that the code uses.
- Spell out an acronym at its first use in the file.
- Write instructions of 20 words or fewer, and descriptions of 25 words or
  fewer.

Two rules protect the meaning, and they matter more than the rules above:

- **Keep the modal verbs.** `should`, `may`, `can`, `would` and `tends to` are
  not idiom. A hedge states how certain the author is, so it is a fact. Do not
  delete a hedge to shorten a sentence, and do not promote `should` to `must`.
- **Keep the connective that carries the argument.** A short sentence is
  worth less than a correct one. When you split `X, because Y` into two
  sentences, the reason disappears and the paragraph states two unrelated
  facts. A two-clause causal sentence is better than that.

`scripts/check_doc_preservation.py` verifies a rewrite of a documentation
file. It checks that the rewrite lost no number, no link, no code block, no
inline code span, no table row and no section. It also checks that the
rewrite broke no in-document link. The check is one-directional: an addition
is allowed, and the script reports it as a note.

The script cannot see the two rules above. It compares tokens, and a rewrite
that turns `should` into `must` keeps every token. A person must read the
change.

### Commit-message prefix enforcement

`scripts/check-commit-msg.py` validates every commit subject. The subject must
start with an allowed conventional-commit prefix (`feat:`, `fix:` and the
others, with an optional scope and an optional `!`).  Install the local hook
one time after you clone the repository:

```bash
sh scripts/setup-hooks.sh
```

This command sets `core.hooksPath = .githooks`. The `commit-msg` hook then
runs automatically.  Do not use `--no-verify`.

**CI also enforces the prefixes.** The `commit-msg-prefix` job in
`.github/workflows/staging-gate.yml` checks every commit in the PR range.
The job fails if any commit has an invalid prefix.  The `Merge ` and
`Revert ` subjects that git generates are exempt.

### Pre-push branch-freshness check

`.githooks/pre-push` stops a push when the merge-base of the branch with
`origin/main` is older than the freshness threshold. The default threshold
is 4 hours. The hook catches the drift pattern of parallel sessions: a
feature branch stays unmerged long enough that PRs land against a stale
baseline. The same `scripts/setup-hooks.sh` step above wires the hook in.

You can configure the threshold in two ways:

- `AELF_PRE_PUSH_FRESHNESS_HOURS=24 git push ...` — one shot for a single invocation.
- `git config aelfrice.prepushFreshnessHours 24` — the repo-local default.

To bypass the check for a one-off emergency push, run this command:

```bash
ALLOW_STALE_BRANCH_PUSH=1 git push ...
```

The override emits a warning to stderr, so the transcript shows the
divergence. The hook always allows a push to `main` itself. The hook always
allows a branch deletion.

### PR body requirements

The `pr-body-issue-link` CI job warns when a PR body contains no GitHub
auto-close keyword, for example `Closes #N`, `Fixes #N` or `Resolves #N`.
The job never blocks the PR.  The warning is advisory. It helps to keep the
traceability between an issue and a PR intact.

Your PR can legitimately have no associated issue. Examples are a dependency
bump, a release commit, and a refactor with no issue filed. In that case, add
the opt-out marker anywhere in the PR body:

```html
<!-- no-issue -->
```

The marker stops the warning. It requires no fake issue link.

### Merging — the `ready-to-merge` label

`main` accepts only a fast-forward (FF) push. `main` also requires a
signature. To get a PR onto `main`, do these steps:

1. Open the PR. Let CI run.
2. When CI is green, and when you or a reviewer are satisfied, add the
   `ready-to-merge` label to the PR.

The `merge-train` workflow (`.github/workflows/merge-train.yml`)
serializes the merges. It picks up the labeled PRs one at a time. For each
PR, it verifies that the branch is a fast-forward on current `main`. It also
verifies that all commits are signed. It then waits for the gating checks to
complete. Finally it makes an FF push to `main`. The concurrency of the
workflow is 1, so no two merges race.

"Gating" is wider than the five contexts that branch protection marks
*required*. `scripts/merge_train_gate.py` blocks on any failing check-run
that is not advisory. The script also has a presence floor. The presence
floor is the set of checks that must have *reported at all*, not merely
not-failed. That floor is the required set plus every check that a
`pull_request` workflow with no `paths:` filter emits (#1458).

A head that never ran `migration-policy-check`, `typos` or `bench-smoke`
therefore does not merge, even though none of those three is required.
Path-filtered workflows stay outside the floor by design. A docs-only PR
never runs `e2e` or `CodeQL`, and a floor on those two would block such a PR
forever.

If the bot rejects the push, it removes the label. It also posts a comment
that explains why. The most common cause is "branch is not fast-forward",
because another PR merged while yours was queued. Rebase locally with
`git rebase github/main`. Force-push the branch. Add the label again.

The bot has no signing key, so it cannot rebase on your behalf. See
`.github/workflows/flag-stale-open-prs.yml` for the original rationale for no
auto-rebase (#341). Authors rebase. The bot only FFs.

After a successful FF the bot dispatches the post-merge workflows itself
(#1423). The bot has no alternative. It makes the FF push with
`secrets.GITHUB_TOKEN`, and GitHub raises no workflow runs from events made
with that token. `on: push: branches: [main]` therefore stopped firing on the
day when the `merge-train` workflow became the merge mechanism.

`release-drafter` and `flag-stale-open-prs` have no other trigger, so they
stopped running at all. `workflow_dispatch` is one of the two documented
exceptions to that guard.

`scripts/push_trigger_workflows.py` derives the dispatch list from the
workflow files. The train therefore picks up a new `push: [main]` workflow
without an edit to the train. A new workflow does need its own
`workflow_dispatch:` trigger to be dispatchable. Without that trigger,
`gh workflow run` returns a 422 error. `tests/test_merge_train_dispatch.py`
fails if a `push: [main]` workflow has not got one.

A dispatch failure only warns. It never fails a merge that has already
landed. `.github/workflows/push-trigger-heartbeat.yml` opens an issue if
`main` outruns any of those workflows by more than 14 days.

The PR-size soft-cap (`.github/workflows/pr-size-soft-cap.yml`) posts an
advisory comment on a PR over 200 LOC or 3 files. A smaller PR is less likely
to lose the FF race. Apply `size:override` for a legitimate large diff, such
as a refactor, a removal, or generated code.

Both workflows shipped as part of #602.

### When a check never reports — how to re-run it by hand

Occasionally a required check never appears on a PR. The causes are:

- GitHub throttles webhook delivery during an incident.
- Somebody deletes a run before a re-run can start.
- An `on:` block stops matching.

The PR is then not mergeable. A push does not repair the PR either, because
the push is exactly the event that GitHub does not deliver.

`ci.yml` and `staging-gate.yml` accept `workflow_dispatch`. Between them
these two workflows carry all five required contexts: `pytest (3.12)`,
`pytest (3.13)`, `secrets-scan`, `pattern-scan` and `history-scan`. A
dispatch goes over the REST API rather than the webhook path. A dispatch
therefore still works when delivery is degraded:

```sh
gh workflow run ci.yml          --repo robotrocketscience/aelfrice --ref <your-branch>
gh workflow run staging-gate.yml --repo robotrocketscience/aelfrice --ref <your-branch>
gh run list --repo robotrocketscience/aelfrice --workflow ci.yml --limit 3
```

Three properties are worth your attention. The third one is a caveat, not a
feature.

- **A dispatch cannot report against a commit it did not test.** These
  workflows deliberately have no `ref` *input*. The check-runs of a run
  attach to the head SHA of the ref of the dispatch. Branch protection and
  `merge-train` both evaluate the checks on the head SHA of the PR. For the
  same reason, `actions/checkout` in these two workflows must likewise never
  pin a `ref:`. A test enforces both rules.
- **A dispatched `ci.yml` always runs the full suite.** The
  `dorny/paths-filter` short-circuit is `pull_request`-only. A dispatch has
  no diff base, and a job that skips must never report a pass that looks like
  a run (#1160).

  For a related reason, no job in either workflow may be guarded to
  `pull_request` only. A guarded job still emits a check-run with the
  conclusion `skipped` and a *later* `started_at`.
  `merge_train_gate.latest_per_name` keeps the newest row per name, and
  `skipped` is not a failing conclusion. Such a row would overwrite an
  earlier real `failure` and clear a red gate. The two jobs that genuinely
  cannot run outside a pull request live in `pr-metadata.yml`, which has no
  `workflow_dispatch`.
- **⚠️ A dispatch of these two does *not* mean the PR is safe to label.**
  The two workflows produce the five *required* contexts. The presence floor
  of `merge-train` (`missing`, #1435) is computed over the required set only.
  Every other gating check is evaluated by an *absence test*, which an absent
  check satisfies. A head that carries only the dispatched rows therefore
  evaluates as green while the other checks never ran. Before you label the
  PR, list what is actually on the head SHA:

  ```sh
  gh api repos/robotrocketscience/aelfrice/commits/<head-sha>/check-runs \
      --jq '[.check_runs[] | {n: .name, c: .conclusion}]'
  ```

  Compare the output against the *check-run* names. A check-run name is a job
  name and not a workflow name. The two differ for most of these workflows,
  so a list of workflow names would never match anything:

  | Workflow | Check-run name(s) |
  | --- | --- |
  | `migration-policy-check.yml` | `migration-policy-check` |
  | `typos.yml` | `typos` |
  | `bench-smoke.yml` | `bench-smoke` |
  | `deadcode.yml` | `deptry`, `vulture` |
  | `e2e.yml` | `e2e (pipx)`, `e2e (uv-tool)`, `e2e (venv-pip)` |
  | `codeql.yml` | `analyze (actions)`, `analyze (python)` |
  | `eval-calibration.yml` | `calibration` |
  | `windows-smoke.yml` | `smoke` |
  | `zizmor.yml` | `zizmor` |
  | `ci.yml` | `pytest (3.12)`, `pytest (3.13)` |
  | `staging-gate.yml` | `secrets-scan`, `pattern-scan`, `history-scan`, `commit-msg-prefix`, `release-docs-check` |
  | `pr-metadata.yml` | `pr-title-prefix`, `pr-body-issue-link` |
  | `pr-size-soft-cap.yml` | `size-check` |
  | `replay-soak-gate.yml` | `consecutive-green` |
  | `e2e.yml` | also `surface-failure` |
  | `auto-add-to-board.yml` | `add-to-board` |
  | `merge-train.yml` | `merge` |

  **Check the `pr-title-prefix` and `pr-body-issue-link` rows first after a
  dispatch-only recovery.** They live in `pr-metadata.yml`, which
  deliberately has no `workflow_dispatch`. They are therefore exactly the
  gating checks a dispatch cannot produce. An absence test evaluates their
  absence, and an absent check satisfies that test. If they are missing, the
  head is not safe to label, however green the dispatched rows look.

  Not all of them belong on every head, so an absent row is not automatically
  a problem. `windows-smoke.yml`, `eval-calibration.yml`, `e2e.yml` and
  `zizmor.yml` carry workflow-level `paths:` filters. `deadcode.yml` and
  `codeql.yml` carry `paths-ignore:`. Those check-runs are legitimately
  missing on a head that touches none of their paths.

  `smoke`, for instance, only appears when the PR touches `src/**`,
  `tests/test_windows_portability_1329.py`, `pyproject.toml` or its own
  workflow file. Read the `on:` block before you conclude that a row should
  have been there. Expect one extra name: code scanning posts its own
  `CodeQL` check-run from the `github-advanced-security` app alongside the
  two `analyze (…)` jobs.

  The work to widen the presence floor to cover the whole gating set is
  tracked separately. This section is the interim instruction, not the fix.

Shipped as part of #1436.

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). The short version: be respectful, focus on the work, and do not harass anybody.

## Security

See [SECURITY.md](SECURITY.md). The project treats a privacy bug as a security bug.
