# Contributing

Thanks for your interest in contributing. aelfrice has one author, and the bar
for a change is this: is the system better afterward, in a way that tests can
defend?

## Status

For the current line, read [the changelog](CHANGELOG.md), which gives the
latest release. For the landing record, read
[the roadmap](docs/concepts/ROADMAP.md). Issues are welcome, and the author
evaluates each pull request (PR) case by case. The bar is that the
change moves the system measurably forward and justifies itself with a test.

The best categories of PR are:

- A bug fix with a regression test that fails before the fix and passes after
  it.
- A documentation fix: a typo, a broken link, or a claim that is stale against
  the current code.
- A change that closes one of the [known
  limitations](docs/user/LIMITATIONS.md). Open an issue first and agree on the
  approach.

These changes are hard to land without agreement in advance:

- A new CLI subcommand.
- A change to the schema.
- A change that adds a hard dependency.
- Bringing back an earlier research-line feature. A benchmark or an experiment
  must show the impact first.

## How to file a useful issue

Write a one-line title in lowercase, with no period at the end.

```
search: locked-belief order is unstable when budget is exhausted
```

Include these items in the body:

- **What happened.** The exact CLI invocation and the exact output.
- **What you expected.** One line.
- **Environment.** The operating system, the Python version
  (`python --version`), the aelfrice version, and, if it's relevant, the host
  agent.
- **A minimal reproduction.** A directory where `aelf onboard <here>`
  reproduces the problem, or the smallest sequence of CLI calls that causes it.

Do not include your real `memory.db` file. That file is at
`<repo>/.git/aelfrice/memory.db`, or at `~/.aelfrice/memory.db` for a directory
that isn't a git repository, and it contains your private beliefs. Reproduce
the problem on a scratch database (`AELFRICE_DB=/tmp/scratch.db`) and share the
scratch database instead.

## Triage labels

The issue tracker uses a small vocabulary of labels. The in-repo
`aelf gate list` aggregator (`src/aelfrice/gate_list.py`) reads `gate:ratify` /
`gate:prereq` / `bench-gated` / `gate:license` through `gh`. Operator-side scan
tooling reads the remaining labels, and this repository doesn't ship that scan
tooling. The scan tooling determines whether an open issue is ready for a claim,
or whether to hide the issue from the active queue. When an issue is not
immediately actionable, apply one or more of these labels as you file it:

- **`gate:operator`** — an operator decision or operator-side data must
  arrive before the issue can move. For example, a tracker that opens once
  enough telemetry has accumulated to baseline against (#749, #488).
- **`gate:prereq`** — blocked on another tracked work item, such as a
  sub-task, an upstream dependency, or a framework that must land first.
- **`gate:lab-corpus`** — blocked on lab-side corpus delivery. The
  public-tree work cannot exercise its acceptance criteria until the corpus
  is committed.
- **`gate:ratify`** — the issue needs ratification of a design decision
  before implementation begins.
- **`gate:umbrella`** — an umbrella issue that coordinates sub-issues. It
  has no implementation surface of its own, and it closes when its children
  close.
- **`bench-gated`** — the implementation has shipped, and the only
  outstanding work is a benchmark run. The result of the run decides whether
  to flip a default, ship a tuning change, or revert (#769, #697, #491).
- **`attn:decisions-needed`** — the operator must adjudicate something
  before the issue can move. This label differs from `gate:operator` in that
  the operator has all the information but hasn't yet made the choice.

An issue that carries one of these labels appears in the scanner inventory
output, but the scan tooling excludes it from the "next actionable" list.
Applying the right label as you file the issue keeps every fresh scan from
re-evaluating it.

## Where to look for work

The [roadmap](docs/concepts/ROADMAP.md) carries the version-by-version landing
record and the current active line. The [list of known
limitations](docs/user/LIMITATIONS.md) lists the gaps against current HEAD. The
issue tracker is the canonical source of the work in flight.

The contributions with the largest effect tend to land in three places:

- **Bench gates.** Several `bench-gated` issues wait on benchmark runs, and
  each run decides whether to flip a default or to revert (see
  `gh issue list --label bench-gated`).
- **Triage drift.** Some issues carry the `Blocked` label even though their
  blockers have since closed. Surface the status flip instead of an
  implementation.
- **Stale-documentation fixes.** Fix any place where the documentation makes
  a false statement about the code at current HEAD. A documentation audit lives
  in [the audits directory](docs/audits/), and the latest pass lists what is
  outstanding.

## What the project will not do

- Vector embeddings or approximate nearest-neighbor (ANN) search in retrieval.
  Both would require a hard dependency on a vector library, and such a
  dependency defeats the local-stdlib design.
- Cloud sync, accounts, or any data path that is not local.
- A web user interface.
- Integration with chat platforms.

## Development setup

```bash
git clone https://github.com/robotrocketscience/aelfrice.git
cd aelfrice
uv sync --all-groups
uv run pytest tests/ -x -q
uv run python scripts/check_pyright_baseline.py
```

Conventions:

- Use a conventional-commit prefix: `feat:`, `fix:`, `perf:`, `refactor:`, `test:`, `docs:`, `build:`, `ci:`, `style:`, `revert:`, `exp:`, `chore:`, `release:`, `gate:`, `audit:`.
- Make atomic commits. Each commit moves the tree from one tested green state to the next.
- Every change of behavior needs a test.
- **`pyright --strict` does not pass, and no file may get worse.** `pyright src/` reports 987 errors over 76 files. `scripts/check_pyright_baseline.py` holds a per-file baseline, and the `pyright-ratchet` workflow fails a pull request that raises any file's count. Drive a file down and regenerate the baseline with `--update`, then commit the lower numbers with the fix. The project deliberately avoids a repo-wide total, because a total lets a fix in one module pay for a regression in another. `tests/` is not gated: `pyproject.toml` includes it, which puts the count at 6,938, and freezing that today would put every test edit in conflict with the ratchet.

### Your local test budgets are 4 times the CI budgets

The suite multiplies every wall-clock timeout at collection by
`AELF_TEST_TIMEOUT_SCALE`, and that variable defaults to **4** on your machine.
Every workflow that runs pytest pins the variable to **1** (#1472). A loaded
laptop was failing tests that had no defect, and `aelf-pr-open.sh` runs pytest
with `-x`, so the first such failure ended the run.

As a result, **a timing failure that you cannot reproduce locally
is expected**, because CI runs the same test on a quarter of your budget. To
reproduce CI exactly, run this command:

```bash
AELF_TEST_TIMEOUT_SCALE=1 uv run pytest tests/ -x -q
```

A malformed value falls back to the default and raises no error. Malformed
means non-numeric, zero, negative, or absurdly large. Zero is the value that
matters, because `pytest-timeout` reads a timeout of 0 as *disabled*. A scale
of zero that multiplied through would remove every budget in the suite without
a message, and every test would still pass.

### The bench-gate tier does not run here

A green `pytest tests/` run on this repository does **not** mean the quality
gates passed. It means the run skipped them.

The quality gates for retrieval, compression, and clustering are in
`tests/bench_gate/`. These tests carry the `bench_gated` marker, and they skip
unless `AELFRICE_CORPUS_ROOT` points at a labeled evaluation corpus. That
corpus is private, and this repository is public, so the whole tier skips on
every public CI run. The run prints a `bench-gate tier` summary line with the
number of skips, so nobody can mistake the skips for passes.

**This disposition is deliberate (#1420 §3), not an oversight.** The project
considered the self-hosted-runner alternative and rejected it. `ci.yml` is
`on: pull_request` and runs `uv run pytest tests/`, so a fork PR executes its
own test files on whatever host runs them. A self-hosted runner on a public
repository is arbitrary code execution by any fork author.

The mitigations are all real: a one-shot non-privileged container, network
isolation, no repository secrets, and an approving label gate on fork PRs.
Together they still amount to a standing security commitment, and the purpose
of that commitment is to move one quality signal earlier. At the size of this
repository, the commitment is not worth it.

**The tier runs at the release cut, and nowhere else** (#1477, operator
ruling 2026-08-11). Step 7 of `docs/concepts/RELEASING.md` makes
`scripts/run_bench_gate.sh` a mandatory step, and the output of that step goes
into the release PR. There is no cron job and no lab-side CI job. The private
repository has no CI at all, so "runs lab-side" was never true of anything.

A job built in the private repository today would exercise a minority of the
tier, because the corpus covers only some of the scaffolded modules. The
release cut is the moment when the project consumes the verdicts, and a
checklist step that blocks the cut is the one schedule that cannot stop
without notice.

Read the summary block of the tier, not the pass count. The block separates
three states:

- Tests that **executed** against the corpus.
- Tests that skipped because a named corpus **module** is missing or empty.
- The whole tier skipped for want of a corpus root.

Only the first state is a verdict. A run that reports "N passed" while most
modules skipped is the normal case today, and the block states which tests are
in which state.

If you change retrieval ranking, compression, or clustering behavior, say so
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
`scripts/collate_changelog.py`, and it is exactly the set that the committed
changelogs already use. Collation preserves indented continuation
paragraphs under the bullet verbatim. `scripts/collate_changelog.py`
refuses a file that breaks either rule, and it does not guess.

The directory is flat, and it holds nothing else. A different suffix
(`.txt`, `.markdown`, an uppercase `.MD`) is an error, and so is an
extensionless file or a subdirectory. The error names the path.
Collation, `scripts/check_changelog_dupes.py`, and `release-docs-check`
all report this error. All three refuse the file, and none of them skips
it. A file that collation does not collect is a file that the release
omits without any message.

The reason (#1475): entries are single lines of 2,000-4,500 characters.
Thirteen of fourteen open PRs were inserting them into the same
eight-line region, and every merge then forced a hand resolution on every
remaining PR. That resolution works on two 4 KB lines. A line has no
granularity inside it, so the resolution can drop an entry and leave no
trace in the diff. Two branches that add files at distinct paths never
conflict.

**Transition.** The `[Unreleased]` block is still valid. Collation emits
the block first and the files second, so a PR that already edits that block
does not need a rebase onto this convention before a merge. Write new
entries as files.

**At release time**, `scripts/collate_changelog.py` folds the block and
the files into the dated section of `CHANGELOG/v<major>.md`. It then
empties the directory; for the full release sequence, read
[the release procedure](docs/concepts/RELEASING.md).
`release-docs-check` fails a release PR that leaves the block or the
directory undrained. `scripts/check_changelog_dupes.py` compares the
entry files against each other and against the block, so it still
catches two PRs that restate the same fix in two files.

### Documentation style

The documentation uses ASD-STE100, which is Simplified Technical English.
Apply these rules when you write or change a documentation file:

- Write one idea in each sentence.
- Write in the active voice, and name the actor.
- Use the imperative for an instruction.
- Use no idiom and no metaphor.
- Use one term for one concept, and use the term that the code uses.
- Spell out an acronym at its first use in the file.
- Write instructions of 20 words or fewer, and descriptions of 25 words or
  fewer.

Two rules protect the meaning, and they matter more than the rules above:

- **Keep the modal verbs.** `should`, `may`, `can`, `would`, and `tends to`
  are not idiom. A hedge states how certain the author is, so it is a fact. Do
  not delete a hedge to shorten a sentence, and do not promote `should` to
  `must`.
- **Keep the connective that carries the argument.** A short sentence is
  worth less than a correct one. When you split `X, because Y` into two
  sentences, the reason disappears, and the paragraph states two unrelated
  facts. A two-clause causal sentence is better than that.

`scripts/check_doc_preservation.py` verifies a rewrite of a documentation
file. It checks that the rewrite lost no number, no link, no code block, no
inline code span, no table row, and no section. It also checks that the
rewrite broke no in-document link. The check is one-directional: an addition
is allowed, and the script reports it as a note.

The script cannot check the two rules above. It compares tokens, so a rewrite
that turns `should` into `must` keeps every token. A person must read the
change.

### Commit-message prefix enforcement

`scripts/check-commit-msg.py` validates every commit subject. The subject must
start with an allowed conventional-commit prefix (`feat:`, `fix:`, and the
others, with an optional scope and an optional `!`). After you clone the
repository, install the local hook one time:

```bash
sh scripts/setup-hooks.sh
```

This command sets `core.hooksPath = .githooks`, and the `commit-msg` hook then
runs automatically. Do not use `--no-verify`.

**CI also enforces the prefixes.** The `commit-msg-prefix` job in
`.github/workflows/staging-gate.yml` checks every commit in the PR range.
If any commit has an invalid prefix, the job fails. The `Merge ` and
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
divergence. The hook always allows a push to `main` itself, and it always
allows a branch deletion.

### PR body requirements

The `pr-body-issue-link` CI job warns when a PR body contains no GitHub
auto-close keyword, for example `Closes #N`, `Fixes #N`, or `Resolves #N`.
The job never blocks the PR. The warning is advisory, and it helps keep the
traceability between an issue and a PR intact.

Your PR can legitimately have no associated issue, for example a dependency
bump, a release commit, or a refactor with no issue filed. In that case, add
the opt-out marker anywhere in the PR body:

```html
<!-- no-issue -->
```

The marker stops the warning, and it requires no fake issue link.

### Merging — the `ready-to-merge` label

`main` accepts only a fast-forward (FF) push, and `main` also requires a
signature. To get a PR onto `main`, follow these steps:

1. Open the PR. Let CI run.
2. When CI is green, and when you or a reviewer are satisfied, add the
   `ready-to-merge` label to the PR.

The `merge-train` workflow (`.github/workflows/merge-train.yml`)
serializes the merges, picking up the labeled PRs one at a time. For each PR,
it verifies that the branch is a fast-forward on current `main` and that all
commits are signed, waits for the gating checks to complete, and then makes an
FF push to `main`. The concurrency of the workflow is 1, so no two merges race.

"Gating" is wider than the five contexts that branch protection marks
*required*. `scripts/merge_train_gate.py` blocks on any failing check-run
that is not advisory. The script also has a presence floor: the set of checks
that must have *reported at all*, not merely not-failed. That floor is the
required set plus every check that a `pull_request` workflow with no `paths:`
filter emits (#1458).

A head that never ran `migration-policy-check`, `typos`, or `bench-smoke`
therefore does not merge, even though none of those three is required.
Path-filtered workflows stay outside the floor by design. A docs-only PR
never runs `e2e` or `CodeQL`, and a floor on those two would block such a PR
forever.

If the bot rejects the push, it removes the label and posts a comment that
explains why. The most common cause is "branch is not fast-forward", because
another PR merged while yours was queued. To recover:

1. Rebase locally with `git rebase github/main`.
2. Force-push the branch.
3. Add the label again.

The bot has no signing key, so it cannot rebase on your behalf. For the
original rationale for no auto-rebase (#341), read
`.github/workflows/flag-stale-open-prs.yml`. Authors rebase, and the bot only
FFs.

After a successful FF the bot dispatches the post-merge workflows itself
(#1423), and it has no alternative. It makes the FF push with
`secrets.GITHUB_TOKEN`, and GitHub raises no workflow runs from events made
with that token. `on: push: branches: [main]` therefore stopped firing on the
day when the `merge-train` workflow became the merge mechanism.

`release-drafter` and `flag-stale-open-prs` have no other trigger, so they
stopped running at all. `workflow_dispatch` is one of the two documented
exceptions to that guard.

`scripts/push_trigger_workflows.py` derives the dispatch list from the
workflow files, so the train picks up a new `push: [main]` workflow
without an edit to the train. A new workflow does need its own
`workflow_dispatch:` trigger to be dispatchable. Without that trigger,
`gh workflow run` returns a 422 error, and
`tests/test_merge_train_dispatch.py` fails if a `push: [main]` workflow does
not have one.

A dispatch failure only warns. It never fails a merge that has already
landed. If `main` outruns any of those workflows by more than 14 days,
`.github/workflows/push-trigger-heartbeat.yml` opens an issue.

The PR-size soft-cap (`.github/workflows/pr-size-soft-cap.yml`) posts an
advisory comment on a PR over 200 lines of code or 3 files. A smaller PR is
less likely to lose the FF race. Apply `size:override` for a legitimate large
diff, such as a refactor, a removal, or generated code.

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
`pytest (3.13)`, `secrets-scan`, `pattern-scan`, and `history-scan`. A
dispatch goes over the REST API rather than the webhook path, so a dispatch
still works when delivery is degraded:

```sh
gh workflow run ci.yml          --repo robotrocketscience/aelfrice --ref <your-branch>
gh workflow run staging-gate.yml --repo robotrocketscience/aelfrice --ref <your-branch>
gh run list --repo robotrocketscience/aelfrice --workflow ci.yml --limit 3
```

Three properties are worth your attention. The third one is a caveat, not a
feature.

- **A dispatch cannot report against a commit it did not test.** These
  workflows deliberately have no `ref` *input*. The check-runs of a run
  attach to the head SHA of the ref of the dispatch, while branch protection
  and `merge-train` both evaluate the checks on the head SHA of the PR. For the
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
  `skipped` is not a failing conclusion, so such a row would overwrite an
  earlier real `failure` and clear a red gate. The two jobs that genuinely
  cannot run outside a pull request live in `pr-metadata.yml`, which has no
  `workflow_dispatch`.
- **⚠️ A dispatch of these two does *not* mean the PR is safe to label.**
  The two workflows produce the five *required* contexts. The presence floor
  of `merge-train` (`missing`, #1435) covers the required set only. An
  *absence test* evaluates every other gating check, and an absent check
  satisfies that test. A head that carries only the dispatched rows therefore
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

  **After a dispatch-only recovery, check the `pr-title-prefix` and
  `pr-body-issue-link` rows first.** They live in `pr-metadata.yml`, which
  deliberately has no `workflow_dispatch`. They are therefore exactly the
  gating checks a dispatch cannot produce. An absence test evaluates their
  absence, and an absent check satisfies that test. If they are missing, the
  head is not safe to label, however green the dispatched rows look.

  Not all of them belong on every head, so an absent row is not automatically
  a problem. `windows-smoke.yml`, `eval-calibration.yml`, `e2e.yml`, and
  `zizmor.yml` carry workflow-level `paths:` filters, and `deadcode.yml` and
  `codeql.yml` carry `paths-ignore:`. Those check-runs are legitimately
  missing on a head that touches none of their paths.

  `smoke`, for instance, only appears when the PR touches `src/**`,
  `tests/test_windows_portability_1329.py`, `pyproject.toml`, or its own
  workflow file. Read the `on:` block before you conclude that a row should
  have been there. Expect one extra name: code scanning posts its own
  `CodeQL` check-run from the `github-advanced-security` app alongside the
  two `analyze (…)` jobs.

  The work to widen the presence floor to cover the whole gating set is
  tracked separately. This section is the interim instruction, not the fix.

Shipped as part of #1436.

## Code of Conduct

Read [the code of conduct](CODE_OF_CONDUCT.md). The short version: be
respectful, focus on the work, and do not harass anybody.

## Security

Read [the security policy](SECURITY.md). The project treats a privacy bug as a
security bug.
