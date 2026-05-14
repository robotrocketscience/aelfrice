# Spec: replay-soak-bot push path — dedicated branch

Spec for issue [#461](https://github.com/robotrocketscience/aelfrice/issues/461). The replay-soak cron landed in [#403 deliverable A](https://github.com/robotrocketscience/aelfrice/issues/403) cannot write its status entry; the PR-level gate (`replay-soak / consecutive-green ≥ 7d`, [#403 C](https://github.com/robotrocketscience/aelfrice/issues/403)) therefore blocks every #264-touching merge with `streak < 7`. This memo records the design decision and implementation plan.

## Live-ruleset facts (verified 2026-05-06)

Single ruleset on `robotrocketscience/aelfrice`, scoped to `refs/heads/main` only:

- `deletion`, `non_fast_forward`, `required_linear_history`
- `required_signatures`
- `pull_request` — `required_approving_review_count: 0`, `allowed_merge_methods: ["rebase"]`
- `required_status_checks`: `secrets-scan`, `pattern-scan`, `history-scan`, `pytest (3.12)`, `pytest (3.13)`
- `current_user_can_bypass: never`

Two of these block the cron's current `git push origin HEAD:main`:
1. `required_signatures` — workflow token's commit is unsigned;
2. `pull_request` — direct push to `main` is forbidden regardless of signing.

The `pull_request` rule is the harder constraint: even a deploy-key-signed bot commit would still be rejected. **Option 1 in the issue body ("sign the bot commit, push direct") does not work standalone** against the live ruleset — it would have to be paired with a bypass actor for the `pull_request` rule, at which point it has collapsed into option 4.

## Decision: option 3, dedicated branch (`refs/heads/replay-soak-status`)

### Why

The ruleset's `conditions.ref_name.include` is `["refs/heads/main"]`. **Any other branch is unconstrained.** A long-lived branch dedicated to soak-status updates therefore needs no signing key, no bypass actor, and no PR machinery — the cron pushes straight to it as the workflow token's default identity. The PR-level gate workflow already does a fresh checkout to read the file; pointing it at the dedicated branch's ref instead of `main` is a one-line change.

The issue body's stated concern about option 3 — "artifact retention windows apply (90 days default) — would need a long-retention bucket or a dedicated branch" — applies only to the *Actions-artifacts* variant. **A dedicated branch has no retention window.** Branch tips persist indefinitely; the audit trail is git-native (a real, fetchable, blameable history of soak entries) and survives the same way `main` does.

### Why not the alternatives

| Option | Verdict | Reason |
|---|---|---|
| **1.** Sign + push direct to main | reject | `pull_request` rule blocks regardless of signature; only viable paired with option 4 |
| **2.** PR-per-soak with auto-merge | reject | Adds ~365 PRs/year (one open + one merge + one close every day) plus ~5 min × 365 = ~30 hr/year of CI on the full pytest matrix per soak. The status-file delta is one JSONL row; the audit trail it produces is the same regardless of whether the commit lands via PR or direct branch push. The PR machinery is pure overhead here. |
| **3.** Dedicated branch | **pick** | Smallest workflow diff; no new long-lived signing key on a public repo; no bypass-roster expansion; git-native audit trail; no retention window. |
| **4.** Ruleset bypass for `aelfrice-soak-bot` | reject | Expands the public repo's bypass roster, which the project has been deliberate about keeping narrow. A compromised bot credential would land unreviewed code on `main`. The benefit (smallest workflow diff) is marginal vs. option 3, and the security-surface cost is real. |

The #403 design recommendation said the soak file should be "git-native, doesn't depend on artifact retention windows, and the soak history becomes part of the audit trail." A dedicated branch satisfies all three; that recommendation does not require the file to live on `main` specifically.

## Implementation plan

### Files changed

1. **`.github/workflows/replay-soak.yml`** — push to `replay-soak-status` instead of `main`. Use `git worktree add` to keep the corpus + code checkout (from `main`) separate from the status-branch worktree (where the commit lands).

2. **`.github/workflows/replay-soak-gate.yml`** — change the `actions/checkout` step's `ref:` from `main` to `replay-soak-status`. Treat a missing branch as `streak = 0` (already the existing fall-through path when `.replay-soak-status.json` does not exist).

3. **One-time bootstrap** — push an empty orphan branch `replay-soak-status` to `github` containing only an empty `.replay-soak-status.json`. No ruleset rules apply; this is a normal `git push`. Done as part of this session before merging.

4. **(Cosmetic, in same PR)** — remove the 0-byte `.replay-soak-status.json` from `main`. The file's authoritative location is now the dedicated branch; leaving the stub on `main` invites future confusion.

### Workflow shape

```yaml
- uses: actions/checkout@…           # main checkout: corpus + code
  with: { fetch-depth: 0, persist-credentials: true }

- name: uv sync
  run: uv sync --frozen --group dev

- name: Fetch replay-soak-status branch into a side worktree
  run: |
    git fetch origin replay-soak-status:refs/remotes/origin/replay-soak-status
    git worktree add .soak-status-branch \
      -B replay-soak-status origin/replay-soak-status

- name: Run replay-soak runner
  id: soak
  run: |
    uv run python scripts/replay_soak_run.py \
      --status-file .soak-status-branch/.replay-soak-status.json
  continue-on-error: true

- name: Commit + push to replay-soak-status
  env:
    GIT_AUTHOR_NAME:     aelfrice-soak-bot
    GIT_AUTHOR_EMAIL:    aelfrice-soak-bot@users.noreply.github.com
    GIT_COMMITTER_NAME:  aelfrice-soak-bot
    GIT_COMMITTER_EMAIL: aelfrice-soak-bot@users.noreply.github.com
  run: |
    set -euo pipefail
    cd .soak-status-branch
    if git diff --quiet .replay-soak-status.json; then
      echo "no soak-status delta to commit"; exit 0
    fi
    git add .replay-soak-status.json
    git commit -m "audit(replay-soak): $(date -u +%Y-%m-%d) entry"
    git push origin HEAD:replay-soak-status
```

The commit on `replay-soak-status` is **not** signed (the rule does not apply to that ref). The audit trail is preserved by branch history; if signing-everything is desired later, a separate hardening pass can add a deploy key without changing the design.

### Gate workflow shape

```yaml
- uses: actions/checkout@…
  with:
    ref: replay-soak-status        # was: main
    fetch-depth: 1
    persist-credentials: false
```

Rest unchanged. The existing `if [ ! -f .replay-soak-status.json ]; then n=0` path covers the brief window between PR merge and the first cron run on the new branch.

## Acceptance trace

- [x] Path picked and rationale recorded.  *(this memo)*
- [ ] `replay-soak.yml` updated; next scheduled run (or a `workflow_dispatch`) lands an entry on `replay-soak-status`.
- [ ] `.replay-soak-status.json` on `replay-soak-status` accumulates ≥1 row.
- [ ] (After 7 daily runs) `scripts/replay_soak_streak.py --quiet` returns ≥7; `replay-soak / consecutive-green ≥ 7d` PR check passes on a no-op derivation-touching PR.
- [ ] #264 claim becomes unblocked.

## Risks and rollback

- **Branch deletion** — anyone with write access can delete `replay-soak-status`, which would reset the streak. The branch is not ruleset-protected. Mitigation: the cron's first commit after deletion re-creates an empty file and the streak rebuilds. If this becomes a real problem, add a separate ruleset entry that includes `refs/heads/replay-soak-status` with `deletion` blocked — independent of this PR.
- **Force-push** — same risk class. Same mitigation if needed.
- **Rollback this PR** — revert the two workflow files and (if it landed) the `main`-side stub deletion. Soak entries already on `replay-soak-status` remain valid history but stop being read by the gate; the streak reverts to 0 until the next on-`main` cron entry succeeds (which it cannot, by the same root cause).

## Out of scope

- Implementing #264. This issue ships the gate plumbing; #264 lands behind it.
- Changing the threshold from 7. The 2026-05-04 ratification on #403 set 7.
- Hardening signing on `replay-soak-status` (deferred — not required for the gate to function).
