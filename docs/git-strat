⏺ Public repo git strategy:

  Branching
  - Branch off main per feature. Conventional prefixes required: feat/,
  fix/, perf/, refactor/, test/, docs/, build/, ci/, style/, chore/,
  exp/.
  - One feature = one branch. If scope changes mid-flight, ask before
  expanding.

  Commits
  - Atomic, easy to follow. Conventional-commits prefix required (same
  allowlist as branches, plus revert:, release:, gate:, audit:).
  - No co-authorship lines. No large data/results files.
  - Pattern: N atomic terse commits → 1 gate: commit at end with
  verification / blockers / rollback for the batch.
  - SSH-signed with ~/.ssh/<ssh_id>.pub
   On fresh clone, re-set the three local config lines.
  - commit-msg hook + scripts/check-commit-msg.py enforces prefix.

  Pushing
  - Push to the github remote.
  - Pre-push hook at .git/hooks/pre-push blocks any content sourced from ($HOME)

  CI gates (.github/workflows/staging-gate.yml)
  - gitleaks, PII pattern scan (PII_PATTERNS_SECRET), commit-history
  audit, pytest matrix.

  Privacy boundary
  - Physical separation, not filtration. Only ~/projects/aelfrice ships
  to GitHub. Private notes/research live in ~/projects/lab
  (local-hosted only; lab pre-push hook rejects any github.com remote). No
  allowlist, no quarantine pipeline — the directory IS the boundary.

  Branch protection on GitHub (verify)
  - Required PR review
  - Required staging-gate checks,
  - Signed commits
  - Linear history
  - No force-push/delete.

  Releases
  1. Bump pyproject.toml + uv lock on release/vX.Y.Z branch, subject
  release: vX.Y.Z.
  2. PR to main; merge after staging-gate passes.
  3. git tag vX.Y.Z && git push github vX.Y.Z → publish.yml
  auto-publishes to PyPI via Trusted Publishing (one-time PyPI setup
  deferred to v1.0.0).

  Merging
  - Reconcile ROADMAP / STATE files on merge if they exist.
