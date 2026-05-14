# Releasing

How to cut a new version. Maintainer reference.

## Versioning

- `0.x.y` — milestones on the v1.0 path. Surface may change.
- `1.0.0` and after — semver in force.

## Cut a release

1. Branch `release/vX.Y.Z` off `main`.
2. Bump `pyproject.toml` `version`. (Single source of truth — no `__version__` in code.)
3. `uv lock`.
4. In `CHANGELOG/v<major>.md` (e.g. `CHANGELOG/v3.md` for any v3.x release), move `[Unreleased]` entries into `[X.Y.Z] — YYYY-MM-DD`. Add the compare-link footnote at the bottom of the same file. Top-level `CHANGELOG.md` is a thin index; do not edit it for routine releases. A new major (`vN+1.0.0`) needs a new `CHANGELOG/v<N+1>.md` and a row added to the index.
5. Update README roadmap status.
6. Run locally:
   ```bash
   uv run pytest tests/ -x -q     # track the actual count in CI
   uv run pyright src/             # strict
   uv run aelf --help              # spot-check CLI
   uv build                        # wheels build clean
   ```
7. Open PR `release: vX.Y.Z`. Body = CHANGELOG entries.
8. `staging-gate` must be green: gitleaks, PII pattern scan, history audit, pytest matrix.
9. Merge — linear history, no merge commits.

## Tag and publish

```bash
git switch main && git pull github main
git tag -s vX.Y.Z -m "release: vX.Y.Z"
git push github vX.Y.Z
```

Tags SSH-signed (`gpg.format = ssh`, key `~/.ssh/id_rrs`). Tag push triggers `.github/workflows/publish.yml`:

1. Build sdist + wheel.
2. Generate Sigstore attestation.
3. Upload to PyPI via [Trusted Publishing](https://docs.pypi.org/trusted-publishers/).

> PyPI publish gated until `v1.0.0`. Workflow wired but not invoked until then.

## Verify

```bash
uv tool install aelfrice==X.Y.Z
aelf --help
aelf stats
```

Clean tool-install in a scratch venv proves the wheel is functional. Compare CLI surface against the previous version to catch drift.

## Hotfixes

```bash
git switch -c release/vX.Y.Z+1 vX.Y.Z
# fix → bump → lock → CHANGELOG/vN.md → PR → gate → merge → tag
```

If `main` has moved on incompatibly, cherry-pick instead.

## Yank

```bash
gh release delete vX.Y.Z --yes
git push github :refs/tags/vX.Y.Z
git tag -d vX.Y.Z
# yank from PyPI manually via web UI
```

Then bump to `vX.Y.Z+1` with the fix.

## Pre-releases

`v0.9.0-rc1`, `v1.0.0-rc1`. PyPI treats `-rc` as pre-release — won't appear as default install candidate.

```bash
uv tool install --pre aelfrice==1.0.0rc1
```

## Branch protection (post-public-flip)

Free-tier GitHub doesn't expose branch-protection APIs while the repo is private. After v1.0 flip, enable on `main`:

- Required PR review (1 approving)
- Required checks: `staging-gate / secrets-scan`, `staging-gate / pii-scan`, `staging-gate / commit-history-audit`, `staging-gate / pytest`
- Required signed commits
- Required linear history
- Block force-push and deletion

Until then, enforced by convention + local pre-push hook.

## Sign keys

```
[gpg "ssh"]
    allowedSignersFile = ~/.ssh/allowed_signers
[gpg]
    format = ssh
[commit]
    gpgsign = true
[user]
    signingkey = ~/.ssh/id_rrs.pub
```

Same key registered as a signing key on GitHub. On a fresh clone, repeat these `git config --local` lines.
