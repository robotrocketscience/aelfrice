# Releasing

How to cut a new version. Maintainer reference.

## Versioning

Semver in force. Current line: v4.x. The historical `0.x.y` milestones on the v1.0 path are no longer cut; pre-v1.0 surface guarantees do not apply (see `CHANGELOG/v0.md` for the historical record).

## Cut a release

1. Branch `release/vX.Y.Z` off `main`.
2. Bump `pyproject.toml` `version`. (Single source of truth — no `__version__` in code.)
3. `uv lock`.
4. Collate the changelog (#1475):

   ```bash
   python3 scripts/collate_changelog.py --version X.Y.Z --date YYYY-MM-DD
   ```

   This folds **both** unreleased surfaces into a new `## [X.Y.Z] - YYYY-MM-DD` section of `CHANGELOG/v<major>.md` (e.g. `CHANGELOG/v3.md` for any v3.x release) and deletes the entry files:

   - the `[Unreleased]` block of `CHANGELOG/v<major>.md` — emitted **first** within each category, so a PR that predates the file convention still releases correctly and never had to be rebased onto it;
   - `CHANGELOG/unreleased/<issue>-<slug>.md`, one file per entry — appended after, sorted by file name.

   Categories come out in the `CATEGORIES` order declared in the script — the Keep-a-Changelog six (Added, Changed, Deprecated, Removed, Fixed, Security) then the house additions (Performance, Documentation, Build, CI, Dependencies, Internal, Reverted, Notes); empty ones are omitted. A heading outside that list is an error, caught on the PR that adds the entry file rather than at the cut. Nothing reads filesystem order, so two maintainers cutting the same release get the same bytes. `--dry-run` prints the result and touches nothing.

   Then **add the compare-link footnote by hand** at the bottom of the same file — the script does not, and `release-docs-check` requires it. If the release wants an opening summary paragraph — every dated section in v0-v4 has one — write it into the dated section by hand at the same time. Do not draft it into `[Unreleased]`: collation has nowhere to put a line that is neither a category heading, an entry, nor a continuation of one, so it is refused by name rather than dropped. Top-level `CHANGELOG.md` is a thin index; do not edit it for routine releases. A new major (`vN+1.0.0`) needs a new `CHANGELOG/v<N+1>.md` and a row added to the index.

   `release-docs-check` refuses a release PR that leaves either surface undrained: content still under `[Unreleased]`, or any path other than `README.md` still in `CHANGELOG/unreleased/` — at any depth, whatever its suffix. That scan is deliberately broader than what collation collects (top-level `*.md`), so a `notes.txt` or an `old/1475-slug.md` cannot be invisible to both at once; collation and the duplicate check refuse the same path by name. The check exists because a stranded entry file is *silent* — nothing renders it, so it would surface only as a duplicate in the next release. Collation and both drain assertions are pinned by `tests/test_collate_changelog.py`.
5. Update README roadmap status.
6. Run locally:
   ```bash
   uv run pytest tests/ -x -q     # track the actual count in CI
   uv run pyright src/             # strict
   uv run aelf --help              # spot-check CLI
   uv build                        # wheels build clean
   ```
7. **Run the bench-gate tier and paste its output into the release PR** (#1477).

   ```bash
   scripts/run_bench_gate.sh          # defaults AELFRICE_CORPUS_ROOT to the lab corpus
   ```

   This is the only scheduled run the quality tier gets, and it is mandatory
   rather than advisory. The retrieval, compression and clustering gates skip
   on every public CI run by design (#1420 §3), so a green `pytest` says
   nothing about them — several defaults are held OFF pending exactly these
   verdicts, and a default parked on a measurement nobody takes is parked
   forever.

   Read the `bench-gate tier` summary block, not the pass count. It reports
   three separate states, and only the first is a verdict: tests **executed**
   against the corpus, tests skipped because a named corpus **module** is
   missing or empty, and the whole tier skipped because no corpus root was
   set at all. The corpus covers a minority of the scaffolded modules, so a
   run that reports "N passed" while most modules skipped is the normal case
   and must be recorded as such — paste the block verbatim rather than
   summarising it.

   If the tier could not run at all, say so in the PR body and name the
   reason. Do not cut on a silent skip.

8. Open PR `release: vX.Y.Z`. Body = CHANGELOG entries + the bench-gate block.
9. `staging-gate` must be green — its jobs are `secrets-scan`, `pattern-scan`, `history-scan`, `release-docs-check`, and the commit-msg / PR-title / PR-body prefix checks — and the `pytest` jobs in the separate `ci.yml` workflow (Python 3.12 / 3.13) must pass.
10. Merge — linear history, no merge commits.

## Tag and publish

```bash
git fetch github main
git tag vX.Y.Z <merge-sha>   # lightweight tag on the merged release commit
git push github vX.Y.Z
```

The **release commit** is SSH-signed (`gpg.format = ssh`, key `~/.ssh/id_rrs`); the **tag itself is lightweight** (points directly at that commit — matching every release tag to date, e.g. `git cat-file -t v3.8.0` → `commit`). Tag push triggers `.github/workflows/publish.yml`:

1. Run pytest (gate).
2. Build sdist + wheel.
3. Generate Sigstore attestation.
4. Upload to PyPI via [Trusted Publishing](https://docs.pypi.org/trusted-publishers/).
5. Promote the drafted GitHub Release to published + Latest.

PyPI publish has been live since v1.0; `pip index versions aelfrice` (or `uv tool install aelfrice`) reflects the current released set.

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

PyPI treats `-rc` as pre-release — won't appear as default install candidate. Naming follows the current major (e.g. `v3.4.0-rc1` for a v3.4.0 candidate). Note: `publish.yml` fires only on final-release tags matching `v[0-9]+.[0-9]+.[0-9]+` — pushing an rc tag triggers nothing (the existing `v0.9.0rc0` tag never reached PyPI). To ship an rc, publish manually (`uv build` + `uv publish`) or extend the workflow's tag filter first.

```bash
uv tool install --pre aelfrice==3.4.0rc1
```

## Branch protection

The public repo enforces `main` protection through a combination of the merge-train workflow (concurrency-1 FF-only pushes, signature-verified, see `.github/workflows/merge-train.yml`) and required checks: the staging-gate jobs (`secrets-scan`, `pattern-scan`, `history-scan`, `release-docs-check`) plus the `pytest` matrix from the separate `ci.yml` workflow. GitHub's native branch-protection APIs may be configured in addition — check `gh api repos/robotrocketscience/aelfrice/branches/main/protection` for the current state.

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
