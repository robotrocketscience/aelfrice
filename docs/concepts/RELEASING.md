# Releasing

This page is the maintainer's reference for cutting a new version.

## Versioning

aelfrice uses semantic versioning (semver), and the current line is v4.x. The project no longer cuts the historical `0.x.y` milestones on the v1.0 path. The pre-v1.0 surface guarantees do not apply. For the historical record, see `CHANGELOG/v0.md`.

## Cut a release

1. Create the branch `release/vX.Y.Z` from `main`.
2. Bump the `version` field in `pyproject.toml`. That field is the single source of truth; the code holds no `__version__`.
3. Run `uv lock`.
4. Collate the changelog (#1475):

   ```bash
   python3 scripts/collate_changelog.py --version X.Y.Z --date YYYY-MM-DD
   ```

   The script folds **both** unreleased surfaces into a new `## [X.Y.Z] - YYYY-MM-DD` section of `CHANGELOG/v<major>.md`, then deletes the entry files. For a v3.x release, that file is `CHANGELOG/v3.md`. The two surfaces are:

   - the `[Unreleased]` block of `CHANGELOG/v<major>.md`. The script emits this block **first** within each category, so a pull request (PR) that predates the file convention still releases correctly, and nobody had to rebase such a PR onto the convention.
   - `CHANGELOG/unreleased/<issue>-<slug>.md`, with one file for each entry. The script appends these entries after the block, sorted by file name.

   The script emits the categories in the `CATEGORIES` order it declares: first the six Keep-a-Changelog categories (Added, Changed, Deprecated, Removed, Fixed, and Security), then the categories this project adds (Performance, Documentation, Build, CI, Dependencies, Internal, Reverted, and Notes). It omits an empty category, and a heading outside that list is an error, caught on the PR that adds the entry file rather than at the cut. It never reads filesystem order, so two maintainers who cut the same release get the same bytes. `--dry-run` prints the result and changes no file.

   Then **add the compare-link footnote by hand** at the bottom of the same file. The script does not add the footnote, and `release-docs-check` requires it. If the release needs an opening summary paragraph, write it by hand into the dated section at the same time. Do not draft the summary into `[Unreleased]`: collation has no place for a line that is neither a category heading, an entry, nor the continuation of an entry, so it refuses such a line by name instead of dropping it. The top-level `CHANGELOG.md` is a thin index, so don't edit it for a routine release. A new major version (`vN+1.0.0`) needs both a new `CHANGELOG/v<N+1>.md` file and a new row in the index.

   `release-docs-check` refuses a release PR that leaves either surface undrained: content still under `[Unreleased]`, or any path other than `README.md` still in `CHANGELOG/unreleased/`, at any depth and with any suffix. That scan is deliberately broader than what collation collects (top-level `*.md`), so a `notes.txt` file, or an `old/1475-slug.md` file, cannot be invisible to both checks at the same time. Collation and the duplicate check refuse the same path by name. The check exists because a stranded entry file is *silent*: nothing renders such a file, so it would appear only as a duplicate in the next release. `tests/test_collate_changelog.py` pins collation and both drain assertions.
5. Update the roadmap status in the README.
6. Run these commands on your machine:
   ```bash
   uv run pytest tests/ -x -q     # track the actual count in CI
   uv run python scripts/check_pyright_baseline.py  # no file may regress
   uv run aelf --help              # spot-check CLI
   uv build                        # wheels build clean
   ```
7. **Run the bench-gate tier** (#1477). **Paste its output into the release PR.**

   ```bash
   scripts/run_bench_gate.sh          # defaults AELFRICE_CORPUS_ROOT to the lab corpus
   ```

   This is the only scheduled run the quality tier gets, and it is mandatory,
   not advisory. By design, the retrieval, compression, and clustering gates
   skip on every public CI run (#1420 §3), so a green `pytest` result says
   nothing about those three gates. Several defaults stay OFF until these
   verdicts arrive, and a default that waits on a measurement nobody takes
   waits forever.

   Read the `bench-gate tier` summary block rather than the pass count. The
   block reports three separate states, and only the first is a verdict:

   - the tests that **executed** against the corpus
   - the tests that skipped because a named corpus **module** is missing or
     empty
   - the whole tier that skipped because no corpus root was set at all

   The corpus covers a minority of the scaffolded modules, so a run that
   reports "N passed" while most modules skipped is the normal case, and you
   must record it as such. Paste the block word for word instead of summarizing it.

   If the tier could not run at all, say so in the PR body and name the reason.
   Do not cut a release on a silent skip.

8. Open a PR with the title `release: vX.Y.Z`. Use the CHANGELOG entries and the bench-gate block as the PR body.
9. `staging-gate` must be green before you merge. Its jobs are `secrets-scan`, `pattern-scan`, `history-scan`, `release-docs-check`, and the prefix checks for the commit message, the PR title, and the PR body. The `pytest` jobs in the separate `ci.yml` workflow, which run Python 3.12 and Python 3.13, must also pass.
10. Merge the PR. Keep the history linear, and create no merge commits.

## Tag and publish

```bash
git fetch github main
git tag vX.Y.Z <merge-sha>   # lightweight tag on the merged release commit
git push github vX.Y.Z
```

You sign the **release commit** with SSH (`gpg.format = ssh`, key `~/.ssh/id_rrs`). The **tag itself is lightweight**: it points directly at that commit. Every release tag to date has this form; for example, `git cat-file -t v3.8.0` prints `commit`. Pushing a tag starts `.github/workflows/publish.yml`, which does these steps:

1. Run pytest as a gate.
2. Build the sdist and the wheel.
3. Generate the Sigstore attestation.
4. Upload the files to PyPI through [Trusted Publishing](https://docs.pypi.org/trusted-publishers/).
5. Promote the drafted GitHub Release to published and to Latest.

The PyPI publish path has been live since v1.0. Either `pip index versions aelfrice` or `uv tool install aelfrice` reflects the current released set.

## Verify

```bash
uv tool install aelfrice==X.Y.Z
aelf --help
aelf stats
```

A clean tool install in a scratch venv proves that the wheel works. To find drift, compare the CLI surface against the previous version.

## Hotfixes

```bash
git switch -c release/vX.Y.Z+1 vX.Y.Z
# fix → bump → lock → CHANGELOG/vN.md → PR → gate → merge → tag
```

If `main` has moved on in an incompatible way, cherry-pick the fix instead.

## Yank

```bash
gh release delete vX.Y.Z --yes
git push github :refs/tags/vX.Y.Z
git tag -d vX.Y.Z
# yank from PyPI manually via web UI
```

Then bump the version to `vX.Y.Z+1` and include the fix.

## Pre-releases

PyPI treats `-rc` as a pre-release, and a pre-release does not appear as the default install candidate. The name follows the current major version: `v3.4.0-rc1` is a candidate for v3.4.0. `publish.yml` fires only on a final-release tag that matches `v[0-9]+.[0-9]+.[0-9]+`, so pushing an rc tag starts nothing. The existing `v0.9.0rc0` tag never reached PyPI. To ship an rc, publish it by hand with `uv build` and `uv publish`, or extend the workflow's tag filter first.

```bash
uv tool install --pre aelfrice==3.4.0rc1
```

## Branch protection

The public repo protects `main` with two mechanisms:

- the merge-train workflow, which makes fast-forward-only pushes at concurrency 1 and verifies the signature (see `.github/workflows/merge-train.yml`)
- the required checks, which are the staging-gate jobs (`secrets-scan`, `pattern-scan`, `history-scan`, and `release-docs-check`) and the `pytest` matrix from the separate `ci.yml` workflow

You can also configure GitHub's native branch-protection APIs. To read the current state, run `gh api repos/robotrocketscience/aelfrice/branches/main/protection`.

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

The same key is also registered as a signing key on GitHub. On a fresh clone, repeat these `git config --local` lines.
