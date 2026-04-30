# 0002 — Two-repo physical separation for public/private boundary

- **Status:** Accepted
- **Date:** 2026-04-30
- **Deciders:** @robotrocketscience

## Context

aelfrice is developed alongside private artifacts: research notes, planning documents, LLC/business records, experiment data, draft posts. The product code is open-source and ships to PyPI; the surrounding workspace is not.

Earlier the project tried a single-repo "filter on publish" pipeline: a quarantine staging area, allowlist (`.github-include`), gitleaks rules, and a publish script that filtered private content out before pushing to GitHub. This pipeline was complex, easy to mis-configure, and a single allowlist typo could leak private content irreversibly — git history is hard to scrub once pushed, and `refs/pull/N/head` pins commits permanently.

## Decision

Use **two physically separate repositories** with no path between them:

- `~/projects/aelfrice` — public product code. Origin is GitHub only.
- `~/projects/aelfrice-lab` — private workspace. Origin is a self-hosted gitea only. A pre-push hook rejects any remote URL containing `github.com`.

The boundary is the directory of origin, not a transformation. Test fixtures, paraphrased examples, and "abstracted" content all count as derived from their source directory.

## Alternatives considered

- **Single repo with publish filter.** The prior approach. Rejected: leak surface area is the entire allowlist, and a leak is irreversible.
- **Single repo with `.gitignore` for private files.** Rejected: ignored files are still trivially committable by `-f`, and there's no structural barrier to mistakes.
- **Submodule for private content.** Rejected: submodules are a pointer mechanism, not an isolation mechanism — a misconfigured push could still expose paths.

## Consequences

- **Positive:** Structural safety. The only path from private content to GitHub requires manually copying files between two working trees. Pre-push hooks add a second layer of defense. No allowlist to maintain.
- **Negative:** Cross-references between the two repos require manual coordination. The lab repo holds a read-only submodule of the public repo for context, but cannot push into it.
- **Neutral:** Some private notes that reference public files now duplicate file paths instead of linking to a single source of truth. Acceptable cost for structural separation.
