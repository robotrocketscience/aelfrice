<!--
v1.0 has shipped. Issues and PRs are open. Please align on approach via an issue first if the change adds new surface, alters the schema, or reintroduces a v2.0 feature. See CONTRIBUTING.md.
-->

## Summary

<!-- One-paragraph description of what this PR does. Focus on the "why," not the diff. -->

## Linked issues

<!-- Closes #N, references #M -->

## Type of change

- [ ] `feat:` — new feature
- [ ] `fix:` — bug fix
- [ ] `perf:` — performance improvement
- [ ] `refactor:` — code restructure with no behavior change
- [ ] `test:` — test-only change
- [ ] `docs:` — documentation-only change
- [ ] `build:` — build system / dependency / lockfile change
- [ ] `ci:` — CI workflow / hook change
- [ ] `release:` — version bump / release tag
- [ ] `chore:` — narrow housekeeping

## Verification

- [ ] `uv run pytest tests/ -x -q` — all green
- [ ] `uv run pyright src/` — strict, no new errors
- [ ] `uv run aelf --help` — surface unchanged (or change documented)
- [ ] CHANGELOG entry added under `[Unreleased]` (if user-visible)
- [ ] Docs updated (if surface or behavior changed)

## Test plan

<!-- Describe how to verify the change manually or what tests cover it. -->

## Notes for reviewer

<!-- Anything tricky, anything you punted, anything that needs a second opinion. -->
