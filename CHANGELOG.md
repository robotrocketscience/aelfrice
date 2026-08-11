# Changelog

All notable changes to aelfrice are documented under [`CHANGELOG/`](CHANGELOG/),
split by major version. New entries land in the current major's file under its
`## [Unreleased]` section; on release, that section is dated and a new
`## [Unreleased]` block opens above it.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Releases by major version

| Line  | File                                  | Status     |
| ----- | ------------------------------------- | ---------- |
| v4.x  | [`CHANGELOG/v4.md`](CHANGELOG/v4.md)  | current    |
| v3.x  | [`CHANGELOG/v3.md`](CHANGELOG/v3.md)  | archived   |
| v2.x  | [`CHANGELOG/v2.md`](CHANGELOG/v2.md)  | archived   |
| v1.x  | [`CHANGELOG/v1.md`](CHANGELOG/v1.md)  | archived   |
| v0.x  | [`CHANGELOG/v0.md`](CHANGELOG/v0.md)  | pre-1.0    |

## Contributing changelog entries

- Add a new entry as its own file under
  [`CHANGELOG/unreleased/`](CHANGELOG/unreleased/), named `<issue>-<slug>.md`.
  One entry per file: two PRs adding entries touch two different paths, so
  they cannot conflict. Do **not** edit the `[Unreleased]` block of the
  current major's file ([`CHANGELOG/v4.md`](CHANGELOG/v4.md) at present) —
  every open PR collides in it.
- On release, `scripts/collate_changelog.py` folds both surfaces — any
  remaining `[Unreleased]` block content and every entry file — into a dated
  `## [X.Y.Z]` section, empties the directory, and the compare-link footnote
  goes at the bottom of the same file. CI checks that the block is drained,
  that the directory is empty, and that no two entries restate each other.
- A new major (e.g. `v4.0.0`) opens a new `CHANGELOG/v4.md` and updates the
  table above; prior majors become archived.
