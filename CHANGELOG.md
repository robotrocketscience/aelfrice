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
| v3.x  | [`CHANGELOG/v3.md`](CHANGELOG/v3.md)  | current    |
| v2.x  | [`CHANGELOG/v2.md`](CHANGELOG/v2.md)  | archived   |
| v1.x  | [`CHANGELOG/v1.md`](CHANGELOG/v1.md)  | archived   |
| v0.x  | [`CHANGELOG/v0.md`](CHANGELOG/v0.md)  | pre-1.0    |

## Contributing changelog entries

- Add new entries under `## [Unreleased]` in the **current** major's file
  ([`CHANGELOG/v3.md`](CHANGELOG/v3.md) at present).
- On release, move `[Unreleased]` content into a dated `## [X.Y.Z]` section and
  add the compare-link footnote at the bottom of the same file. CI checks both.
- A new major (e.g. `v4.0.0`) opens a new `CHANGELOG/v4.md` and updates the
  table above; prior majors become archived.
