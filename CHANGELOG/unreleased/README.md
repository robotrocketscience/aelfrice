# Unreleased changelog entries

One file per entry, named `<issue>-<slug>.md`. At release time
`scripts/collate_changelog.py` folds every file here into the dated
section of `CHANGELOG/v<major>.md` and empties this directory.

Why: entries are 2,000-4,500-character single lines, and thirteen of
fourteen open PRs were inserting them into the same eight-line region
of `CHANGELOG/v4.md` (#1475). Two branches adding files at distinct
paths never conflict; two branches inserting into one block always do,
and the resolution — two 4 KB lines with no intra-line granularity —
can drop an entry invisibly.

## Format

```markdown
### Fixed

- **One-line title ([#1475](https://github.com/robotrocketscience/aelfrice/issues/1475)).** Body prose.
```

Exactly one `### <Category>` heading and exactly one top-level `- `
bullet per file. Category is one of `Added`, `Changed`, `Deprecated`,
`Removed`, `Fixed`, `Security`, `Performance`, `Documentation`,
`Build`, `CI`, `Dependencies`, `Internal`, `Reverted`, `Notes` —
the list `CATEGORIES` in the script, and the only headings the
committed changelogs use. Indented
continuation paragraphs under the bullet are preserved verbatim.

This directory is flat and holds nothing else. A different suffix
(`.txt`, `.markdown`, an uppercase `.MD`), an extensionless file or a
subdirectory is an error naming the path — collation refuses it,
`scripts/check_changelog_dupes.py` refuses it, and `release-docs-check`
lists it as stranded. All three refuse rather than skip: a file
collation will not collect is one the release omits without a word,
which is the one failure this convention has to keep loud.

Collation order is stated in the script's docstring: categories in
`CATEGORIES` order; within a category, entries still in the
`[Unreleased]` block of `CHANGELOG/v<major>.md` first, then these files
sorted by name. Nothing reads filesystem order.

## Transition

`[Unreleased]` in `CHANGELOG/v<major>.md` is still valid — collation
emits it first. A PR already editing that block does not need to be
rebased onto this convention to merge. New entries should be files.

This README also keeps the directory tracked: git stores no empty
trees, so without it the first entry-adding PR would have to create the
directory too.
