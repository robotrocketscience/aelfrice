# Configuration: `.aelfrice.toml`

Most users never need this file. aelfrice is designed to work silently in the background, keeping LLM sessions current with the project's ground truth. The defaults are tuned so that `pip install aelfrice && aelf onboard .` does the right thing for almost every project.

This document is for the case where the defaults are *not* doing the right thing — when a project has a documentation idiom or naming convention that the default filter mishandles, when a power user wants finer control over what enters the belief store, or when a contributor is debugging onboarding behaviour. If that's not you, close this tab and let the agent do its job.

---

## What is `.aelfrice.toml`?

A single optional TOML file at the root of a project (or any ancestor directory). Editing it changes how `aelf onboard` ingests beliefs — nothing else. Retrieval, the hook, MCP tools, locks, and the Bayesian feedback math are not affected by anything in this file.

If the file does not exist, aelfrice ships with safe defaults; this is the recommended state for almost all projects.

## Where it lives

`scan_repo` (the function backing `aelf onboard`) walks up from the scan root, checking each directory for `.aelfrice.toml`, until it finds one or reaches the filesystem root. The first file found wins.

This means:

- A project with `.aelfrice.toml` at its top directory is configured for any onboard run inside that project.
- A nested subproject can shadow its parent by placing its own `.aelfrice.toml` inside the subproject directory.
- A user without the file gets the default behaviour, which is the v1.0 ship behaviour.

The walk stops at filesystem root. There is no global / per-user `.aelfrice.toml` — configuration is per-project (or per-ancestor-of-project) on purpose, so checking out a different repo cannot silently inherit unrelated config.

## Schema

The file has one table at v1.0.1: `[noise]`. Future versions add more (project identity, store path, etc. — see [ROADMAP](ROADMAP.md)). Unknown keys and unknown tables are ignored, so this file is forward-compatible.

```toml
# .aelfrice.toml
[noise]
# Turn off any of the four built-in categories. Each disabled
# category will not contribute to skipped_noise; its content
# may still be filtered out by the classifier or by other
# categories. Subset of: headings | checklists | fragments | license
disable = []

# Below this many whitespace tokens, treat a paragraph as a
# fragment and drop it. The default catches stubs and labels
# while leaving real prose alone. Lower this if your project
# has many terse beliefs you want to keep ("lock fast",
# "prefer composition"). Set to 0 to disable the check.
min_words = 4

# Drop paragraphs that contain any of these whole words. Match
# is case-insensitive and word-bounded — adding "jso" drops
# paragraphs containing the standalone token "jso" but NOT
# "json", "jsonify", or "jsodb". Use this for initials,
# codenames, internal jargon you don't want surfacing in
# retrieval.
exclude_words = []

# Drop paragraphs that contain any of these substrings. Match
# is case-insensitive but otherwise literal — punctuation and
# whitespace inside the phrase are matched verbatim. Use this
# for status flags, header strings, or templated lines you
# always want filtered.
exclude_phrases = []
```

## What each key does and what it affects

### `disable`

A list naming the built-in noise categories you want to turn off.

| Token | What it disables | What you'll see |
|---|---|---|
| `headings` | The "every line is a markdown heading" filter. Pure heading blocks like `# Section\n## Subsection` will pass through. | Markdown headings in your docs become belief candidates. The classifier may still drop them as low-content; if not, expect short labels in your store. |
| `checklists` | The "every line is `- [ ]`" filter. Pure task-list blocks pass through. | TODO list items become belief candidates. Useful only if your TODO entries actually contain durable behavioural rules. |
| `fragments` | The `min_words` short-paragraph filter. | Short labels like `INSTRUCTIONS:`, `DRAFT`, terse one-liners pass to the classifier. |
| `license` | The seven-signature license-preamble filter. | LICENSE.md text and equivalent files become belief candidates. Most projects do not want this; aelfrice deliberately drops legal boilerplate to keep the store about your code. |

A category disabled here is silent — `ScanResult.skipped_noise` will not count anything from that category, and the candidate flows through to classification. Other categories still fire normally.

Unrecognised tokens are silently ignored. Typos (`fragmints`, `lisence`) do not turn the whole filter off; the misnamed category remains enabled.

### `min_words`

Integer, default `4`. Any paragraph with fewer than this many whitespace-separated tokens is dropped as a fragment.

| Setting | Behaviour | When to use |
|---|---|---|
| `4` (default) | Drops 0–3-word paragraphs. | Most projects. |
| `3` or lower | Lets 3-word and shorter beliefs through. | Projects that lock terse rules ("prefer composition", "no global state", "fail loud"). |
| `0` | Disables the fragment check entirely. | Mostly for debugging. Equivalent to `disable = ["fragments"]`. |
| Negative | Clamped to 0 silently. | — |

A non-integer value (string, float, `true`, etc.) is rejected with a warning to stderr and the default of 4 is used.

### `exclude_words`

A list of strings, default `[]`. Each string is treated as a whole word and matched case-insensitively.

The matching uses word boundaries: a word like `"jso"` matches the standalone token `jso` (or `jso.`, `jso,`, `jso-files` — anywhere `jso` is followed by a non-letter / non-digit / non-underscore). It does *not* match `json`, `jsonify`, `jsodb`, or any longer alphanumeric run that contains `jso` as a substring.

This is the right choice for:

- **Initials / contributor names.** Adding `["jso"]` filters paragraphs naming that contributor without breaking docs that mention `json`.
- **Codenames.** Internal project codenames you'd rather not surface to the LLM.
- **Status keywords.** `["DRAFT", "WIP"]` drops paragraphs flagged as work-in-progress.

Empty strings in the list are skipped (would otherwise match every paragraph). Non-string entries are skipped with a warning.

### `exclude_phrases`

A list of strings, default `[]`. Each string is matched case-insensitively as a literal substring anywhere in the paragraph. Punctuation and whitespace inside the phrase are matched verbatim.

This is the right choice for:

- **Templated header lines.** `["Last updated:", "Generated on"]` drops paragraphs that are auto-stamped boilerplate.
- **Inline status flags.** `["TODO:", "FIXME"]` drops paragraphs that contain a development marker mid-text.
- **Multi-word fixed strings** that don't follow word-boundary semantics.

The trade-off vs. `exclude_words`: phrase match is literal substring, no word boundaries. `["foo"]` here would drop a paragraph containing `foobar`. If that's not what you want, use `exclude_words` instead.

## When a change takes effect

Edits to `.aelfrice.toml` apply on the next `aelf onboard` run. They do **not** retroactively re-filter beliefs that are already in your store — once a belief is in the database, it stays there until you `aelf demote` or `aelf delete` it (the latter lands in v2.0). This is intentional: the config controls ingestion, not retention.

If you want to remove existing noise from a store, the cleanest path is:

```bash
rm "$(python -c 'from aelfrice.cli import db_path; print(db_path())')"  # resolves to .git/aelfrice/memory.db inside a repo, ~/.aelfrice/memory.db otherwise, or AELFRICE_DB if set
aelf onboard /path/to/project   # re-onboard with new config
```

Project-level locks, manually inserted beliefs, and feedback history will be lost. For a less destructive cleanup, query the store directly with `sqlite3` and `DELETE` rows that match the noise pattern you've added.

## What this file does NOT do

- **Does not affect retrieval.** The `[noise]` table only runs at onboard time. Once a belief is in the store, retrieval (BM25, L0 locks, the hook) treats it identically regardless of how or whether the noise filter would have flagged it.
- **Does not affect `aelf lock`, `aelf remember`, or the MCP `aelf:remember` tool.** Manually-asserted beliefs bypass the noise filter — you're explicitly saying "this matters."
- **Does not affect the harness conflict.** The Claude Code auto-memory write path is governed by the harness directive in `~/.claude/CLAUDE.md`, not by this file. See [LIMITATIONS § Harness conflict](LIMITATIONS.md#harness-conflict--claude-code-auto-memory-write-path).
- **Does not affect classifier behaviour.** Candidates that pass the noise filter go to `aelfrice.classification.classify_sentence`, which has its own (non-configurable at v1.0.1) rules for `persist=False`. If a candidate is being dropped and you don't expect it to, check `ScanResult.skipped_non_persisting` before blaming `[noise]`.
- **Does not retroactively re-filter.** See above.
- **Does not redefine the four categories.** You can disable them, not modify what they match. To filter on a custom rule, use `exclude_words` or `exclude_phrases`. If you need true regex semantics, you have to extend the module — that is a deliberate constraint, not an oversight.
- **Does not load from `pyproject.toml`, environment variables, or CLI flags.** Single surface, single file.

## Worked examples

### Filter a contributor's initials without breaking `json` mentions

```toml
[noise]
exclude_words = ["jso"]
```

Drops paragraphs naming the contributor `jso`. Leaves docs about `json`, `jsonify`, etc. untouched.

### Let terse beliefs through on a dense rule project

```toml
[noise]
min_words = 2
```

Two-word beliefs ("lock fast", "fail loud") and longer pass; one-word labels ("DRAFT", "INSTRUCTIONS:") still drop.

### Filter templated boilerplate that the default doesn't catch

```toml
[noise]
exclude_phrases = [
    "Last updated:",
    "Generated by tool-x",
    "DO NOT EDIT",
]
```

These signatures are project-specific and the default filter doesn't know about them. Adding them here keeps the store free of the auto-stamped lines.

### Disable a category for a license-heavy project

```toml
[noise]
disable = ["license"]
```

Useful if your project's documentation discusses licenses substantively (a legal-tech project, an OSS-compliance tool). The default would filter genuine prose containing the license preamble phrases; disabling lets it through.

### Combine all four

```toml
[noise]
disable = ["headings"]
min_words = 3
exclude_words = ["jso", "internal-codename"]
exclude_phrases = ["TODO:", "FIXME"]
```

A power user with strong opinions. Keeps headings (maybe their docs use heading-only paragraphs as semantic markers), tightens the fragment threshold, filters internal terminology, and drops dev-marker lines. All four take effect simultaneously.

## Defaults reference

If you write `[noise]` with no fields, you get the v1.0.1 ship behaviour — same as not having a config file at all:

| Field | Default |
|---|---|
| `disable` | `[]` (all four categories enabled) |
| `min_words` | `4` |
| `exclude_words` | `[]` |
| `exclude_phrases` | `[]` |

## Resilience contract

If `.aelfrice.toml` is malformed (invalid TOML), unreadable (permission error), or contains wrong-typed values for known fields, the noise filter degrades silently to defaults rather than failing the onboard run. Failures are traced to stderr so a power user debugging can see them; a casual user is not blocked.

Specifically:

- Malformed TOML → defaults loaded, `malformed TOML in <path>: <error>` to stderr.
- Wrong-typed field (e.g. `min_words = "three"`) → that field defaults, `ignoring [noise] <field>` to stderr; other fields still load.
- Non-string entry in a string-list field → that entry skipped with a warning, list still loads.
- Unknown field → silently ignored (forward-compat for future schema additions).
- Missing file → defaults loaded, no warning.

## See also

- [COMMANDS § `aelf onboard`](COMMANDS.md) — the CLI surface.
- [ARCHITECTURE § Modules](ARCHITECTURE.md) — `noise_filter.py` placement in the module map.
- [LIMITATIONS § Onboarding](LIMITATIONS.md) — what's still on the v1.x horizon for onboard behaviour.
- The module docstring in `src/aelfrice/noise_filter.py` — same schema, slightly more terse, with the regex internals exposed for contributors.
