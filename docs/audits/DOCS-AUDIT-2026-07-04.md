# Docs audit — 2026-07-04

> Documentation audit against HEAD (`33b7ddd5`, v3.8.0 + post-release main), run under #1075
> as the v4.0.0 release gate. This pass covers the **current-behavior doc-file surface** —
> root docs, `docs/concepts/*`, `docs/user/*`, `docs/feature-*.md`, benchmarks READMEs, and all
> 26 `src/aelfrice/slash_commands/*.md` — in 5 parallel batches. Every checkable claim (command,
> flag, default, env var, budget, version gate, file path, API signature, config key, lane name)
> verified against code; every MEDIUM finding re-verified against a named code location before
> the fix. 0 CRITICAL, 0 HIGH, 9 MEDIUM, 8 LOW, 1 code-side drift issue.
>
> **Scope carried forward (NOT done in this pass):** (1) the in-code documentation layer —
> docstrings, argparse/`--help` text, MCP tool descriptions and schemas across all 552 `.py`
> files — matching the #959/#964 precedent from the 2026-06-10 audit; (2) the remaining
> doc-file buckets — `CHANGELOG*`, `docs/design/*` (incl. `historical/`), `docs/adr/*`, prior
> `docs/audits/*` snapshots, `docs/experiments/*`, `tests/` + `.github/` docs, `CITATION.cff`
> (~90 files). The v4.0.0 gate is not closeable until both are audited. See § Deferred scope.

## Scope

| Batch | Surface | Findings |
|---|---|---|
| A | README, CONTRIBUTING, SECURITY, CODE_OF_CONDUCT, docs/README, docs/concepts/* | 5 (4 MED, 1 LOW) |
| B | docs/user/* | 9 (5 MED, 3 LOW, 1 code-side) |
| C | docs/feature-*.md + benchmarks READMEs | 3 LOW |
| D | src/aelfrice/slash_commands/*.md (first 13 alphabetical) | 0 (2 LOW notes, no defect) |
| E | src/aelfrice/slash_commands/*.md (files 14–26) | 0 |
| — | in-code layer (docstrings, --help, MCP descriptions; 552 .py) | **deferred — see below** |

## Cross-cutting theme — the two newest features are under-documented

The dominant pattern (independently surfaced by batches A and B): the two features that landed
into `[Unreleased]` after the 2026-06-10 audit are missing from the enumerative surfaces.

1. **`agent_context` hook (#1068)** — `aelf-agent-context-hook`, `PreToolUse:^(Agent|Task)$`,
   worker-context injection, default-on (`hook_manifest.json:77`, `default_on: true`; opt-out
   `--no-agent-context`). Missing from: README hook count ("Nine" → **Ten**), INSTALL script
   count ("ten console scripts / nine hooks" → **eleven / ten**) + INSTALL hooks table,
   COMMANDS setup default-on list, ARCHITECTURE default-on hooks table.
2. **Temporal-spine surface (#1064)** — the hidden `spine` subcommand (`cli.py:7109`,
   `add_parser("spine", …SUPPRESS)`, action `backfill`, `--dry-run`) and the default-off
   `use_temporal_spine` retrieval lane. Missing from: COMMANDS `--advanced` hidden-subcommand
   list, SLASH_COMMANDS hidden-CLI list, the ARCHITECTURE/README retrieval-lane maps. CONFIG
   documents the keys but tags them a version that will be skipped (below).

## Version-label drift

3. **ROADMAP "current line is v3.7.0"** — `pyproject.toml` is `3.8.0`, `v3.8.0` is tagged, README
   names v3.8.0 as latest stable. ROADMAP:12 + the "Versions at a glance" table stop at v3.7.0.
4. **CONFIG "v3.9+ (#1064)"** (5 sites: lines 147, 161, 177, 503, 520) — the temporal-spine keys
   predict a v3.9 that this audit's own release gate skips. Reconciled to **v4.0.0** per #1075.
5. **RELEASING documents `git tag -s` (signed annotated tags)** — actual practice is *lightweight*
   tags on the merge SHA (`git cat-file -t v3.6.0/v3.7.0/v3.8.0` → `commit`; `git tag -v v3.8.0`
   fails "non-tag object"). Consistent with every release tag to date. Non-breaking
   (`publish.yml` fires on the ref regardless); doc corrected to describe lightweight tagging.

## Findings ledger

### MEDIUM (fixed in place)
| # | File:line | Claim | Reality | Fix |
|---|---|---|---|---|
| M1 | README.md:145 | "Nine default-on hooks" | manifest ships 10 `default_on:true` | "Ten" + add PreToolUse:Agent context hook; add `--no-agent-context` to the opt-out list (README:123) |
| M2 | docs/concepts/ARCHITECTURE.md:158–170 | default-on hooks table (8 rows) | missing `claude_memory_mirror` + `agent_context` | add both rows |
| M3 | docs/concepts/ROADMAP.md:12 | "current line is v3.7.0" | v3.8.0 shipped | bump to v3.8.0 |
| M4 | docs/concepts/RELEASING.md:31 | `git tag -s … -m` signed annotated | lightweight tags on merge SHA | document `git tag vX.Y.Z <merge-sha>` |
| M5 | docs/user/INSTALL.md:24 | "ten console scripts … nine hook entry-points" | pyproject has 11 scripts / 10 hooks | "eleven / ten" + add `aelf-agent-context-hook` |
| M6 | docs/user/INSTALL.md:140–151 | hooks table (10 rows) | missing `agent-context` | add row (default-on, `--no-agent-context`) |
| M7 | docs/user/COMMANDS.md:62 | setup default-on hook list | omits `agent_context` | add it |
| M8 | docs/user/COMMANDS.md:105 | `--advanced` hidden-subcommand list | omits `spine` | add `spine` |
| M9 | docs/user/CONFIG.md:147,161,177,503,520 | temporal-spine keys "v3.9+" | next release is v4.0.0 (#1075) | reconcile to v4.0.0 |

### LOW (fixed in place)
| # | File:line | Note |
|---|---|---|
| L1 | ARCHITECTURE.md:107–120 | retrieval-lane map omits the default-off temporal-spine lane (#1064) — added a default-off line. README's "How it works" intro map left as-is: it correctly describes *default* behaviour (three lanes + opt-in BFS), and the spine is off by default. |
| L2 | docs/user/SLASH_COMMANDS.md:5 | hidden-CLI list omits `spine`; wrongly includes `export-obsidian` (a *visible* subparser, `cli.py:8200`, not SUPPRESS) |
| L3 | docs/user/MCP.md:245 | line cite `mcp_server.py:806-807` drifted → `814-815` (behavior claim TRUE) |
| L4 | docs/user/QUICKSTART.md:11–19 | onboard story doesn't mention the #1072/#1073 stderr pointer to `/aelf:onboard`. **No edit** — the prose makes no false claim (the pointer is an additive stderr line); recorded as a completeness note. README/onboard.md/`--llm-classify` help already carry the primary-path framing from #1073. |
| L5 | docs/feature-zeta-posterior-rerank.md:46 | "scale … defaults to 14.5" — `scale` has no kwarg default; 14.5 is `ZETA_SCALE_DEFAULT` callers pass |
| L6 | benchmarks/context-rebuilder/README.md:2,258 | link *display text* drops the `design/` path segment (targets correct) |
| L7 | benchmarks/context-rebuilder/fixtures/README.md:18–19,28 | bare-prose path refs drop `design/` (markdown link target correct) |
| L8 | slash_commands/lock.md, confirm.md | flag-taking commands have undocumented (`--reference`/`--frozen`/`--id`/`--doc`; `--source`/`--note`) flags unreachable via the slash surface; quoting is correct as-is (single free-text positional) — documentation-completeness note, no defect |

### Code-side drift (filed separately — doc is right, code is wrong)
- **C1 — `cli.py:6347`**: `search --budget` help reads `"output token budget (default 2000)"`
  while `default=DEFAULT_TOKEN_BUDGET` = **2400** (`retrieval.py:125`). COMMANDS.md/QUICKSTART
  correctly say 2,400. Fix the help string, not the docs. → filed as **#1077**.

## Verified-clean (zero findings)

- **Root/concepts:** CONTRIBUTING.md, SECURITY.md, CODE_OF_CONDUCT.md, docs/README.md,
  docs/concepts/README.md, COMPARISON.md, BENCHMARKS.md, PHILOSOPHY.md, HARNESS_INTEGRATION.md.
  README.md otherwise clean (deps disclosed, 1500/2400 budgets, 9 origins, edge model, opt-out
  flags) apart from M1/L1.
- **docs/user:** PRIVACY.md (the #958 INEDIBLE-on-ingest-transcript divergence is now resolved —
  `is_inedible_path` wired into both `ingest.py:458` single-file and `ingest.py:596` batch paths),
  MCP.md (15 tools, `wonder_gc dry_run=False` — prior audit's fix held; only the L3 line cite),
  LIMITATIONS.md, docs/user/README.md.
- **docs/feature:** feature-hot-path.md, feature-posterior-temperature.md,
  feature-rerank-relevance-corpus.md (feature-zeta only the L5 wording).
- **benchmarks:** benchmarks/README.md.
- **slash_commands (all 26):** every `/aelf:*` command maps to an existing subparser; every
  documented flag exists; `$ARGUMENTS` quoting correct throughout (single-positional commands
  quoted, flag-taking commands unquoted); no `v3.4.0-standalone` claims; delete/gc/rebuild flag
  surfaces verified against handlers.

## Traps correctly avoided (checked, not filed)
- **BM25F `-raw` sign** at the γ/ζ scorer call sites is CORRECT (scorers negate internally;
  `scoring.py:218/258/301`). Re-flagged in past sweeps; #978 covers the real gap. Not a finding.
- **`replay_results.jsonl`** in the context-rebuilder README is explicitly written by the
  operator's inline step, not claimed as a harness output. Not a finding.
- **MCP `wonder_gc dry_run`** default — TRUE `False` in both doc and code now. Not a finding.

## Deferred scope (blocks v4.0.0 gate closure)

Two tranches of the #1075 inventory were NOT audited in this pass. Per the issue's coverage
rule, absence from this ledger means *unaudited*, not clean — both tranches block the v4.0.0
tag:

1. **In-code documentation layer** — Python docstrings, argparse/`--help` strings, MCP tool
   descriptions + schemas, hook self-descriptions across all 552 `.py` files. Prior audits
   (#959, #964) deferred the same layer to follow-ups; #1075 asks for it in-scope, so it must
   be completed as its own batched pass + fix PR. The `cli.py:6347` code-side drift (C1) is
   one instance already surfaced incidentally; a full pass will find more.
2. **Remaining doc-file buckets (~90 files)** — `CHANGELOG.md` + `CHANGELOG/v0.md`–`v3.md`
   (verify entries against code at the corresponding tags, per the #957 finding that some
   entries were spec-written); `docs/design/*` incl. `historical/` (71 files — status banners
   + present-tense current-behavior claims; bodies are historical record per
   `docs/design/README.md`); `docs/adr/*`; prior `docs/audits/*` snapshots (links/banners
   only; bodies frozen); `docs/experiments/*`; `tests/` READMEs; `.github/` docs;
   `CITATION.cff`.

Acceptance criteria still open after this pass: both tranches audited; v4.0.0 released.
