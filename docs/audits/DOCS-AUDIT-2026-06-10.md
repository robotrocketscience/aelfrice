# Docs audit — 2026-06-10

> Full audit of the documentation surface against HEAD (`78f95f89`, v3.5.0), run under #957.
> ~2,500 checkable claims verified across 130+ files in 32 parallel batches; every CRITICAL/HIGH
> finding independently re-verified against the code before fixing (0 refuted). 291 findings:
> 4 CRITICAL, 62 HIGH, 107 MEDIUM, 118 LOW. All fixes land in the #957 PR as per-file atomic
> commits; this document is the audit record and disposition ledger.

## Scope

| Batch | Surface | Findings |
|---|---|---|
| A | README, docs/README, CONTRIBUTING, SECURITY, CODE_OF_CONDUCT, CHANGELOG(+v3), docs/concepts/* | 91 |
| B | docs/user/* + docs/feature-*.md | 103 |
| C | src/aelfrice/slash_commands/*.md | 24 |
| D | docs/design/* (status banners + present-tense current-behavior claims only) | 37 |
| E | docs/adr, docs/audits, docs/experiments, benchmarks READMEs | 25 |

Out of scope: docs/design/historical/ links (#956), design-doc bodies (historical record per
docs/design/README.md framing), lab-side measurement numbers (unverifiable from this repo).

## The four CRITICALs

1. **BENCHMARKS.md prescribed a nonexistent `--mode judge` harness invocation** — the κ run
   protocol's steps 1–2 exited with an argparse error if followed. Rewritten around the shipped
   host-agent eval-replay (#600) + `judges.llm_judge` operator flow.
2. **MCP.md documented `aelf:wonder_gc` `dry_run` default as `true`; code defaults `False`** —
   an agent calling it bare, expecting the documented preview, soft-deletes speculative beliefs.
3. **PRIVACY.md claimed INEDIBLE directory exclusion holds on every ingest path** — it holds for
   `aelf onboard` only; `ingest-transcript` checks file basenames alone. Doc now discloses the
   real behavior; the code-side fix is tracked as **#958**.
4. **benchmarks/context-rebuilder/README.md judge snippets read `replay_results.jsonl`**, a file
   no tool ever writes (FileNotFoundError on step 1). Snippets now export rows from the harness
   `--out` report.

## Cross-cutting themes

1. **v3.4/v3.5 version-gate drift** — graph/scope-out shipped v3.3.0; the belief-hygiene verbs
   shipped v3.5.0 (v3.4.0 was never tagged standalone); #814 first shipped in the v3.2.0 tag.
2. **Flag-surface drift** — graph/scope-out/tail/clamp-ghosts/setup/feed rows in COMMANDS and
   SLASH_COMMANDS named flags that don't exist or missed ones that do.
3. **"Stdlib-only / zero deps" residue** — numpy/scipy (v1.5) and snowballstemmer (v1.7) are
   hard deps; PHILOSOPHY/ARCHITECTURE/PRIVACY/INSTALL all restated.
4. **Budget confusion** — the per-prompt hook budget is 1,500; 2,400 is the `retrieve()` /
   `aelf search` default. README/PHILOSOPHY/PRIVACY all said 2,400 for the hook.
5. **Retrieval-lane naming** — the per-prompt lanes are L0 + L2.5 (entity index) + L1; BFS is
   the opt-in L3; HRR is a `retrieve_v2` structural-marker route, not a per-prompt lane.
6. **Spec-written changelog entries** — three v3.3.0 entries (#876/#875/#870) described designs,
   not what shipped; rewritten against code verified at the tags.
7. **"Status: spec" banners on shipped features** — 16 design docs re-bannered with shipped
   version + issue refs, bodies untouched (extends the 2026-05-26 audit's F40 batch).
8. **Process-line quoting in slash files** — flag-taking commands used `"$ARGUMENTS"`/`"$@"`,
   breaking flag tokenisation; normalised to unquoted `$ARGUMENTS`.

## Code-side drift surfaced (not doc bugs; filed separately)

- #958 — INEDIBLE directory exclusion missing on ingest-transcript paths.
- Stale docstrings/help-text batch (see the follow-up issue filed with this audit): mcp_server.py
  v1.2.0 `edges`-drop notes + module tool list missing wonder tools; cli.py `_cmd_eval` "bundled
  with the wheel"; context_rebuilder.py:628 docstring defaults; pre_issue_create_hook.py:323
  misleading inline-bypass hint; cli.py `--llm-classify` help "Default: off" vs default-on
  resolver; mab_reader.py missing the BENCHMARKS.md-mandated context-only sentence.

## Deferred (recorded, not fixed here)

- StructMemEval upstream URL differs between benchmarks/README.md and temporal_blend_sweep.py —
  needs upstream verification before reconciling.
- AMA-Bench dataset/license row stays "TBD on activation" pending the activation-time check.
- docs/feature-posterior-temperature.md's quoted `_l1_hits` snippet omits the #817 ζ branch
  (cosmetic; the mutual-exclusion contract is documented in prose).
- docs/design/ingest_enrichment.md re-bannered in place; relocating it to historical/ is left to
  the next historical/ sweep.

## Method note

Audit agents verified claims by reading src/ + tests/ and running read-only `--help`
invocations only (no state-changing commands). CRITICAL/HIGH findings each got an independent
adversarial verifier prompted to refute the finding; 0 of 66 were refuted. Three batches lost
their verifier pass to a rate-limit window and were recovered from the run journal; their
findings were spot-verified manually during application.
