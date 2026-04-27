# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Pre-`1.0.0` releases are atomic milestones building toward the first
installable release; see the roadmap in [README.md](README.md).

## [Unreleased]

## [1.0.2] - 2026-04-27

Patch release: per-project install routing, `aelf doctor` settings
linter, and CI guardrails for release docs. Closes the v1.0.1 gap
where one machine couldn't cleanly route per-project venv hooks
alongside a global `pipx` install, and the README roadmap drifted out
of sync minutes after the wheel was on PyPI.

### Added

- `aelf doctor`: scan user-scope and project-scope Claude Code
  `settings.json` for hook + statusline commands whose program token
  doesn't resolve. Catches dangling absolute paths, bare names not on
  `$PATH`, and missing scripts under `bash /…` interpreter wrappers.
  Exits `1` on any broken finding so it can gate CI (#81).
- `staging-gate.yml` `release-docs-check` job: when a PR bumps
  `pyproject.toml` `version`, enforce `CHANGELOG.md` has a matching
  `## [X.Y.Z]` section + compare-link footnote, and that `README.md`
  has no roadmap row marking the released version as `next` /
  `planned`. No-op on non-release PRs (#80).
- `post-release-docs-issue.yml`: on `release.published`, opens a
  tracking issue `docs sweep for vX.Y.Z` with a per-doc checklist
  for the second-order docs the gate can't verify automatically
  (RELEASING.md test counts, ROADMAP.md narrative, etc.) (#80).

### Changed

- `aelf setup` no longer requires `--scope`. Default is auto-detect:
  `project` (writes `<cwd>/.claude/settings.json`) when `cwd/.venv`
  matches the active interpreter's `sys.prefix`, else `user` (writes
  `~/.claude/settings.json`). Explicit `--scope` still wins (#81).
- `aelf setup --command` defaults to an absolute path: project scope
  -> the active venv's `aelf-hook`; user scope -> the first
  `aelf-hook` on `$PATH` (typically a `pipx`-installed
  `~/.local/bin/aelf-hook`). Lets one machine route per-project venvs
  to their own hook AND fall back to a global pipx install outside
  any project, without bare-name `$PATH` collisions (#81).
- `aelf setup` now silently removes legacy dangling
  `/usr/local/bin/aelf{,-hook}` symlinks if their target no longer
  exists. Real files and live symlinks are never touched (#81).
- `aelf unsetup` defaults to basename-match cleanup (every entry
  whose program basename is `aelf-hook`), so an install written with
  the new auto-resolved absolute path can be torn down by bare
  `aelf unsetup` without specifying the path (#81).

## [1.0.1] - 2026-04-27

Patch release: power-user ergonomics + retrieval-side feedback loop.
Closes the v1.0.0 gap where hook retrievals never wrote audit rows,
adds a no-config noise filter (with a TOML escape hatch), and ships
`aelf --version` as a first-class CLI flag.

### Added

- `tests/regression/test_onboard_perf_50k_loc.py`: regression benchmark
  asserting `scan_repo` finishes in under 60s on a synthetic ~55k-LOC
  project (250 .py + 60 doc files). Marked `regression`. Held against
  the `:memory:` store. Current measured time ~0.8s on Apple Silicon;
  the 60s budget is a regression alarm, not a target (#NN).

### Added

- `aelfrice.hook_search` module: `search_for_prompt(store, prompt, ...)`
  and `record_retrieval(store, beliefs, ...)`. Closes the v1.0.1
  retrieval-side feedback-loop gap: every UserPromptSubmit hook
  retrieval now writes one `feedback_history` row per returned belief,
  tagged `source='hook'` with `HOOK_RETRIEVAL_VALENCE` (0.1) positive
  valence (#70).
- `propagate: bool = True` kwarg on `aelfrice.feedback.apply_feedback`.
  Default preserves the corrective-feedback contract (positive signal
  on a contradictor pressures the contradicted user lock); pass `False`
  for non-corrective signals (hook-driven retrievals) to update the
  posterior without pressure-walking locked beliefs (#70).
- `aelf --version` flag prints `aelf <__version__>` and exits 0.
  Closes the long-standing argparse error users hit when probing the
  installed version (#71).
- `aelfrice.noise_filter` module: pure-stdlib pure-function predicate
  `is_noise(text, config)` that drops candidate paragraphs matching
  one of four well-known non-belief shapes — markdown heading blocks,
  checklist blocks, three-word fragments, or license-header
  boilerplate (seven canonical signatures). Wired into
  `scanner.scan_repo` before classification. New
  `ScanResult.skipped_noise` counter (default 0 for back-compat)
  reports how many candidates were filtered per scan (#72).
- Power-user configuration via a single `.aelfrice.toml` at the
  project root (or any ancestor); discovered automatically by
  `scan_repo` walking up from the scan root. The `[noise]` table
  exposes: `disable` (subset of `headings`, `checklists`,
  `fragments`, `license`), `min_words` (override fragment
  threshold), `exclude_words` (whole-word case-insensitive — "jso"
  drops "jso" but never "json"), and `exclude_phrases` (literal
  substring case-insensitive). No regex in the user-facing schema.
  Library callers can pass an explicit `NoiseConfig` to
  `scan_repo(..., noise_config=...)` to bypass file discovery (#72).
- `aelf upgrade` subcommand: prints the right pip-upgrade command for
  the user's install context (venv / pipx / system), includes the
  published wheel SHA-256 + PyPI URL for hash-pinned installs (#73).
- `aelf uninstall` subcommand with mutually-exclusive `--keep-db` /
  `--purge` / `--archive PATH` disposition flags. `--purge` requires
  three redundant gates (explicit flag, typed `PURGE`, final `[y/N]`).
  `--archive` encrypts the DB with AES-128-CBC + HMAC via Fernet,
  scrypt-derived key from a user password (Salt embedded; password is
  the only secret needed for recovery). Public `decrypt_archive()` API
  for later recovery (#73).
- `aelf statusline` subcommand: emits an orange-coloured one-line
  update-banner prefix snippet for Claude Code's statusbar (and any
  shell-driven status bar). Empty output when no update is pending.
  Truecolor → 256-color → basic-yellow fallback, NO_COLOR honoured (#73).
- Two-component update notifier ported from GSD's pattern: detached
  background PyPI check writes a JSON cache at
  `~/.cache/aelfrice/update_check.json`; statusline + post-command
  banners read that cache only. Cache TTL: 24h. Opt out via
  `AELF_NO_UPDATE_CHECK=1` (#73).
- `aelf setup` auto-wires the statusline alongside the hook by default
  (`--no-statusline` opts out). Composes deterministically with an
  existing `statusLine`: empty slot → install; already ours → no-op;
  simple existing → wrap with `; aelf statusline 2>/dev/null`; complex
  (shell metacharacters) → leave alone with a one-line hint (#73).
- `aelf unsetup` reverses the statusline composition surgically (#73).
- New optional extra: `pip install 'aelfrice[archive]'` adds the
  `cryptography` dep that `aelf uninstall --archive` needs (#73).
- Slash commands: `/aelf:upgrade`, `/aelf:uninstall`, `/aelf:statusline` (#73).

### Changed

- `aelfrice.hook.user_prompt_submit` now routes through
  `aelfrice.hook_search.search_for_prompt` instead of calling
  `aelfrice.retrieval.retrieve()` directly. Same non-blocking contract,
  same OPEN_TAG/CLOSE_TAG output envelope, same payload schema; the
  behavioural difference is the audit-row write per returned belief
  (#70).
- `aelf setup` output now shows two lines (hook + statusline) on a
  fresh install. `aelf unsetup` shows the matching teardown lines (#73).
- README "Install" section rewritten with explicit verification
  commands; new "Upgrade" and "Uninstall" sections (#73).
- `docs/INSTALL.md` rewritten with explicit Statusline composition
  table, full uninstall reference (including archive recovery), and
  Update notifier opt-out instructions (#73).

## [1.0.0] - 2026-04-27

First installable PyPI release. The full v1.0 surface is the cumulative
shipped contents of v0.1.0–v0.9.0rc0; this release tags it, builds the
sdist + wheel, and publishes via the GitHub Actions Trusted Publisher
workflow.

### Added

- PyPI publication: `pip install aelfrice` (`pip install "aelfrice[mcp]"`
  for the MCP server extra).

### Changed

- Trove `Development Status` classifier promoted from `4 - Beta` to
  `5 - Production/Stable`.
- README headline and tagline rewritten for a post-rebuild audience:
  install instruction front-and-centre, "Status: under active rebuild"
  warning removed, "What works today" section retitled to v1.0.0.

## [0.9.0rc0] - 2026-04-26

Benchmark-harness milestone — final gate before `v1.0.0`. The `v0.9.0-rc`
roadmap row is shipped as PEP 440 release candidate `0.9.0rc0`.

### Added

- `aelfrice.benchmark` module — deterministic 16-belief × 16-query
  synthetic harness. Public surface: `BENCHMARK_NAME`,
  `seed_corpus(store)`, `run_benchmark(store, *, aelfrice_version,
  top_k=5)`, frozen `BenchmarkReport` dataclass with `hit_at_1`,
  `hit_at_3`, `hit_at_5`, `mrr`, `p50_latency_ms`, `p99_latency_ms`
  (#50).
- `aelf bench` CLI subcommand — prints the report as a single JSON
  document. Defaults to in-memory store for full reproducibility;
  `--db PATH` for an explicit on-disk SQLite file; `--top-k N` to
  override hit@k accounting depth (#50).
- `slash_commands/bench.md` keeping the CLI ↔ slash 1:1 invariant
  green at 11 commands (#50).
- `tests/regression/test_benchmark_cli_end_to_end.py` —
  `@pytest.mark.regression` end-to-end coverage of `aelf bench`
  default invocation, `--db PATH`, and `--top-k` override (#50).

### Notes

- The harness is the **measurement instrument**, not proof of the
  central feedback claim. Retrieval ranking in the v1.0 line is
  BM25-only (`store.search_beliefs` orders by `bm25(beliefs_fts)`,
  posterior `alpha`/`beta` are not consumed), so `apply_feedback`
  does not currently move benchmark scores. A v1.x retrieval
  upgrade that consumes posterior is the precondition for using
  this harness to claim feedback drives accuracy.

## [0.8.0] - 2026-04-26

Project-metadata milestone — everything required for PyPI publish (gated
until v1.0.0) is now present and verified.

### Added

- `LICENSE` file at repo root (MIT) matching the `pyproject.toml` license
  declaration. `uv build` bundles it into both wheel and sdist (#45).
- `CHANGELOG.md` following the [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/)
  format with retroactive sections for v0.1.0–v0.7.0 (#46).
- `docs/INSTALL.md` — install, configure, quickstart, Claude Code wiring,
  development workflow, uninstall (#47).
- `docs/ARCHITECTURE.md` — design principles, module map, data model,
  Bayesian update path, retrieval flow with ASCII diagram, Claude Code
  integration diagram, test layers, explicit "out of scope through v1.0.0"
  list (#47).
- README `## docs` section linking to both new files (#47).
- `pyproject.toml` PyPI-ready metadata pass: sharpened `description`,
  `authors` with email, explicit `license-files = ["LICENSE"]`, ten
  `keywords`, thirteen Trove `classifiers` (Beta dev status, Python 3 /
  3.12 / 3.13, MIT, Topic / Audience / `Typing :: Typed`), and
  `[project.urls]` (Homepage, Repository, Documentation, Changelog,
  Issues) (#48).

## [0.7.0] - 2026-04-26

Claude Code wiring milestone — aelfrice retrieval can now be installed
as a `UserPromptSubmit` hook with a single `aelf setup`.

### Added

- `aelfrice.setup` module: idempotent `install_user_prompt_submit_hook` /
  `uninstall_user_prompt_submit_hook` / `default_settings_path`
  functions that mutate a Claude Code `settings.json`. Atomic on-disk
  write via sibling tempfile + `os.replace` (#39).
- `aelfrice.hook` module: `aelfrice.hook:main` reads the
  `UserPromptSubmit` JSON payload from stdin, runs aelfrice retrieval,
  and writes an `<aelfrice-memory>...</aelfrice-memory>` block to stdout.
  Non-blocking by contract — every failure mode (empty stdin, malformed
  JSON, missing/blank/wrong-type prompt field, retrieval exceptions)
  exits 0 with no stdout (#40).
- `aelf setup` and `aelf unsetup` CLI subcommands wrapping the install /
  uninstall functions, with `--scope user|project`, `--project-root`,
  `--settings-path`, `--command`, `--timeout`, `--status-message`. CLI
  surface grows from 8 to 10 commands; matching `setup.md` and
  `unsetup.md` slash commands ship in `src/aelfrice/slash_commands/`
  (#41).
- `aelf-hook = "aelfrice.hook:main"` script in `[project.scripts]`. CLI
  default `--command` switches to `aelf-hook` (#42).
- End-to-end regression test in `tests/regression/` exercising
  `aelf setup` → real subprocess spawn of the recorded hook command →
  verify retrieval output → `aelf unsetup` (#43).

### Changed

- `aelfrice.__version__` and `uv.lock` synced to `0.6.0` after v0.6.0
  shipped (#38).

## [0.6.0] - 2026-04-26

CLI / MCP / slash-commands milestone — the user-facing surface.

### Added

- `aelfrice.cli` with eight subcommands (`onboard`, `search`, `lock`,
  `locked`, `demote`, `feedback`, `stats`, `health`) and the `aelf`
  console script in `[project.scripts]`. Folds `config.py` into the CLI
  and reorganises `scoring.py` / `store.py` (#32).
- `aelfrice.mcp_server` with eight FastMCP tools mirroring the CLI
  surface; `pip install aelfrice[mcp]` extra adds the `fastmcp`
  dependency (#35).
- `src/aelfrice/slash_commands/` directory with eight markdown slash
  commands matched 1:1 to CLI subcommands; an invariant test enforces
  the correspondence (#36).
- `aelfrice.health` module with regime classifier
  (insufficient-data / supersede / ignore / balanced) backed by
  confidence, lock density, and edge density features (#31).
- Polymorphic onboard state machine in `aelfrice.classification` (#34).
- `onboard_sessions` schema + CRUD helpers in `aelfrice.store` (#33).

### Fixed

- FTS5 query special characters are now escaped in `search_beliefs`
  (#30).

## [0.5.0] - 2026-04-26

Scanner milestone — onboarding from a project directory.

### Added

- `aelfrice.scanner` package with `scan_repo` orchestrator combining
  three extractors (filesystem walk, git log, AST) with the
  classification module and the store (#28).
- Filesystem-walk extractor (#25), git-log extractor (#26), and AST
  extractor (#27).
- `aelfrice.classification` with `TYPE_PRIORS` and a regex fallback
  (#24).
- End-to-end regression test for the onboarding flow (#29).

## [0.4.0] - 2026-04-26

Feedback-loop milestone — the central `apply_feedback` endpoint.

### Added

- `apply_feedback` endpoint in `aelfrice.feedback` performing
  Beta-Bernoulli updates and writing to the `feedback_history` table
  (#19).
- `feedback_history` table + `Store` helpers (#18).
- Demotion-pressure-on-contradiction-edge: contradicting feedback
  against a locked belief now increments `demotion_pressure` (#20).
- Auto-demote locked belief when `demotion_pressure` crosses threshold
  (#21).
- No-LLM correction detector in `aelfrice.correction` (#22).
- End-to-end regression test for the feedback loop (#23).

## [0.3.0] - 2026-04-26

Retrieval milestone — two-layer L0 locked + L1 FTS5 BM25.

### Added

- `aelfrice.retrieval` module with L0 locked-belief auto-load and L1
  FTS5 BM25 keyword search. Token-budgeted output. No HRR, no BFS
  multi-hop, no entity-index in the v1.0 line (#14).
- Lock test enforcing L0-before-L1 ordering invariant in retrieval
  output (#15).
- Property test: token-budget invariant holds across budget magnitudes
  (#17).

### Changed

- Pytest configured with `pytest-timeout` (5s default) and registered
  markers; subprocess-driven tests (v0.6.0+) override per-test (#16).

## [0.2.0] - 2026-04-26

Scoring milestone — Beta-Bernoulli posterior + decay.

### Added

- `aelfrice.scoring` with posterior mean, type-specific half-life decay
  (with lock-floor short-circuit), and a basic relevance combiner; plus
  the `test_bayesian_inertia` property test (#8).
- `test_decay_required` property test confirming decay actually moves
  posteriors over time (#9).
- `test_lock_floor_sharp` property test confirming a `user`-locked
  belief does not decay (#10).

### Changed

- Pyright strict mode enabled across `src/aelfrice` and `tests` (#12).

### Fixed

- `requirement` belief half-life corrected to 720h per spec (#11).

### Documentation

- README test-count corrected (#13).

## [0.1.0] - 2026-04-26

Foundation milestone — store, models, config.

### Added

- `aelfrice.models`: `Belief` / `Edge` dataclasses plus enum-style
  module-level constants for belief / edge / lock types; `aelfrice.config`
  with type half-lives and broker-attenuation parameters (#4).
- `aelfrice.store` with SQLite WAL journaling, FTS5 full-text search,
  CRUD for beliefs and edges, broker-confidence-attenuated
  `propagate_valence`, and `demotion_pressure` read/write (#5).
- Property test: `propagate_valence` broker-attenuation invariant (#6).
- Round-trip test for `demotion_pressure` reads + writes (#7).
- Initial repo scaffold: pyproject, README, GitHub Actions workflows,
  scan configs (commit `67b4343`).

[Unreleased]: https://github.com/robotrocketscience/aelfrice/compare/v1.0.2...HEAD
[1.0.2]: https://github.com/robotrocketscience/aelfrice/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/robotrocketscience/aelfrice/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/robotrocketscience/aelfrice/compare/v0.9.0rc0...v1.0.0
[0.9.0rc0]: https://github.com/robotrocketscience/aelfrice/compare/v0.8.0...v0.9.0rc0
[0.8.0]: https://github.com/robotrocketscience/aelfrice/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/robotrocketscience/aelfrice/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/robotrocketscience/aelfrice/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/robotrocketscience/aelfrice/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/robotrocketscience/aelfrice/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/robotrocketscience/aelfrice/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/robotrocketscience/aelfrice/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/robotrocketscience/aelfrice/releases/tag/v0.1.0
