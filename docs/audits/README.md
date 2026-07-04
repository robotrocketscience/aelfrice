# Audits

Point-in-time analysis snapshots. Each file is dated and frozen; if a finding is still relevant, file an issue or open a follow-up audit rather than editing the snapshot.

- [`CLI_SURFACE_AUDIT.md`](CLI_SURFACE_AUDIT.md) — historical v1.3-era inventory of the then-22 `aelf` subcommands and the visible/hidden consolidation proposal; for the current surface see [docs/user/COMMANDS.md](../user/COMMANDS.md).
- [`DOCS-AUDIT-2026-05-26.md`](DOCS-AUDIT-2026-05-26.md) — full documentation audit (130 files vs HEAD, run at v3.3.0).
- [`DOCS-AUDIT-2026-06-10.md`](DOCS-AUDIT-2026-06-10.md) — full documentation audit (≈2,500 claims vs HEAD, run at v3.5.0; #957).
- [`DOCS-AUDIT-2026-07-04.md`](DOCS-AUDIT-2026-07-04.md) — doc-file-surface audit vs HEAD `33b7ddd5` (v3.8.0 + post-release main), v4.0.0 release gate; #1075. 0 CRITICAL/HIGH, 9 MEDIUM, 8 LOW, 1 code-side. In-code documentation layer (552 .py) and remaining doc-file buckets (CHANGELOG, docs/design, ADRs, prior audits, experiments, tests/.github docs, CITATION.cff — ~90 files) deferred to follow-ups.
- [`DOCS-AUDIT-2026-07-04-incode.md`](DOCS-AUDIT-2026-07-04-incode.md) — in-code documentation layer (docstrings / `--help` / MCP descriptions across 103 `src/aelfrice/` .py) vs HEAD `33b7ddd5`, v4.0.0 release gate; #1075. 4 CRITICAL + 43 HIGH fixed in place; 48 MEDIUM/LOW + 1 discretion-sensitive HIGH confirmed + deferred; 1 rejected on adversarial re-verify. Closes #1077. `tests/` + `benchmarks/` .py docstrings deferred.
