# aelfrice docs

aelfrice is a persistent memory layer for AI coding agents: a local SQLite store + `UserPromptSubmit` hook that injects matched beliefs into every prompt before the model sees it. Deterministic, auditable, no embeddings. See [`../README.md`](../README.md) for the install pitch.

Documentation, organized by audience.

| Bucket                                       | What's in it                                                       |
| -------------------------------------------- | ------------------------------------------------------------------ |
| [`user/`](user/)                             | Operational reference: install, commands, slash commands, MCP, config, privacy, limitations. Start here if you're using aelfrice. |
| [`concepts/`](concepts/)                     | Background and project context: architecture, philosophy, roadmap, benchmarks, releasing, harness integration. |
| [`design/`](design/)                         | Internal design specs and feature notes. Not user-facing — read these if you're contributing to a specific subsystem. (In-flight feature specs — `feature-*.md` — sit at the `docs/` root until they graduate.) |
| [`adr/`](adr/)                               | Architecture decision records. One file per decision; see [`adr/README.md`](adr/README.md). |
| [`audits/`](audits/)                         | Point-in-time analysis snapshots (e.g. CLI surface audits). Frozen at the date stamped in each file. |
| [`assets/`](assets/)                         | Images and other binary assets referenced from docs and the README. |
| [`bake_off_results/`](bake_off_results/)     | Raw JSON from internal benchmark bake-offs. |
| [`experiments/`](experiments/)               | One-off experiment write-ups (`EXP-NNN-...`). |

The release changelog lives at the repo root: [`../CHANGELOG.md`](../CHANGELOG.md).
