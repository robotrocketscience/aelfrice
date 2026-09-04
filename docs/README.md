# aelfrice docs

aelfrice is a persistent memory layer for AI coding agents. A local SQLite store holds the beliefs, and a `UserPromptSubmit` hook injects the matched beliefs into each prompt before the model reads the prompt. aelfrice is deterministic and auditable, and it uses no embeddings. For an overview of the project, read the top-level [`../README.md`](../README.md).

The following table lists the documentation directories, organized by audience.

| Directory                                    | Contents                                                           |
| -------------------------------------------- | ------------------------------------------------------------------ |
| [`user/`](user/)                             | The operational reference: installation, commands, slash commands, configuration, privacy, and limitations. Start here if you use aelfrice. |
| [`concepts/`](concepts/)                     | Project background and context: architecture, philosophy, roadmap, benchmarks, release procedure, and harness integration. |
| [`design/`](design/)                         | Internal design specifications and feature notes. These documents aren't written for users; read them if you contribute to a subsystem. Specifications for features still in development (`feature-*.md`) stay in the `docs/` root directory until they graduate. |
| [`adr/`](adr/)                               | Architecture decision records (ADR), one file per decision. For more information, read [`adr/README.md`](adr/README.md). |
| [`audits/`](audits/)                         | Point-in-time snapshots, such as audits of the CLI surface. Each snapshot is frozen at the date given in its file. |
| [`assets/`](assets/)                         | Images and other binary assets used by the documentation and the README file. |
| [`bake_off_results/`](bake_off_results/)     | Raw JSON output from the internal benchmark comparisons. |
| [`experiments/`](experiments/)               | Reports for individual experiments (`EXP-NNN-...`). |

For the release changelog, read [`../CHANGELOG.md`](../CHANGELOG.md) in the root directory of the repository.
