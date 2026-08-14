# aelfrice docs

aelfrice is a persistent memory layer for AI coding agents. A local SQLite store holds the beliefs. A `UserPromptSubmit` hook injects the matched beliefs into each prompt before the model reads the prompt. aelfrice is deterministic and auditable. aelfrice uses no embeddings. Read [`../README.md`](../README.md).

The table below lists the documentation. The documentation is organized by audience.

| Directory                                    | What is in it                                                      |
| -------------------------------------------- | ------------------------------------------------------------------ |
| [`user/`](user/)                             | The operational reference: the installation, the commands, the slash commands, the configuration, the privacy and the limitations. Read this directory first if you use aelfrice. |
| [`concepts/`](concepts/)                     | The background and the context of the project: the architecture, the philosophy, the roadmap, the benchmarks, the release procedure and the harness integration. |
| [`design/`](design/)                         | The internal design specifications and the notes on the features. These documents are not for the user. Read them if you contribute to one subsystem. The specifications of the features in development (`feature-*.md`) stay in the `docs/` root directory until they graduate. |
| [`adr/`](adr/)                               | The architecture decision records (ADR). There is one file for each decision. For more data, read [`adr/README.md`](adr/README.md). |
| [`audits/`](audits/)                         | The analysis snapshots at one point in time, for example the audits of the CLI surface. Each snapshot is frozen at the date that its file gives. |
| [`assets/`](assets/)                         | The images and the other binary assets that the documentation and the README file use. |
| [`bake_off_results/`](bake_off_results/)     | The raw JSON output of the internal benchmark comparisons. |
| [`experiments/`](experiments/)               | The reports of the individual experiments (`EXP-NNN-...`). |

The changelog for the releases is in the root directory of the repository: [`../CHANGELOG.md`](../CHANGELOG.md).
