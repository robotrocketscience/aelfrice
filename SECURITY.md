# Security policy

## Reporting a vulnerability

Send an email to **security@robotrocketscience.com**. If you prefer the GitHub flow, open a GitHub Security Advisory instead. Do **not** open a public issue for a security bug or a privacy bug.

Please include:

- A description of the issue and of its impact. The impact covers data exposure, integrity, and availability.
- The steps to reproduce the issue, or a proof-of-concept.
- The version of aelfrice that you run (`aelf --version`).
- Your environment: the operating system, the Python version, and the host agent if the host agent is relevant.

We acknowledge receipt within **48 hours**. We aim to give an initial assessment within **5 business days**.

## Scope

In scope:

- The published Python package on PyPI (`pip index versions aelfrice`).
- Any file in `src/aelfrice/` on `main`.
- The argument handling of the CLI (`aelfrice.cli`).
- The integrity of the SQLite schema and of the full-text search version 5 (FTS5) index.

Out of scope:

- A vulnerability in your host agent. Report it upstream to the vendor of the host.
- A vulnerability in the cloud large language model (LLM) that receives your prompt.
- A third-party tool that you use to inspect the database, such as sqlite3 or datasette.

## What aelfrice promises

- **No telemetry.** The shipped package contains no network code in the retrieval, scoring, scanner, store or feedback paths. The one outbound call in the default configuration is the update notifier (`lifecycle.py`). A time-to-live (TTL) gate controls the notifier. The notifier makes one GET request to `https://pypi.org/pypi/aelfrice/json`, and this request transmits no user data. To disable the notifier, set `AELF_NO_UPDATE_CHECK=1`. See [docs/user/PRIVACY.md](docs/user/PRIVACY.md).
- **All data is local.** Your beliefs live in a single SQLite file. `src/aelfrice/db_paths.py` resolves the path of that file. `$AELFRICE_DB` overrides the path. Otherwise the per-project path `<git-common-dir>/aelfrice/memory.db` applies. Otherwise `~/.aelfrice/memory.db` applies as a legacy fallback for a current directory that is not in a git repository. aelfrice does not back up this file, does not sync this file, and does not transmit any portion of it.
- **Auditable update mathematics.** Every Bayesian update goes through one function (`apply_feedback`, about 60 lines). The ordering of production retrieval enters through one function (`retrieve` in `src/aelfrice/retrieval.py`). Both functions are pure Python. Neither function does input or output beyond the local SQLite file. You can review both functions.

For details that you can verify, see [docs/user/PRIVACY.md](docs/user/PRIVACY.md).

## Disclosure

We follow [coordinated disclosure](https://en.wikipedia.org/wiki/Coordinated_vulnerability_disclosure):

1. You report the issue privately.
2. We acknowledge the report. We triage the report. We develop a fix.
3. We coordinate a release with you on a target date.
4. We publish the release with a security advisory that describes the issue. If the reporter wants credit, we credit the reporter.

We do not currently run a paid bug bounty.

## Credit for reporters

We add entries here as advisories are published.
