# Security policy

## Reporting a vulnerability

Send an email to **security@robotrocketscience.com**. If you prefer the GitHub flow, open a GitHub Security Advisory instead. Do **not** open a public issue for a security bug or a privacy bug.

In your report, include:

- A description of the issue and its impact on data exposure, integrity, and availability.
- Steps to reproduce the issue, or a proof-of-concept.
- The version of aelfrice you run (`aelf --version`).
- Your environment: the operating system, the Python version, and the host agent, if relevant.

We acknowledge receipt within **48 hours**, and we aim to send an initial assessment within **5 business days**.

## Scope

In scope:

- The published Python package on PyPI (`pip index versions aelfrice`).
- Any file in `src/aelfrice/` on `main`.
- The CLI's argument handling (`aelfrice.cli`).
- The integrity of the SQLite schema and the full-text search version 5 (FTS5) index.

Out of scope:

- A vulnerability in your host agent. Report it upstream to the host's vendor.
- A vulnerability in the cloud large language model (LLM) that receives your prompt.
- A third-party tool you use to inspect the database, such as sqlite3 or datasette.

## What aelfrice promises

- **No telemetry.** The shipped package contains no network code in the retrieval, scoring, scanner, store, or feedback paths. By default, two outbound calls are enabled. The first is the update notifier (`lifecycle.py`), which a time-to-live (TTL) gate controls. The notifier sends one GET request to `https://pypi.org/pypi/aelfrice/json`, and that request carries no user data. To turn the notifier off, set `AELF_NO_UPDATE_CHECK=1`. The second is the pre-issue duplicate guard, which runs `gh issue list --search` with tokens from your issue title, and only when you run `gh issue create`. To turn the guard off, set `AELFRICE_NO_PRE_ISSUE_GUARD=1` or run `aelf setup --no-pre-issue-guard`. For the details, see [the privacy reference](docs/user/PRIVACY.md).
- **All data is local.** Your beliefs live in a single SQLite file, and `src/aelfrice/db_paths.py` resolves its path: `$AELFRICE_DB` if you set it, otherwise the per-project path `<git-common-dir>/aelfrice/memory.db`, otherwise `~/.aelfrice/memory.db` as a legacy fallback when the current directory isn't in a git repository. aelfrice doesn't back this file up, doesn't sync it, and doesn't transmit any part of it.
- **Auditable update mathematics.** Every Bayesian update runs through one function (`apply_feedback`, under 200 lines), and production retrieval gets its ordering from one function (`retrieve` in `src/aelfrice/retrieval.py`). Both are pure Python, and neither does any input or output beyond the local SQLite file, so you can review both.

For details you can verify, see [the privacy reference](docs/user/PRIVACY.md).

## Disclosure

We follow [coordinated disclosure](https://en.wikipedia.org/wiki/Coordinated_vulnerability_disclosure):

1. You report the issue privately.
2. We acknowledge the report, triage it, and develop a fix.
3. We coordinate a target release date with you.
4. We publish the release with a security advisory that describes the issue. If you want credit, we credit you.

We don't currently run a paid bug bounty.

## Credit for reporters

We list reporter credits here as advisories go out.
