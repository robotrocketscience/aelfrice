# Security policy

## Reporting a vulnerability

Email **security@robotrocketscience.com** (or open a GitHub Security Advisory if you prefer the GitHub flow). Do **not** open a public issue for security or privacy bugs.

Please include:

- A description of the issue and its impact (data exposure, integrity, availability).
- Reproduction steps or a proof-of-concept.
- The version of aelfrice you're running (`aelf --version`).
- Your environment (OS, Python version, MCP host if relevant).

We will acknowledge receipt within **48 hours** and aim to provide an initial assessment within **5 business days**.

## Scope

In scope:

- The published Python package on PyPI (`pip index versions aelfrice`).
- Anything in `src/aelfrice/` on `main`.
- The MCP server's input handling (`aelfrice.mcp_server`).
- The CLI's argument handling (`aelfrice.cli`).
- The SQLite schema and FTS5 index integrity.

Out of scope:

- Vulnerabilities in `fastmcp` (please report upstream).
- Vulnerabilities in your MCP host (Claude Code, Codex). Report upstream to the host vendor.
- Vulnerabilities in the cloud LLM at the other end of your prompt.
- Third-party tools you use to inspect the database (sqlite3, datasette, etc.).

## What aelfrice promises

- **No telemetry.** The shipped package contains no network code in the retrieval, scoring, scanner, store, or feedback paths. The optional `[mcp]` extra adds `fastmcp`, which speaks MCP over stdio (local IPC), not the network. The single default outbound call is the TTL-gated update notifier (`lifecycle.py`): one GET to `https://pypi.org/pypi/aelfrice/json` that transmits no user data; disable with `AELF_NO_UPDATE_CHECK=1`. See [docs/user/PRIVACY.md](docs/user/PRIVACY.md).
- **All data is local.** Your beliefs live in a single SQLite file, resolved per `src/aelfrice/db_paths.py`: `$AELFRICE_DB` overrides; otherwise `<git-common-dir>/aelfrice/memory.db` per-project; otherwise `~/.aelfrice/memory.db` as a legacy fallback for non-git cwds. aelfrice does not back this up, sync this, or transmit any portion of it.
- **Auditable update math.** Every Bayesian update goes through one function (`apply_feedback`, ~60 lines). Production retrieval ordering enters through one function (`retrieve` in `src/aelfrice/retrieval.py`). Both are pure Python with no I/O beyond the local SQLite file, and reviewable.

See [docs/user/PRIVACY.md](docs/user/PRIVACY.md) for verifiable details.

## Disclosure

We follow [coordinated disclosure](https://en.wikipedia.org/wiki/Coordinated_vulnerability_disclosure):

1. You report privately.
2. We acknowledge, triage, and develop a fix.
3. We coordinate a release with you on a target date.
4. We publish the release with credit to the reporter (if desired) and a security advisory describing the issue.

We do not currently run a paid bug bounty.

## Hall of fame

Will be added here as advisories land.
