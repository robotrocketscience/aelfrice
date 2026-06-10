# Quickstart

Five minutes from `uv tool install` to your first locked rule.

```bash
uv tool install aelfrice    # requires uv — https://docs.astral.sh/uv/
```

aelfrice is uv-only as of v3.0.1 ([#730](https://github.com/robotrocketscience/aelfrice/issues/730)); see [INSTALL](INSTALL.md) for the migration line if you have an older pipx-based install.

## 1. Onboard

```bash
$ cd ~/projects/my-app
$ aelf onboard .
onboarded .: 287 added, 0 skipped (already present), 14 skipped (non-persisting), 412 candidates seen
```

The scanner reads three sources: `.md` / `.rst` / `.txt` / `.adoc` prose, `git log`, and Python AST. Everything else is ignored. Markdown headings, license boilerplate, and three-word fragments are filtered out.

## 2. Lock the rules that matter

```bash
$ aelf lock "never push to main; use scripts/publish.sh"
locked: a1f3c2d09e1b4f7a

$ aelf locked
a1f3c2d09e1b4f7a: never push to main; use scripts/publish.sh
```

Fresh locks start at `(α, β) = (9.0, 0.5)` ≈ 0.95 posterior. They short-circuit decay and always come back at L0.

## 3. Search

```bash
$ aelf search "deploy to production"
[locked] a1f3c2d09e1b4f7a: never push to main; use scripts/publish.sh
         91e02d3c: scripts/publish.sh runs the release checklist before tagging
```

L0 (locked) is always returned first. L1 is BM25-ranked FTS5 hits, token-budgeted (default 2,400).

## 4. Wire into Claude Code

```bash
$ aelf setup
installed UserPromptSubmit hook in <project>/.claude/settings.json (project scope)
installed transcript-ingest hooks in <project>/.claude/settings.json
installed SessionStart hook in <project>/.claude/settings.json
installed Stop hook in <project>/.claude/settings.json
installed commit-ingest PostToolUse hook in <project>/.claude/settings.json
installed statusline in <project>/.claude/settings.json
...
```

Bare `aelf setup` wires the full default hook set — transcript capture, commit ingest, session-start injection, and the rest — not just retrieval; pass `--no-transcript-ingest` / `--no-commit-ingest` (etc.) to opt out per lane. See [INSTALL § default-on hooks](INSTALL.md).

Restart Claude Code. The next prompt mentioning "deploy" or "push" will already have the locked rule attached as `<aelfrice-memory>` above your message.

## 5. Inspect

```bash
$ aelf status
aelfrice <version>
beliefs: 287
threads: 42
locked: 1
feedback events: 0
hrr.persist_state: on 1181696 bytes, last build 0.4s

$ aelf doctor graph
audit:
  [ok ] orphan_threads     all threads resolve to existing beliefs
  [ok ] fts_sync           FTS5 mirror in sync (287 rows)
  [ok ] locked_contradicts no unresolved contradictions between locked beliefs
  ...(plus informational metrics: edges-by-type, credal gap, feedback coverage)

$ aelf doctor
scanned user: ~/.claude/settings.json
summary: 1 ok, 0 broken, 0 skipped
```

## 6. Feedback

When a belief proves useful or harmful:

```bash
$ aelf feedback a1f3c2d09e1b4f7a used
applied used to a1f3c2d09e1b4f7a: alpha 9.000->10.000, beta 0.500->0.500

$ aelf feedback 91e02d3c harmful
applied harmful to 91e02d3c: alpha 0.600->0.600, beta 1.000->2.000
```

`used` bumps α; `harmful` bumps β. Locks resist passive feedback by design ([#814](https://github.com/robotrocketscience/aelfrice/issues/814) removed the v2.x auto-demote mechanism at v3.2); change a wrong lock with `aelf unlock` / `aelf delete`, or re-lock the corrected statement.

> Partial Bayesian re-rank shipped at v1.3; BM25F default-on at v1.7. See [LIMITATIONS](LIMITATIONS.md) for what's still partial.

## Next

[Install](INSTALL.md) · [Commands](COMMANDS.md) · [Architecture](../concepts/ARCHITECTURE.md) · [Philosophy](../concepts/PHILOSOPHY.md)
