# Quickstart

This procedure needs five minutes. It starts at `uv tool install`. It ends at your first locked rule.

```bash
uv tool install aelfrice    # requires uv — https://docs.astral.sh/uv/
```

aelfrice supports only uv as of v3.0.1 ([#730](https://github.com/robotrocketscience/aelfrice/issues/730)). If you have an older install that uses pipx, read [INSTALL](INSTALL.md) for the migration line.

## 1. Onboard

```bash
$ cd ~/projects/my-app
$ aelf onboard .
onboarded .: 287 added, 0 skipped (already present), 14 skipped (non-persisting), 412 candidates seen
```

The scanner reads three sources:

- the prose in `.md`, `.rst`, `.txt` and `.adoc` files
- `git log`
- the Python abstract syntax tree (AST)

The scanner ignores all other sources. The scanner also filters out the markdown headings, the license boilerplate and the three-word fragments.

## 2. Lock the rules that matter

```bash
$ aelf lock "never push to main; use scripts/publish.sh"
locked: a1f3c2d09e1b4f7a

$ aelf locked
a1f3c2d09e1b4f7a: never push to main; use scripts/publish.sh
```

A new lock starts at `(α, β) = (9.0, 0.5)`. That value is a posterior of approximately 0.95. A lock short-circuits decay. aelfrice always returns a locked belief at L0.

## 3. Search

```bash
$ aelf search "deploy to production"
[locked] a1f3c2d09e1b4f7a: never push to main; use scripts/publish.sh
         91e02d3c: scripts/publish.sh runs the release checklist before tagging
```

The search always returns the L0 (locked) beliefs first. L1 gives the hits from full-text search version 5 (FTS5). Best Matching 25 (BM25) ranks the L1 hits. A token budget limits the L1 hits. The default token budget is 2,400.

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

Bare `aelf setup` installs the full default hook set. That set contains these items:

- the transcript capture
- the commit ingest
- the session-start injection
- the other default hooks

The set covers capture as well as retrieval. To opt out of one lane, pass the option for that lane, such as `--no-transcript-ingest` or `--no-commit-ingest`. See [INSTALL § default-on hooks](INSTALL.md).

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

`used` increases α. `harmful` increases β. Locks resist passive feedback. This behaviour is intentional. [#814](https://github.com/robotrocketscience/aelfrice/issues/814) removed the v2.x auto-demote mechanism at v3.2. To change a wrong lock, use `aelf unlock` or `aelf delete`. You can also lock the corrected statement again.

> The partial Bayesian re-rank shipped at v1.3. BM25F became default-on at v1.7. See [LIMITATIONS](LIMITATIONS.md) for the parts that are still partial.

## Next

[Install](INSTALL.md) · [Commands](COMMANDS.md) · [Architecture](../concepts/ARCHITECTURE.md) · [Philosophy](../concepts/PHILOSOPHY.md)
