# Quickstart

Five minutes from `pip install` to your first locked rule.

```bash
pip install aelfrice
```

## 1. Onboard

```bash
$ cd ~/projects/my-app
$ aelf onboard .
onboarded .: 287 added, 14 skipped (non-persisting), 412 candidates
```

The scanner reads three sources: `.md` / `.rst` / `.txt` / `.adoc` prose, `git log`, and Python AST. Everything else is ignored. Markdown headings, license boilerplate, and three-word fragments are filtered out.

## 2. Lock the rules that matter

```bash
$ aelf lock "never push to main; use scripts/publish.sh"
locked: a1f3c2d09e1b4f7a

$ aelf locked
[locked] a1f3c2d09e1b4f7a  never push to main; use scripts/publish.sh
```

Fresh locks start at `(α, β) = (9.0, 0.5)` ≈ 0.95 posterior. They short-circuit decay and always come back at L0.

## 3. Search

```bash
$ aelf search "deploy to production"
[locked] a1f3c2d09e1b4f7a  never push to main; use scripts/publish.sh
         91e02d3c          scripts/publish.sh wraps gitleaks + PII scan + tag verify
```

L0 (locked) is always returned first. L1 is BM25-ranked FTS5 hits, token-budgeted (default 2,400).

## 4. Wire into Claude Code

```bash
$ aelf setup
installed UserPromptSubmit hook in <project>/.claude/settings.json
```

Restart Claude Code. The next prompt mentioning "deploy" or "push" will already have the locked rule attached as `<aelfrice-memory>` above your message.

## 5. Inspect

```bash
$ aelf stats
beliefs=287  threads=42  locked=1  feedback_events=0  avg_confidence=0.71

$ aelf health
audit:
  [ok ] orphan_threads     all threads resolve to existing beliefs
  [ok ] fts_sync           FTS5 mirror in sync (287 rows)
  [ok ] locked_contradicts no unresolved contradictions between locked beliefs
  [ok ] corpus_volume      287 belief(s) (>= 50; project ~12d old)

$ aelf doctor
scanned user: ~/.claude/settings.json
summary: 1 ok, 0 broken, 0 skipped
```

## 6. Feedback

When a belief proves useful or harmful:

```bash
$ aelf feedback a1f3c2d09e1b4f7a used
α 9.0→10.0, β 0.5→0.5

$ aelf feedback 91e02d3c harmful
α 1.0→1.0, β 0.5→1.5
```

`used` bumps α; `harmful` bumps β. Five harmful events through `CONTRADICTS` edges to a lock auto-demote it.

> Partial Bayesian re-rank shipped at v1.3; BM25F default-on at v1.7. See [LIMITATIONS](LIMITATIONS.md) for what's still partial.

## Next

[Install](INSTALL.md) · [Commands](COMMANDS.md) · [Architecture](../concepts/ARCHITECTURE.md) · [Philosophy](../concepts/PHILOSOPHY.md)
