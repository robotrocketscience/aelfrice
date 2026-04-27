# Quickstart

Five minutes from `pip install` to your first feedback loop.

```bash
pip install aelfrice
```

## 1. Onboard

```bash
$ aelf onboard ~/projects/my-app
onboarded: 287 added, 14 skipped (non-persisting), 412 candidates
```

The scanner reads three sources: `.md`/`.rst`/`.txt`/`.adoc` prose, git log, Python AST. Everything else is ignored.

> v1.0 caveat: the synchronous onboard path's noise filter isn't yet wired. Expect markdown headings and short fragments in the output until v1.x. See [LIMITATIONS](LIMITATIONS.md#known-issues-at-v10).

## 2. Search

```bash
$ aelf search "deploy to production"
[locked] a1f3c2…  Never push directly to main; use scripts/publish.sh
         91e02d…  scripts/publish.sh wraps gitleaks + PII scan + tag verify
```

L0 = locked beliefs (always first). L1 = FTS5 BM25 hits, token-budgeted (default 2000).

## 3. Lock + feedback

```bash
$ aelf lock "Never push directly to main; use scripts/publish.sh"
locked: a1f3c2d09e1b4f7a

$ aelf feedback a1f3c2d09e1b4f7a used
α 9.0→10.0, β 0.5→0.5
```

Fresh locks start at `(α, β) = (9.0, 0.5)` ≈ 0.95 confidence. `used` bumps α; `harmful` bumps β. Locks short-circuit decay; auto-demote after 5 contradictions.

> v1.0 caveat: the updated posterior is recorded but doesn't yet drive retrieval ranking. The math is in place; v1.x wires it into ordering.

## 4. Wire it up

```bash
$ aelf setup
installed UserPromptSubmit hook in ~/.claude/settings.json
```

Every prompt to Claude Code is now preceded by an `<aelfrice-memory>` block.

## 5. Inspect

```bash
$ aelf stats
beliefs=287  edges=42  locked=1  feedback_events=1

$ aelf health
brain mode: early-onboarding (confidence 0.78)

$ aelf bench
{"hit_at_1": 0.875, "hit_at_3": 1.0, ..., "p50_latency_ms": 0.4, ...}
```

## Next

[COMMANDS](COMMANDS.md) · [MCP](MCP.md) · [SLASH_COMMANDS](SLASH_COMMANDS.md) · [ARCHITECTURE](ARCHITECTURE.md) · [PHILOSOPHY](PHILOSOPHY.md)
