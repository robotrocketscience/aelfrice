---
name: aelf:category
description: Manage keyword-triggered belief categories — group rules and bind them to triggers.
allowed-tools:
  - Bash
---
<objective>
Manage belief categories (#1126): named groups of beliefs (repo-rules,
git-workflow, prose-and-docs, …) bound to an activation trigger. When a
category's trigger fires — a keyword phrase in the prompt, or an
always-on category — its member rules are surfaced into context. This is
the conditional, right-rule-at-the-right-moment complement to a static
CLAUDE.md / AGENTS.md.

Injection is default-off and advisory (it never blocks a tool call).
Enable it with `AELFRICE_BELIEF_CATEGORIES=1` or `[belief_categories]
enabled = true` in `.aelfrice.toml`.
</objective>

<process>
Parse `$ARGUMENTS` as an `aelf category` sub-action and pass it through.

- No arguments, or `list` → run `uv run aelf category list` and display
  the output verbatim.
- `init` → run `uv run aelf category init` (seeds the 5 starter
  categories idempotently).
- `add <name> [--always-on] [--keyword PHRASE ...] [--tool-glob G ...]
  [--file-glob G ...] [--lock none|locked|advisory]` → run
  `uv run aelf category add <name> ...` with the given flags.
- `show <name>` → run `uv run aelf category show <name>`.
- `set-trigger <name> [--keyword ...] [--tool-glob ...] [--file-glob ...]`
  → run `uv run aelf category set-trigger <name> ...`.
- `assign <belief_id> <name>` / `unassign <belief_id> <name>` → run the
  matching `uv run aelf category assign|unassign <belief_id> <name>`.
- `delete <name>` → run `uv run aelf category delete <name>`.

To put a rule into a category at lock time instead, use
`uv run aelf lock "<rule>" --category <name>`.

Display the command's output verbatim. Do not add commentary.
</process>
