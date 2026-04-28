---
name: aelf:migrate
description: Copy beliefs from the legacy global ~/.aelfrice/memory.db into the active project's per-project DB.
allowed-tools:
  - Bash
---
<objective>
Migrate from v1.0's single global DB layout to v1.1.0's per-project
layout. Reads the legacy DB at ~/.aelfrice/memory.db read-only, copies
project-relevant beliefs (and their edges) into the active project's
.git/aelfrice/memory.db. Dry-run by default — re-run with --apply to
actually write. The default filter copies only beliefs whose content
references the absolute project root path; pass --all to copy
everything regardless.

Never deletes from the source. Idempotent: re-running on a clean
target is a no-op.
</objective>

<process>
Run: `uv run aelf migrate`
Display the dry-run report verbatim. If the user wants to proceed,
they re-run with `aelf migrate --apply` themselves.
</process>
