---
name: aelf:show
description: Print one belief by id, or by an unambiguous id prefix — content in full, origin, lock level, retention class, posterior, timestamps, and scope. It changes no belief, and it reaches retired ones.
argument-hint: The belief ID to print (a prefix works when only one belief starts with it)
allowed-tools:
  - Bash
---
<objective>
Turn a belief id back into the belief. Every injected `<belief>` element
carries an id, and `show` is the way back to the full text when the block
you are reading carries only a fragment of it.

The lookup is by id, not by content. `aelf search` is full-text search over
what a belief says, so an id there matches on whatever it tokenizes into;
`show` addresses the row. A partial id resolves only when exactly one belief
starts with it. Otherwise the command names the candidates and exits
non-zero — it never picks one for you.

A retired belief is reachable and prints `status: retired`, because the id in
an old block may name one. An unknown id exits 1 with a "no belief with id"
message on stderr.

The command changes no belief, and it reads a store you cannot write: it
opens through `open_store_for_read()` (#1416), which falls back to a
read-only handle when the writable open is refused for lack of write
access. That is the whole of what the routing buys. Against a store you
*can* write, the handle is the ordinary writable one, so the open still
pays the schema battery, the migrations, the scope-id mint, and the
expired-lock sweep — `show` is not a write-free command there.
</objective>

<process>
Run: `uv run aelf show $ARGUMENTS`
Display the output verbatim. Do not add commentary.
</process>
