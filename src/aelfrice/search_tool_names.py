"""The tools the search-tool hook runs before (#1628).

A leaf module on purpose, holding the one copy of this fact. Two modules
need it: `aelfrice.hook_search_tool` decides at fire time whether a
payload's `tool_name` is a search tool, and `aelfrice.setup` writes the
matcher that makes the host call the hook at all. Before #1628 each kept
its own copy, so a tool added to one and not the other produced a hook
that fired for a tool nothing matched, or a matcher for a tool the hook
ignored, and no test failed.

Neither module imports the other. `setup` importing `hook_search_tool`
would put `setup` on the import cycle through `aelfrice.hook` (#1631),
and `hook_search_tool` importing `setup` would load the installer on
every hook fire. This module imports only `typing`.
"""
from __future__ import annotations

from typing import Final

# #1626: every tool that performs a search, not just the local ones.
#
# The value of the hook is ORDERING: aelfrice runs first, so the model
# already holds the relevant brain-graph context before it chooses grep,
# the web, or anything else. A search the model runs afterwards, or not
# at all, is worth much less. Covering only Grep and Glob left the web
# tools reaching out with no brain-graph context at all, and whether
# that path is taken is exactly the model choice this product exists to
# remove.
SEARCH_TOOL_NAMES: Final[tuple[str, ...]] = (
    "Grep",
    "Glob",
    "WebSearch",
    "WebFetch",
)

SEARCH_TOOL_MATCHER: Final[str] = "|".join(SEARCH_TOOL_NAMES)
"""The `PreToolUse` matcher `aelf setup` installs, derived from the names.

The matcher is a `|`-separated alternation of tool names, so joining the
names is the whole derivation. That holds only while no name contains a
regex metacharacter, which the binding test asserts."""
