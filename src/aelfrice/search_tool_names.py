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
every hook fire. This module imports only `typing` and the leaf
`aelfrice.hosts`.
"""
from __future__ import annotations

from typing import Final, Mapping

from aelfrice.hosts import CLAUDE_HOST, CODEX_HOST, GEMINI_HOST

# #1626: every tool that performs a search, not just the local ones.
#
# The hook attaches the relevant brain-graph context to every search the
# model makes. The host shows it next to the tool's result, after the
# model has chosen the tool and its query (#1646; hooks reference,
# https://code.claude.com/docs/en/hooks), so it shapes the next step
# rather than this search. Covering only Grep and Glob left the web
# tools returning with no brain-graph context at all, and whether memory
# gets consulted is exactly the model choice this product exists to
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

SHELL_TOOL_NAME: Final[str] = "Bash"
"""The tool name a shell command arrives under, which the Bash lane reads."""

# #1812: the tool names the search-tool hook acts on, per host. The hook
# process reads the host from its caller, and the console script passes
# the default, so a Codex payload is read with the claude names. That is
# also correct for Codex, which reports shell commands as `Bash` (#1055)
# and installs no search-tool matcher. The gemini entries are the Gemini
# CLI tool names; `aelf setup --host gemini` installs no tool hook yet,
# because the input fields of its tool payloads are not documented.
SEARCH_TOOL_NAMES_BY_HOST: Final[Mapping[str, tuple[str, ...]]] = {
    CLAUDE_HOST: SEARCH_TOOL_NAMES,
    CODEX_HOST: SEARCH_TOOL_NAMES,
    GEMINI_HOST: (
        "search_file_content",
        "glob",
        "google_web_search",
        "web_fetch",
    ),
}
SHELL_TOOL_NAME_BY_HOST: Final[Mapping[str, str]] = {
    CLAUDE_HOST: SHELL_TOOL_NAME,
    CODEX_HOST: SHELL_TOOL_NAME,
    GEMINI_HOST: "run_shell_command",
}
