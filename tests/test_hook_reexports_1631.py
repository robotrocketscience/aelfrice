"""#1631: which moved names `aelfrice.hook` still exports, and which it doesn't.

The memory-block switch, its constants and the UserPromptSubmit telemetry
reader moved to `aelfrice.hook_audit`. They are public, so `aelfrice.hook`
keeps exporting them for `from aelfrice.hook import ...` callers; static
analysis reads such a re-export as unused, which is why this pins it. The
private `_escape_attr` has no caller in `aelfrice.hook` and no external
contract, so it is not re-exported: a later sweep that adds it back
restores a dead alias.
"""
from __future__ import annotations

import pytest

from aelfrice import hook, hook_audit

PUBLIC_REEXPORTS = (
    "ENV_MEMORY_BLOCK",
    "MEMORY_BLOCK_ENABLED_KEY",
    "MEMORY_BLOCK_SECTION",
    "memory_block_enabled",
    "read_user_prompt_submit_telemetry",
)


@pytest.mark.parametrize("name", PUBLIC_REEXPORTS)
def test_the_public_name_resolves_to_its_new_home(name: str) -> None:
    assert getattr(hook, name) is getattr(hook_audit, name)


def test_the_private_attr_escaper_is_not_reexported() -> None:
    assert not hasattr(hook, "_escape_attr")
