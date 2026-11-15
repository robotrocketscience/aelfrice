"""Approve Codex hook handlers in a test `$CODEX_HOME` (#1812).

Codex starts a user handler only when a `[hooks.state]` entry in
config.toml holds the hash it computes for the handler now, so a test that
expects doctor to count a run approves the handler first, as `/hooks` does.
The hash comes from `host_codex._trust_hash`, which
`tests/test_codex_doctor_launch_1812.py` pins against hashes that Codex's
own code computed for fixed handlers.
"""
from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import cast

from aelfrice.host_codex import (
    _EVENT_KEY_LABELS,
    _run_command,
    _scan_json_depth,
    _toml_nesting_error,
    _trust_hash,
)

#: The line that starts the entries `approve_codex_hooks` adds, so a second
#: call replaces them instead of repeating a key.
MARKER = "# approvals added by tests/codex_approvals.py\n"


def approval_entries(
    codex_dir: Path, *, windows: bool = False,
    skip: frozenset[tuple[str, str, int, int]] = frozenset(),
) -> str:
    """`[hooks.state]` entries that approve each `command` handler Codex
    reads in hooks.json and config.toml, except the (file name, event,
    group, handler) tuples in `skip`."""
    lines: list[str] = []
    for name in ("hooks.json", "config.toml"):
        path = codex_dir / name
        try:
            text = path.read_text(encoding="utf-8").split(MARKER)[0]
            if name == "hooks.json":
                # Cut what nests past serde_json's limit, as doctor does.
                text, _cuts, too_deep = _scan_json_depth(text)
                doc = None if too_deep else json.loads(text)
            else:
                doc = None if _toml_nesting_error(text) else tomllib.loads(text)
        except (OSError, ValueError, RecursionError):
            continue
        events = doc.get("hooks") if isinstance(doc, dict) else None
        if not isinstance(events, dict):
            continue
        for event, groups in cast(dict[str, object], events).items():
            if event not in _EVENT_KEY_LABELS or not isinstance(groups, list):
                continue
            for g, group in enumerate(cast(list[object], groups)):
                handlers = (
                    cast(dict[str, object], group).get("hooks")
                    if isinstance(group, dict) else None
                )
                if not isinstance(handlers, list):
                    continue
                for h, handler in enumerate(cast(list[object], handlers)):
                    command = _run_command(handler, windows=windows)
                    if command is None or (name, event, g, h) in skip:
                        continue
                    digest, _why = _trust_hash(event, group, handler, command)
                    if digest is None:
                        continue
                    key = f"{path}:{_EVENT_KEY_LABELS[event]}:{g}:{h}"
                    lines.append(
                        f"[hooks.state.{json.dumps(key)}]\n"
                        f"trusted_hash = {json.dumps(digest)}\n",
                    )
    return "".join(lines)


def approve_codex_hooks(
    codex_dir: Path, *, windows: bool = False,
    skip: frozenset[tuple[str, str, int, int]] = frozenset(),
) -> None:
    """Append `approval_entries` to config.toml, replacing the ones an
    earlier call added. A config.toml that `tomllib` can't parse, or that
    the entries would make unparseable, is left as it is."""
    config = codex_dir / "config.toml"
    try:
        base = config.read_text(encoding="utf-8").split(MARKER)[0]
    except FileNotFoundError:
        base = ""
    except (OSError, ValueError):
        return
    entries = approval_entries(codex_dir, windows=windows, skip=skip)
    if not entries:
        return
    text = base + ("" if not base or base.endswith("\n") else "\n") + MARKER + entries
    if _toml_nesting_error(base) is not None:
        return
    try:
        tomllib.loads(base)
        tomllib.loads(text)
    except (tomllib.TOMLDecodeError, RecursionError):
        return
    config.write_text(text, encoding="utf-8")
