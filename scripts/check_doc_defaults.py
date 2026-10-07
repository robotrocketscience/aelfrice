"""Check documented flag defaults against the resolvers that decide them (#1654).

Usage: uv run python scripts/check_doc_defaults.py [ROOT]

Two kinds of claim are checked, each by calling the resolver with no
arguments, every AELFRICE_* variable cleared, and HOME and the working
directory pointed at an empty temporary directory, so no config file
applies:

1. A source comment of the form ``default-ON (is_flag_enabled)`` or
   ``Default-OFF (is_flag_enabled)``. The resolver is found by name
   wherever the imported ``aelfrice`` package defines it, so the
   scanned ROOT and the code under test are the same tree in a checkout.
2. A ``Boolean, default `true|false``` line under a ``### `key``` heading
   in docs/user/CONFIG.md, for the keys in ``CONFIG_RESOLVERS``. A key
   with no entry is printed as advisory, not checked.

Read-only. Exits 1 on any mismatch, or when no claim was checked at all.
"""
from __future__ import annotations

import importlib
import os
import re
import sys
import tempfile
from pathlib import Path

MARKER = re.compile(r"default-(on|off)[\s#]*\((is_[a-z0-9_]+)\)", re.IGNORECASE)
HEADING = re.compile(r"^## `\[([a-z0-9_.]+)\]`")
KEY = re.compile(r"^### `([a-z0-9_]+)`")
BOOL_LINE = re.compile(r"^Boolean, default `(true|false)`")

#: (section, key) -> "module:resolver". Each resolver must take no
#: required arguments and resolve env, then TOML, then its default.
CONFIG_RESOLVERS: dict[tuple[str, str], str] = {
    ("retrieval", "entity_index_enabled"): "retrieval:is_entity_index_enabled",
    ("retrieval", "use_entity_persist_demote"): "retrieval:is_entity_persist_demote_enabled",
    ("retrieval", "use_supersession_demote"): "retrieval:is_supersession_demote_enabled",
    ("retrieval", "use_origin_tiebreak"): "retrieval:is_origin_tiebreak_enabled",
    ("retrieval", "use_fan_effect"): "retrieval:is_fan_effect_enabled",
    ("retrieval", "bfs_enabled"): "retrieval:is_bfs_enabled",
    ("retrieval", "use_bm25f_anchors"): "retrieval:resolve_use_bm25f_anchors",
    ("retrieval", "use_heat_kernel"): "retrieval:is_heat_kernel_enabled",
    ("retrieval", "use_hrr_structural"): "retrieval:is_hrr_structural_enabled",
    ("retrieval", "hrr_persist"): "retrieval:is_hrr_persist_enabled",
    ("retrieval", "use_type_aware_compression"): "retrieval:resolve_use_type_aware_compression",
    ("implicit_feedback", "enqueue_on_retrieve"): "deferred_feedback:is_enqueue_on_retrieve_enabled",
    ("belief_categories", "enabled"): "category:is_enabled",
}


def _call(target: str) -> bool:
    module, name = target.split(":")
    return bool(getattr(importlib.import_module(f"aelfrice.{module}"), name)())


def _resolver_module(name: str) -> str | None:
    """The module of the imported package that defines `name`."""
    src = Path(importlib.import_module("aelfrice").__file__ or "").parent
    pattern = re.compile(rf"^def {name}\(", re.MULTILINE)
    for path in sorted(src.rglob("*.py")):
        if pattern.search(path.read_text(encoding="utf-8")):
            return path.relative_to(src).with_suffix("").as_posix().replace("/", ".")
    return None


def check(root: Path) -> tuple[list[str], list[str], int]:
    """Return (mismatches, advisories, number of claims checked)."""
    src = root / "src" / "aelfrice"
    bad: list[str] = []
    advisory: list[str] = []
    checked = 0
    for path in sorted(src.rglob("*.py")):
        for m in MARKER.finditer(path.read_text(encoding="utf-8")):
            where = _resolver_module(m.group(2))
            rel = path.relative_to(root)
            if where is None:
                bad.append(f"{rel}: no resolver named {m.group(2)}")
                continue
            checked += 1
            want, got = m.group(1).lower() == "on", _call(f"{where}:{m.group(2)}")
            if got != want:
                bad.append(f"{rel}: says default-{m.group(1)}, {m.group(2)}() returns {got}")
    section = key = ""
    config = root / "docs" / "user" / "CONFIG.md"
    for n, line in enumerate(config.read_text(encoding="utf-8").splitlines(), 1):
        if h := HEADING.match(line):
            section, key = h.group(1), ""
        elif k := KEY.match(line):
            key = k.group(1)
        elif b := BOOL_LINE.match(line):
            target = CONFIG_RESOLVERS.get((section, key))
            if target is None:
                advisory.append(f"CONFIG.md:{n}: [{section}] {key} has no mapped resolver")
                continue
            checked += 1
            want, got = b.group(1) == "true", _call(target)
            if got != want:
                bad.append(f"CONFIG.md:{n}: [{section}] {key} says {b.group(1)}, {target}() returns {got}")
    return bad, advisory, checked


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else Path(__file__).parents[1]).resolve()
    for name in [k for k in os.environ if k.startswith("AELFRICE_")]:
        del os.environ[name]
    empty = tempfile.mkdtemp(prefix="aelf-doc-defaults-")
    os.environ["HOME"] = empty
    os.chdir(empty)
    bad, advisory, checked = check(root)
    for line in advisory:
        print(f"advisory: {line}")
    for line in bad:
        print(f"MISMATCH: {line}")
    print(f"check_doc_defaults: {checked} claims checked, {len(bad)} mismatched")
    return 1 if bad or checked == 0 else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
