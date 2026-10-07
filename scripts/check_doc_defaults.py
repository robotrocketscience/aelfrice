"""Check documented flag defaults against the resolvers that decide them (#1654).

Usage: uv run python scripts/check_doc_defaults.py [ROOT]

Two kinds of claim are checked, each by calling the resolver the way
production does with no config: every AELFRICE_* variable cleared, and
HOME and the working directory pointed at an empty temporary directory.

1. A source comment of the form ``default-ON (is_flag_enabled)`` or
   ``Default-OFF (``is_flag_enabled``)``. Each must be listed in
   ``MARKERS`` by file and resolver, which names the module to call.
2. A ``Boolean, default `true|false``` line under a ``### `key``` heading
   in docs/user/CONFIG.md, for the keys in ``CONFIG_RESOLVERS``. A
   Boolean line whose key has no entry is printed as advisory.

Both tables are required, not optional: a listed marker or key that the
scan doesn't find fails the check, so deleting or rewording a claim can't
turn it into a silent skip. Read-only. Exits 1 on any failure.
"""
from __future__ import annotations

import importlib
import os
import re
import sys
import tempfile
from pathlib import Path

MARKER = re.compile(r"default-(on|off)[\s#]*\(`{0,2}(is_[a-z0-9_]+)`{0,2}\)", re.IGNORECASE)
HEADING = re.compile(r"^## `\[([a-z0-9_.]+)\]`")
KEY = re.compile(r"^### `([a-z0-9_]+)`")
BOOL_LINE = re.compile(r"^Boolean, default `(true|false)`")

#: (file under ROOT, resolver named in the marker) -> "module:resolver".
MARKERS: dict[tuple[str, str], str] = {
    ("src/aelfrice/ingest.py", "is_auto_relationship_detection_enabled"):
        "relationship_detector:is_auto_relationship_detection_enabled",
    ("src/aelfrice/ingest.py", "is_temporal_spine_write_enabled"):
        "temporal_spine:is_temporal_spine_write_enabled",
    ("src/aelfrice/retrieval.py", "is_entity_persist_demote_enabled"):
        "retrieval:is_entity_persist_demote_enabled",
    ("src/aelfrice/temporal_spine.py", "is_temporal_spine_write_enabled"):
        "temporal_spine:is_temporal_spine_write_enabled",
}

#: (section, key) -> ("module:resolver", positional args). Each resolver
#: returns the default when no override applies; most read an env var and
#: then `.aelfrice.toml`, and some read only the env var or the dict passed
#: in. The args mirror a call with no config file; `retrieve()` passes a
#: literal `False` for `use_origin_tiebreak`, which the env tier still
#: overrides.
CONFIG_RESOLVERS: dict[tuple[str, str], tuple[str, tuple[object, ...]]] = {
    ("retrieval", "entity_index_enabled"): ("retrieval:is_entity_index_enabled", ()),
    ("retrieval", "use_entity_persist_demote"): ("retrieval:is_entity_persist_demote_enabled", ()),
    ("retrieval", "use_supersession_demote"): ("retrieval:is_supersession_demote_enabled", ()),
    ("retrieval", "use_origin_tiebreak"): ("retrieval:is_origin_tiebreak_enabled", ()),
    ("retrieval", "use_fan_effect"): ("retrieval:is_fan_effect_enabled", ()),
    ("retrieval", "bfs_enabled"): ("retrieval:is_bfs_enabled", ()),
    ("retrieval", "use_bm25f_anchors"): ("retrieval:resolve_use_bm25f_anchors", ()),
    ("retrieval", "use_heat_kernel"): ("retrieval:is_heat_kernel_enabled", ()),
    ("retrieval", "use_hrr_structural"): ("retrieval:is_hrr_structural_enabled", ()),
    ("retrieval", "hrr_persist"): ("retrieval:is_hrr_persist_enabled", ()),
    ("retrieval", "use_type_aware_compression"): ("retrieval:resolve_use_type_aware_compression", ()),
    ("implicit_feedback", "enqueue_on_retrieve"): ("deferred_feedback:is_enqueue_on_retrieve_enabled", ()),
    # hook.py passes the loaded TOML, which is {} when no file exists.
    ("belief_categories", "enabled"): ("category:is_enabled", ({},)),
    # The TOML tier only; the Stop hook reads the env var first (#1740).
    ("core_gate", "session_end"): (
        "hook:_core_gate_session_end_toml_enabled", (None, sys.stderr),
    ),
}


def _call(target: str, args: tuple[object, ...] = ()) -> bool:
    module, name = target.split(":")
    return bool(getattr(importlib.import_module(f"aelfrice.{module}"), name)(*args))


def check(root: Path) -> tuple[list[str], list[str], int]:
    """Return (failures, advisories, number of claims checked)."""
    bad: list[str] = []
    advisory: list[str] = []
    checked = 0
    seen_markers: set[tuple[str, str]] = set()
    for path in sorted((root / "src" / "aelfrice").rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        for m in MARKER.finditer(path.read_text(encoding="utf-8")):
            target = MARKERS.get((rel, m.group(2)))
            if target is None:
                bad.append(f"{rel}: marker for {m.group(2)} is not listed in MARKERS")
                continue
            seen_markers.add((rel, m.group(2)))
            checked += 1
            want, got = m.group(1).lower() == "on", _call(target)
            if got != want:
                bad.append(f"{rel}: says default-{m.group(1)}, {target}() returns {got}")
    bad += [f"{f}: no default-ON/OFF marker for {r}" for f, r in MARKERS if (f, r) not in seen_markers]
    section = key = ""
    seen_keys: set[tuple[str, str]] = set()
    config = root / "docs" / "user" / "CONFIG.md"
    for n, line in enumerate(config.read_text(encoding="utf-8").splitlines(), 1):
        if h := HEADING.match(line):
            section, key = h.group(1), ""
        elif k := KEY.match(line):
            key = k.group(1)
        elif b := BOOL_LINE.match(line):
            entry = CONFIG_RESOLVERS.get((section, key))
            if entry is None:
                advisory.append(f"CONFIG.md:{n}: [{section}] {key} has no mapped resolver")
                continue
            seen_keys.add((section, key))
            checked += 1
            want, got = b.group(1) == "true", _call(*entry)
            if got != want:
                bad.append(f"CONFIG.md:{n}: [{section}] {key} says {b.group(1)}, {entry[0]}() returns {got}")
    bad += [f"CONFIG.md: no `Boolean, default` line for [{s}] {k}"
            for s, k in CONFIG_RESOLVERS if (s, k) not in seen_keys]
    return bad, advisory, checked


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else Path(__file__).parents[1]).resolve()
    for name in [k for k in os.environ if k.startswith("AELFRICE_")]:
        del os.environ[name]
    origin, home = os.getcwd(), os.environ.get("HOME")
    with tempfile.TemporaryDirectory(prefix="aelf-doc-defaults-") as empty:
        os.environ["HOME"] = empty
        os.chdir(empty)
        try:
            bad, advisory, checked = check(root)
        finally:
            os.chdir(origin)
            if home is None:
                del os.environ["HOME"]
            else:
                os.environ["HOME"] = home
    for line in advisory:
        print(f"advisory: {line}")
    for line in bad:
        print(f"FAIL: {line}")
    print(f"check_doc_defaults: {checked} claims checked, {len(bad)} failed")
    return 1 if bad or checked == 0 else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
