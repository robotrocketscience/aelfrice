"""#1639 AC4: the census behind the issue's table counts what it claims."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "hook_output_census.py"


@pytest.fixture(scope="module")
def census_mod() -> ModuleType:
    spec = importlib.util.spec_from_file_location("hook_output_census", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hook_output_census"] = mod
    spec.loader.exec_module(mod)
    return mod


def _att(content: str, event: str = "UserPromptSubmit",
         ts: str = "2026-09-25T00:00:00Z") -> str:
    return json.dumps({
        "type": "attachment", "timestamp": ts,
        "attachment": {"type": "hook_success", "hookEvent": event,
                       "content": content},
    }) + "\n"


def _lock(bid: str) -> str:
    return f'<belief id="{bid}" lock="user">rule</belief>'


@pytest.mark.timeout(30)
def test_census_counts_inline_saved_and_lost_locks(
    census_mod: ModuleType, tmp_path: Path,
) -> None:
    root = tmp_path / "projects"
    (root / "p").mkdir(parents=True)
    full_a = tmp_path / "a.txt"   # both locks; preview keeps one
    full_a.write_text("<aelfrice-memory>" + _lock("aa") + _lock("bb"))
    full_b = tmp_path / "b.txt"   # one lock; preview keeps it
    full_b.write_text("<aelfrice-memory>" + _lock("cc"))
    stub = ("<persisted-output>\nOutput too large ({kb}KB). Full output "
            "saved to: {path}\n\nPreview (first 2KB):\n<aelfrice-memory>{pv}")
    lines = [
        _att("<aelfrice-memory>" + "x" * 500),                       # inline
        _att(stub.format(kb=12.0, path=full_a, pv=_lock("aa"))),     # lost bb
        _att(stub.format(kb=20.0, path=full_b, pv=_lock("cc")),
             event="SessionStart"),                                  # kept
        _att(stub.format(kb=30.0, path=tmp_path / "gone.txt", pv="")),  # gone
        _att("unrelated hook output"),                               # skipped
        _att("<aelfrice-memory>old", ts="2026-01-01T00:00:00Z"),     # --since
    ]
    (root / "p" / "t.jsonl").write_text("".join(lines))

    got = census_mod.census(root)
    assert got == {
        "inline": 2,
        "saved": 3,
        "saved_by_event": {"SessionStart": 1, "UserPromptSubmit": 2},
        "saved_size_kb": {"min": 12.0, "median": 20.0, "max": 30.0},
        "saved_with_locks": {"lost_from_preview": 1, "all_in_preview": 1,
                             "file_gone": 1},
        "max_inline_chars": len("<aelfrice-memory>" + "x" * 500),
    }
    since = census_mod.census(root, census_mod._parse_ts("2026-09-01T00:00:00Z"))
    assert since["inline"] == 1


@pytest.mark.timeout(30)
def test_missing_root_exits_2(census_mod: ModuleType, tmp_path: Path) -> None:
    assert census_mod.main(["--root", str(tmp_path / "absent")]) == 2
