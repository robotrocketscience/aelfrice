"""hook-activity schema reservation: enforce PostToolUseFailure:* namespace.

Companion to docs/hook_activity_schema.md. The doc reserves the literal
event-name prefix `PostToolUseFailure` inside `~/.aelfrice/hook-activity.jsonl`
for the HOME-side failure-signal hook. This test guards the reservation
deterministically: if any future writer under `src/aelfrice/` emits the
literal string, this test fails with a descriptive collision message.

Pure stdlib + filesystem walk. Runtime: well under 1 s.
"""
from __future__ import annotations

from pathlib import Path

# The literal event-name prefix reserved by docs/hook_activity_schema.md.
# Producing hook lives in the user's HOME repo, not in src/aelfrice/.
RESERVED_EVENT_PREFIX: str = "PostToolUseFailure"

# Project root resolved from this test file's location:
# tests/test_hook_activity_schema.py -> repo root.
_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_SRC_DIR: Path = _REPO_ROOT / "src" / "aelfrice"


def _python_files(root: Path) -> list[Path]:
    """Every .py file under root, excluding __pycache__ trees. Sorted."""
    files: list[Path] = []
    for p in root.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        files.append(p)
    return sorted(files)


def test_no_aelfrice_writer_emits_post_tool_use_failure_event_name() -> None:
    """No file under src/aelfrice/ contains the literal `PostToolUseFailure`.

    The HOME-side hook owns this event-name namespace. Any aelfrice-side
    occurrence is a collision and would invalidate the schema reservation
    in docs/hook_activity_schema.md.
    """
    assert _SRC_DIR.is_dir(), f"expected src tree at {_SRC_DIR}"

    offenders: list[tuple[Path, int, str]] = []
    for path in _python_files(_SRC_DIR):
        # utf-8 decode with strict errors; aelfrice source is ascii-clean.
        text = path.read_text(encoding="utf-8")
        if RESERVED_EVENT_PREFIX not in text:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if RESERVED_EVENT_PREFIX in line:
                offenders.append((path.relative_to(_REPO_ROOT), lineno, line.strip()))

    if offenders:
        formatted = "\n".join(
            f"  {rel}:{lineno}: {snippet}" for rel, lineno, snippet in offenders
        )
        msg = (
            f"Found {len(offenders)} aelfrice-side reference(s) to the reserved "
            f"event-name prefix '{RESERVED_EVENT_PREFIX}'. "
            "This namespace is reserved for the HOME-side failure-signal hook "
            "by docs/hook_activity_schema.md; aelfrice writers must not emit "
            "it. Collisions:\n"
            f"{formatted}"
        )
        raise AssertionError(msg)
