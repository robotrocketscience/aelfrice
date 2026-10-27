"""`.github/dependabot.yml` asks for weekly Python dependency updates (#1542).

#1542 waited on a Dependabot PR that could not come: the repository had no
`dependabot.yml`, so Dependabot raised security updates only, and a security
update needs an alert with a patched version. Version updates on the `uv`
ecosystem cover the gap. These tests hold the configuration to the terms it
landed under: the `uv` updater and not `pip`, since only `uv` rewrites
uv.lock and `uv sync --frozen` fails on a stale lock; a weekly schedule;
minor and patch bumps grouped into one PR; at most five open PRs; the
labels Dependabot's security updates already apply; and no label or option
that would merge a PR without a person adding `ready-to-merge`.

Parsed by line rather than with PyYAML on purpose: `yaml` is not in this
project's dependency set, so an `import yaml` passes locally and fails
under CI's `uv sync --frozen`.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DEPENDABOT = _REPO / ".github" / "dependabot.yml"

_ITEM_START = "  - package-ecosystem:"


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _code_lines(text: str) -> list[str]:
    """Every line that is neither blank nor a whole-line comment."""
    return [
        line
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _updates(lines: list[str]) -> list[list[str]]:
    """Each `- package-ecosystem:` item under `updates:`, as its lines."""
    items: list[list[str]] = []
    for line in lines:
        if line.startswith(_ITEM_START):
            items.append([line])
        elif items and _indent(line) >= 4:
            items[-1].append(line)
    return items


def _value(lines: list[str], key: str) -> str | None:
    """The scalar after `key:` on the first line that carries it."""
    for line in lines:
        match = re.match(rf"^\s*(?:-\s+)?{re.escape(key)}:\s*(.+?)\s*$", line)
        if match:
            return match.group(1).strip().strip("\"'")
    return None


def _list_items(lines: list[str], key: str) -> list[str]:
    """The `- value` lines nested under `key:`."""
    items: list[str] = []
    depth: int | None = None
    for line in lines:
        if depth is None:
            if re.match(rf"^\s*{re.escape(key)}:\s*$", line):
                depth = _indent(line)
            continue
        if _indent(line) <= depth:
            break
        match = re.match(r"^\s*-\s+(.+?)\s*$", line)
        if match:
            items.append(match.group(1).strip().strip("\"'"))
    return items


def _uv_item() -> list[str]:
    assert _DEPENDABOT.is_file(), _DEPENDABOT
    lines = _code_lines(_DEPENDABOT.read_text(encoding="utf-8"))
    assert _value(lines, "version") == "2", lines
    updates = _updates(lines)
    assert updates, "no `- package-ecosystem:` item under updates:"
    uv = [item for item in updates if _value(item, "package-ecosystem") == "uv"]
    assert len(uv) == 1, [_value(item, "package-ecosystem") for item in updates]
    return uv[0]


def test_uv_updates_are_weekly_grouped_and_capped() -> None:
    """The `uv` updater, weekly, minor and patch in one group, five PRs open
    at most, and the `build(deps):` subject the commit checks accept."""
    item = _uv_item()
    assert _value(item, "directory") == "/", item
    assert _value(item, "interval") == "weekly", item
    assert _value(item, "open-pull-requests-limit") == "5", item
    assert _value(item, "prefix") == "build", item
    assert _value(item, "include") == "scope", item
    groups = [line for line in item if "update-types:" in line]
    assert len(groups) == 1, item
    update_types = re.findall(r"[\"']([a-z]+)[\"']", groups[0].split(":", 1)[1])
    assert sorted(update_types) == ["minor", "patch"], groups


def test_uv_updates_carry_the_dependency_labels_and_wait_for_the_train() -> None:
    """The PR labels are the ones security updates already apply. Neither
    label starts the merge train; a person adds `ready-to-merge` after
    reading the diff, and nothing in the file merges on its own."""
    item = _uv_item()
    labels = _list_items(item, "labels")
    assert labels == ["dependencies", "python:uv"], labels
    assert "ready-to-merge" not in labels, labels
    merging = [
        line
        for line in _code_lines(_DEPENDABOT.read_text(encoding="utf-8"))
        if "auto-merge" in line.lower() or "automerge" in line.lower()
    ]
    assert not merging, merging
