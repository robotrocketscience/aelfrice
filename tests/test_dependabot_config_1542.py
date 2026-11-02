"""`.github/dependabot.yml` asks for weekly Python dependency updates (#1542).

#1542 waited on a Dependabot PR that could not come: the repository had no
`dependabot.yml`, so Dependabot raised security updates only, and a security
update needs an alert with a patched version. Version updates on the `uv`
ecosystem cover the gap. These tests hold the configuration to the terms it
landed under: the `uv` updater and not `pip`, since only `uv` rewrites
uv.lock, and CI passes a PR that leaves the lock stale because its `uv run`
steps re-lock in the runner; a weekly schedule; minor and patch bumps
grouped into one PR; at most five open PRs; the labels Dependabot's
security updates already apply; and no label or option that would merge a
PR without a person adding `ready-to-merge`.

Parsed by line rather than with PyYAML on purpose: `yaml` is not in this
project's dependency set, so an `import yaml` passes locally and fails
under CI's `uv sync --frozen`.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DEPENDABOT = _REPO / ".github" / "dependabot.yml"
_WORKFLOWS = _REPO / ".github" / "workflows"


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
    """Each item of the `updates:` list, as its lines. An item starts at
    any `-` line at the list's indent, whatever key comes first in it, and
    the list ends at the first line that is not indented past that dash."""
    items: list[list[str]] = []
    dash: int | None = None
    inside = False
    for line in lines:
        if not inside:
            inside = re.match(r"^updates:\s*$", line) is not None
            continue
        stripped = line.strip()
        starts = stripped == "-" or stripped.startswith("- ")
        if dash is None:
            if not starts:
                break
            dash = _indent(line)
        if starts and _indent(line) == dash:
            items.append([line])
        elif _indent(line) > dash:
            items[-1].append(line)
        else:
            break
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
    assert updates, "no item under updates:"
    uv = [item for item in updates if _value(item, "package-ecosystem") == "uv"]
    assert len(uv) == 1, [_value(item, "package-ecosystem") for item in updates]
    return uv[0]


def _scalars(raw: str) -> list[str]:
    """The values of a scalar or a flow list such as `["minor", "patch"]`."""
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        parts = raw[1:-1].split(",")
    else:
        parts = [raw]
    return [part.strip().strip("\"'") for part in parts if part.strip()]


def _groups(item: list[str]) -> dict[str, dict[str, list[str]]]:
    """Each group under `groups:`, as its keys and their values. A key's
    values come from a flow list on its line or from `- value` lines below."""
    groups: dict[str, dict[str, list[str]]] = {}
    depth: int | None = None
    name_depth: int | None = None
    rules: dict[str, list[str]] = {}
    key: str | None = None
    for line in item:
        if depth is None:
            if re.match(r"^\s*groups:\s*$", line):
                depth = _indent(line)
            continue
        if _indent(line) <= depth:
            break
        if name_depth is None:
            name_depth = _indent(line)
        stripped = line.strip()
        if _indent(line) == name_depth:
            assert stripped.endswith(":"), line
            rules = groups.setdefault(stripped[:-1].strip("\"'"), {})
            key = None
            continue
        entry = re.match(r"^-\s+(.+)$", stripped)
        if entry:
            assert key is not None, line
            rules[key].extend(_scalars(entry.group(1)))
            continue
        pair = re.match(r"^([A-Za-z-]+):\s*(.*)$", stripped)
        assert pair, line
        name: str = pair.group(1)
        rules[name] = _scalars(pair.group(2))
        key = name
    return groups


def test_no_python_updater_besides_uv() -> None:
    """No `pip`, `pipenv`, or `poetry` item sits next to the `uv` one. Each
    of those bumps pyproject.toml without rewriting uv.lock, CI passes the
    PR, and the stale lock merges. An item counts whichever key it lists
    first, and an item whose ecosystem this parser cannot read, such as a
    flow mapping, fails the test rather than passing unread."""
    _uv_item()
    items = _updates(_code_lines(_DEPENDABOT.read_text(encoding="utf-8")))
    ecosystems = [_value(item, "package-ecosystem") for item in items]
    assert None not in ecosystems, items
    assert not {"pip", "pipenv", "poetry"} & set(ecosystems), ecosystems


def test_uv_updates_are_weekly_and_capped() -> None:
    """The `uv` updater, weekly, five PRs open at most, and the
    `build(deps):` subject the commit checks accept."""
    item = _uv_item()
    assert _value(item, "directory") == "/", item
    assert _value(item, "interval") == "weekly", item
    assert _value(item, "open-pull-requests-limit") == "5", item
    assert _value(item, "prefix") == "build", item
    assert _value(item, "include") == "scope", item


def test_uv_updates_wait_out_a_seven_day_cooldown() -> None:
    """Dependabot proposes a release no sooner than seven days after it is
    published. Without `cooldown`, or with fewer days, zizmor's
    dependabot-cooldown audit raises a code-scanning alert on main. Every
    `*-days` key holds at least seven, and no `include` or `exclude` list
    narrows the packages the cooldown covers."""
    item = _uv_item()
    depth: int | None = None
    rules: dict[str, str] = {}
    for line in item:
        if depth is None:
            if re.match(r"^\s*cooldown:\s*$", line):
                depth = _indent(line)
            continue
        if _indent(line) <= depth:
            break
        pair = re.match(r"^\s*([A-Za-z-]+):\s*(.*?)\s*$", line)
        assert pair, line
        rules[pair.group(1)] = pair.group(2)
    assert "default-days" in rules, rules
    days = {"default-days", "semver-major-days", "semver-minor-days", "semver-patch-days"}
    assert set(rules) <= days, rules
    short = {key: value for key, value in rules.items() if not value.isdigit() or int(value) < 7}
    assert not short, short


def test_uv_group_takes_every_minor_and_patch_version_update() -> None:
    """One group holds the minor and patch version updates of every
    package. `applies-to: security-updates` moves the group off version
    updates, and `patterns`, `exclude-patterns`, `dependency-type`, or
    `group-by` limit or split what it holds. The schema accepts each of
    them, and each leaves weekly bumps outside the group, one PR per
    package."""
    groups = _groups(_uv_item())
    assert len(groups) == 1, groups
    (rules,) = groups.values()
    assert sorted(rules.get("update-types", [])) == ["minor", "patch"], rules
    assert rules.get("applies-to", ["version-updates"]) == ["version-updates"], rules
    assert set(rules) <= {"update-types", "applies-to"}, rules


def test_stale_lock_reason_matches_how_ci_installs() -> None:
    """The reason given for `uv` over `pip` is that CI passes a PR with a
    stale uv.lock and the stale lock merges. Every workflow installs with
    `uv sync --frozen`, which installs the locked pins without checking
    them against pyproject.toml. The test steps then call plain `uv run`,
    which re-locks in the runner and installs the newer versions, so the
    tests run on the new pins. No workflow runs `uv lock --check` or
    `uv sync --locked`, and none keeps `uv run` from syncing with
    `--frozen`, `--no-sync`, or a `UV_FROZEN` or `UV_NO_SYNC` variable.
    The config comment must say that `uv run` re-locks, and neither it nor
    a docstring here may say that a stale lock makes `--frozen` fail or that
    the tests see the old versions. If a workflow changes how it installs,
    the reason changes, and this test asks for the comment to change with
    it."""
    syncs: list[str] = []
    test_runs: list[str] = []
    checks: list[str] = []
    for workflow in sorted(_WORKFLOWS.glob("*.yml")):
        for line in _code_lines(workflow.read_text(encoding="utf-8")):
            where = f"{workflow.name}: {line.strip()}"
            if "uv sync" in line:
                syncs.append(where)
            if re.search(r"\buv run\b.*\bpytest\b", line):
                test_runs.append(where)
            if re.search(
                r"uv lock\b.*--check|--locked\b|UV_(?:LOCKED|FROZEN|NO_SYNC)\b",
                line,
            ):
                checks.append(where)
    assert syncs, "no workflow runs `uv sync`"
    assert all("--frozen" in sync for sync in syncs), syncs
    assert test_runs, "no workflow runs pytest under `uv run`"
    pinned = [run for run in test_runs if re.search(r"--(?:frozen|no-sync)\b", run)]
    assert not pinned, pinned
    assert not checks, checks
    comment = " ".join(
        line.lstrip().lstrip("#").strip()
        for line in _DEPENDABOT.read_text(encoding="utf-8").splitlines()
        if line.lstrip().startswith("#")
    )
    assert "uv sync --frozen" in comment, comment
    assert re.search(r"`uv run`.*\bre-locks\b", comment), comment
    false_claim = re.compile(
        r"--frozen`?\s+fails|\bon the old (?:versions|pins)\b", re.IGNORECASE
    )
    docs = [__doc__ or ""] + [
        test.__doc__ or ""
        for name, test in sorted(globals().items())
        if name.startswith("test_") and callable(test)
    ]
    for text in [comment] + [" ".join(doc.split()) for doc in docs]:
        assert not false_claim.search(text), text


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
