"""`.github/dependabot.yml` asks for weekly Python dependency updates (#1542).

#1542 waited on a Dependabot PR that could not come: the repository had no
`dependabot.yml`, so Dependabot raised security updates only, and a security
update needs an alert with a patched version. Version updates on the `uv`
ecosystem cover the gap. These tests hold the configuration to the terms it
landed under: the `uv` updater and not `pip`, since only `uv` rewrites
uv.lock, and CI passes a PR that leaves the lock stale because its `uv run`
steps re-lock in the runner; a weekly schedule; minor and patch bumps
grouped into one PR, with no `ignore` or `allow` filter narrowing them;
at most five open PRs; the labels Dependabot's
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


Entry = tuple[str, list[str]]


def _uncomment(raw: str) -> str:
    """`raw` without a trailing YAML comment. A comment starts at a `#` at
    the start of `raw` or after whitespace, outside a quoted scalar, so
    `"pip" # note` reads as `"pip"`, the value Dependabot sees."""
    quote: str | None = None
    for index, char in enumerate(raw):
        if quote is not None:
            if char == quote:
                quote = None
        elif char in "\"'" and (index == 0 or raw[index - 1] in " [,"):
            quote = char
        elif char == "#" and (index == 0 or raw[index - 1].isspace()):
            return raw[:index].strip()
    return raw.strip()


def _mapping(lines: list[str]) -> dict[str, Entry]:
    """The keys of a block mapping at the indent of its first line, each as
    the value on its line and the lines nested under it. A key indented
    deeper belongs to the key above it, so a `cooldown:` under `schedule:`
    is a key of `schedule`, not of the item. A `- ` line at the key indent
    continues the key above it, as YAML allows for a list. A line this
    parser cannot read, or a key given twice, fails the test."""
    assert lines, "empty mapping"
    column = _indent(lines[0])
    keys: dict[str, Entry] = {}
    last: str | None = None
    for line in lines:
        stripped = line.strip()
        dash = stripped == "-" or stripped.startswith("- ")
        if _indent(line) > column or (dash and _indent(line) == column):
            assert last is not None, line
            keys[last][1].append(line)
            continue
        pair = re.match(r"^([A-Za-z][\w-]*|\"[^\"]*\"|'[^']*'):(?:\s+(.*))?$", stripped)
        assert _indent(line) == column and pair, line
        last = pair.group(1).strip("\"'")
        assert last not in keys, f"{last} given twice"
        keys[last] = (_uncomment(pair.group(2) or ""), [])
    return keys


def _entries(lines: list[str]) -> list[list[str]]:
    """Each `- ` entry of a block sequence, as its lines, with the dash
    blanked so the keys of a mapping entry line up. An entry starts at any
    dash at the list's indent, whatever key comes first in it."""
    assert lines, "empty list"
    column = _indent(lines[0])
    entries: list[list[str]] = []
    for line in lines:
        stripped = line.strip()
        if _indent(line) == column and (stripped == "-" or stripped.startswith("- ")):
            entries.append([])
            line = line[:column] + " " + line[column + 1 :]
            if not line.strip():
                continue
        assert entries and _indent(line) > column, line
        entries[-1].append(line)
    return entries


def _text(fields: dict[str, Entry], key: str) -> str | None:
    """The scalar on the line of `key`, or None when `key` is absent."""
    if key not in fields:
        return None
    inline, nested = fields[key]
    assert not nested, (key, nested)
    return inline.strip("\"'")


def _block(fields: dict[str, Entry], key: str) -> dict[str, Entry]:
    """The mapping nested under `key`, which must hold a block of its own."""
    assert key in fields, (key, sorted(fields))
    inline, nested = fields[key]
    assert not inline and nested, (key, inline)
    return _mapping(nested)


def _scalars(raw: str) -> list[str]:
    """The values of a scalar or a flow list such as `["minor", "patch"]`."""
    raw = _uncomment(raw)
    if raw.startswith("[") and raw.endswith("]"):
        parts = raw[1:-1].split(",")
    else:
        parts = [raw]
    return [part.strip().strip("\"'") for part in parts if part.strip()]


def _sequence(fields: dict[str, Entry], key: str) -> list[str]:
    """The values under `key`: a scalar or flow list on its line, or one
    `- value` line per value below it."""
    assert key in fields, (key, sorted(fields))
    inline, nested = fields[key]
    if inline:
        assert not nested, (key, nested)
        return _scalars(inline)
    values: list[str] = []
    for entry in _entries(nested):
        assert len(entry) == 1, entry
        values.append(_uncomment(entry[0]).strip("\"'"))
    return values


def _items() -> list[dict[str, Entry]]:
    """Each item of the top-level `updates:` list, as its keys."""
    assert _DEPENDABOT.is_file(), _DEPENDABOT
    document = _mapping(_code_lines(_DEPENDABOT.read_text(encoding="utf-8")))
    assert _text(document, "version") == "2", document
    assert "updates" in document and not document["updates"][0], document
    return [_mapping(entry) for entry in _entries(document["updates"][1])]


def _uv_item() -> dict[str, Entry]:
    items = _items()
    uv = [item for item in items if _text(item, "package-ecosystem") == "uv"]
    assert len(uv) == 1, [_text(item, "package-ecosystem") for item in items]
    return uv[0]


def _groups(item: dict[str, Entry]) -> dict[str, dict[str, list[str]]]:
    """Each group under the item's `groups:` key, as its keys and their
    values. A key's values come from a flow list on its line or from
    `- value` lines below."""
    groups = _block(item, "groups")
    return {
        name: {key: _sequence(rules, key) for key in rules}
        for name, rules in ((name, _block(groups, name)) for name in groups)
    }


def test_no_python_updater_besides_uv() -> None:
    """No `pip`, `pipenv`, or `poetry` item sits next to the `uv` one. Each
    of those bumps pyproject.toml without rewriting uv.lock, CI passes the
    PR, and the stale lock merges. An item counts whichever key it lists
    first, a trailing comment does not hide its ecosystem, and an item
    whose ecosystem this parser cannot read, such as a flow mapping, fails
    the test rather than passing unread."""
    ecosystems = [_text(item, "package-ecosystem") for item in _items()]
    assert None not in ecosystems, ecosystems
    assert ecosystems.count("uv") == 1, ecosystems
    assert not {"pip", "pipenv", "poetry"} & set(ecosystems), ecosystems


def test_uv_updates_are_weekly_and_capped() -> None:
    """The `uv` updater, weekly, five PRs open at most, and the
    `build(deps):` subject the commit checks accept. Each key counts only
    at the depth Dependabot reads it: `directory` and
    `open-pull-requests-limit` on the item, `interval` under `schedule`,
    and `prefix` and `include` under `commit-message`."""
    item = _uv_item()
    assert _text(item, "directory") == "/", item
    assert _text(_block(item, "schedule"), "interval") == "weekly", item
    assert _text(item, "open-pull-requests-limit") == "5", item
    message = _block(item, "commit-message")
    assert _text(message, "prefix") == "build", message
    assert _text(message, "include") == "scope", message


def test_uv_item_sets_no_key_that_narrows_the_weekly_bump() -> None:
    """The `uv` item sets only the keys these tests read. `ignore` drops
    packages or update types from the weekly bump and `allow` limits it to
    the packages it names, so either one leaves dependencies without the
    minor and patch updates the group promises. A key outside the set,
    such as `target-branch` or `versioning-strategy`, fails as well, so a
    new key lands with a test of what it changes."""
    reviewed = {
        "package-ecosystem",
        "directory",
        "schedule",
        "cooldown",
        "open-pull-requests-limit",
        "labels",
        "commit-message",
        "groups",
    }
    item = _uv_item()
    assert set(item) <= reviewed, sorted(set(item) - reviewed)


def test_uv_updates_wait_out_a_seven_day_cooldown() -> None:
    """Dependabot proposes a release no sooner than seven days after it is
    published. Without `cooldown`, or with fewer days, zizmor's
    dependabot-cooldown audit raises a code-scanning alert on main. Every
    `*-days` key holds at least seven, and no `include` or `exclude` list
    narrows the packages the cooldown covers. The block sits on the item
    itself, the only place Dependabot's schema defines `cooldown`, so a
    `cooldown:` nested under `schedule:` fails."""
    cooldown = _block(_uv_item(), "cooldown")
    rules = {key: _text(cooldown, key) or "" for key in cooldown}
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
    labels = _sequence(item, "labels")
    assert labels == ["dependencies", "python:uv"], labels
    assert "ready-to-merge" not in labels, labels
    merging = [
        line
        for line in _code_lines(_DEPENDABOT.read_text(encoding="utf-8"))
        if "auto-merge" in line.lower() or "automerge" in line.lower()
    ]
    assert not merging, merging
