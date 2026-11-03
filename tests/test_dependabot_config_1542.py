"""`.github/dependabot.yml` asks for weekly Python and action updates (#1542, #1786).

#1542 waited on a Dependabot PR that could not come: the repository had no
`dependabot.yml`, so Dependabot raised security updates only, and a security
update needs an alert with a patched version. Version updates on the `uv`
ecosystem cover the gap. These tests hold the configuration to the terms it
landed under: the `uv` updater and not `pip`, since only `uv` rewrites
uv.lock, and CI passes a PR that leaves the lock stale because its `uv run`
steps re-lock in the runner; a `github-actions` item beside it (#1786); a
weekly schedule and a seven-day cooldown on both; minor and patch bumps
grouped into one PR per ecosystem, with no `ignore` or `allow` filter
narrowing them; at most five open `uv` PRs; and the labels Dependabot's
security updates already apply, neither of which starts the merge train.
`tests/test_workflow_supply_chain_1786.py` holds the rule that nothing
under `.github/` enables auto-merge, this file included.

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
    the test rather than passing unread. The items are exactly
    `github-actions` and `uv`, one each, so a new ecosystem lands with a
    test of its own."""
    ecosystems = [_text(item, "package-ecosystem") for item in _items()]
    assert None not in ecosystems, ecosystems
    assert ecosystems.count("uv") == 1, ecosystems
    assert not {"pip", "pipenv", "poetry"} & set(ecosystems), ecosystems
    assert sorted(ecosystems) == ["github-actions", "uv"], ecosystems


def test_every_item_is_weekly_with_the_build_subject() -> None:
    """Each item, weekly, at the repository root, with the `build(deps):`
    subject the commit checks accept. Each key counts only at the depth
    Dependabot reads it: `directory` on the item, `interval` under
    `schedule`, and `prefix` and `include` under `commit-message`."""
    for item in _items():
        ecosystem = _text(item, "package-ecosystem")
        assert _text(item, "directory") == "/", (ecosystem, item)
        assert _text(_block(item, "schedule"), "interval") == "weekly", (ecosystem, item)
        message = _block(item, "commit-message")
        assert _text(message, "prefix") == "build", (ecosystem, message)
        assert _text(message, "include") == "scope", (ecosystem, message)


def test_uv_updates_are_capped_at_five_open_prs() -> None:
    """`open-pull-requests-limit` counts on the item, not under a nested
    key."""
    assert _text(_uv_item(), "open-pull-requests-limit") == "5"


def test_no_item_sets_a_key_that_narrows_the_weekly_bump() -> None:
    """Each item sets only the keys these tests read. `ignore` drops
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
    for item in _items():
        extra = sorted(set(item) - reviewed)
        assert not extra, (_text(item, "package-ecosystem"), extra)


def test_every_item_waits_out_a_seven_day_cooldown() -> None:
    """Dependabot proposes a release no sooner than seven days after it is
    published, in each ecosystem. Without `cooldown`, or with fewer days,
    zizmor's dependabot-cooldown audit raises a code-scanning alert on
    main; zizmor 1.31.1 flags the `github-actions` item as well as the
    `uv` one. Every `*-days` key holds at least seven, and no `include` or
    `exclude` list narrows the packages the cooldown covers. The
    `github-actions` ecosystem supports `default-days` only, not the
    `semver-*-days` keys. The block sits on the item itself, the only
    place Dependabot's schema defines `cooldown`, so a `cooldown:` nested
    under `schedule:` fails."""
    semver = {"semver-major-days", "semver-minor-days", "semver-patch-days"}
    items = _items()
    assert len(items) >= 2, items
    for item in items:
        ecosystem = _text(item, "package-ecosystem")
        cooldown = _block(item, "cooldown")
        rules = {key: _text(cooldown, key) or "" for key in cooldown}
        assert "default-days" in rules, (ecosystem, rules)
        days = {"default-days"} if ecosystem == "github-actions" else {"default-days", *semver}
        assert set(rules) <= days, (ecosystem, rules)
        short = {key: value for key, value in rules.items() if not value.isdigit() or int(value) < 7}
        assert not short, (ecosystem, short)


def test_each_group_takes_every_minor_and_patch_version_update() -> None:
    """In each item, one group holds the minor and patch version updates of
    every package, and no major: a grouped major would hide a breaking bump
    inside a routine PR, and a group with no `update-types` takes every
    type. `applies-to: security-updates` moves the group off version
    updates, and `patterns`, `exclude-patterns`, `dependency-type`, or
    `group-by` limit or split what it holds. The schema accepts each of
    them, and each leaves weekly bumps outside the group, one PR per
    package."""
    for item in _items():
        ecosystem = _text(item, "package-ecosystem")
        groups = _groups(item)
        assert len(groups) == 1, (ecosystem, groups)
        (rules,) = groups.values()
        assert sorted(rules.get("update-types", [])) == ["minor", "patch"], (ecosystem, rules)
        assert rules.get("applies-to", ["version-updates"]) == ["version-updates"], (ecosystem, rules)
        assert set(rules) <= {"update-types", "applies-to"}, (ecosystem, rules)


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
    reading the diff. The scan for an auto-merge rule in this file is
    `test_nothing_under_github_enables_auto_merge` in
    `tests/test_workflow_supply_chain_1786.py`, which reads every spelling
    this file and the workflows can use."""
    item = _uv_item()
    labels = _sequence(item, "labels")
    assert labels == ["dependencies", "python:uv"], labels
    assert "ready-to-merge" not in labels, labels
