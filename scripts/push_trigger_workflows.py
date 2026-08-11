#!/usr/bin/env python3
"""Enumerate the workflows whose `on: push` names a branch (#1423).

`merge-train.yml` lands every merge with
`git push origin "${HEAD_SHA}:refs/heads/main"` using `secrets.GITHUB_TOKEN`,
and GitHub does not start workflow runs from events raised by that token —
the documented recursion guard. So since the merge train became the merge
path, **no** `on: push: branches: [main]` workflow has run: the newest
push-event run across seven of them is `3421cd73`, 2026-07-21, and `main` has
moved 666+ commits since. Two of the seven (`release-drafter.yml`,
`flag-stale-open-prs.yml`) have no other trigger at all, so they were not
running late — they were not running.

The fix is for the train to dispatch them itself after a successful
fast-forward; `workflow_dispatch` and `repository_dispatch` are the two
documented exceptions to the `GITHUB_TOKEN` guard. That needs a list of what
to dispatch, and a hand-maintained list in the workflow is the failure this
issue already had once: the next `push:` workflow someone adds is silently
omitted. So the list is derived from the workflow files, here, and both
`merge-train.yml` (which dispatches) and `push-trigger-heartbeat.yml` (which
notices when dispatching has stopped working) read it from this one place.

Pure stdlib on purpose, and parsed as text rather than with PyYAML: `yaml` is
not in this project's dependency set — it reaches a local venv only as a
transitive dep of optional extras, so `import yaml` passes locally and fails
under CI's `uv sync --frozen --group dev --extra archive`. The same trap is
recorded in `tests/test_ci_manual_dispatch.py`.

The parse is deliberately narrow. `branches:` under `on: push:` is the whole
question; `tags:` is not (`publish.yml` triggers on `v*` tag pushes, which the
merge train never creates and which are unaffected by the token guard), and
neither is `branches-ignore:`.

Usage:

    python3 scripts/push_trigger_workflows.py            # one filename per line
    python3 scripts/push_trigger_workflows.py --branch release
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

_DEFAULT_WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _nested(lines: Sequence[str], start: int, at_indent: int) -> list[str]:
    """Lines under `lines[start]`, i.e. more-indented; blanks/comments dropped."""
    out: list[str] = []
    for line in lines[start + 1 :]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if _indent(line) <= at_indent:
            break
        out.append(line)
    return out


def _flow_list(value: str) -> list[str]:
    """`[main, 'release/*']` -> `['main', 'release/*']`."""
    inner = value.strip()[1:-1]
    return [item.strip().strip("\"'") for item in inner.split(",") if item.strip()]


def _key_at(lines: Sequence[str], name: str, at_indent: int) -> int | None:
    want = " " * at_indent + name + ":"
    for i, line in enumerate(lines):
        if line.rstrip() == want:
            return i
    return None


def push_branches(text: str) -> list[str]:
    """The branch patterns under a workflow's `on: push: branches:`.

    Empty for a workflow with no `push` trigger, and — importantly — empty for
    one whose `push` trigger names only `tags:`.
    """
    lines = text.splitlines()
    # `on` is the one YAML key GitHub's schema spells three ways, because bare
    # `on` is the YAML 1.1 boolean `true`; accept the quoted forms too.
    on_at = None
    for i, line in enumerate(lines):
        if _indent(line) == 0 and line.rstrip() in ("on:", '"on":', "'on':"):
            on_at = i
            break
    if on_at is None:
        return []

    on_block = _nested(lines, on_at, 0)
    push_at = _key_at(on_block, "push", 2)
    if push_at is None:
        return []

    push_block = _nested(on_block, push_at, 2)
    for i, line in enumerate(push_block):
        if _indent(line) != 4:
            continue
        stripped = line.strip()
        # `branches-ignore:` does not match: the colon is part of the prefix.
        if not stripped.startswith("branches:"):
            continue
        value = stripped[len("branches:") :].strip()
        if value.startswith("[") and value.endswith("]"):
            return _flow_list(value)
        branches: list[str] = []
        for follow in _nested(push_block, i, 4):
            item = follow.strip()
            if not item.startswith("- "):
                break
            branches.append(item[2:].strip().strip("\"'"))
        return branches
    return []


def workflows_pushed_on(workflows_dir: Path, branch: str) -> list[str]:
    """Sorted filenames of workflows whose `on: push: branches:` lists `branch`.

    Matched literally rather than as a glob: a wildcard pattern is a deliberate
    superset and should not be dispatched on every merge to one branch.
    """
    names = []
    for path in sorted(workflows_dir.iterdir()):
        if path.suffix not in (".yml", ".yaml") or not path.is_file():
            continue
        if branch in push_branches(path.read_text(encoding="utf-8")):
            names.append(path.name)
    return sorted(names)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--workflows-dir",
        type=Path,
        default=_DEFAULT_WORKFLOWS,
        help="directory of workflow files (default: .github/workflows)",
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="branch name to match against `on: push: branches:` (default: main)",
    )
    args = parser.parse_args(argv)
    for name in workflows_pushed_on(args.workflows_dir, args.branch):
        print(name)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
