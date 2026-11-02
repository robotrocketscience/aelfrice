"""Supply-chain rules for `.github/`, pinned to the tree (#1786).

The 2026-10-21 CI audit found 16 `uses:` lines that named a tag, three
`uvx` calls that ran whatever PyPI served that day, a `zizmor` check that
could not fail (`continue-on-error: true`), and `harden-runner` in every
Linux job regardless of what the job could do with its token. Each of
those is a rule that a reviewer can forget, so each is a test here.

Parsed by line, like the neighbouring workflow tests: PyYAML is not in the
dependency set CI installs (#1436).
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_GITHUB = _REPO / ".github"
_WORKFLOWS = _GITHUB / "workflows"
_DEPENDABOT = _GITHUB / "dependabot.yml"
_ZIZMOR = _WORKFLOWS / "zizmor.yml"
_CONTRIBUTING = _REPO / "CONTRIBUTING.md"
_SCRIPTS = _REPO / "scripts"



def _workflow_files(directory: Path) -> list[Path]:
    """Every file GitHub runs as a workflow: both `.yml` and `.yaml`."""
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix in (".yml", ".yaml")
    )


_WORKFLOW_FILES = _workflow_files(_WORKFLOWS)

_USES = re.compile(r"^\s*(?:-\s+)?uses:\s*(?P<ref>\S+)(?P<rest>.*)$")
# owner/repo[/path]@<40-hex>, then a comment carrying the full tag.
_PINNED = re.compile(
    r"^[\w.-]+/[\w.-]+(?:/[\w./-]+)?@[0-9a-f]{40}$"
)
_VERSION_COMMENT = re.compile(r"^\s+#\s*v\d+\.\d+\.\d+\s*$")
# Every `uvx` word, then its first argument after any flags, whatever that
# argument looks like. A quoted or variable argument is still captured, so it
# fails the pin check below rather than dropping out of the scan. Flags that
# take a value are rare in this repository and would be read as the tool;
# the assertion then names them.
_UVX = re.compile(r"\buvx\b(?:\s+--?[\w-]+)*(?:\s+(?P<arg>\S+))?")
_PINNED_TOOL = re.compile(r"^[A-Za-z0-9_.-]+@\d+(?:\.\d+)+$")
_HARDEN = "step-security/harden-runner@"


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _code_lines(text: str) -> list[tuple[int, str]]:
    """(1-based line number, line) for every line that is not a whole-line comment."""
    return [
        (n, line)
        for n, line in enumerate(text.splitlines(), start=1)
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _jobs(text: str) -> dict[str, list[str]]:
    """Each job's lines, keyed by job id (two-space keys under `jobs:`)."""
    lines = text.splitlines()
    start = lines.index("jobs:")
    jobs: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines[start + 1 :]:
        if line.strip() and not line.lstrip().startswith("#") and _indent(line) == 0:
            break
        match = re.match(r"^  (?P<id>[A-Za-z0-9_-]+):\s*$", line)
        if match:
            current = str(match.group("id"))
            jobs[current] = []
        elif current is not None:
            jobs[current].append(line)
    return jobs


def _permissions_block(lines: list[str], key_indent: int) -> str | None:
    """The `permissions:` value at `key_indent`, as text, or None if absent.

    A scalar (`read-all`, `write-all`) is returned as is; a mapping is
    returned as its lines joined, so `"write" in block` is the check.
    """
    prefix = " " * key_indent + "permissions:"
    for i, line in enumerate(lines):
        if not line.startswith(prefix):
            continue
        rest = line[len(prefix) :].split("#")[0].strip()
        if rest:
            return rest
        block: list[str] = []
        for follow in lines[i + 1 :]:
            if follow.strip() and not follow.lstrip().startswith("#") and _indent(follow) <= key_indent:
                break
            block.append(follow.split("#")[0])
        return "\n".join(block)
    return None


def _steps(job_lines: list[str]) -> list[str]:
    """The job's steps, one string per `- ` item under `steps:`, comments dropped."""
    steps: list[str] = []
    item_indent: int | None = None
    in_steps = False
    for line in job_lines:
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = _indent(line)
        if stripped.startswith("steps:"):
            in_steps = True
            continue
        if not in_steps:
            continue
        if item_indent is None and stripped.startswith("- "):
            item_indent = indent
        if item_indent is not None and indent < item_indent:
            break
        if indent == item_indent and stripped.startswith("- "):
            steps.append(line + "\n")
        elif steps:
            steps[-1] += line + "\n"
    return steps


def _runs_on(job_lines: list[str]) -> str:
    for line in job_lines:
        match = re.match(r"^    runs-on:\s*(.+?)\s*$", line)
        if match:
            return match.group(1)
    return ""


# --- `uses:` lines -----------------------------------------------------------


def test_the_scan_sees_the_workflow_directory() -> None:
    """Non-vacuity: the per-file tests below pass trivially on an empty glob."""
    assert len(_WORKFLOW_FILES) >= 30, [p.name for p in _WORKFLOW_FILES]
    uses = sum(
        1
        for path in _WORKFLOW_FILES
        for _n, line in _code_lines(path.read_text(encoding="utf-8"))
        if _USES.match(line)
    )
    assert uses >= 100, uses


def test_the_scan_reads_yaml_workflows_too(tmp_path: Path) -> None:
    """GitHub runs `.yaml` workflows as well as `.yml` ones, so a rule that
    globbed only `*.yml` would skip a workflow saved with the longer suffix."""
    for name in ("a.yml", "b.yaml", "notes.md"):
        (tmp_path / name).write_text("on: push\n", encoding="utf-8")
    (tmp_path / "dir.yml").mkdir()
    assert [p.name for p in _workflow_files(tmp_path)] == ["a.yml", "b.yaml"]


@pytest.mark.parametrize("path", _WORKFLOW_FILES, ids=lambda p: p.name)
def test_every_uses_names_a_commit_sha_and_its_tag(path: Path) -> None:
    """A tag moves; a sha does not. The `# vX.Y.Z` comment tells a reviewer
    which release the sha names. Dependabot finds the version from the tags
    that point at the sha and rewrites the comment on a bump."""
    bad: list[str] = []
    for n, line in _code_lines(path.read_text(encoding="utf-8")):
        match = _USES.match(line)
        if not match:
            continue
        ref = match.group("ref")
        if ref.startswith("./"):
            continue  # a local action ships in the same commit
        if not _PINNED.match(ref) or not _VERSION_COMMENT.match(match.group("rest")):
            bad.append(f"{path.name}:{n}: {line.strip()}")
    assert not bad, "\n".join(["uses: lines without a sha pin and a `# vX.Y.Z` comment:", *bad])


# --- uvx ---------------------------------------------------------------------


def _uvx_calls() -> list[tuple[str, str]]:
    """(location, first argument) for every `uvx` word in a workflow.

    The argument is empty when nothing follows `uvx` on the line.
    """
    calls: list[tuple[str, str]] = []
    for path in _WORKFLOW_FILES:
        for n, line in _code_lines(path.read_text(encoding="utf-8")):
            code = line.split(" #")[0]
            for match in _UVX.finditer(code):
                calls.append((f"{path.name}:{n}", match.group("arg") or ""))
    return calls


def test_every_uvx_call_pins_an_exact_version() -> None:
    calls = _uvx_calls()
    assert len(calls) >= 3, calls
    bad = [f"{where}: uvx {arg}" for where, arg in calls if not _PINNED_TOOL.match(arg)]
    assert not bad, "\n".join(["uvx calls whose tool is not a bare `tool@X.Y.Z`:", *bad])


# --- zizmor ------------------------------------------------------------------


def test_zizmor_blocks_at_high_severity() -> None:
    """CI-11: the `zizmor` check fails on a high-severity finding.

    Two runs: the SARIF run keeps every severity for code scanning and must
    not fail (`--no-exit-codes`), the gate run carries `--min-severity high`
    and nothing may swallow its exit status.
    """
    text = _ZIZMOR.read_text(encoding="utf-8")
    code = "\n".join(line for _n, line in _code_lines(text))
    assert "continue-on-error" not in code, code
    job = _jobs(text)["zizmor"]
    steps = _steps(job)
    gates = [s for s in steps if "uvx zizmor@" in s and "--min-severity high" in s]
    assert len(gates) == 1, steps
    assert "--format" not in gates[0], gates[0]
    # An `if:` on the job or on the gate step can skip the gate, and a
    # skipped step reports success.
    job_if = [line for line in job if re.match(r"^    if:", line)]
    assert not job_if, job_if
    gate_if = [
        line for line in gates[0].splitlines() if re.match(r"^(?:-\s+)?if:", line.strip())
    ]
    assert not gate_if, gates[0]
    # The gate's `run:` is the zizmor command alone: `|| true`, `; exit 0`
    # or a block script around it would swallow the exit status.
    gate_run = [line.strip() for line in gates[0].splitlines() if line.strip().startswith("run:")]
    assert len(gate_run) == 1, gates[0]
    assert re.fullmatch(r"run:\s*uvx zizmor@\S+ --min-severity high \.", gate_run[0]), gate_run[0]
    sarif = [s for s in steps if "--format sarif" in s]
    assert len(sarif) == 1, steps
    assert "--no-exit-codes" in sarif[0], sarif[0]
    assert "upload-sarif@" in "".join(steps), steps


def _trigger(text: str, event: str) -> list[str] | None:
    """The settings under `on: <event>:`, one stripped line each, comments
    dropped; an inline value on the event line comes first. None when the
    workflow has no such event."""
    lines = text.splitlines()
    settings: list[str] | None = None
    for line in lines[lines.index("on:") + 1 :]:
        stripped = line.split(" #")[0].strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = _indent(line)
        if indent == 0:
            break
        if indent == 2:
            if settings is not None:
                break
            key, _, value = stripped.partition(":")
            if key == event:
                settings = [value.strip()] if value.strip() else []
        elif settings is not None:
            settings.append(stripped)
    return settings


@pytest.mark.parametrize(
    ("on", "expected"),
    [
        ("on:\n  pull_request:\n  push:\n    branches: [main]\n", []),
        ("on:\n  pull_request:\n    paths:\n      - 'a/**'\njobs:\n", ["paths:", "- 'a/**'"]),
        ("on:\n  pull_request:\n    # note\n    branches: [x]  # why\n", ["branches: [x]"]),
        ("on:\n  pull_request: {types: [opened]}\n", ["{types: [opened]}"]),
        ("on:\n  push:\n", None),
    ],
)
def test_the_trigger_reader_returns_every_setting(on: str, expected: list[str] | None) -> None:
    assert _trigger(on, "pull_request") == expected


def test_zizmor_runs_on_every_pr() -> None:
    """zizmor reads files outside `.github/workflows/`: an `action.yml` or
    `action.yaml` anywhere in the tree, `dependabot.yaml` as well as
    `dependabot.yml`, the pre-commit configs, and its own `zizmor.yml`, which
    can switch a rule off. A `paths:` list cannot track that set, and any
    filter on `pull_request` (`paths:`, `branches:`, `types:`) lets some PR
    through with no `zizmor` check at all. An absent check does not block the
    train, so `pull_request` carries no settings."""
    text = _ZIZMOR.read_text(encoding="utf-8")
    assert _trigger(text, "pull_request") == [], text
    assert _trigger(text, "push") == ["branches: [main]"], text


# --- harden-runner -----------------------------------------------------------


def _job_ids() -> list[tuple[str, str]]:
    return [
        (path.name, job)
        for path in _WORKFLOW_FILES
        for job in _jobs(path.read_text(encoding="utf-8"))
    ]


@pytest.mark.parametrize("workflow", [p.name for p in _WORKFLOW_FILES])
def test_every_workflow_declares_permissions(workflow: str) -> None:
    """Without a `permissions:` block the token's reach is a repository
    setting, and the harden-runner rule below cannot classify the job."""
    lines = (_WORKFLOWS / workflow).read_text(encoding="utf-8").splitlines()
    assert _permissions_block(lines, 0) is not None, workflow


@pytest.mark.parametrize(("workflow", "job"), _job_ids(), ids=lambda v: str(v))
def test_harden_runner_runs_exactly_where_the_token_can_do_harm(workflow: str, job: str) -> None:
    """F12: `harden-runner` is the first step of a job that holds a write
    permission or reads a `secrets.*` value, and absent from every other
    job. Windows jobs are exempt: the action is Linux-only."""
    text = (_WORKFLOWS / workflow).read_text(encoding="utf-8")
    job_lines = _jobs(text)[job]
    permissions = _permissions_block(job_lines, 4) or _permissions_block(text.splitlines(), 0) or ""
    secret = any("secrets." in line.split("#")[0] for line in job_lines)
    wants = "write" in permissions or secret
    steps = _steps(job_lines)
    hardened = [i for i, s in enumerate(steps) if _HARDEN in s]
    if "windows" in _runs_on(job_lines):
        assert not hardened, f"{workflow}:{job} runs harden-runner on Windows"
        return
    if wants:
        assert hardened == [0], (
            f"{workflow}:{job} holds a write permission or a secret but "
            f"harden-runner is not its first step (found at {hardened})"
        )
    else:
        assert not hardened, (
            f"{workflow}:{job} is read-only and references no secret; "
            "harden-runner has nothing to protect there"
        )


def test_the_harden_runner_rule_is_not_vacuous() -> None:
    """Both branches of the rule above must be exercised by the tree."""
    kept = 0
    dropped = 0
    for path in _WORKFLOW_FILES:
        text = path.read_text(encoding="utf-8")
        for job_lines in _jobs(text).values():
            if any(_HARDEN in s for s in _steps(job_lines)):
                kept += 1
            else:
                dropped += 1
    assert kept >= 10, kept
    assert dropped >= 10, dropped


# --- dependabot.yml ----------------------------------------------------------


def _updates(text: str) -> list[list[str]]:
    """Each `- package-ecosystem:` item under `updates:`, comments dropped."""
    items: list[list[str]] = []
    for _n, line in _code_lines(text):
        if line.startswith("  - package-ecosystem:"):
            items.append([line])
        elif items and _indent(line) >= 4:
            items[-1].append(line)
    return items


def _value(lines: list[str], key: str) -> str | None:
    for line in lines:
        match = re.match(rf"^\s*(?:-\s+)?{re.escape(key)}:\s*(.+?)\s*$", line)
        if match:
            return match.group(1).strip().strip("\"'")
    return None


def test_dependabot_covers_actions_and_uv_weekly_with_grouped_minor_and_patch() -> None:
    """CI-09: both ecosystems, weekly, minor and patch grouped, the subject
    prefix the commit checks accept. `uv` and not `pip`: only the `uv`
    updater rewrites uv.lock, and `uv sync --frozen` fails on a stale lock."""
    assert _DEPENDABOT.is_file(), _DEPENDABOT
    text = _DEPENDABOT.read_text(encoding="utf-8")
    assert _value(text.splitlines(), "version") == "2", text
    updates = _updates(text)
    ecosystems = sorted(_value(item, "package-ecosystem") or "" for item in updates)
    assert ecosystems == ["github-actions", "uv"], ecosystems
    for item in updates:
        name = _value(item, "package-ecosystem")
        assert _value(item, "directory") == "/", (name, item)
        assert _value(item, "interval") == "weekly", (name, item)
        assert _value(item, "prefix") == "build", (name, item)
        assert _value(item, "include") == "scope", (name, item)
        groups = [line for line in item if "update-types:" in line]
        assert len(groups) == 1, (name, item)
        # Exactly minor and patch: a grouped major would hide a breaking
        # bump inside a routine PR.
        types = re.findall(r"\w+", _value(groups, "update-types") or "")
        assert sorted(types) == ["minor", "patch"], (name, groups)


# `gh pr merge` with `--auto` anywhere after it on the same command: the PR
# argument and other flags may come first.
_GH_MERGE_AUTO = re.compile(r"\bgh\s+pr\s+merge\b.*\s--auto\b")


def _auto_merge_lines(text: str) -> list[tuple[int, str]]:
    """(line number, command) for every code line that enables auto-merge.

    A shell line ending in `\\` continues on the next line, so the two are
    read as one command, numbered by its first line.
    """
    found: list[tuple[int, str]] = []
    pending: tuple[int, str] | None = None
    for n, line in _code_lines(text):
        start, command = pending if pending else (n, "")
        command += " " + line.strip()
        if command.endswith("\\"):
            pending = (start, command[:-1])
            continue
        pending = None
        lowered = command.lower()
        if "auto-merge" in lowered or "automerge" in lowered or _GH_MERGE_AUTO.search(lowered):
            found.append((start, command.strip()))
    return found


@pytest.mark.parametrize(
    "command",
    [
        "run: gh pr merge --auto --squash \"$PR_URL\"",
        "run: gh pr merge \"$PR_URL\" --auto --squash",
        "run: gh pr merge --squash --auto \"$PR_URL\"",
        "run: gh pr merge \"$PR_URL\" \\\n  --squash \\\n  --auto",
        "uses: peter-evans/enable-pull-request-automerge@0000000000000000000000000000000000000000",
        "run: gh api graphql -f query='mutation { enablePullRequestAutoMerge }'",
    ],
)
def test_the_auto_merge_scan_catches_each_spelling(command: str) -> None:
    text = "jobs:\n  merge:\n    steps:\n      - " + command + "\n"
    assert _auto_merge_lines(text), command


def test_the_auto_merge_scan_passes_a_plain_merge() -> None:
    text = "jobs:\n  merge:\n    steps:\n      - run: gh pr merge --squash \"$PR_URL\"\n"
    assert _auto_merge_lines(text) == []


def _dependabot_section() -> str:
    """CONTRIBUTING.md § Dependency updates from Dependabot, heading to next heading."""
    text = _CONTRIBUTING.read_text(encoding="utf-8")
    start = text.index("### Dependency updates from Dependabot")
    end = text.find("\n#", start + 1)
    return text[start : end if end != -1 else len(text)]


def test_the_dependabot_section_names_the_author_check_where_it_runs() -> None:
    """The section once credited the merge train with an author check. The
    train has none: the author rule is `check_commit_identity.py` in the
    `history-scan` job, which the train waits on as a required check."""
    section = " ".join(_dependabot_section().split())
    assert "The train has no author check of its own." in section, section
    assert "`history-scan` job runs `scripts/check_commit_identity.py`" in section, section
    staging = _jobs((_WORKFLOWS / "staging-gate.yml").read_text(encoding="utf-8"))
    assert any(
        "python3 scripts/check_commit_identity.py" in line for line in staging["history-scan"]
    ), staging["history-scan"]
    train = (_WORKFLOWS / "merge-train.yml").read_text(encoding="utf-8")
    assert "check_commit_identity" not in train
    gate = (_SCRIPTS / "merge_train_gate.py").read_text(encoding="utf-8")
    start = gate.index("FLOOR_NAMES: frozenset[str] = frozenset({")
    floor = gate[start : gate.index("})", start)]
    assert '"history-scan",' in floor, floor
    spec = importlib.util.spec_from_file_location(
        "check_commit_identity_1786", _SCRIPTS / "check_commit_identity.py"
    )
    assert spec is not None and spec.loader is not None
    identity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(identity)
    assert identity.name_is_allowed("dependabot[bot]")
    assert identity.email_is_allowed("49699333+dependabot[bot]@users.noreply.github.com")


def test_both_dependabot_subjects_are_documented_and_accepted() -> None:
    """With `include: scope`, Dependabot writes `deps-dev` when every package
    in a PR is a development dependency (the pyproject `dev` group) and `deps`
    otherwise. Both docs name both subjects, and the prefix check that
    `commit-msg-prefix` and `pr-title-prefix` run accepts both."""
    subjects = ("build(deps):", "build(deps-dev):")
    section = _dependabot_section()
    header = "\n".join(
        line
        for line in _DEPENDABOT.read_text(encoding="utf-8").splitlines()
        if line.startswith("#")
    )
    for subject in subjects:
        assert f"`{subject}" in section, (subject, section)
        assert f"`{subject}" in header, (subject, header)
    spec = importlib.util.spec_from_file_location(
        "check_commit_msg_1786", _SCRIPTS / "check-commit-msg.py"
    )
    assert spec is not None and spec.loader is not None
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    for subject in subjects:
        assert check.validate_subject(f"{subject} bump pytest from 8.4.1 to 9.0.0"), subject


def test_the_docs_give_the_version_comment_its_real_job() -> None:
    """dependabot-core's github_actions updater takes a sha-pinned action's
    version from git (`git_checker.version_for_pinned_sha` in file_parser.rb,
    `most_specific_tag_equivalent_to_pinned_ref` in update_checker.rb) and
    only rewrites the comment. The docs once said it read the comment to tell
    a patch bump from a major one."""
    header = "\n".join(
        line.lstrip("# ")
        for line in _DEPENDABOT.read_text(encoding="utf-8").splitlines()
        if line.startswith("#")
    )
    texts = {
        "dependabot.yml header": header,
        "CONTRIBUTING.md": _dependabot_section(),
        "pin test docstring": test_every_uses_names_a_commit_sha_and_its_tag.__doc__ or "",
    }
    for where, text in texts.items():
        flat = " ".join(text.split())
        assert "tags that point at the sha" in flat, (where, flat)
        assert "tells a reviewer which release" in flat, (where, flat)
        assert not re.search(r"reads? (?:the|that|its)\b[^.]*comment", flat), (where, flat)
        assert "tell a patch" not in flat, (where, flat)


def test_nothing_under_github_enables_auto_merge() -> None:
    """Dependabot PRs ride the merge train like every other PR. The repository
    setting is outside the tree; this holds the tree's half of the rule."""
    offenders = [
        f"{path.name}:{n}: {command}"
        for path in [_DEPENDABOT, *_WORKFLOW_FILES]
        for n, command in _auto_merge_lines(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, "\n".join(offenders)
