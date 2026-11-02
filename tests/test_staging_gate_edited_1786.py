"""`staging-gate.yml` does not run on `pull_request.edited` (#1786).

Kept out of `tests/test_ci_manual_dispatch.py`: that module's top-level test
count is published as the #1436 mutation count (see
`benchmarks/published_constants.py`), and these tests are not among those
mutations. The workflow parsing helpers come from that module.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from tests.test_ci_manual_dispatch import _block, _indent, _text

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from merge_train_gate import FLOOR_NAMES, evaluate  # noqa: E402

# The presence-floor names `ci.yml` emits (see `merge_train_gate.FLOOR_NAMES`).
_CI_FLOOR_NAMES = frozenset({
    "derived-figures (3.12)",
    "derived-figures (3.13)",
    "pytest (3.12)",
    "pytest (3.13)",
})

# #1786: `pull_request.edited` fires on a title, body or base change. Of the
# two workflows checked below, only `pr-metadata.yml` reads the title or body,
# so with `edited` in its trigger list `staging-gate.yml` re-scanned an
# unchanged diff on every title or body edit and, through the concurrency
# group, cancelled the in-flight run of the same PR.
#
# The regex counts a read of the event payload in any of these forms: the
# dotted or bracketed field (`github.event.pull_request.title`,
# `github['event']['pull_request']['title']`), the whole PR object or event
# passed to `toJSON`, the event file (`GITHUB_EVENT_PATH`, `github.event_path`),
# which holds the title and body, and the `actions/github-script` view of that
# file: `context.payload.pull_request.title`, or `context.payload` or its
# `pull_request` object taken whole, as a destructuring or a `JSON.stringify`
# does. It does not count an API fetch: a workflow that reads the body as
# `merge-train.yml` does, with `gh pr view --json body`, does not trip it.
#
# A `${{ }}` expression looks up function and property names without regard
# to case, so the `github` forms and `toJSON` match in any case:
# `toJson(github.event)` and `github.event.pull_request.Title` both count. The
# `context.payload` form is JavaScript and `GITHUB_EVENT_PATH` is a shell
# variable, and both keep their exact case.
_EVENT = r"github(?:\.event|\[\s*['\"]event['\"]\s*\])"
_EVENT_PATH = r"github(?:\.event_path\b|\[\s*['\"]event_path['\"]\s*\])"
_PULL_REQUEST = r"(?:\.pull_request|\[\s*['\"]pull_request['\"]\s*\])"
_TITLE_OR_BODY = r"(?:\??\.(?:title|body)\b|\[\s*['\"](?:title|body)['\"]\s*\])"
_PAYLOAD_PULL_REQUEST = r"(?:\??\.pull_request|\[\s*['\"]pull_request['\"]\s*\])"
_PR_METADATA_FIELD_RE = re.compile(
    rf"(?i:{_EVENT}{_PULL_REQUEST}{_TITLE_OR_BODY}"
    rf"|toJSON\(\s*(?:{_EVENT}{_PULL_REQUEST}?|github)\s*\)"
    rf"|\b{_EVENT_PATH})"
    rf"|\bGITHUB_EVENT_PATH\b"
    rf"|(?<![\w$])context\.payload(?:{_PAYLOAD_PULL_REQUEST}{_TITLE_OR_BODY}"
    rf"|(?:{_PAYLOAD_PULL_REQUEST})?(?![\w$]|\s*(?:\??\.|\[)))"
)
# A repo script a step runs, such as `python3 scripts/check_commit_identity.py`.
_REPO_SCRIPT_RE = re.compile(r"(?<![\w./-])(?:\./)?((?:scripts|\.github)/[\w./-]+\.(?:py|sh|js|mjs|cjs))\b")


def _trigger_list(text: str, key: str) -> list[str]:
    """The `key:` list under the `pull_request:` trigger of workflow `text`; [] when absent.

    Reads the flow form (`[a, b]`, on one line or several), the block form
    (`- a` lines) and a lone scalar (`key: a`). A list line in any other
    shape fails the test instead of reading as [].
    """
    on_block = _block(text, "on", 0)
    lines = _block("\n".join(on_block), "pull_request", 2)
    for i, line in enumerate(lines):
        m = re.match(rf"( +){key}:(?:\s+(.*?))?\s*(?:\s#.*)?$", line)
        if not m:
            continue
        value = m.group(2) or ""
        if value.startswith("["):
            j = i
            while not value.endswith("]"):
                j += 1
                assert j < len(lines), f"unclosed `{key}:` flow list: {line!r}"
                value += " " + re.sub(r"\s+#.*$", "", lines[j]).strip()
            return [t.strip().strip("'\"") for t in value[1:-1].split(",") if t.strip()]
        if value:
            return [value.strip("'\"")]
        indent = len(m.group(1))
        items = []
        for item in lines[i + 1 :]:
            if _indent(item) < indent or (
                _indent(item) == indent and not item.lstrip().startswith("-")
            ):
                break
            im = re.match(r"\s*-\s+(.*?)\s*(?:\s#.*)?$", item)
            assert im, f"unread `{key}:` list line: {item!r}"
            items.append(im.group(1).strip("'\""))
        return items
    return []


def _pull_request_list(workflow: str, key: str) -> list[str]:
    """The `key:` list under the `pull_request:` trigger of `workflow`; [] when absent."""
    return _trigger_list(_text(workflow), key)


def _pull_request_types(workflow: str) -> list[str]:
    """The `types:` list under the `pull_request:` trigger; [] when absent."""
    return _pull_request_list(workflow, "types")


def _code_lines(text: str) -> list[str]:
    """The lines of `text` that are not `#` or `//` comments."""
    return [line for line in text.splitlines() if not line.lstrip().startswith(("#", "//"))]


def _reads_title_or_body(text: str, root: Path = _REPO) -> bool:
    """True when workflow `text`, or a repo script it runs, reads the PR title or body.

    A line that hands the whole PR object or event to a step counts as a read.
    A script under `root` that a non-comment line names is read as well: a step
    gets `GITHUB_EVENT_PATH` from the runner, so the script can open the event
    file without the workflow line naming it. A third-party action
    (`uses: owner/repo@ref`) also gets the event file, and its code is not
    read here.
    """
    lines = _code_lines(text)
    if any(_PR_METADATA_FIELD_RE.search(line) for line in lines):
        return True
    for line in lines:
        for m in _REPO_SCRIPT_RE.finditer(line):
            script = root / m.group(1)
            if script.is_file() and any(
                _PR_METADATA_FIELD_RE.search(code)
                for code in _code_lines(script.read_text(encoding="utf-8"))
            ):
                return True
    return False


def test_staging_gate_does_not_run_on_pull_request_edited() -> None:
    """No job in `staging-gate.yml` reads the PR title or body.

    `edited` also fires on a retarget, which changes `base.ref` and
    `base.sha` and so the range the scan jobs read. A PR retargeted onto
    `main` from a feature branch has no required check-runs until it is
    pushed (`test_a_pr_retargeted_onto_main_waits_for_a_push`), and that push
    fires `synchronize`, which scans the new range.
    """
    assert _pull_request_types("staging-gate.yml") == ["opened", "synchronize", "reopened"], (
        "staging-gate.yml must not trigger on pull_request.edited: no job in it "
        "reads the title or body, and a retarget needs a push, which fires "
        "synchronize, before the merge train accepts the PR (#1786)"
    )


def test_a_pr_retargeted_onto_main_waits_for_a_push() -> None:
    """A retargeted PR has no `ci.yml` rows, so the train waits for a push.

    `ci.yml` and `staging-gate.yml` run only for a PR whose base is `main`, so a
    PR stacked on another branch gets no row from either. A retarget onto
    `main` fires `edited`, which neither runs on, so the head stays without the
    `ci.yml` floor names and the merge train reports them `missing` until a
    push fires `synchronize`.
    """
    for workflow in ("ci.yml", "staging-gate.yml"):
        assert _pull_request_list(workflow, "branches") == ["main"], workflow
        assert "edited" not in _pull_request_types(workflow), workflow
    rows = [
        {"name": name, "status": "completed", "conclusion": "success",
         "started_at": "2026-11-02T10:00:00Z"}
        for name in sorted(FLOOR_NAMES - _CI_FLOOR_NAMES)
    ]
    assert evaluate(rows, set())["missing"] == sorted(_CI_FLOOR_NAMES)


_ON_PR = "on:\n  pull_request:\n    branches: [main]\n"


@pytest.mark.parametrize(
    ("body", "types"),
    [
        ("    types: [opened, edited]\n", ["opened", "edited"]),
        ("    types: [opened,\n      edited]\n", ["opened", "edited"]),
        ("    types:\n      - opened\n      - edited\n", ["opened", "edited"]),
        ("    types:\n    - opened\n    - 'edited'  # title or body\n", ["opened", "edited"]),
        ("    types: edited\n", ["edited"]),
        ("    types:\n      - opened\n    paths: [src/**]\n", ["opened"]),
        ("    paths: [src/**]\n", []),
    ],
)
def test_the_types_list_is_read_in_flow_block_and_scalar_form(body: str, types: list[str]) -> None:
    """Every YAML form of `types:` reads, so `edited` in a block list is not missed."""
    assert _trigger_list(_ON_PR + body + "  workflow_dispatch:\n", "types") == types


@pytest.mark.parametrize("workflow", ["pr-metadata.yml", "staging-gate.yml"])
def test_edited_is_a_trigger_exactly_where_the_title_or_body_is_read(workflow: str) -> None:
    """A workflow runs on `edited` exactly when it reads the PR title or body.

    The rule covers the title and body only. A retarget also fires `edited`
    and changes `base.ref` and `base.sha`, which `staging-gate.yml` reads;
    `test_staging_gate_does_not_run_on_pull_request_edited` covers why that
    workflow leaves a retarget to the `synchronize` run of the next push.
    """
    reads = _reads_title_or_body(_text(workflow))
    assert ("edited" in _pull_request_types(workflow)) == reads, (
        f"{workflow} {'reads' if reads else 'does not read'} the PR title or body, "
        f"so `edited` must {'stay in' if reads else 'stay out of'} its trigger "
        f"types: {_pull_request_types(workflow)} (#1786)"
    )


@pytest.mark.parametrize(
    ("line", "reads"),
    [
        ("# names github.event.pull_request.title", False),
        ("      # names github.event.pull_request.body", False),
        ("          TITLE: ${{ github.event.pull_request.title }}", True),
        ("        run: echo \"${{ github.event.pull_request.body }}\"", True),
    ],
)
def test_a_comment_naming_the_title_or_body_is_not_a_read(line: str, reads: bool) -> None:
    """Only a non-comment line counts, so a comment cannot add or keep `edited`."""
    assert _reads_title_or_body(f"on:\n  pull_request:\n{line}\n") is reads


@pytest.mark.parametrize(
    ("line", "reads"),
    [
        ("          T: ${{ github.event.pull_request['title'] }}", True),
        ("          B: ${{ github.event['pull_request'][\"body\"] }}", True),
        ("        run: echo '${{ toJSON(github.event.pull_request) }}' | jq -r .title", True),
        ("        run: echo '${{ toJSON(github.event) }}' | jq -r .pull_request.body", True),
        ("        run: echo '${{ toJSON(github) }}' | jq -r .event.pull_request.title", True),
        ("        run: jq -r .pull_request.title \"$GITHUB_EVENT_PATH\"", True),
        ("          E: ${{ github.event_path }}", True),
        ("          T: ${{ github['event']['pull_request']['title'] }}", True),
        ("          B: ${{ github['event'].pull_request.body }}", True),
        ("        run: echo '${{ toJSON(github['event']) }}' | jq -r .pull_request.title", True),
        ("          E: ${{ github['event_path'] }}", True),
        ("          R: ${{ github['event']['pull_request']['base']['ref'] }}", False),
        ("          script: core.info(context.payload.pull_request.title)", True),
        ("            const body = context.payload.pull_request.body;", True),
        ("            const b = context.payload['pull_request']['body'];", True),
        ("            const t = github.context.payload.pull_request?.title;", True),
        ("            const { title, body } = context.payload.pull_request;", True),
        ("            core.info(JSON.stringify(context.payload));", True),
        ("            const n = context.payload.pull_request.number;", False),
        ("            const ref = context.payload.pull_request.base.ref;", False),
        ("            const r = context.payload.pull_request_review.state;", False),
        ("            const a = context.payload.action;", False),
        ("          BASE_REF: ${{ github.event.pull_request.base.ref }}", False),
        ("          N: ${{ github.event.pull_request.number }}", False),
        ("        run: echo '${{ toJSON(github.event.pull_request.base) }}'", False),
        ("        run: gh pr view \"$PR\" --json body", False),
        ("        run: echo '${{ toJson(github.event) }}' | jq -r .pull_request.title", True),
        ("        run: echo '${{ tojson(github.event.pull_request) }}'", True),
        ("        run: echo '${{ TOJSON(github) }}' | jq -r .event.pull_request.body", True),
        ("          T: ${{ github.event.pull_request.Title }}", True),
        ("          B: ${{ GitHub.Event['Pull_Request']['BODY'] }}", True),
        ("          E: ${{ github.Event_Path }}", True),
        ("          R: ${{ GitHub.Event.Pull_Request.Base.Ref }}", False),
        ("            const t = context.payload.pull_request.Title;", False),
    ],
)
def test_every_event_payload_form_of_the_title_or_body_is_a_read(line: str, reads: bool) -> None:
    """A read through brackets, `toJSON`, the event file or `context.payload` counts.

    An expression read counts in any letter case, as the runner resolves it.
    """
    assert _reads_title_or_body(f"on:\n  pull_request:\n{line}\n") is reads


_READS_EVENT_FILE = (
    "import json, os\n"
    "event = json.load(open(os.environ['GITHUB_EVENT_PATH'], encoding='utf-8'))\n"
    "print(event['pull_request']['title'])\n"
)


@pytest.mark.parametrize(
    ("step", "script", "reads"),
    [
        ("      - run: python3 scripts/step.py --strict", _READS_EVENT_FILE, True),
        ("        run: bash ./scripts/step.py", _READS_EVENT_FILE, True),
        ("      # - run: python3 scripts/step.py", _READS_EVENT_FILE, False),
        ("      - run: python3 scripts/other.py", _READS_EVENT_FILE, False),
        ("      - run: python3 scripts/step.py", "# GITHUB_EVENT_PATH is not read\nprint(1)\n", False),
        ("      - run: python3 scripts/step.py", "print('scan')\n", False),
    ],
)
def test_a_repo_script_a_step_runs_is_read_for_the_title_or_body(
    tmp_path: Path, step: str, script: str, reads: bool,
) -> None:
    """A script that opens the event file reads the title or body for its workflow."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "step.py").write_text(script, encoding="utf-8")
    workflow = f"on:\n  pull_request:\njobs:\n  j:\n    steps:\n{step}\n"
    assert _reads_title_or_body(workflow, tmp_path) is reads
