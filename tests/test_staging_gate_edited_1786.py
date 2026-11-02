"""`staging-gate.yml` does not run on `pull_request.edited` (#1786).

Kept out of `tests/test_ci_manual_dispatch.py`: that module's top-level test
count is published as the #1436 mutation count (see
`benchmarks/published_constants.py`), and these tests are not among those
mutations. The workflow parsing helpers come from that module.
"""

from __future__ import annotations

import re

import pytest

from tests.test_ci_manual_dispatch import _block, _text

# #1786: `pull_request.edited` fires on a title, body or base change. Of the
# two workflows checked below, only `pr-metadata.yml` reads the title or body,
# so with `edited` in its trigger list `staging-gate.yml` re-scanned an
# unchanged diff on every title or body edit and, through the concurrency
# group, cancelled the in-flight run of the same PR.
#
# The regex counts a read of the event payload in any of these forms: the
# dotted or bracketed field (`github.event.pull_request.title`,
# `github.event.pull_request['title']`), the whole PR object or event passed
# to `toJSON`, the event file (`GITHUB_EVENT_PATH`, `github.event_path`),
# which holds the title and body, and the `actions/github-script` view of that
# file: `context.payload.pull_request.title`, or `context.payload` or its
# `pull_request` object taken whole, as a destructuring or a `JSON.stringify`
# does. It does not count an API fetch: a workflow that reads the body as
# `merge-train.yml` does, with `gh pr view --json body`, does not trip it.
_PULL_REQUEST = r"(?:\.pull_request|\[\s*['\"]pull_request['\"]\s*\])"
_TITLE_OR_BODY = r"(?:\??\.(?:title|body)\b|\[\s*['\"](?:title|body)['\"]\s*\])"
_PAYLOAD_PULL_REQUEST = r"(?:\??\.pull_request|\[\s*['\"]pull_request['\"]\s*\])"
_PR_METADATA_FIELD_RE = re.compile(
    rf"github\.event{_PULL_REQUEST}{_TITLE_OR_BODY}"
    rf"|toJSON\(\s*github(?:\.event{_PULL_REQUEST}?)?\s*\)"
    r"|\bGITHUB_EVENT_PATH\b|\bgithub\.event_path\b"
    rf"|(?<![\w$])context\.payload(?:{_PAYLOAD_PULL_REQUEST}{_TITLE_OR_BODY}"
    rf"|(?:{_PAYLOAD_PULL_REQUEST})?(?![\w$]|\s*(?:\??\.|\[)))"
)


def _pull_request_types(workflow: str) -> list[str]:
    """The `types:` list under the `pull_request:` trigger; [] when absent."""
    on_block = _block(_text(workflow), "on", 0)
    for line in _block("\n".join(on_block), "pull_request", 2):
        m = re.match(r"\s+types:\s*\[(.*)\]", line)
        if m:
            return [t.strip().strip("'\"") for t in m.group(1).split(",") if t.strip()]
    return []


def _reads_title_or_body(text: str) -> bool:
    """True when a non-comment line of workflow `text` reads the PR title or body.

    A line that hands the whole PR object or event to a step counts as a read.
    """
    return any(
        _PR_METADATA_FIELD_RE.search(line)
        for line in text.splitlines()
        if not line.lstrip().startswith("#")
    )


def test_staging_gate_does_not_run_on_pull_request_edited() -> None:
    """No job in `staging-gate.yml` reads the PR title or body.

    `edited` also fires on a retarget, which changes `base.ref` and
    `base.sha` and so the range the scan jobs read. A PR retargeted onto
    `main` from a feature branch has no required check-runs until it is
    pushed (see `merge_train_gate.base_refusal`), and that push fires
    `synchronize`, which scans the new range.
    """
    assert _pull_request_types("staging-gate.yml") == ["opened", "synchronize", "reopened"], (
        "staging-gate.yml must not trigger on pull_request.edited: no job in it "
        "reads the title or body, and a retarget needs a push, which fires "
        "synchronize, before the merge train accepts the PR (#1786)"
    )


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
    ],
)
def test_every_event_payload_form_of_the_title_or_body_is_a_read(line: str, reads: bool) -> None:
    """A read through brackets, `toJSON`, the event file or `context.payload` counts."""
    assert _reads_title_or_body(f"on:\n  pull_request:\n{line}\n") is reads
