"""The Windows smoke lane must test the artifact users install (#1493).

`windows-smoke.yml`'s original job answers "does the source tree run on
Windows". #1417 is about a different thing: `aelfrice==4.2.0` **from
PyPI** cannot start on native Windows, and the evidence that `main` fixes
it is a source-checkout run. The packaged artifact had never been
exercised on that platform at all.

The job is not a required check and cannot be — the workflow's own header
refuses that until its coverage widens — so nothing else would notice it
being edited into something that installs the checkout again. These tests
are that notice. They parse the YAML as text on purpose: PyYAML is not
importable in this repo's CI.

Every assertion here is paired with the `smoke` job as a control. Without
that pairing a test that "a job installs a wheel" would pass on a file
where both jobs did, and the distinction between the two jobs — which is
the entire content of this change — would go unpinned.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_WORKFLOW = _REPO / ".github" / "workflows" / "windows-smoke.yml"

_JOB_RE = re.compile(r"^  ([a-z][a-z0-9_-]*):$", re.MULTILINE)


def _jobs() -> dict[str, str]:
    """Job name -> body, split on two-space-indented keys under `jobs:`.

    `on:`'s `pull_request:` sits at the same indent, so the split is taken
    from the `jobs:` line onward rather than over the whole file.
    """
    text = _WORKFLOW.read_text(encoding="utf-8")
    start = text.index("\njobs:\n")
    body = text[start:]
    starts = [(m.group(1), m.start()) for m in _JOB_RE.finditer(body)]
    assert starts, "no jobs found — the split regex no longer matches"
    out: dict[str, str] = {}
    for i, (name, at) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else len(body)
        out[name] = body[at:end]
    return out


def test_both_jobs_are_present_and_distinct() -> None:
    """The control. Every test below compares the two."""
    jobs = _jobs()

    assert "smoke" in jobs, "the source-checkout job is gone"
    assert "wheel" in jobs, "the wheel-install job is gone"
    assert jobs["smoke"] != jobs["wheel"]


def test_the_wheel_job_installs_a_built_wheel_and_the_other_does_not() -> None:
    """`uv sync` here would silently make this a second source-tree job.

    That is not hypothetical: it is what the lane did before #1493, and
    the assertions it ran were all green while it did.
    """
    jobs = _jobs()

    assert "uv build --wheel" in jobs["wheel"]
    assert ".whl" in jobs["wheel"]
    assert "uv pip install" in jobs["wheel"]
    assert "uv sync" not in jobs["wheel"], (
        "the wheel job syncs the checkout — it is testing the source tree "
        "again, which is the defect #1493 exists to close"
    )
    assert "uv sync" in jobs["smoke"], (
        "the control job no longer syncs, so this file no longer "
        "distinguishes the two lanes"
    )


def test_the_wheel_job_refuses_an_ambiguous_build() -> None:
    """One wheel, named. A bare glob installs whatever it matched.

    `uv pip install dist/*.whl` on a build that emitted two files would
    install one of them and report success, and which one it was would
    depend on the shell's sort order.
    """
    body = _jobs()["wheel"]

    assert "-ne 1" in body, "nothing checks how many wheels were built"
    assert "expected exactly one wheel" in body


@pytest.mark.parametrize("step_marker", [
    "The installed console script starts",
    "doctor reads the Codex host without crashing",
    "setup is idempotent, and unsetup removes what it installed",
])
def test_every_cli_step_runs_outside_the_checkout(step_marker: str) -> None:
    """A step run in the checkout can be satisfied by the working tree.

    With today's `src/` layout a cwd import cannot reach the package, so
    this is structural rather than currently load-bearing — which is the
    reason to pin it. A flat layout, or a stray top-level `aelfrice/`,
    would make the working tree satisfy every assertion in the job and
    nothing would report the difference.
    """
    body = _jobs()["wheel"]
    at = body.index(step_marker)
    end = body.find("\n      - name:", at)
    step = body[at:end if end != -1 else len(body)]

    assert "working-directory: ${{ runner.temp }}" in step, (
        f"step {step_marker!r} runs in the checkout"
    )


def test_the_idempotence_step_compares_two_counts() -> None:
    """Exit codes prove nothing here — #1412 duplicated and exited 0.

    Both runs of a duplicating setup succeed. The only signal that
    separates the fixed behaviour from the broken one is the handler
    count doctor prints, read twice and compared.
    """
    at = _jobs()["wheel"].index("setup is idempotent")
    step = _jobs()["wheel"][at:]

    assert "aelfrice_handlers=" in step, "no count is read at all"
    assert step.count("setup --host codex") >= 2, "setup is run only once"
    assert 'second" != "$first' in step or "second != first" in step, (
        "the two readings are never compared"
    )
    assert "unsetup --host codex" in step
    assert "-ne 0" in step, "nothing asserts the count returns to zero"


def test_a_setup_that_installed_nothing_is_a_failure() -> None:
    """Otherwise the idempotence check passes on 0 == 0.

    A setup that silently installed nothing gives two equal readings and
    satisfies the comparison above, so the comparison needs a floor
    under it.
    """
    step = _jobs()["wheel"][_jobs()["wheel"].index("setup is idempotent"):]

    assert "setup installed nothing" in step
    assert "-eq 0" in step


def test_the_codex_home_is_a_scratch_directory() -> None:
    """The job must not write into the runner's real home.

    Both Codex steps point `$CODEX_HOME` at a path under `runner.temp`,
    and at *different* paths — the doctor step's home is deliberately
    empty, which is the state it asserts about.
    """
    body = _jobs()["wheel"]
    homes = re.findall(r"CODEX_HOME: (.+)$", body, re.MULTILINE)

    assert len(homes) == 2, f"expected two CODEX_HOME settings, got {homes}"
    assert all("runner.temp" in h for h in homes), homes
    assert len(set(homes)) == 2, "both Codex steps share one home directory"
