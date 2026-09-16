"""A `gh` the merge-train tests put on `PATH`, so nothing reaches the network.

`scripts/merge_train_linked_issues.py` decides which issues a merged pull
request closes by rendering the body through GitHub's `/markdown` endpoint.
Both of its test modules drive the shipped command rather than only an injected
callable, which needs a `gh` of their own: `fake_gh` writes one that records the
argv and stdin it was given, prints canned HTML and exits with a chosen code.

Not a test module. Named without the `test_` prefix so pytest does not collect
it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# One interpreter per CLI test, reading a string and printing numbers -- no
# network, no store, no lock. Scaled by the suite's own knob so a loaded
# machine reports contention as slowness rather than as a failure (#1307).
CLI_TIMEOUT = 30 * int(os.environ.get("AELF_TEST_TIMEOUT_SCALE", "4"))


def issue_anchor(
    number: int | str, repo: str, text: str | None = None, *, kind: str = "issue"
) -> str:
    """One issue-link anchor shaped like the ones GitHub's renderer emits.

    `kind` is `data-hovercard-type`, which is what says whether the number is
    an issue or a pull request; GitHub puts it on every issue-link anchor, and
    only `issue` is a closing reference. It defaults to `issue` because that
    is the case almost every test is about, and the tests that are about the
    attribute itself pass it. Passing `kind=""` omits the attribute, which is
    the shape GitHub does not emit.
    """
    shown = text if text is not None else f"#{number}"
    hovercard = f'data-hovercard-type="{kind}" ' if kind else ""
    return (
        '<a class="issue-link js-issue-link" '
        f'data-url="https://github.com/{repo}/issues/{number}" '
        f"{hovercard}"
        f'href="https://github.com/{repo}/pull/{number}">{shown}</a>'
    )


def fake_gh(
    tmp_path: Path, stdout: str, *, returncode: int = 0, stderr: str = ""
) -> Path:
    """Write a `gh` into `tmp_path/bin` and return that directory.

    The call it received lands in `tmp_path/call.json` as
    `{"argv": [...], "stdin": "..."}`, which is how a test checks that the
    shipped code really invoked `subprocess.run` with the pinned arguments.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    call = tmp_path / "call.json"
    script = bin_dir / "gh"
    script.write_text(
        "#!" + sys.executable + "\n"
        "import json, sys, pathlib\n"
        f"pathlib.Path({str(call)!r}).write_text(\n"
        "    json.dumps({'argv': sys.argv[1:], 'stdin': sys.stdin.read()})\n"
        ")\n"
        f"sys.stderr.write({stderr!r})\n"
        f"sys.stdout.write({stdout!r})\n"
        f"raise SystemExit({returncode})\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return bin_dir


def recorded_call(tmp_path: Path) -> dict[str, object]:
    """The argv and stdin the fake `gh` under `tmp_path` was handed."""
    return json.loads((tmp_path / "call.json").read_text(encoding="utf-8"))


def run_cli(
    script: Path,
    args: list[str],
    *,
    bin_dir: Path,
    stdin: str = "",
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the tool with `PATH` holding only the fake `gh`.

    The environment is replaced rather than extended, so a stray
    `GITHUB_REPOSITORY` on the developer's machine cannot decide a test and
    the real `gh` is unreachable by construction.
    """
    environment = {"PATH": str(bin_dir), "HOME": str(bin_dir.parent)}
    environment.update(env or {})
    return subprocess.run(
        [sys.executable, str(script), *args],
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
        timeout=CLI_TIMEOUT,
        env=environment,
    )
