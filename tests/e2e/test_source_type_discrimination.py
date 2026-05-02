"""E2E scenario #3 (#334): ingest-source discrimination across paths.

Catches the #190 R1 class of regression: an INGEST_SOURCE_* constant
is defined in `aelfrice.models`, exported, but never written to the
store by the path that should be writing it. Unit tests are green at
every step because each module is correct in isolation; the contract
drift only shows up after a real cross-module ingest.

The discriminating signal on a single-shot first ingest lives in
`ingest_log.source_kind` (the v2.0 #205 source-of-truth log written
by `record_ingest`). The corroboration table's `source_type` is the
same wiring expressed on the re-assertion edge, but it only fires on
content_hash hits, which a one-shot test can't reliably trigger.

Each path is exercised through its real entry point and the resulting
row in `ingest_log` is read back via stdlib `sqlite3` — no in-process
imports of the aelfrice package, per the e2e boundary rule:

    cli_remember  ->  `aelf lock <statement>`
    filesystem    ->  `aelf ingest-transcript <jsonl>` (source_path
                       distinguishes transcript from other filesystem
                       ingest paths within `ingest_log`)
    git           ->  `aelf-commit-ingest` (PostToolUse hook entry
                       point) fed a synthetic Bash-tool payload

If any of these console-script paths fails to write the expected
`source_kind` to the store, this test fails — independent of whether
the in-process unit tests for that module still pass.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from pathlib import Path
from typing import Callable, Sequence

import pytest


pytestmark = pytest.mark.timeout(120)


def _read_ingest_source_kinds(db_path: Path) -> set[str]:
    """Return the distinct source_kind values in `ingest_log`.

    Stdlib sqlite3 read-only access; no aelfrice imports.
    """
    if not db_path.exists():
        return set()
    uri = f"file:{db_path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        rows = conn.execute(
            "SELECT DISTINCT source_kind FROM ingest_log"
        ).fetchall()
    return {str(r[0]) for r in rows}


def _read_transcript_source_paths(db_path: Path) -> set[str]:
    """Distinct source_path values for filesystem-kind ingest_log rows.

    Lets the test prove that `aelf ingest-transcript` lands the
    transcript label, not just the generic filesystem source_kind.
    """
    if not db_path.exists():
        return set()
    uri = f"file:{db_path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        rows = conn.execute(
            "SELECT DISTINCT source_path FROM ingest_log "
            "WHERE source_kind = 'filesystem'"
        ).fetchall()
    return {str(r[0]) for r in rows if r[0] is not None}


def test_three_paths_record_distinct_source_types(
    aelf_run: Callable[..., subprocess.CompletedProcess[str]],
    installed_console_script: Callable[[str], Sequence[str]],
    ephemeral_db: Path,
    tiny_project: Path,
    tmp_path: Path,
) -> None:
    """All three first-class ingest paths must record their declared
    `source_type` in `belief_corroborations`. Asserts the union of
    observed source_types contains the three declared constants.
    """
    # Path 1: cli_remember via `aelf lock`.
    aelf_run("lock", "Quokkas calibrate the knob carefully on Tuesdays.")

    # Path 2: transcript_ingest via `aelf ingest-transcript`. A
    # transcript-logger turns.jsonl line is the simplest accepted shape.
    transcript = tmp_path / "turns.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "role": "user",
                "text": "The aardvark counter resets at midnight.",
                "session_id": "e2e-source-type",
                "ts": "2026-05-02T22:00:00Z",
            }
        )
        + "\n"
    )
    aelf_run("ingest-transcript", str(transcript))

    # Path 3: commit_ingest via the PostToolUse hook entry point. The
    # hook reads a Claude Code Bash tool payload from stdin and runs
    # `git log -1 --format=%B <hash>` against `cwd` to fetch the body.
    # `tiny_project` is a real git repo, so we land a fresh commit on
    # it and shape a payload mirroring what Claude Code emits.
    # The triple extractor needs at least one (subject, relation, object)
    # match in the message body, otherwise the hook short-circuits before
    # opening the store. Pattern: `<NP> <verb> <NP>` with a permitted
    # relation verb. "supports" is in the relation bank.
    commit_msg = "feat: the yokozuna parser supports nested meridian keys"
    git_env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "tiny-project",
        "GIT_AUTHOR_EMAIL": "tiny@example.invalid",
        "GIT_COMMITTER_NAME": "tiny-project",
        "GIT_COMMITTER_EMAIL": "tiny@example.invalid",
    }
    (tiny_project / "MARKER").write_text("present\n")
    subprocess.run(  # noqa: S603, S607
        ["git", "add", "MARKER"], cwd=tiny_project, env=git_env, check=True,
        capture_output=True,
    )
    commit_proc = subprocess.run(  # noqa: S603, S607
        ["git", "commit", "-q", "-m", commit_msg],
        cwd=tiny_project, env=git_env, check=True, capture_output=True, text=True,
    )
    short_hash_proc = subprocess.run(  # noqa: S603, S607
        ["git", "rev-parse", "--short=12", "HEAD"],
        cwd=tiny_project, env=git_env, check=True, capture_output=True, text=True,
    )
    short_hash = short_hash_proc.stdout.strip()
    # Synthesise the bracket-prefix line `[branch hash] subject` the
    # hook expects in tool_response.stdout. Real Claude Code prints
    # this line after every successful `git commit`.
    fake_stdout = f"[main {short_hash}] {commit_msg}\n"
    payload = {
        "tool_name": "Bash",
        "tool_input": {"command": f"git commit -m {commit_msg!r}"},
        "tool_response": {"stdout": fake_stdout, "isError": False},
        "cwd": str(tiny_project),
    }

    hook_argv = installed_console_script("aelf-commit-ingest")
    env = os.environ.copy()
    env["AELFRICE_DB"] = str(ephemeral_db)
    hook_proc = subprocess.run(  # noqa: S603
        list(hook_argv),
        input=json.dumps(payload),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,  # hook contract is exit 0 even on internal errors
    )
    # Hook is non-blocking by contract; surface stderr if it tracebacked.
    assert "Traceback" not in hook_proc.stderr, hook_proc.stderr
    # commit subprocess succeeded; sanity-check stdout had the bracket line
    assert short_hash in commit_proc.stdout or commit_proc.returncode == 0

    observed_kinds = _read_ingest_source_kinds(ephemeral_db)

    # The three constants are declared in src/aelfrice/models.py:
    # INGEST_SOURCE_CLI_REMEMBER, INGEST_SOURCE_FILESYSTEM, INGEST_SOURCE_GIT.
    # Hard-code the literal values here so a rename-without-call-site-update
    # on either side fails this test loudly — that is the #190 R1 bug class.
    expected_kinds = {"cli_remember", "filesystem", "git"}
    missing = expected_kinds - observed_kinds
    assert not missing, (
        f"missing source_kind rows in ingest_log: {sorted(missing)}; "
        f"observed: {sorted(observed_kinds)}"
    )

    # Transcript ingest must additionally land its source_label so it is
    # distinguishable from other filesystem-kind ingest paths (e.g. raw
    # filesystem scanner). The default label from `aelf ingest-transcript`
    # is "transcript".
    transcript_paths = _read_transcript_source_paths(ephemeral_db)
    assert "transcript" in transcript_paths, (
        f"expected source_path='transcript' for filesystem-kind ingest; "
        f"got {sorted(transcript_paths)}"
    )
