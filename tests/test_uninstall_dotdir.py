"""Disposition of `~/.aelfrice/` on uninstall (#1186).

#1173 made uninstall artifact-complete for the *store* directory and
deliberately left `~/.aelfrice/` alone, because `projects/` there holds
every other project's belief corpus. The cost of that was the LLM consent
sentinel: it survived the uninstall, so a reinstall read a stale grant as
current and never re-prompted.

These tests pin both halves — that the package's own install state and
capture logs go, and that another project's store, the user's config, and
anything the package did not write provably stay.
"""
from __future__ import annotations

import argparse
import io
import time
from pathlib import Path
from typing import Any

import pytest

from aelfrice import auto_install, cli, lifecycle

_SECRET = "the corpus content that must not survive a purge"


@pytest.fixture()
def dotdir(tmp_path: Path) -> Path:
    """A populated `~/.aelfrice/` holding one of everything."""
    home = tmp_path / "home" / ".aelfrice"
    home.mkdir(parents=True)

    # Install state — goes in every mode.
    (home / "llm-classify-consented").write_text(
        '{"scopes": ["stored_beliefs"]}', encoding="utf-8",
    )
    (home / "spine-backfilled").touch()
    (home / "installed-manifest-version").write_text("4.2.0", encoding="utf-8")
    (home / "migrated-to-uv").touch()
    (home / ".auto-install.lock").touch()
    (home / "claude-memory-reconciled").touch()
    (home / "logs").mkdir()
    (home / "logs" / "hook-failures.log").write_text("boom\n", encoding="utf-8")

    # Data — goes only when the user asked for data to go.
    (home / "telemetry.jsonl").write_text('{"v": 1}\n', encoding="utf-8")
    (home / "transcripts").mkdir()
    (home / "transcripts" / "turns.jsonl").write_text(
        _SECRET + "\n", encoding="utf-8",
    )

    # Preserved — user decisions and other stores.
    (home / "config.json").write_text('{"project_warm": {}}', encoding="utf-8")
    (home / "opt-out-hooks.json").write_text('{"hooks": ["Stop"]}', encoding="utf-8")
    (home / "projects" / "other-project").mkdir(parents=True)
    (home / "projects" / "other-project" / "memory.db").write_bytes(b"someone else")
    (home / "shared" / "global").mkdir(parents=True)
    (home / "shared" / "global" / "memory.db").write_bytes(b"federated peer")

    return home


# --- The load-bearing AC: consent does not outlive the uninstall ----------


@pytest.mark.parametrize("include_data", [True, False])
def test_consent_sentinel_goes_in_both_dispositions(
    dotdir: Path, include_data: bool,
) -> None:
    """AC1, at the mechanism level.

    `include_data` is the `--keep-db` axis. Consent is install state, not
    data, so it goes either way: keeping the beliefs is not consent to keep
    shipping them to a vendor after the package is gone.
    """
    result = lifecycle.dispose_dotdir(dotdir, include_data=include_data)

    assert not (dotdir / "llm-classify-consented").exists()
    assert (dotdir / "llm-classify-consented") in result.removed


def test_every_install_state_path_goes_under_keep_db(dotdir: Path) -> None:
    """The whole install-state set, not just the sentinel."""
    lifecycle.dispose_dotdir(dotdir, include_data=False)

    for relpath in lifecycle._DOTDIR_INSTALL_STATE:
        assert not (dotdir / relpath).exists(), f"{relpath} survived"


# --- AC2: the install-state vs data split --------------------------------


def test_keep_db_preserves_the_capture_logs(dotdir: Path) -> None:
    """AC2: `--keep-db` must not delete telemetry or the legacy transcripts.

    Reported as preserved rather than dropped from the report, so the user
    can see they are still there.
    """
    result = lifecycle.dispose_dotdir(dotdir, include_data=False)

    assert (dotdir / "telemetry.jsonl").exists()
    assert (dotdir / "transcripts" / "turns.jsonl").read_text().strip() == _SECRET
    assert (dotdir / "telemetry.jsonl") in result.preserved
    assert (dotdir / "transcripts") in result.preserved


def test_purge_takes_the_capture_logs(dotdir: Path) -> None:
    """The legacy transcripts hold verbatim user prose; a purge must reach it."""
    result = lifecycle.dispose_dotdir(dotdir, include_data=True)

    assert not (dotdir / "telemetry.jsonl").exists()
    assert not (dotdir / "transcripts").exists()
    assert (dotdir / "transcripts") in result.removed


# --- AC3: what must provably survive -------------------------------------


@pytest.mark.parametrize("include_data", [True, False])
def test_other_stores_and_user_config_are_untouched(
    dotdir: Path, include_data: bool,
) -> None:
    """AC3. `projects/` is the one that would be catastrophic.

    A blanket sweep of `~/.aelfrice/` would delete every other project's
    belief corpus — strictly worse than the bug being fixed. `shared/` is
    the same class under another name (federation peers, #655).
    """
    result = lifecycle.dispose_dotdir(dotdir, include_data=include_data)

    assert (dotdir / "projects" / "other-project" / "memory.db").read_bytes() == (
        b"someone else"
    )
    assert (dotdir / "shared" / "global" / "memory.db").read_bytes() == (
        b"federated peer"
    )
    assert (dotdir / "config.json").exists()
    assert (dotdir / "opt-out-hooks.json").exists()

    for kept in ("projects", "shared", "config.json", "opt-out-hooks.json"):
        assert (dotdir / kept) in result.preserved
        assert (dotdir / kept) not in result.removed


# --- AC5: unrecognised paths are reported, never deleted -----------------


def test_unrecognised_paths_are_reported_and_kept(dotdir: Path) -> None:
    """AC5. This directory can hold files the package never wrote."""
    stranger = dotdir / "my-notes.txt"
    stranger.write_text("mine, not aelfrice's", encoding="utf-8")
    stranger_dir = dotdir / "some-tool"
    stranger_dir.mkdir()

    result = lifecycle.dispose_dotdir(dotdir, include_data=True)

    assert stranger.exists()
    assert stranger_dir.exists()
    assert set(result.unrecognised) == {stranger, stranger_dir}
    assert not (set(result.unrecognised) & set(result.removed))


def test_plan_deletes_nothing(dotdir: Path) -> None:
    """`dotdir_plan` is the disclosure half and must be read-only.

    The CLI prints it at the gate, before the user has confirmed.
    """
    before = sorted(p.name for p in dotdir.iterdir())

    planned, preserved, unrecognised = lifecycle.dotdir_plan(
        dotdir, include_data=True,
    )

    assert planned  # it found something to report
    assert sorted(p.name for p in dotdir.iterdir()) == before
    assert not (set(planned) & set(preserved))
    assert not (set(planned) & set(unrecognised))


# --- The nested log file and its directory --------------------------------


def test_logs_dir_is_pruned_once_emptied(dotdir: Path) -> None:
    """Removing the only file in `logs/` should not leave the directory."""
    lifecycle.dispose_dotdir(dotdir, include_data=True)

    assert not (dotdir / "logs").exists()


def test_the_plan_discloses_the_directory_it_will_prune(dotdir: Path) -> None:
    """The gate must name `logs/` too, not just the file inside it.

    Removing a path the manifest never mentioned is the #1173 defect, so
    the prune is planned rather than done as a side effect. Ordered after
    its contents, which is also the order removal needs.
    """
    planned, _preserved, _unknown = lifecycle.dotdir_plan(
        dotdir, include_data=True,
    )

    assert (dotdir / "logs") in planned
    assert planned.index(dotdir / "logs") > planned.index(
        dotdir / "logs" / "hook-failures.log"
    )


def test_a_shared_parent_is_left_when_anything_in_it_is_kept(
    dotdir: Path,
) -> None:
    """The subset test is what makes the prune safe, so pin it directly."""
    (dotdir / "logs" / "not-ours.log").write_text("keep", encoding="utf-8")

    planned, _preserved, _unknown = lifecycle.dotdir_plan(
        dotdir, include_data=True,
    )

    assert (dotdir / "logs") not in planned


def test_logs_dir_survives_when_it_holds_a_stranger(dotdir: Path) -> None:
    """Prune-if-empty, never rmtree: something else in there is not ours."""
    (dotdir / "logs" / "someone-elses.log").write_text("keep", encoding="utf-8")

    lifecycle.dispose_dotdir(dotdir, include_data=True)

    assert not (dotdir / "logs" / "hook-failures.log").exists()
    assert (dotdir / "logs" / "someone-elses.log").read_text() == "keep"


# --- Overlap with the store disposition ----------------------------------


def test_skip_prevents_double_handling_of_store_artifacts(dotdir: Path) -> None:
    """`~/.aelfrice/` *is* the store directory on the non-git fallback.

    There `transcripts/` and `claude-memory-reconciled` belong to both
    sets, and `artifact_paths` already claims them. Passing them as `skip`
    is what stops the two dispositions reporting the same path twice.
    """
    overlap = (dotdir / "transcripts", dotdir / "claude-memory-reconciled")

    result = lifecycle.dispose_dotdir(
        dotdir, include_data=True, skip=overlap,
    )

    for path in overlap:
        assert path not in result.removed
        assert path not in result.preserved
        assert path not in result.unrecognised
        assert path.exists(), "skip must mean 'not mine to touch'"


def test_absent_dotdir_is_a_noop(tmp_path: Path) -> None:
    """A host that never wrote the directory must not error."""
    result = lifecycle.dispose_dotdir(
        tmp_path / "nope" / ".aelfrice", include_data=True,
    )

    assert result == lifecycle.DotdirDisposition()


# --- AC4: the removal set is single-sourced ------------------------------


def test_install_state_matches_its_owning_modules() -> None:
    """AC4. A rename at the source must fail here, not orphan a file.

    Same contract as `test_sibling_filenames_match_their_owning_modules`
    for the store directory: `lifecycle` spells these as literals (import
    graph), and this test is what keeps the two in step.
    """
    from aelfrice import claude_memory, llm_classifier, mcp_cleanup, temporal_spine

    expected = {
        llm_classifier.SENTINEL_FILENAME,
        temporal_spine.SPINE_BACKFILLED_SENTINEL.name,
        auto_install.STAMP_PATH.name,
        lifecycle.MIGRATED_TO_UV_SENTINEL.name,
        auto_install.AUTO_INSTALL_LOCK_FILENAME,
        claude_memory._RECONCILE_SENTINEL_NAME,
        mcp_cleanup.MCP_CLEANUP_SENTINEL.name,
    }
    named = set(lifecycle._DOTDIR_INSTALL_STATE)
    assert expected <= named, (
        "an install-state filename changed at its source but not in the "
        f"uninstall removal set: {sorted(expected - named)}"
    )


def test_the_hook_failure_log_is_addressed_relative_to_the_dotdir() -> None:
    """`doctor.HOOK_FAILURES_LOG` is the only nested entry.

    Pinned separately because it carries a parent segment, and getting it
    wrong would silently leave the log (and the `logs/` directory) behind.
    """
    from aelfrice import doctor

    relative = doctor.HOOK_FAILURES_LOG.relative_to(auto_install.AELFRICE_DOTDIR)

    assert str(relative) in lifecycle._DOTDIR_INSTALL_STATE


def test_data_paths_match_their_owning_modules() -> None:
    from aelfrice import telemetry, transcript_logger

    expected = {
        telemetry.DEFAULT_TELEMETRY_PATH.name,
        transcript_logger.LEGACY_TRANSCRIPTS_DIR.name,
    }
    named = set(lifecycle._DOTDIR_DATA)
    assert expected <= named, (
        "a captured-data path changed at its source but not in the "
        f"uninstall removal set: {sorted(expected - named)}"
    )


def test_preserved_paths_match_their_owning_modules() -> None:
    """The keep-list is single-sourced too — a rename must not start deleting.

    If `doctor._AELFRICE_PROJECTS_DIR` were renamed and this list were
    not, `projects/` would fall through to "unrecognised" (still not
    deleted, but no longer positively recognised) and the guarantee in
    AC3 would rest on nothing.
    """
    from aelfrice import doctor, project_warm

    expected = {
        doctor._AELFRICE_PROJECTS_DIR.name,
        project_warm._CONFIG_FILENAME,
        auto_install.OPT_OUT_PATH.name,
    }
    named = set(lifecycle._DOTDIR_PRESERVED)
    assert expected <= named, (
        "a preserved path changed at its source but not in the uninstall "
        f"keep-list: {sorted(expected - named)}"
    )


def test_the_three_sets_are_disjoint() -> None:
    """No path may be both removed and preserved."""
    install = {lifecycle._dotdir_top_level(r) for r in lifecycle._DOTDIR_INSTALL_STATE}
    data = set(lifecycle._DOTDIR_DATA)
    preserved = set(lifecycle._DOTDIR_PRESERVED)

    assert not (install & data)
    assert not (install & preserved)
    assert not (data & preserved)


# --- End to end, through the CLI -----------------------------------------


def _args(**over: object) -> argparse.Namespace:
    base: dict[str, Any] = {
        "keep_db": False, "purge": False, "archive": None, "yes": True,
        "keep_hook": True, "settings_path": None, "password_stdin": False,
        "host": "claude",
    }
    base.update(over)
    return argparse.Namespace(**base)


@pytest.fixture()
def cli_sandbox(
    dotdir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """`aelf uninstall` wired to a sandboxed dotdir and a store outside it."""
    store_dir = tmp_path / "repo" / ".git" / "aelfrice"
    store_dir.mkdir(parents=True)
    db = store_dir / "memory.db"
    db.write_bytes(b"SQLite format 3\x00")

    monkeypatch.setattr(auto_install, "AELFRICE_DOTDIR", dotdir)
    monkeypatch.setattr(cli, "db_path", lambda: db)
    monkeypatch.setattr(cli, "_clear_update_cache", lambda: None)
    monkeypatch.setattr(cli, "_read_password", lambda _a: "hunter2")
    return dotdir


@pytest.mark.parametrize(
    "mode", ["keep_db", "purge", "archive"],
)
def test_cli_removes_consent_in_every_disposition_mode(
    cli_sandbox: Path, tmp_path: Path, mode: str,
) -> None:
    """AC1 end to end: no mode may leave the grant behind.

    This is the whole point of #1186 — a reinstall after any of the three
    must re-prompt.
    """
    sentinel = cli_sandbox / "llm-classify-consented"
    assert sentinel.exists()

    if mode == "archive":
        args = _args(archive=str(tmp_path / "out.age"))
    else:
        args = _args(**{mode: True})

    out = io.StringIO()
    code = cli._cmd_uninstall(args, out)

    assert code == 0, out.getvalue()
    assert not sentinel.exists(), f"consent survived --{mode}"


def test_cli_purge_gate_discloses_the_dotdir_paths(
    cli_sandbox: Path,
) -> None:
    """Deleting a path the manifest never named is the #1173 defect again.

    The gate must say so before the prompt, so `--yes` is not the only
    thing standing between the user and an undisclosed deletion.
    """
    out = io.StringIO()
    code = cli._cmd_uninstall(_args(purge=True), out)
    text = out.getvalue()

    assert code == 0
    assert str(cli_sandbox) in text
    for name in ("llm-classify-consented", "telemetry.jsonl", "transcripts"):
        assert name in text, f"{name} missing from the dotdir disclosure"


def test_cli_reports_paths_it_declined_to_delete(cli_sandbox: Path) -> None:
    """The user needs to know what is left, to finish by hand if they want."""
    (cli_sandbox / "my-notes.txt").write_text("mine", encoding="utf-8")

    out = io.StringIO()
    code = cli._cmd_uninstall(_args(purge=True), out)
    text = out.getvalue()

    assert code == 0
    assert "not written by aelfrice" in text
    assert "my-notes.txt" in text
    assert (cli_sandbox / "my-notes.txt").exists()


def test_cli_keep_db_leaves_the_capture_logs(cli_sandbox: Path) -> None:
    """AC2 end to end."""
    out = io.StringIO()
    code = cli._cmd_uninstall(_args(keep_db=True), out)

    assert code == 0
    assert (cli_sandbox / "telemetry.jsonl").exists()
    assert (cli_sandbox / "transcripts").exists()
    assert not (cli_sandbox / "llm-classify-consented").exists()


def test_a_stray_file_under_a_named_directory_is_reported(dotdir: Path) -> None:
    """`logs/` is aelfrice's, but a file inside it need not be.

    The subset check already refuses to prune a `logs/` holding anything
    the package did not write, so nothing was ever at risk. But `logs`
    is an accounted top-level name, so the report walked straight past
    it and the stray was neither deleted nor mentioned — "reported,
    never deleted" kept only its second half, in the one case where the
    user most needs the first.
    """
    stray = dotdir / "logs" / "operator-notes.log"
    stray.write_text("mine", encoding="utf-8")

    planned, _preserved, unrecognised = lifecycle.dotdir_plan(
        dotdir, include_data=True,
    )

    assert stray in unrecognised, (
        f"a file the package never wrote went unreported: {unrecognised}"
    )
    assert stray not in planned
    # The directory itself is neither pruned nor reported in its stead —
    # naming `logs/` would read as though aelfrice were disowning a
    # directory it does use.
    assert (dotdir / "logs") not in planned
    assert (dotdir / "logs") not in unrecognised
    # And the package's own log inside it is still removed.
    assert (dotdir / "logs" / "hook-failures.log") in planned


def test_a_named_directory_holding_only_our_own_files_is_still_pruned(
    dotdir: Path,
) -> None:
    """The stray-reporting path must not disarm the prune it sits beside."""
    planned, _preserved, unrecognised = lifecycle.dotdir_plan(
        dotdir, include_data=True,
    )

    assert (dotdir / "logs") in planned
    assert not any(u.parent.name == "logs" for u in unrecognised)


def test_the_disclosure_names_only_the_categories_it_will_delete(
    cli_sandbox: Path,
) -> None:
    """A gate that overstates is as wrong as one that understates.

    With `include_data=False` the capture logs move to `preserved`, so
    naming them would announce a deletion that is not going to happen.
    Exercised directly rather than through the CLI: both gates that call
    this pass `include_data=True` today, so the False branch is currently
    unreachable from the command line. The function takes the flag, so it
    should honour it — and a `--keep-db` disclosure is the obvious next
    caller.
    """
    out = io.StringIO()
    cli._disclose_dotdir_removals(include_data=False, skip=(), out=out)
    text = out.getvalue()

    assert "install state" in text
    assert "capture logs" not in text, (
        "the disclosure named capture logs it is not going to delete"
    )
    assert "telemetry.jsonl" not in text
    assert "llm-classify-consented" in text


def test_purge_disclosure_does_name_the_capture_logs(cli_sandbox: Path) -> None:
    """The converse: when they do go, the gate has to say so."""
    out = io.StringIO()
    code = cli._cmd_uninstall(_args(purge=True), out)
    text = out.getvalue()

    assert code == 0
    assert "install state and capture logs" in text


def test_the_removal_report_does_not_call_captured_data_install_state(
    cli_sandbox: Path,
) -> None:
    """This module keeps data and install state distinct; the report must too."""
    out = io.StringIO()
    code = cli._cmd_uninstall(_args(purge=True), out)
    text = out.getvalue()

    assert code == 0
    assert "removed" in text
    assert "cleared install state" not in text, (
        "capture-log paths were reported as install state"
    )


def test_cli_archive_gate_discloses_the_dotdir_paths(
    cli_sandbox: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--archive` deletes the dotdir paths rather than encrypting them.

    Companion to the `--purge` disclosure test: the archive gate reaches
    the same disposition by a different branch, and it is the branch that
    had to learn to prompt when the store itself has no extras.
    """
    monkeypatch.setattr(cli, "_read_password", lambda _args: "pw")

    out = io.StringIO()
    code = cli._cmd_uninstall(
        _args(archive=str(tmp_path / "out.age")), out,
    )
    text = out.getvalue()

    assert code == 0, text
    assert str(cli_sandbox) in text
    for name in ("llm-classify-consented", "telemetry.jsonl", "transcripts"):
        assert name in text, f"{name} missing from the archive-gate disclosure"


# --- #1202: classifying a large directory must stay linear ---------------


def _fill_logs(home: Path, n: int) -> None:
    """`n` strays under `logs/`, which a long-lived store really reaches."""
    logs = home / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (logs / f"ingest-{i:06d}.log").write_text("x", encoding="utf-8")


def test_many_strays_are_each_reported_once(tmp_path: Path) -> None:
    """Correctness half of #1202: the set-based dedup keeps the contract."""
    home = tmp_path / ".aelfrice"
    home.mkdir()
    _fill_logs(home, 500)
    (home / "logs" / "hook-failures.log").write_text("boom\n", encoding="utf-8")

    _planned, _preserved, unrecognised = lifecycle.dotdir_plan(
        home, include_data=True,
    )

    strays = [p for p in unrecognised if p.name.startswith("ingest-")]
    assert len(strays) == 500
    assert len(set(strays)) == 500, "a path was reported twice"
    assert strays == sorted(strays), "ordering must survive the dedup"


def test_dotdir_plan_dedupes_without_rescanning(tmp_path: Path) -> None:
    """`unrecognised` deduped against itself as a *list* — O(n^2) (#1202).

    Counts path comparisons instead of timing them (#1473). The previous
    version asserted the shape of the wall-clock curve — doubling the input
    roughly doubles a linear walk — taking `min` of repeated samples on the
    reasoning that load can only make a sample slower. That reasoning holds
    for one measurement and fails for a *ratio* of two: under contention the
    larger case's floor inflates more than the smaller one's, so the ratio
    drifts upward. Measured 2026-08-19 under sustained load, this test failed
    3 runs in 5 while the other nine default-run clock assertions all held.

    The comparison count has no such failure mode. `stray not in <list>`
    calls `PurePath.__eq__` once per element already collected, so the
    quadratic version reaches ~n^2/2 calls, while the shipped set-based dedup
    reaches **zero** — a hash lookup that does not collide never compares. The
    bound below therefore sits three orders of magnitude clear of the
    regression it guards, and reads the same on an idle and a loaded box.
    """
    import pathlib as _pathlib

    home = tmp_path / "dot" / ".aelfrice"
    n = 2000
    _fill_logs(home, n)

    original_eq = _pathlib.PurePath.__eq__
    calls = 0

    def counting_eq(self: Any, other: Any) -> Any:
        nonlocal calls
        calls += 1
        return original_eq(self, other)

    _pathlib.PurePath.__eq__ = counting_eq  # type: ignore[method-assign]
    try:
        _, _, unrecognised = lifecycle.dotdir_plan(home, include_data=True)
    finally:
        _pathlib.PurePath.__eq__ = original_eq  # type: ignore[method-assign]

    assert len(unrecognised) >= n, (
        f"expected at least {n} strays to classify, got {len(unrecognised)} "
        "— the fixture is not exercising the dedup"
    )
    assert calls < n, (
        f"dotdir_plan made {calls} path comparisons for {n} strays. A "
        f"set-based dedup makes ~0; a list rescan makes ~{n * n // 2}. This "
        "is the #1202 quadratic walk returning."
    )
