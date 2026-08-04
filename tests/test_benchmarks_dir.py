"""Smoke tests for the academic benchmark suite scaffold (P1).

Distinct from `tests/test_benchmark.py` (singular), which covers the
in-tree synthetic regression harness at `src/aelfrice/benchmark.py`.
This file covers the top-level `benchmarks/` directory: contamination
gate, score utilities, and the `aelf bench` dispatcher's handling of
inert academic-suite targets.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest

from aelfrice.cli import main as cli_main


def _run_cli(*argv: str) -> tuple[int, str]:
    buf = io.StringIO()
    code = cli_main(argv=list(argv), out=buf)
    return code, buf.getvalue()


def test_verify_clean_passes_on_clean_file(tmp_path: Path) -> None:
    """verify_clean accepts a retrieval file with only safe keys."""
    from benchmarks import verify_clean

    clean_file = tmp_path / "retrieval.json"
    clean_file.write_text(json.dumps([
        {"id": "q1", "question": "what?", "retrieved_context": "..."},
        {"id": "q2", "question": "why?", "retrieved_context": "..."},
    ]))
    assert verify_clean.verify_file(str(clean_file)) is True


def test_verify_clean_rejects_contaminated_file(tmp_path: Path) -> None:
    """verify_clean rejects a retrieval file containing ground truth."""
    from benchmarks import verify_clean

    bad_file = tmp_path / "retrieval.json"
    bad_file.write_text(json.dumps([
        {"id": "q1", "question": "what?", "answer": "leaked!"},
    ]))
    assert verify_clean.verify_file(str(bad_file)) is False


def test_verify_clean_rejects_missing_file(tmp_path: Path) -> None:
    from benchmarks import verify_clean

    assert verify_clean.verify_file(str(tmp_path / "nope.json")) is False


@pytest.mark.parametrize(
    "target",
    ["mab", "locomo", "longmemeval", "structmemeval", "amabench"],
)
def test_aelf_bench_inert_targets_exit_2_with_pointer(target: str) -> None:
    """Each scaffolded but not-yet-runnable target exits 2 with a
    pointer to benchmarks/README.md so users know where to look.

    `all` is no longer in this list — it dispatches via
    benchmarks.run.main_all() per the v2.0 reproducibility harness
    (#437). See test_aelf_bench_all_requires_out below.
    """
    code, output = _run_cli("bench", target)
    assert code == 2
    assert "benchmarks/README.md" in output
    assert target in output


def test_aelf_bench_all_requires_out() -> None:
    """`aelf bench all` without --out exits 2 with a clear message (#437)."""
    code, output = _run_cli("bench", "all")
    assert code == 2
    assert "--out PATH is required" in output


def test_aelf_bench_unknown_target_exits_2() -> None:
    code, output = _run_cli("bench", "not-a-real-target")
    assert code == 2
    assert "unknown target" in output


def test_aelf_bench_default_still_runs_synthetic_harness() -> None:
    """Backward-compat: bare `aelf bench` must remain the synthetic
    regression harness (it's the v0.9.0-rc → v1.0.0 contract).
    """
    code, output = _run_cli("bench")
    assert code == 0
    report = json.loads(output)
    assert report["benchmark_name"] == "aelfrice-bench-v1"
    assert report["corpus_size"] == 16


def test_aelf_bench_synthetic_target_explicit_runs_synthetic() -> None:
    """`aelf bench synthetic` is an alias for the default."""
    code, output = _run_cli("bench", "synthetic")
    assert code == 0
    report = json.loads(output)
    assert report["corpus_size"] == 16


def test_aelf_bench_verify_clean_target_redirects_to_module() -> None:
    """The dev-only target now points users at `python -m benchmarks.verify_clean`."""
    code, output = _run_cli("bench", "verify-clean")
    assert code == 2
    assert "python -m benchmarks.verify_clean" in output


def test_aelf_bench_longmemeval_score_target_redirects_to_module() -> None:
    code, output = _run_cli("bench", "longmemeval-score")
    assert code == 2
    assert "python -m benchmarks.longmemeval_score" in output


def test_aelf_bench_posterior_residual_target_redirects_to_module() -> None:
    code, output = _run_cli("bench", "posterior-residual")
    assert code == 2
    assert "python -m benchmarks.posterior_ranking" in output


def test_aelf_bench_unknown_target_lists_moved_dev_targets() -> None:
    code, output = _run_cli("bench", "no-such-target")
    assert code == 2
    assert "verify-clean" in output
    assert "longmemeval-score" in output
    assert "posterior-residual" in output


def test_benchmarks_package_imports() -> None:
    """The benchmarks/ directory is a valid (empty) package."""
    import benchmarks
    assert benchmarks is not None


@pytest.mark.timeout(120)
def test_posterior_channel_audit_holds_the_documented_defaults() -> None:
    """`benchmarks/posterior_channel_audit.py` exits 0 on the shipped defaults.

    The script is the regression guard behind the LIMITATIONS entry that
    says no automatic channel moves a posterior (#1267). A guard nothing
    invokes cannot keep a doc entry honest, and no workflow runs
    `benchmarks/*.py` — `bench-smoke` only runs two named test modules.
    Running it from the required `pytest` job is what makes the claim
    enforceable.

    Driven as a subprocess rather than imported: the script deletes every
    ambient `AELFRICE_*` variable at import time so it measures defaults
    rather than the developer's opt-ins, and that must not leak into the
    rest of the test session.
    """
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "benchmarks" / "posterior_channel_audit.py"
    assert script.is_file(), f"missing {script}"

    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=110,
    )
    assert proc.returncode == 0, (
        f"posterior-channel audit failed (exit {proc.returncode}); a "
        f"documented default moved:\n{proc.stdout}\n{proc.stderr}"
    )


@pytest.mark.timeout(120)
def test_posterior_channel_audit_ignores_an_ambient_aelfrice_toml(
    tmp_path: Path,
) -> None:
    """An ambient `.aelfrice.toml` must not change the audit's verdict (#1295).

    Clearing `AELFRICE_*` pins only the env tier, but every resolver the
    script drives is env -> kwarg -> TOML -> default, and the TOML lookup
    walks **up from the working directory**. Before #1295 a developer with
    `[implicit_feedback] enqueue_on_retrieve = true` at or above the repo
    got `FAIL: retrieval enqueues exposures by default` on a tree where no
    default had moved — and because #1290 wired this script into the
    required `pytest` job, that false failure blocks a merge.

    Running from a directory that carries exactly that config is the
    distinguishing arm: it exits 1 against the unpinned script and 0 once
    every call site passes `start=` / `config_start=`. Without this the
    fix is unverifiable and regresses silently — the same shape as the
    vacuous check #1290 replaced.
    """
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "benchmarks" / "posterior_channel_audit.py"
    assert script.is_file(), f"missing {script}"

    (tmp_path / ".aelfrice.toml").write_text(
        "[implicit_feedback]\nenqueue_on_retrieve = true\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=110,
    )
    assert proc.returncode == 0, (
        "an ambient .aelfrice.toml changed the audit's verdict; the TOML "
        f"tier is not pinned (exit {proc.returncode}):\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


@pytest.mark.timeout(120)
def test_posterior_channel_audit_scratch_walk_follows_symlinks(
    tmp_path: Path,
) -> None:
    """The scratch-walk check must walk the chain the resolver reads.

    `_read_toml_value` resolves its `start`; `Path.parents` is lexical.
    Before #1311 `_scratch_walk_hits` walked the unresolved chain, so it
    could report the pin clean while an ambient `.aelfrice.toml` above
    the *resolved* scratch directory still fed the resolvers — the audit
    would then silently measure a developer's config and report it as a
    shipped default. That is the failure the pin exists to prevent, so
    the pin needs a guard of its own.

    Constructed so the two chains genuinely diverge rather than alias:
    `TMPDIR` is `<base>/a/link`, a symlink to `<base>/b/real`, and the
    config sits at `<base>/b`. The lexical walk sees `<base>/a` and
    `<base>`; the resolved walk sees `<base>/b`. Note a symlink to a
    sibling *inside the same parent* would not distinguish anything —
    `Path.exists()` follows symlinks, so both chains would find the same
    file.

    `grace_window_seconds` is used rather than `enqueue_on_retrieve`
    deliberately: it does not by itself fail any channel, so the run's
    exit code isolates the scratch-walk check instead of conflating it
    with the enqueue assertion.

    Terminates deterministically: one subprocess with an explicit
    `timeout=`, under a `pytest.mark.timeout` ceiling above it. No
    polling, no retry, no unbounded wait.
    """
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "benchmarks" / "posterior_channel_audit.py"
    assert script.is_file(), f"missing {script}"

    base = tmp_path.resolve()
    (base / "b" / "real").mkdir(parents=True)
    (base / "a").mkdir()
    (base / "clean").mkdir()
    (base / "a" / "link").symlink_to(base / "b" / "real")
    (base / "b" / ".aelfrice.toml").write_text(
        "[implicit_feedback]\ngrace_window_seconds = 7\n", encoding="utf-8"
    )

    lexical = base / "a" / "link"
    assert not any(
        (parent / ".aelfrice.toml").is_file()
        for parent in (lexical, *lexical.parents)
    ), "fixture is wrong: the config must NOT be on the lexical chain"
    resolved = lexical.resolve()
    assert any(
        (parent / ".aelfrice.toml").is_file()
        for parent in (resolved, *resolved.parents)
    ), "fixture is wrong: the config must be on the resolved chain"

    env = {**os.environ, "TMPDIR": str(lexical)}
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=base / "clean",
        env=env,
        capture_output=True,
        text=True,
        timeout=110,
    )

    assert proc.returncode == 1, (
        "the audit did not notice a config on the resolved scratch chain; "
        "the scratch walk is lexical again and the pin is unverified "
        f"(exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
    )
    # The failure list is rendered on stderr, not stdout.
    assert "the scratch walk is not clean" in proc.stderr, (
        "exit 1 came from something other than the scratch-walk check:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
