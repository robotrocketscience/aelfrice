"""Tests for `aelf core` CLI subcommand (#439).

Five-belief fixture matrix per spec (docs/feature-aelf-core.md):
  b-locked       lock=user  alpha=1 beta=1 corr=0  → yes (LOCK)
  b-corr         lock=none  alpha=1 beta=1 corr=3  → yes (CORR)
  b-posterior    lock=none  alpha=4 beta=1 corr=0  → yes (μ=0.8, α+β=5)
  b-thin-post    lock=none  alpha=2 beta=1 corr=0  → no  (μ=0.667 but α+β=3)
  b-prior        lock=none  alpha=1 beta=1 corr=0  → no

Tests cover all 8 spec scenarios.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice.cli import main
from aelfrice.models import (
    BELIEF_FACTUAL,
    CORROBORATION_SOURCE_CLI_REMEMBER,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    Belief,
)
from aelfrice.store import MemoryStore


@pytest.fixture(autouse=True)
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "aelf.db"
    monkeypatch.setenv("AELFRICE_DB", str(p))
    return p


def _run(*argv: str) -> tuple[int, str]:
    buf = io.StringIO()
    code = main(argv=list(argv), out=buf)
    return code, buf.getvalue()


def _make_belief(
    bid: str,
    content: str,
    lock_level: str = LOCK_NONE,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash="hash_" + bid[:6],
        alpha=alpha,
        beta=beta,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at="2026-05-06T00:00:00Z" if lock_level != LOCK_NONE else None,
        demotion_pressure=0,
        created_at="2026-05-06T00:00:00Z",
        last_retrieved_at=None,
        origin=ORIGIN_AGENT_INFERRED,
    )


def _seed_store(db: Path) -> dict[str, Belief]:
    beliefs = {
        "b-locked": _make_belief("b0locked0000000000", "locked ground truth", lock_level=LOCK_USER),
        "b-corr": _make_belief("b0corr00000000000a", "independently corroborated"),
        "b-posterior": _make_belief("b0posterior000000a", "strong posterior", alpha=4.0, beta=1.0),
        "b-thin": _make_belief("b0thinpost000000aa", "thin posterior", alpha=2.0, beta=1.0),
        "b-prior": _make_belief("b0prior000000000aa", "flat prior"),
    }
    s = MemoryStore(str(db))
    try:
        for b in beliefs.values():
            s.insert_belief(b)
        # b-corr: record 3 corroboration rows via the public API
        for _ in range(3):
            s.record_corroboration(
                beliefs["b-corr"].id,
                source_type=CORROBORATION_SOURCE_CLI_REMEMBER,
            )
    finally:
        s.close()
    return beliefs


# --- scenario 1: default text output -----------------------------------------


def test_core_default_includes_locked(isolated_db: Path) -> None:
    b = _seed_store(isolated_db)
    code, out = _run("core")
    assert code == 0
    assert b["b-locked"].id in out


def test_core_default_includes_corroborated(isolated_db: Path) -> None:
    b = _seed_store(isolated_db)
    code, out = _run("core")
    assert code == 0
    assert b["b-corr"].id in out


def test_core_default_includes_posterior(isolated_db: Path) -> None:
    b = _seed_store(isolated_db)
    code, out = _run("core")
    assert code == 0
    assert b["b-posterior"].id in out


def test_core_default_excludes_thin_posterior(isolated_db: Path) -> None:
    b = _seed_store(isolated_db)
    _, out = _run("core")
    assert b["b-thin"].id not in out


def test_core_default_excludes_flat_prior(isolated_db: Path) -> None:
    b = _seed_store(isolated_db)
    _, out = _run("core")
    assert b["b-prior"].id not in out


def test_core_tag_block_lock(isolated_db: Path) -> None:
    b = _seed_store(isolated_db)
    _, out = _run("core")
    assert f"{b['b-locked'].id} [LOCK]" in out


def test_core_tag_block_corr(isolated_db: Path) -> None:
    _seed_store(isolated_db)
    _, out = _run("core")
    assert "CORR=3" in out


def test_core_tag_block_posterior(isolated_db: Path) -> None:
    _seed_store(isolated_db)
    _, out = _run("core")
    assert "μ=0.800" in out


# --- scenario 2: --json round-trip -------------------------------------------


def test_core_json_parses(isolated_db: Path) -> None:
    b = _seed_store(isolated_db)
    _, out = _run("core", "--json")
    rows = json.loads(out)
    assert isinstance(rows, list)
    # JSON output must apply the same core-filter as text mode:
    # b-thin (μ=0.667, α+β=3) and b-prior (μ=0.5, α+β=2) are excluded
    # at default thresholds.
    row_ids = {row["id"] for row in rows}
    assert b["b-thin"].id not in row_ids
    assert b["b-prior"].id not in row_ids


def test_core_json_signals_lock(isolated_db: Path) -> None:
    b = _seed_store(isolated_db)
    _, out = _run("core", "--json")
    rows = json.loads(out)
    locked_row = next(r for r in rows if r["id"] == b["b-locked"].id)
    assert "lock" in locked_row["signals"]


def test_core_json_signals_corroboration(isolated_db: Path) -> None:
    b = _seed_store(isolated_db)
    _, out = _run("core", "--json")
    rows = json.loads(out)
    corr_row = next(r for r in rows if r["id"] == b["b-corr"].id)
    assert "corroboration" in corr_row["signals"]


def test_core_json_signals_posterior(isolated_db: Path) -> None:
    b = _seed_store(isolated_db)
    _, out = _run("core", "--json")
    rows = json.loads(out)
    post_row = next(r for r in rows if r["id"] == b["b-posterior"].id)
    assert "posterior" in post_row["signals"]


def test_core_json_signals_nonempty(isolated_db: Path) -> None:
    _seed_store(isolated_db)
    _, out = _run("core", "--json")
    rows = json.loads(out)
    for row in rows:
        assert len(row["signals"]) >= 1


# --- scenario 3: --locked-only -----------------------------------------------


def test_core_locked_only_exits_zero(isolated_db: Path) -> None:
    _seed_store(isolated_db)
    code, _ = _run("core", "--locked-only")
    assert code == 0


def test_core_locked_only_returns_locked(isolated_db: Path) -> None:
    b = _seed_store(isolated_db)
    _, out = _run("core", "--locked-only")
    assert b["b-locked"].id in out


def test_core_locked_only_excludes_corr(isolated_db: Path) -> None:
    b = _seed_store(isolated_db)
    _, out = _run("core", "--locked-only")
    assert b["b-corr"].id not in out


def test_core_locked_only_excludes_posterior(isolated_db: Path) -> None:
    b = _seed_store(isolated_db)
    _, out = _run("core", "--locked-only")
    assert b["b-posterior"].id not in out


# --- scenario 4: --no-locked -------------------------------------------------


def test_core_no_locked_suppresses_locked(isolated_db: Path) -> None:
    b = _seed_store(isolated_db)
    _, out = _run("core", "--no-locked")
    assert b["b-locked"].id not in out


def test_core_no_locked_includes_corr(isolated_db: Path) -> None:
    b = _seed_store(isolated_db)
    _, out = _run("core", "--no-locked")
    assert b["b-corr"].id in out


def test_core_no_locked_includes_posterior(isolated_db: Path) -> None:
    b = _seed_store(isolated_db)
    _, out = _run("core", "--no-locked")
    assert b["b-posterior"].id in out


# --- scenario 5: mutual exclusion exit 2 -------------------------------------


def test_core_locked_only_and_no_locked_exit_two(
    isolated_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        _run("core", "--locked-only", "--no-locked")
    assert exc.value.code == 2


# --- scenario 6: --limit 1 returns locked first ------------------------------


def test_core_limit_one_returns_locked(isolated_db: Path) -> None:
    b = _seed_store(isolated_db)
    _, out = _run("core", "--limit", "1")
    assert b["b-locked"].id in out
    assert b["b-corr"].id not in out
    assert b["b-posterior"].id not in out


# --- scenario 7: empty store prints sentinel ----------------------------------


def test_core_empty_store_exits_zero(isolated_db: Path) -> None:
    code, _ = _run("core")
    assert code == 0


def test_core_empty_store_message(isolated_db: Path) -> None:
    _, out = _run("core")
    assert out.strip() == "no core beliefs"


# --- scenario 8: threshold flags ---------------------------------------------


def test_core_min_corroboration_raised_drops_corr(isolated_db: Path) -> None:
    b = _seed_store(isolated_db)
    _, out = _run("core", "--min-corroboration", "4")
    assert b["b-corr"].id not in out


def test_core_disabled_posterior_and_corr_includes_all_nonprior(isolated_db: Path) -> None:
    """--min-posterior 0.0 --min-alpha-beta 0 includes b-thin (μ=0.667, α+β=3)."""
    b = _seed_store(isolated_db)
    _, out = _run("core", "--min-posterior", "0.0", "--min-alpha-beta", "0")
    assert b["b-thin"].id in out


# --- scenario 9: defensive — α+β==0 must not crash sort ----------------------


def test_core_zero_alpha_beta_does_not_crash(isolated_db: Path) -> None:
    """Belief with alpha=beta=0 must not raise ZeroDivisionError in sort path."""
    s = MemoryStore(str(isolated_db))
    try:
        s.insert_belief(
            _make_belief("b0zeroab0000000000", "zero ab edge case", alpha=0.0, beta=0.0),
        )
        s.insert_belief(
            _make_belief(
                "b0postlive00000000",
                "live posterior",
                alpha=4.0,
                beta=1.0,
            ),
        )
    finally:
        s.close()
    code, out = _run("core")
    assert code == 0
    assert "b0postlive00000000" in out


def test_core_zero_alpha_beta_with_min_ab_zero_does_not_crash(
    isolated_db: Path,
) -> None:
    """`--min-alpha-beta 0` must not enable the alpha/ab division on an α+β==0 belief.

    The default test above passes only because `min_alpha_beta=4` short-circuits
    the qualify check before the division. This variant exercises the actual
    unguarded sites at the JSON and text emission paths.
    """
    s = MemoryStore(str(isolated_db))
    try:
        s.insert_belief(
            _make_belief("b0zeroab0000000000", "zero ab edge case", alpha=0.0, beta=0.0),
        )
    finally:
        s.close()
    code, _ = _run("core", "--min-alpha-beta", "0")
    assert code == 0
