"""E2E scenario #5 (#334): v1.4 fixture DB -> current binary -> reads still work.

Catches: schema-migration regressions on real DB shapes. Unit-test
suites historically run against fresh-init DBs which exercise only the
current schema; they do not catch a migration that drops or corrupts
existing rows. This test opens a real v1.4 snapshot built from the
v1.4.0 PyPI release (see fixtures/build-v14-snapshot.sh) and asserts:

    1. The current `aelf search` reads the pre-existing locked beliefs.
    2. The post-migration DB has gained the v1.5/v1.6 tables that the
       current schema requires (belief_corroborations, ingest_log).
    3. The pre-existing belief rows survive the upgrade (count + content).

Boundary rule: invokes the binary as installed (subprocess), not via
in-process imports. The fixture is a binary file, treated read-only —
each test copies it into the ephemeral DB path before running.
"""
from __future__ import annotations

import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Callable

import pytest


pytestmark = pytest.mark.timeout(120)

FIXTURE = Path(__file__).parent / "fixtures" / "v14-snapshot.db"

# The three statements seeded into the v1.4 snapshot by build-v14-snapshot.sh.
# Hard-coded so a regeneration that changes the seed corpus trips review.
SEEDED_STATEMENTS = (
    "Quokkas calibrate the knob carefully on Tuesdays.",
    "The aardvark counter resets at midnight.",
    "Wibble pickling requires the canonical protocol header bytes.",
)


def _table_names(db_path: Path) -> set[str]:
    uri = f"file:{db_path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    return {str(r[0]) for r in rows}


def _belief_statements(db_path: Path) -> set[str]:
    uri = f"file:{db_path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        rows = conn.execute("SELECT content FROM beliefs").fetchall()
    return {str(r[0]) for r in rows}


@pytest.fixture
def v14_db(tmp_path: Path) -> Path:
    """Working copy of the v1.4 snapshot. The fixture file is read-only;
    each test gets its own mutable copy so the binary can migrate it.
    """
    if not FIXTURE.exists():
        pytest.skip(
            f"v1.4 snapshot missing at {FIXTURE}; "
            "run tests/e2e/fixtures/build-v14-snapshot.sh"
        )
    dest = tmp_path / "v14-working.sqlite3"
    shutil.copyfile(FIXTURE, dest)
    return dest


def test_v14_snapshot_seeds_are_searchable_after_migration(
    installed_aelf,
    v14_db: Path,
) -> None:
    """Current binary opens the v1.4 DB and `aelf search` returns each
    seeded statement. Failure means migration corrupted the belief rows
    or broke the FTS index that the search path relies on.
    """
    pre_tables = _table_names(v14_db)
    # The v1.4 snapshot must lack the post-v1.4 tables; if it doesn't,
    # the fixture is stale and the test is no longer testing migration.
    assert "belief_corroborations" not in pre_tables, (
        f"fixture {FIXTURE} already has belief_corroborations; "
        "regenerate via build-v14-snapshot.sh against aelfrice==1.4.0"
    )
    assert "ingest_log" not in pre_tables, (
        f"fixture {FIXTURE} already has ingest_log; "
        "regenerate via build-v14-snapshot.sh against aelfrice==1.4.0"
    )

    # Search distinctive tokens unique to each seeded statement. The
    # statement bodies appear verbatim in `aelf search` output.
    for token in ("quokka", "aardvark", "wibble"):
        proc = subprocess.run(  # noqa: S603 — argv list, not shell
            [*installed_aelf, "search", token],
            env={"AELFRICE_DB": str(v14_db), "PATH": _path()},
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        assert token in proc.stdout.lower(), (
            f"expected {token!r} hit in search output; got:\n{proc.stdout!r}"
        )


def test_v14_migration_grows_tables_and_preserves_belief_rows(
    installed_aelf,
    v14_db: Path,
) -> None:
    """After the current binary touches the v1.4 DB, the schema must
    have the post-v1.4 tables (regression on additive migration) and
    the original belief rows must be intact (regression on content).
    """
    pre_statements = _belief_statements(v14_db)
    assert pre_statements == set(SEEDED_STATEMENTS), (
        f"fixture {FIXTURE} seed corpus drifted; "
        f"got {sorted(pre_statements)}"
    )

    # Any read-write operation triggers the migration on first connect.
    # `aelf locked` lists locked beliefs and is read-mostly.
    subprocess.run(  # noqa: S603
        [*installed_aelf, "locked"],
        env={"AELFRICE_DB": str(v14_db), "PATH": _path()},
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )

    post_tables = _table_names(v14_db)
    required_post_tables = {
        "belief_corroborations",  # v1.5 corroboration tracking
        "ingest_log",  # v2.0 #205 ingest source-of-truth
        "belief_versions",  # version-vector backfill
    }
    missing = required_post_tables - post_tables
    assert not missing, (
        f"migration did not add expected tables: {sorted(missing)}; "
        f"observed: {sorted(post_tables)}"
    )

    post_statements = _belief_statements(v14_db)
    assert post_statements == set(SEEDED_STATEMENTS), (
        f"migration altered belief rows; pre={sorted(pre_statements)}, "
        f"post={sorted(post_statements)}"
    )


def _path() -> str:
    """Minimal PATH for subprocess. `installed_aelf` may be an absolute
    path or `uv run aelf`; either way the resolver is the parent's PATH.
    """
    import os

    return os.environ.get("PATH", "")


def test_search_after_migration_returns_seeded_belief(
    aelf_run: Callable[..., subprocess.CompletedProcess[str]],
    tmp_path: Path,
) -> None:
    """End-to-end variant using the suite's `aelf_run` fixture, which
    pins AELFRICE_DB through the same env-overlay path as the other
    e2e tests. Catches regressions where the migration would succeed
    via direct argv invocation but fail under the fixture's env shape.
    """
    if not FIXTURE.exists():
        pytest.skip(
            f"v1.4 snapshot missing at {FIXTURE}; "
            "run tests/e2e/fixtures/build-v14-snapshot.sh"
        )
    # `aelf_run` reads AELFRICE_DB from its own ephemeral_db fixture; we
    # have to overwrite that target with the v1.4 snapshot before the
    # first invocation so the migration runs on the real fixture data.
    db = tmp_path / "aelf.sqlite3"
    shutil.copyfile(FIXTURE, db)

    result = aelf_run("search", "wibble")
    assert "wibble" in result.stdout.lower(), (
        f"expected wibble hit after migration; stdout:\n{result.stdout!r}"
    )
