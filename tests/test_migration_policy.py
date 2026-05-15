"""Tests for `scripts/check_migration_policy.py` (#840).

The gate refuses destructive `_MIGRATIONS` entries that ship without a
paired reader update in the same diff. These tests pin the four
positive-pass cases (no change, ADD COLUMN only, destructive + reader,
destructive + dataclass) and the destructive-pattern set
(DROP COLUMN / DROP TABLE / RENAME COLUMN / RENAME TO).

The historical-regression case is the #833 escape itself: pre-#814
base + post-#814 head with the DROP COLUMN added but the v3.1.0
`_row_to_belief` still unchanged. The gate must flag this. The same
shape with the v3.2.0 reader (read removed) must pass.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "check_migration_policy.py"
)


@pytest.fixture(scope="module")
def policy_module():
    """Load the script as a module (it isn't a package member)."""
    spec = importlib.util.spec_from_file_location(
        "check_migration_policy", str(_SCRIPT_PATH)
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_migration_policy"] = mod
    spec.loader.exec_module(mod)
    return mod


# --- store.py templates --------------------------------------------------


_STORE_TEMPLATE = '''\
"""Synthetic store.py shape for policy-gate tests."""
from __future__ import annotations
import sqlite3

_MIGRATIONS: tuple[str, ...] = (
{migrations}
)


def _row_to_belief(row: sqlite3.Row) -> "Belief":
{reader_body}
'''


_DEFAULT_READER = """\
    keys = row.keys()
    return Belief(id=row["id"], content=row["content"])
"""


_V3_1_0_READER = """\
    keys = row.keys()
    return Belief(
        id=row["id"],
        content=row["content"],
        demotion_pressure=row["demotion_pressure"],
    )
"""


_V3_2_0_READER = """\
    keys = row.keys()
    return Belief(id=row["id"], content=row["content"])
"""


def _store(migrations: list[str], reader_body: str = _DEFAULT_READER) -> str:
    lines = "\n".join(f'    "{m}",' for m in migrations)
    return _STORE_TEMPLATE.format(migrations=lines, reader_body=reader_body)


_MODELS_DEFAULT = '''\
"""Synthetic models.py."""
from dataclasses import dataclass

@dataclass
class Belief:
    id: str
    content: str
'''


_MODELS_WITH_DEMOTION = '''\
"""Synthetic models.py — pre-#814 Belief carries demotion_pressure."""
from dataclasses import dataclass

@dataclass
class Belief:
    id: str
    content: str
    demotion_pressure: int = 0
'''


# --- tests ---------------------------------------------------------------


def test_no_new_migrations_passes(policy_module) -> None:
    base = _store(["ALTER TABLE beliefs ADD COLUMN session_id TEXT"])
    head = _store(["ALTER TABLE beliefs ADD COLUMN session_id TEXT"])
    rc, msg = policy_module.check(
        base_store=base, head_store=head,
        base_models=_MODELS_DEFAULT, head_models=_MODELS_DEFAULT,
    )
    assert rc == 0, msg
    assert "no new migration entries" in msg


def test_add_column_alone_passes(policy_module) -> None:
    base = _store(["ALTER TABLE beliefs ADD COLUMN session_id TEXT"])
    head = _store([
        "ALTER TABLE beliefs ADD COLUMN session_id TEXT",
        "ALTER TABLE beliefs ADD COLUMN scope TEXT NOT NULL DEFAULT 'project'",
    ])
    rc, msg = policy_module.check(
        base_store=base, head_store=head,
        base_models=_MODELS_DEFAULT, head_models=_MODELS_DEFAULT,
    )
    assert rc == 0, msg
    assert "none destructive" in msg


def test_833_regression_fails(policy_module) -> None:
    """The #833 escape: DROP COLUMN added but reader untouched."""
    base = _store(
        ["ALTER TABLE beliefs ADD COLUMN session_id TEXT"],
        reader_body=_V3_1_0_READER,
    )
    head = _store(
        [
            "ALTER TABLE beliefs ADD COLUMN session_id TEXT",
            "ALTER TABLE beliefs DROP COLUMN demotion_pressure",
        ],
        reader_body=_V3_1_0_READER,  # unchanged — the bug
    )
    rc, msg = policy_module.check(
        base_store=base, head_store=head,
        base_models=_MODELS_WITH_DEMOTION,
        head_models=_MODELS_WITH_DEMOTION,
    )
    assert rc == 1, msg
    assert "DROP COLUMN" in msg
    assert "DROP COLUMN demotion_pressure" in msg


def test_drop_column_with_reader_change_passes(policy_module) -> None:
    base = _store(
        ["ALTER TABLE beliefs ADD COLUMN session_id TEXT"],
        reader_body=_V3_1_0_READER,
    )
    head = _store(
        [
            "ALTER TABLE beliefs ADD COLUMN session_id TEXT",
            "ALTER TABLE beliefs DROP COLUMN demotion_pressure",
        ],
        reader_body=_V3_2_0_READER,  # the fix
    )
    rc, msg = policy_module.check(
        base_store=base, head_store=head,
        base_models=_MODELS_WITH_DEMOTION,
        head_models=_MODELS_DEFAULT,
    )
    assert rc == 0, msg
    assert "paired reader change present" in msg
    assert "_row_to_belief changed=True" in msg


def test_drop_column_with_dataclass_change_alone_passes(
    policy_module,
) -> None:
    """Belief dataclass change is enough by itself."""
    base = _store(
        ["ALTER TABLE beliefs ADD COLUMN session_id TEXT"],
        reader_body=_DEFAULT_READER,
    )
    head = _store(
        [
            "ALTER TABLE beliefs ADD COLUMN session_id TEXT",
            "ALTER TABLE beliefs DROP COLUMN demotion_pressure",
        ],
        reader_body=_DEFAULT_READER,  # unchanged
    )
    rc, msg = policy_module.check(
        base_store=base, head_store=head,
        base_models=_MODELS_WITH_DEMOTION,
        head_models=_MODELS_DEFAULT,  # dataclass dropped the field
    )
    assert rc == 0, msg
    assert "Belief dataclass changed=True" in msg


def test_drop_table_without_reader_fails(policy_module) -> None:
    base = _store(["ALTER TABLE beliefs ADD COLUMN session_id TEXT"])
    head = _store([
        "ALTER TABLE beliefs ADD COLUMN session_id TEXT",
        "DROP TABLE legacy_corroborations",
    ])
    rc, msg = policy_module.check(
        base_store=base, head_store=head,
        base_models=_MODELS_DEFAULT, head_models=_MODELS_DEFAULT,
    )
    assert rc == 1
    assert "DROP TABLE" in msg


def test_rename_column_without_reader_fails(policy_module) -> None:
    base = _store(["ALTER TABLE beliefs ADD COLUMN session_id TEXT"])
    head = _store([
        "ALTER TABLE beliefs ADD COLUMN session_id TEXT",
        "ALTER TABLE beliefs RENAME COLUMN session_id TO sess_id",
    ])
    rc, msg = policy_module.check(
        base_store=base, head_store=head,
        base_models=_MODELS_DEFAULT, head_models=_MODELS_DEFAULT,
    )
    assert rc == 1
    assert "RENAME COLUMN" in msg


def test_rename_table_without_reader_fails(policy_module) -> None:
    base = _store(["ALTER TABLE beliefs ADD COLUMN session_id TEXT"])
    head = _store([
        "ALTER TABLE beliefs ADD COLUMN session_id TEXT",
        "ALTER TABLE old_beliefs RENAME TO beliefs_v2",
    ])
    rc, msg = policy_module.check(
        base_store=base, head_store=head,
        base_models=_MODELS_DEFAULT, head_models=_MODELS_DEFAULT,
    )
    assert rc == 1
    assert "RENAME TO" in msg


def test_destructive_pattern_is_case_insensitive(policy_module) -> None:
    base = _store(["ALTER TABLE beliefs ADD COLUMN session_id TEXT"])
    head = _store([
        "ALTER TABLE beliefs ADD COLUMN session_id TEXT",
        "alter table beliefs drop column demotion_pressure",
    ])
    rc, _ = policy_module.check(
        base_store=base, head_store=head,
        base_models=_MODELS_DEFAULT, head_models=_MODELS_DEFAULT,
    )
    assert rc == 1


def test_models_absent_falls_back_to_reader_only(policy_module) -> None:
    """If --base-models / --head-models are omitted, the gate still
    works — Belief-dataclass touch can't be detected, but a reader
    touch still satisfies the gate."""
    base = _store(
        ["ALTER TABLE beliefs ADD COLUMN session_id TEXT"],
        reader_body=_V3_1_0_READER,
    )
    head = _store(
        [
            "ALTER TABLE beliefs ADD COLUMN session_id TEXT",
            "ALTER TABLE beliefs DROP COLUMN demotion_pressure",
        ],
        reader_body=_V3_2_0_READER,
    )
    rc, msg = policy_module.check(
        base_store=base, head_store=head,
        base_models=None, head_models=None,
    )
    assert rc == 0, msg
    assert "_row_to_belief changed=True" in msg


def test_base_missing_constant_treats_as_empty(policy_module) -> None:
    """First-time introduction of `_MIGRATIONS` on `head` (no prior
    constant on `base`) — every head entry is "added" but the gate
    only fires on destructive shape; pure ADD COLUMN passes."""
    base = '"""no migrations yet."""\n'
    head = _store(["ALTER TABLE beliefs ADD COLUMN session_id TEXT"])
    rc, msg = policy_module.check(
        base_store=base, head_store=head,
        base_models=_MODELS_DEFAULT, head_models=_MODELS_DEFAULT,
    )
    assert rc == 0, msg


def test_head_missing_constant_is_internal_error(policy_module) -> None:
    base = _store(["ALTER TABLE beliefs ADD COLUMN session_id TEXT"])
    head = '"""no migrations on head."""\n'
    rc, msg = policy_module.check(
        base_store=base, head_store=head,
        base_models=_MODELS_DEFAULT, head_models=_MODELS_DEFAULT,
    )
    assert rc == 2
    assert "head store parse error" in msg


def test_non_string_entry_is_internal_error(policy_module) -> None:
    head = """\
_MIGRATIONS: tuple = (
    "ALTER TABLE beliefs ADD COLUMN session_id TEXT",
    123,
)
"""
    base = _store(["ALTER TABLE beliefs ADD COLUMN session_id TEXT"])
    rc, _ = policy_module.check(
        base_store=base, head_store=head,
        base_models=_MODELS_DEFAULT, head_models=_MODELS_DEFAULT,
    )
    assert rc == 2


def test_real_main_store_self_consistent(policy_module) -> None:
    """Smoke: the live `src/aelfrice/store.py` parses cleanly via the
    same extractor the gate uses, and self-comparison (base==head)
    reports zero new entries. Locks in the contract that the script
    can read the production file without falling back to
    rc=2 'internal error'."""
    live = Path(__file__).resolve().parents[1] / "src" / "aelfrice" / "store.py"
    source = live.read_text(encoding="utf-8")
    rc, msg = policy_module.check(
        base_store=source, head_store=source,
        base_models=None, head_models=None,
    )
    assert rc == 0, msg
    assert "no new migration entries" in msg
