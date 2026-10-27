"""`aelf setup`, `unsetup`, and `doctor` pick their executor by host (#1812).

The executors live in per-host tables in `aelfrice.cli`. These tests pin
that each supported host has an entry, that the host on the namespace
selects it, and that a host with no entry fails instead of running some
other host's executor.
"""
from __future__ import annotations

import argparse
import io

import pytest

import aelfrice.cli as cli
from aelfrice.hosts import SUPPORTED_HOSTS


@pytest.mark.timeout(30)
def test_setup_and_unsetup_have_an_executor_for_every_host() -> None:
    assert tuple(cli._SETUP_BY_HOST) == SUPPORTED_HOSTS
    assert tuple(cli._UNSETUP_BY_HOST) == SUPPORTED_HOSTS


@pytest.mark.timeout(30)
def test_the_tables_hold_the_existing_executors() -> None:
    assert cli._SETUP_BY_HOST == {
        "claude": cli._cmd_setup_claude,
        "codex": cli._cmd_setup_codex,
    }
    assert cli._UNSETUP_BY_HOST == {
        "claude": cli._cmd_unsetup_claude,
        "codex": cli._cmd_unsetup_codex,
    }
    assert cli._DOCTOR_BY_HOST == {"codex": cli._cmd_doctor_codex}


def _recorders(
    monkeypatch: pytest.MonkeyPatch, table: dict[str, object],
) -> list[str]:
    calls: list[str] = []
    for host in list(table):
        monkeypatch.setitem(
            table, host,
            lambda _a, _o, host=host: (calls.append(host), 0)[1],
        )
    return calls


@pytest.mark.timeout(30)
@pytest.mark.parametrize("host", SUPPORTED_HOSTS)
@pytest.mark.parametrize(
    ("table_name", "entry"),
    [("_SETUP_BY_HOST", "_cmd_setup_locked"),
     ("_UNSETUP_BY_HOST", "_cmd_unsetup_locked")],
)
def test_the_named_host_selects_its_executor(
    monkeypatch: pytest.MonkeyPatch, host: str, table_name: str, entry: str,
) -> None:
    calls = _recorders(monkeypatch, getattr(cli, table_name))
    rc = getattr(cli, entry)(argparse.Namespace(host=host), io.StringIO())
    assert rc == 0
    assert calls == [host]


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    ("table_name", "entry"),
    [("_SETUP_BY_HOST", "_cmd_setup_locked"),
     ("_UNSETUP_BY_HOST", "_cmd_unsetup_locked")],
)
def test_a_namespace_without_a_host_selects_claude(
    monkeypatch: pytest.MonkeyPatch, table_name: str, entry: str,
) -> None:
    calls = _recorders(monkeypatch, getattr(cli, table_name))
    assert getattr(cli, entry)(argparse.Namespace(), io.StringIO()) == 0
    assert calls == ["claude"]


@pytest.mark.timeout(30)
@pytest.mark.parametrize("entry", ["_cmd_setup_locked", "_cmd_unsetup_locked"])
def test_a_host_with_no_executor_fails_rather_than_falling_through(
    monkeypatch: pytest.MonkeyPatch, entry: str,
) -> None:
    _recorders(monkeypatch, cli._SETUP_BY_HOST)
    _recorders(monkeypatch, cli._UNSETUP_BY_HOST)
    with pytest.raises(ValueError, match="no executor for host 'other'"):
        getattr(cli, entry)(argparse.Namespace(host="other"), io.StringIO())


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    ("host", "expected"), [("claude", "full doctor"), ("codex", "codex")],
)
def test_doctor_runs_the_host_scan_or_the_full_doctor(
    monkeypatch: pytest.MonkeyPatch, host: str, expected: str,
) -> None:
    calls = _recorders(monkeypatch, cli._DOCTOR_BY_HOST)
    # `--classify-orphans` is the first branch of the full doctor, so
    # reaching it shows the host table let the run through.
    monkeypatch.setattr(
        cli, "_cmd_doctor_classify_orphans",
        lambda _a, _o: (calls.append("full doctor"), 0)[1],
    )
    args = argparse.Namespace(host=host, classify_orphans=True)
    assert cli._cmd_doctor(args, io.StringIO()) == 0
    assert calls == [expected]
