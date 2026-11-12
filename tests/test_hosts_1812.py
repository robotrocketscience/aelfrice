"""`aelfrice.hosts` names the supported hosts and the default (#1812)."""
from __future__ import annotations

import argparse
import subprocess
import sys

import pytest

from aelfrice.hosts import (
    CLAUDE_HOST,
    CODEX_HOST,
    DEFAULT_HOST,
    GEMINI_HOST,
    HOST_ENV,
    SUPPORTED_HOSTS,
    host_from_env,
    host_of,
)


@pytest.mark.timeout(30)
def test_the_default_host_is_claude() -> None:
    assert DEFAULT_HOST == "claude"
    assert CLAUDE_HOST == "claude"
    assert CODEX_HOST == "codex"


@pytest.mark.timeout(30)
def test_supported_hosts_lists_claude_codex_then_gemini() -> None:
    assert SUPPORTED_HOSTS == ("claude", "codex", "gemini")


@pytest.mark.timeout(30)
def test_host_of_reads_the_namespace() -> None:
    assert host_of(argparse.Namespace(host="codex")) == "codex"
    assert host_of(argparse.Namespace(host="claude")) == "claude"


@pytest.mark.timeout(30)
def test_host_of_falls_back_to_the_default_without_a_host() -> None:
    assert host_of(argparse.Namespace()) == DEFAULT_HOST


@pytest.mark.timeout(30)
def test_the_host_marker_names_gemini() -> None:
    assert GEMINI_HOST == "gemini"
    assert HOST_ENV == "AELFRICE_HOST"
    assert host_from_env({HOST_ENV: "gemini"}) == "gemini"
    assert host_from_env({HOST_ENV: "codex"}) == "codex"


@pytest.mark.timeout(30)
@pytest.mark.parametrize("env", [{}, {HOST_ENV: ""}, {HOST_ENV: "other"}])
def test_an_absent_or_unknown_marker_reads_as_the_default(
    env: dict[str, str],
) -> None:
    assert host_from_env(env) == DEFAULT_HOST


@pytest.mark.timeout(30)
def test_host_from_env_reads_the_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(HOST_ENV, "gemini")
    assert host_from_env() == "gemini"
    monkeypatch.delenv(HOST_ENV)
    assert host_from_env() == DEFAULT_HOST


@pytest.mark.timeout(60)
def test_hosts_is_a_leaf_module() -> None:
    """Importing it loads no other aelfrice module, so a hook can read it
    without paying for the installer or the store."""
    probe = (
        "import sys, aelfrice.hosts; "
        "print(sorted(m for m in sys.modules if m.startswith('aelfrice')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, timeout=45, check=True,
    )
    assert result.stdout.strip() == "['aelfrice', 'aelfrice.hosts']"
