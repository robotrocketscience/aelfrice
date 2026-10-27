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
    SUPPORTED_HOSTS,
    host_of,
)


@pytest.mark.timeout(30)
def test_the_default_host_is_claude() -> None:
    assert DEFAULT_HOST == "claude"
    assert CLAUDE_HOST == "claude"
    assert CODEX_HOST == "codex"


@pytest.mark.timeout(30)
def test_supported_hosts_lists_claude_then_codex() -> None:
    assert SUPPORTED_HOSTS == ("claude", "codex")


@pytest.mark.timeout(30)
def test_host_of_reads_the_namespace() -> None:
    assert host_of(argparse.Namespace(host="codex")) == "codex"
    assert host_of(argparse.Namespace(host="claude")) == "claude"


@pytest.mark.timeout(30)
def test_host_of_falls_back_to_the_default_without_a_host() -> None:
    assert host_of(argparse.Namespace()) == DEFAULT_HOST


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
