"""The search-tool and memory-mirror hooks read tool names per host (#1812).

Both hooks take `host=` from their entry point down to the check that
reads `tool_name`. The console scripts pass no host, so an installed hook
reads the default and behaves as before. These tests pin the threading:
the entry point hands the host down, and the host selects the table entry
the check reads.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice import hook_claude_memory_mirror as mirror
from aelfrice import hook_search_tool as search
from aelfrice.hosts import SUPPORTED_HOSTS
from aelfrice.search_tool_names import (
    SEARCH_TOOL_NAMES,
    SEARCH_TOOL_NAMES_BY_HOST,
    SHELL_TOOL_NAME_BY_HOST,
)

_GREP_COMMAND = {"command": "grep -rn needle_token src"}


@pytest.mark.timeout(30)
def test_every_host_has_search_and_shell_tool_names() -> None:
    assert tuple(SEARCH_TOOL_NAMES_BY_HOST) == SUPPORTED_HOSTS
    assert tuple(SHELL_TOOL_NAME_BY_HOST) == SUPPORTED_HOSTS
    assert set(SEARCH_TOOL_NAMES_BY_HOST.values()) == {SEARCH_TOOL_NAMES}
    assert set(SHELL_TOOL_NAME_BY_HOST.values()) == {"Bash"}


@pytest.mark.timeout(30)
def test_the_host_selects_the_search_tool_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(SEARCH_TOOL_NAMES_BY_HOST, "codex", ("Probe",))
    payload: dict[str, object] = {"tool_name": "Probe"}
    assert search._is_search_tool_call(payload, "codex")
    assert not search._is_search_tool_call(payload, "claude")
    assert not search._is_search_tool_call(payload)


@pytest.mark.timeout(30)
def test_the_host_selects_the_shell_tool_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(SHELL_TOOL_NAME_BY_HOST, "codex", "Shell")
    payload: dict[str, object] = {"tool_name": "Shell", "tool_input": _GREP_COMMAND}
    assert search._extract_bash_query(payload, "codex") is not None
    assert search._extract_bash_query(payload, "claude") is None
    assert search._extract_bash_query(payload) is None


@pytest.mark.timeout(30)
def test_do_search_reads_the_search_lane_with_its_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(SEARCH_TOOL_NAMES_BY_HOST, "codex", ("Probe",))
    seen: list[dict[str, object]] = []
    monkeypatch.setattr(
        search, "_extract_query", lambda payload: (seen.append(payload), None)[1],
    )
    payload: dict[str, object] = {"tool_name": "Probe"}
    search._do_search(payload, io.StringIO(), host="codex")
    assert seen == [payload]
    search._do_search(payload, io.StringIO())
    assert seen == [payload]


@pytest.mark.timeout(30)
def test_do_search_reads_the_shell_lane_with_its_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(SHELL_TOOL_NAME_BY_HOST, "codex", "Shell")
    seen: list[str] = []
    monkeypatch.setattr(
        search, "_extract_bash_query",
        lambda payload, host: (seen.append(host), None)[1],
    )
    payload: dict[str, object] = {"tool_name": "Shell", "tool_input": _GREP_COMMAND}
    search._do_search(payload, io.StringIO(), host="codex")
    assert seen == ["codex"]
    search._do_search(payload, io.StringIO())
    assert seen == ["codex"]


def _search_payload() -> io.StringIO:
    return io.StringIO(json.dumps({
        "tool_name": "Grep", "tool_input": {"pattern": "needle"},
    }))


@pytest.mark.timeout(30)
@pytest.mark.parametrize("host", SUPPORTED_HOSTS)
def test_search_tool_main_hands_the_host_down(
    monkeypatch: pytest.MonkeyPatch, host: str,
) -> None:
    seen: list[str] = []

    def fake(
        payload: dict[str, object], stdout: object,
        stderr: object = None, *, host: str,
    ) -> None:
        seen.append(host)

    monkeypatch.setattr(search, "_do_search", fake)
    rc = search.main(
        stdin=_search_payload(), stdout=io.StringIO(), stderr=io.StringIO(),
        host=host,
    )
    assert rc == 0
    assert seen == [host]


@pytest.mark.timeout(30)
def test_search_tool_main_defaults_to_claude(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    def fake(
        payload: dict[str, object], stdout: object,
        stderr: object = None, *, host: str,
    ) -> None:
        seen.append(host)

    monkeypatch.setattr(search, "_do_search", fake)
    search.main(
        stdin=_search_payload(), stdout=io.StringIO(), stderr=io.StringIO(),
    )
    assert seen == ["claude"]


def _fact_payload(tmp_path: Path) -> dict[str, object]:
    fact = tmp_path / ".claude" / "projects" / "p" / "memory" / "fact.md"
    return {"tool_name": "Write", "tool_input": {"file_path": str(fact)}}


@pytest.mark.timeout(30)
def test_the_mirror_reads_write_tools_for_the_claude_host_only(
    tmp_path: Path,
) -> None:
    payload = _fact_payload(tmp_path)
    assert mirror._file_path_for_memory_write(payload) is not None
    assert mirror._file_path_for_memory_write(payload, "claude") is not None
    # A host with no entry keeps no claude-memory files to mirror.
    assert mirror._file_path_for_memory_write(payload, "codex") is None


@pytest.mark.timeout(30)
@pytest.mark.parametrize("host", SUPPORTED_HOSTS)
def test_do_mirror_checks_the_path_with_its_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, host: str,
) -> None:
    seen: list[str] = []
    monkeypatch.setattr(
        mirror, "_file_path_for_memory_write",
        lambda payload, host: (seen.append(host), None)[1],
    )
    mirror._do_mirror(_fact_payload(tmp_path), host=host)
    assert seen == [host]


@pytest.mark.timeout(30)
@pytest.mark.parametrize("host", [None, *SUPPORTED_HOSTS])
def test_mirror_main_hands_the_host_down(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, host: str | None,
) -> None:
    seen: list[str] = []
    monkeypatch.setattr(
        mirror, "_do_mirror",
        lambda payload, *, host: seen.append(host),
    )
    stdin = io.StringIO(json.dumps(_fact_payload(tmp_path)))
    if host is None:
        rc = mirror.main(stdin=stdin, stderr=io.StringIO())
    else:
        rc = mirror.main(stdin=stdin, stderr=io.StringIO(), host=host)
    assert rc == 0
    assert seen == [host or "claude"]
