"""#1422 Part 2: the upgrade cleanup for the removed MCP surface.

The cleanup edits nothing by default. That is the property most of these
tests defend, because the file it looks at is one aelfrice never wrote and
which routinely holds the user's *other* MCP servers — the failure that
matters is not "we missed our entry", it is "we touched someone else's".
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aelfrice.mcp_cleanup import (
    find_registrations,
    is_aelfrice_mcp_entry,
    maybe_clean_up_mcp,
    mcp_extra_is_installed,
    remove_registration,
)

_NOW = datetime(2026, 8, 6, 12, 0, 0, tzinfo=timezone.utc)


def _write(path: Path, document: object) -> Path:
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


# --- the recognition predicate ---------------------------------------------
#
# Every shape below was published by this project at some point. A predicate
# written from the *current* docs alone matches only the first two.

@pytest.mark.parametrize(
    ("command", "args"),
    [
        ("aelf", ["mcp"]),                                              # MCP.md
        ("uv", ["run", "--project", "/abs", "aelf", "mcp"]),            # MCP.md
        ("aelf", ["serve"]),                                            # INSTALL.md @99160871
        ("uv", ["run", "--project", "/abs", "python", "-m", "aelfrice.mcp_server"]),
        ("python3", ["-m", "aelfrice.mcp_server"]),
        ("/home/u/.venv/bin/python", ["-m", "aelfrice.mcp_server"]),
        ("/opt/homebrew/bin/aelf", ["mcp"]),                            # absolute path
        ("aelf-mcp", []),                                               # documented, never shipped
    ],
)
def test_published_shapes_are_recognised(command: str, args: list[str]) -> None:
    assert is_aelfrice_mcp_entry(command, args) is True


@pytest.mark.parametrize(
    ("command", "args"),
    [
        ("aelf", ["status"]),        # `aelf` has 40+ other verbs
        ("aelf", []),                # bare command is not enough
        ("node", ["server.js"]),
        ("uv", ["run", "some-other-mcp-server"]),
        ("python3", ["-m", "someone_else.mcp_server"]),
        ("mcp-server-aelfrice-lookalike", ["mcp"]),
    ],
)
def test_other_servers_are_not_recognised(command: str, args: list[str]) -> None:
    """The expensive direction: a false positive deletes someone else's server."""
    assert is_aelfrice_mcp_entry(command, args) is False


# --- scanning ---------------------------------------------------------------


def test_our_entry_is_found_among_other_servers(tmp_path: Path) -> None:
    config = _write(tmp_path / "cfg.json", {
        "mcpServers": {
            "github": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"]},
            "aelfrice": {"command": "aelf", "args": ["mcp"]},
            "postgres": {"command": "uvx", "args": ["mcp-server-postgres"]},
        },
    })
    found, notes = find_registrations([config])
    assert [r.key for r in found] == ["aelfrice"]
    assert notes == []


def test_a_renamed_key_is_still_found(tmp_path: Path) -> None:
    """Users rename the map key, so the key is never part of the rule."""
    config = _write(tmp_path / "cfg.json", {
        "mcpServers": {"my-memory": {"command": "aelf", "args": ["mcp"]}},
    })
    found, _ = find_registrations([config])
    assert [r.key for r in found] == ["my-memory"]


def test_our_key_pointing_elsewhere_is_reported_not_claimed(tmp_path: Path) -> None:
    """Named `aelfrice` but running something else: report, never touch."""
    config = _write(tmp_path / "cfg.json", {
        "mcpServers": {"aelfrice": {"command": "node", "args": ["custom.js"]}},
    })
    found, notes = find_registrations([config])
    assert found == []
    assert any("did not publish" in n for n in notes)


@pytest.mark.parametrize(
    "content",
    [
        '{"mcpServers": {"a": {"command": "aelf", "args": ["mcp"]}},}',  # trailing comma
        '// leading comment\n{"mcpServers": {}}',                        # JSONC
        '{"mcpServers": {"a": {"command": "aelf"',                       # truncated
    ],
)
def test_unparseable_configs_report_and_do_not_claim_entries(
    tmp_path: Path, content: str
) -> None:
    """AC12: guessing at a malformed config is worse than leaving it."""
    config = tmp_path / "cfg.json"
    config.write_text(content, encoding="utf-8")
    found, notes = find_registrations([config])
    assert found == []
    assert any("not parseable" in n for n in notes)


@pytest.mark.parametrize(
    "document",
    [{}, {"mcpServers": {}}, {"mcpServers": []}, {"mcpServers": "nope"}, []],
)
def test_odd_shapes_never_raise(tmp_path: Path, document: object) -> None:
    config = _write(tmp_path / "cfg.json", document)
    found, _ = find_registrations([config])
    assert found == []


def test_a_missing_file_is_silent(tmp_path: Path) -> None:
    found, notes = find_registrations([tmp_path / "absent.json"])
    assert (found, notes) == ([], [])


# --- the opt-in edit --------------------------------------------------------


def test_removal_backs_up_first_and_preserves_other_servers(tmp_path: Path) -> None:
    """AC10/AC11: only our entry goes, and the original is recoverable."""
    config = _write(tmp_path / "cfg.json", {
        "mcpServers": {
            "github": {"command": "npx", "args": ["-y", "server-github"]},
            "aelfrice": {"command": "aelf", "args": ["mcp"]},
        },
        "otherTopLevelKey": {"kept": True},
    })
    original = config.read_text(encoding="utf-8")

    found, _ = find_registrations([config])
    changed, message = remove_registration(found[0], now=_NOW)

    assert changed is True
    document = json.loads(config.read_text(encoding="utf-8"))
    assert "aelfrice" not in document["mcpServers"]
    assert document["mcpServers"]["github"]["command"] == "npx"
    assert document["otherTopLevelKey"] == {"kept": True}

    backup = config.with_name("cfg.json.aelfrice-20260806T120000Z.bak")
    assert backup.exists()
    assert backup.read_text(encoding="utf-8") == original
    assert str(backup) in message


def test_removal_drops_an_emptied_mcpservers_key(tmp_path: Path) -> None:
    config = _write(tmp_path / "cfg.json", {
        "mcpServers": {"aelfrice": {"command": "aelf", "args": ["mcp"]}},
        "theme": "dark",
    })
    found, _ = find_registrations([config])
    remove_registration(found[0], now=_NOW)

    document = json.loads(config.read_text(encoding="utf-8"))
    assert "mcpServers" not in document
    assert document["theme"] == "dark"


def test_removal_is_idempotent(tmp_path: Path) -> None:
    config = _write(tmp_path / "cfg.json", {
        "mcpServers": {"aelfrice": {"command": "aelf", "args": ["mcp"]}},
    })
    found, _ = find_registrations([config])
    assert remove_registration(found[0], now=_NOW)[0] is True
    changed, message = remove_registration(found[0], now=_NOW)
    assert changed is False
    assert "already gone" in message


# --- the automatic pass -----------------------------------------------------


def test_the_automatic_pass_never_edits(tmp_path: Path) -> None:
    """The property this whole design rests on: report, do not touch."""
    config = _write(tmp_path / "cfg.json", {
        "mcpServers": {"aelfrice": {"command": "aelf", "args": ["mcp"]}},
    })
    before = config.read_text(encoding="utf-8")

    result = maybe_clean_up_mcp(
        sentinel_path=tmp_path / "sentinel",
        config_paths=[config],
        receipt_path=tmp_path / "absent-receipt.toml",
    )

    assert result.ran is True
    assert config.read_text(encoding="utf-8") == before
    assert any("aelf migrate --remove-mcp-config" in n for n in result.notes)


def test_the_sentinel_short_circuits_the_second_run(tmp_path: Path) -> None:
    sentinel = tmp_path / "sentinel"
    config = _write(tmp_path / "cfg.json", {"mcpServers": {}})

    first = maybe_clean_up_mcp(
        sentinel_path=sentinel, config_paths=[config],
        receipt_path=tmp_path / "absent.toml",
    )
    assert first.ran is True
    assert sentinel.exists()

    second = maybe_clean_up_mcp(
        sentinel_path=sentinel, config_paths=[config],
        receipt_path=tmp_path / "absent.toml",
    )
    assert second.ran is False
    assert "sentinel" in second.reason


def test_the_uv_receipt_decides_whether_the_extra_is_installed(
    tmp_path: Path,
) -> None:
    """AC9: a stdlib TOML read, no subprocess, no `uv tool install` call."""
    with_extra = tmp_path / "with.toml"
    with_extra.write_text(
        '[tool]\nrequirements = [{ name = "aelfrice", extras = ["mcp"] }]\n',
        encoding="utf-8",
    )
    without = tmp_path / "without.toml"
    without.write_text(
        '[tool]\nrequirements = [{ name = "aelfrice" }]\n', encoding="utf-8"
    )
    malformed = tmp_path / "bad.toml"
    malformed.write_text("this is not toml = = =\n", encoding="utf-8")

    assert mcp_extra_is_installed(with_extra) is True
    assert mcp_extra_is_installed(without) is False
    assert mcp_extra_is_installed(malformed) is False
    assert mcp_extra_is_installed(tmp_path / "absent.toml") is False


def test_the_extra_is_advised_never_reinstalled(tmp_path: Path) -> None:
    """The inversion trap: this population IS the uv-tool install.

    `maybe_migrate_to_uv` may shell out precisely because it never runs on a
    uv-tool install. Here the dead extra lives in one, so the cleanup prints
    the command instead of running a package operation mid-process.
    """
    receipt = tmp_path / "uv-receipt.toml"
    receipt.write_text(
        '[tool]\nrequirements = [{ name = "aelfrice", extras = ["mcp"] }]\n',
        encoding="utf-8",
    )
    result = maybe_clean_up_mcp(
        sentinel_path=tmp_path / "sentinel",
        config_paths=[tmp_path / "absent.json"],
        receipt_path=receipt,
    )
    assert result.extra_installed is True
    assert any("uv tool install --force aelfrice" in n for n in result.notes)
