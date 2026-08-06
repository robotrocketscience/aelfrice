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


# --- locally-scoped registrations, backup collisions, re-arming ----------
#
# Four defects found in review. Each is a user-machine failure on a
# destructive one-shot path, so each gets a test that fails without its fix.


def test_a_locally_scoped_registration_is_found_and_removable(
    tmp_path: Path,
) -> None:
    """A host stores a *local*-scope server nested, not at the top level.

    Scanning only `mcpServers` reports "nothing to clean up" on that shape
    and then latches the sentinel, so the user is never told again. Both
    halves are asserted: finding it, and removing it from the right map —
    a scan that found it without carrying the container could not remove
    it.
    """
    config = _write(tmp_path / "cfg.json", {
        "mcpServers": {},
        "projects": {
            "/home/u/proj": {
                "mcpServers": {"aelfrice": {"command": "aelf", "args": ["mcp"]}},
            },
        },
    })

    found, _notes = find_registrations([config])
    assert [r.key for r in found] == ["aelfrice"]
    assert found[0].project == "/home/u/proj"
    assert found[0].location() == "projects./home/u/proj.mcpServers.aelfrice"

    changed, _message = remove_registration(found[0], now=_NOW)
    assert changed is True
    document = json.loads(config.read_text(encoding="utf-8"))
    assert "mcpServers" not in document["projects"]["/home/u/proj"]
    # The top-level empty map is untouched: this removed one entry, not
    # everything that looked like one.
    assert document["mcpServers"] == {}


def test_both_scopes_are_scanned_in_one_pass(tmp_path: Path) -> None:
    """A config may carry a global and a local registration at once.

    Asserted so a fix that merely *switched* which map is read would fail
    here rather than trading one blind spot for another.
    """
    config = _write(tmp_path / "cfg.json", {
        "mcpServers": {"aelfrice": {"command": "aelf", "args": ["mcp"]}},
        "projects": {
            "/p": {"mcpServers": {"aelf-local": {"command": "aelf", "args": ["mcp"]}}},
        },
    })
    found, _notes = find_registrations([config])
    assert sorted(r.location() for r in found) == [
        "mcpServers.aelfrice",
        "projects./p.mcpServers.aelf-local",
    ]


def test_two_removals_from_one_file_keep_both_backups(tmp_path: Path) -> None:
    """The first backup must survive the second removal.

    The stamp is second-resolution, so two removals in one run resolved to
    the same filename and the second wrote *already-edited* content over
    it — destroying the only copy of the pre-edit config while the message
    still named it as the undo path. `now` is pinned here precisely so the
    collision is forced rather than left to timing.
    """
    original = {"mcpServers": {
        "aelfrice": {"command": "aelf", "args": ["mcp"]},
        "aelfrice-src": {
            "command": "uv",
            "args": ["run", "--project", "/abs", "aelf", "mcp"],
        },
        "other": {"command": "somethingelse", "args": []},
    }}
    config = _write(tmp_path / "cfg.json", original)

    found, _notes = find_registrations([config])
    assert len(found) == 2
    for registration in found:
        changed, _message = remove_registration(registration, now=_NOW)
        assert changed is True

    backups = sorted(tmp_path.glob("cfg.json.aelfrice-*.bak"))
    assert len(backups) == 2, "the second removal overwrote the first backup"
    # One backup must hold the untouched original — that is the whole
    # point of naming it as the undo path.
    restored = [
        sorted(json.loads(b.read_text(encoding="utf-8"))["mcpServers"])
        for b in backups
    ]
    assert sorted(original["mcpServers"]) in restored
    assert sorted(json.loads(config.read_text(encoding="utf-8"))["mcpServers"]) == [
        "other"
    ]


def test_the_same_file_reached_by_two_candidate_paths_is_scanned_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`cwd` is the user's home often enough to matter.

    Scanning one file twice duplicated every note and made a successful
    removal report failure on the second pass.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.chdir(tmp_path)
    _write(tmp_path / ".mcp.json", {
        "mcpServers": {"aelfrice": {"command": "aelf", "args": ["mcp"]}},
    })

    found, _notes = find_registrations()
    assert len(found) == 1, "the same file was scanned twice"


def test_an_unreadable_config_re_arms_rather_than_latching(
    tmp_path: Path,
) -> None:
    """A scan that could not read its input has not proved anything.

    Latching the sentinel on it suppresses the one-shot report for good on
    exactly the machines that still need it — and the module docstring
    already promises the opposite.
    """
    broken = tmp_path / "broken.json"
    broken.write_text("{ not json,", encoding="utf-8")
    sentinel = tmp_path / "sentinel"

    result = maybe_clean_up_mcp(
        config_paths=[broken],
        sentinel_path=sentinel,
        receipt_path=tmp_path / "absent.toml",
    )
    assert result.ran is True
    assert sentinel.exists() is False, "an incomplete scan latched the sentinel"

    # The control: a clean scan still latches, or this test would pass
    # against a build that never wrote the sentinel at all.
    clean = _write(tmp_path / "clean.json", {"mcpServers": {}})
    result = maybe_clean_up_mcp(
        config_paths=[clean],
        sentinel_path=sentinel,
        receipt_path=tmp_path / "absent.toml",
    )
    assert sentinel.exists() is True


def test_a_windows_launcher_is_recognised(tmp_path: Path) -> None:
    """`aelf.exe` is the same command, and saying otherwise is a lie.

    Unrecognised plus a key named `aelfrice` makes the routine print that
    aelfrice did not publish that command, about one it did.
    """
    assert is_aelfrice_mcp_entry(r"C:\Users\u\.local\bin\aelf.exe", ["mcp"]) is True
    assert is_aelfrice_mcp_entry("aelf.exe", ["mcp"]) is True
    # Still needs the verb: the suffix strip must not widen the match.
    assert is_aelfrice_mcp_entry("aelf.exe", ["status"]) is False


def test_the_with_fastmcp_install_shape_is_detected(tmp_path: Path) -> None:
    """`uv tool install --with fastmcp aelfrice` was published too.

    uv records it as a sibling requirement, not as an extra, so checking
    only `extras` reported "nothing installed" to that whole population
    while the dead dependency sat on disk — and the CHANGELOG promises the
    pass reports it.
    """
    receipt = tmp_path / "uv-receipt.toml"
    receipt.write_text(
        'requirements = [{ name = "aelfrice" }, { name = "fastmcp" }]\n'
        "[tool]\n"
        'requirements = [{ name = "aelfrice" }, { name = "fastmcp" }]\n',
        encoding="utf-8",
    )
    assert mcp_extra_is_installed(receipt) is True

    # The control: a plain install must still read as clean, or the check
    # would fire for everyone.
    plain = tmp_path / "plain.toml"
    plain.write_text(
        "[tool]\nrequirements = [{ name = \"aelfrice\" }]\n", encoding="utf-8",
    )
    assert mcp_extra_is_installed(plain) is False


def test_a_scalar_tool_key_reads_as_not_installed(tmp_path: Path) -> None:
    """A hand-edited receipt must not escape as AttributeError.

    `receipt.get("tool", {}).get(...)` calls `.get` on whatever the receipt
    holds. `aelf setup`'s broad handler happens to swallow the resulting
    AttributeError, so only a direct call — the one `/aelf:upgrade` makes —
    shows the docstring's "unparseable reads as not-an-mcp-install"
    contract being broken.
    """
    receipt = tmp_path / "uv-receipt.toml"
    receipt.write_text('tool = "aelfrice"\n', encoding="utf-8")
    assert mcp_extra_is_installed(receipt) is False


def test_the_success_message_names_a_project_scoped_entry_correctly(
    tmp_path: Path,
) -> None:
    """The message is the undo instruction; naming the wrong key misleads.

    A locally-scoped registration lives at `projects.<dir>.mcpServers.<key>`.
    Reporting it as `mcpServers.<key>` sends the user to a key that does not
    exist in their file. `Registration.location()` exists to spell this, and
    every other message in the module already uses it.
    """
    project = "/home/u/proj"
    config = _write(tmp_path / "cfg.json", {
        "projects": {project: {"mcpServers": {
            "aelfrice": {"command": "aelf", "args": ["mcp"]},
        }}},
    })
    found, _ = find_registrations([config])
    assert len(found) == 1
    changed, message = remove_registration(found[0], now=_NOW)
    assert changed is True
    assert f"projects.{project}.mcpServers.aelfrice" in message
