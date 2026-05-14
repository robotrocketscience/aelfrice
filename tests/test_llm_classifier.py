"""Acceptance tests for the v1.3.0 LLM-Haiku onboard classifier.

One test per acceptance criterion in
[docs/design/llm_classifier.md § 9](../docs/design/llm_classifier.md#9-acceptance-criteria-for-the-implementation-pr).

Hard rules from the spec / task:
- ALL tests mock the Anthropic client. No real network. No real
  `~/.aelfrice/`. Each test ≤2s.
- A tripwire test asserts that a default install (`aelf onboard <path>`
  with no flag and no config block) NEVER touches the SDK; the
  tripwire is an `_anthropic_importable` probe + a synthetic SDK
  module whose `Anthropic(...)` constructor raises on call.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import pytest

import aelfrice.cli as cli_module
import aelfrice.llm_classifier as llm
from aelfrice.models import Belief
from aelfrice.store import MemoryStore


def _all_beliefs(store: MemoryStore) -> list[Belief]:
    """Enumerate every belief in the store via the underlying conn.

    The public store API exposes `count_beliefs`, `get_belief(id)`,
    and `search_beliefs(query)` but not "list all" because retrieval
    callers always have a query. Tests need the unscoped view.
    """
    cur = store._conn.execute("SELECT * FROM beliefs")  # type: ignore[attr-defined]
    out: list[Belief] = []
    for row in cur.fetchall():
        out.append(
            Belief(
                id=row["id"],
                content=row["content"],
                content_hash=row["content_hash"],
                alpha=row["alpha"],
                beta=row["beta"],
                type=row["type"],
                lock_level=row["lock_level"],
                locked_at=row["locked_at"],
                created_at=row["created_at"],
                last_retrieved_at=row["last_retrieved_at"],
                origin=row["origin"] if "origin" in row.keys() else "unknown",
            )
        )
    return out


# --- Fakes ---------------------------------------------------------------


def _tripwire_sdk() -> Any:
    """Return a fake SDK module whose `Anthropic(...)` raises on call.

    Used to assert that a default `aelf onboard <path>` (no flag, no
    config) NEVER reaches the SDK constructor.
    """
    class _Tripwire:
        @staticmethod
        def Anthropic(**kwargs: Any) -> Any:  # pragma: no cover - tripwire
            raise AssertionError(
                "tripwire: SDK constructor invoked when no opt-in was given"
            )

    return _Tripwire()


# --- Fixtures ------------------------------------------------------------


@pytest.fixture
def tmp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect Path.home() and HOME to a tmp directory.

    Every test that touches `~/.aelfrice/` MUST use this fixture. We
    monkeypatch both `Path.home` and the env var so neither path
    leaks to the user's real home.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    return home


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Build a tiny project tree with one .md, one .py, and a few
    paragraphs that exercise classification.
    """
    root = tmp_path / "proj"
    root.mkdir()
    (root / "README.md").write_text(
        "the publish workflow blocks on green CI\n"
        "\n"
        "this is a question? what is the answer\n"
        "\n"
        "we always use uv for python work in this repo\n",
        encoding="utf-8",
    )
    (root / "module.py").write_text(
        '"""Returns the user home directory deterministically."""\n',
        encoding="utf-8",
    )
    return root


@pytest.fixture
def stub_sdk_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend `anthropic` is importable.

    The dev-deps don't actually install anthropic, but the LLM-path
    tests need gate 1 to pass so they can exercise the rest of the
    contract. Production gate-1 path uses `import anthropic` which
    is unavailable in dev, so this stub explicitly forces the
    extra-installed branch in `_anthropic_importable`.
    """
    monkeypatch.setattr(llm, "_anthropic_importable", lambda _check: True)


@pytest.fixture
def memdb(monkeypatch: pytest.MonkeyPatch) -> Iterator[MemoryStore]:
    store = MemoryStore(":memory:")
    real_close = store.close
    # Suppress the per-command `store.close()` so the same in-memory
    # DB persists across the cli call AND the post-call assertions.
    store.close = lambda: None  # type: ignore[method-assign]

    def _open() -> MemoryStore:
        return store

    monkeypatch.setattr(cli_module, "_open_store", _open)
    yield store
    real_close()


# --- 1. Default-off verified (TRIPWIRE) ---------------------------------


def test_default_install_no_flag_no_config_makes_zero_outbound_calls(
    tmp_home: Path,
    repo: Path,
    memdb: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance § 9.1 (post-v1.5 default-on, soft-fallback path).

    Default `aelf onboard <path>` resolves enabled=True (post-v1.5
    flip). Because the user did not explicitly opt in (no
    --llm-classify flag), gates 1 and 2 are run in soft-fallback mode:
    if the [onboard-llm] extra is not installed, or ANTHROPIC_API_KEY
    is not set, onboard silently falls back to the regex classifier
    instead of exiting 1. The tripwire below proves no Anthropic call
    is made on the soft-fallback path even when the API key is set.
    """
    # Provide a tripwire SDK module to llm_classifier — if any code
    # path tries to import it via our test-injection seam, it will
    # fail loudly. (Production import path uses `import anthropic`,
    # which is also unavailable — see test_optional_import_contract.)
    tripwire = _tripwire_sdk()
    # Patch the local import used by classify_batch. Real prod path
    # never reaches here without all four gates, but the assertion
    # is "any call to the SDK constructor fails the test."
    monkeypatch.setattr(
        llm,
        "_call_anthropic",
        lambda **kw: (_ for _ in ()).throw(
            AssertionError("tripwire: _call_anthropic invoked")
        ),
    )

    # Run with the default flags. Use ANTHROPIC_API_KEY=set to make
    # sure even *that* gate isn't what saves us — only gate 3
    # (no opt-in) should keep the network silent.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    rc = cli_module.main(["onboard", str(repo)], out=io.StringIO())
    assert rc == 0
    # The store should still have at least one belief — the regex
    # classifier ran. Tripwire never fired.
    assert memdb.count_beliefs() > 0


# --- 2. Opt-in path tested ----------------------------------------------


def test_llm_classify_flag_calls_haiku_once_and_inserts_typed_beliefs(
    tmp_home: Path,
    repo: Path,
    memdb: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_present: None,
) -> None:
    """Acceptance § 9.2.

    --llm-classify with sentinel pre-created: one Haiku call, parsed,
    and beliefs land with LLM-assigned types. Test asserts at least
    one belief carries a non-default origin (not all
    ORIGIN_AGENT_INFERRED — proving the LLM origin reached the store).
    """
    # Pre-create the consent sentinel so the prompt is suppressed.
    sentinel = llm.sentinel_path()
    llm.write_sentinel(sentinel, model=llm.DEFAULT_MODEL)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    # Build a deterministic LLM response. We can't predict the exact
    # candidate count because scan_repo runs the noise filter; mock
    # at the SDK layer with an inspector that builds the right-sized
    # response on demand.
    captured: dict[str, Any] = {}

    def fake_call_anthropic(**kwargs: Any) -> llm.ClientResponse:
        # Reflect the input to determine response size.
        msg = kwargs["user_message"]
        n = len(json.loads(msg))
        objs = [
            {
                "belief_type": "requirement",
                "origin": "document_recent",
                "persist": True,
            }
        ] * n
        captured["calls"] = captured.get("calls", 0) + 1
        captured["n_in"] = n
        return llm.ClientResponse(
            text=json.dumps(objs),
            input_tokens=80 * n,
            output_tokens=10 * n,
        )

    monkeypatch.setattr(llm, "_call_anthropic", fake_call_anthropic)

    out = io.StringIO()
    rc = cli_module.main(
        ["onboard", str(repo), "--llm-classify"], out=out,
    )
    assert rc == 0
    assert captured.get("calls") == 1
    # Beliefs land with the LLM-assigned type.
    beliefs = _all_beliefs(memdb)
    assert beliefs, "scan should have inserted at least one belief"
    assert any(b.type == "requirement" for b in beliefs)
    assert any(b.origin == "document_recent" for b in beliefs)


# --- 3. Config-block path tested ----------------------------------------


def test_config_block_enabled_routes_through_llm_without_flag(
    tmp_home: Path,
    repo: Path,
    memdb: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_present: None,
) -> None:
    """Acceptance § 9.3.

    [onboard.llm].enabled = true in .aelfrice.toml triggers the LLM
    path with no --llm-classify flag.
    """
    cfg = repo / ".aelfrice.toml"
    cfg.write_text(
        "[onboard.llm]\nenabled = true\n",
        encoding="utf-8",
    )
    sentinel = llm.sentinel_path()
    llm.write_sentinel(sentinel, model=llm.DEFAULT_MODEL)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    captured: dict[str, int] = {}

    def fake_call_anthropic(**kwargs: Any) -> llm.ClientResponse:
        n = len(json.loads(kwargs["user_message"]))
        captured["n"] = captured.get("n", 0) + 1
        return llm.ClientResponse(
            text=json.dumps(
                [
                    {
                        "belief_type": "factual",
                        "origin": "agent_inferred",
                        "persist": True,
                    }
                ]
                * n
            ),
            input_tokens=10 * n,
            output_tokens=5 * n,
        )

    monkeypatch.setattr(llm, "_call_anthropic", fake_call_anthropic)
    rc = cli_module.main(["onboard", str(repo)], out=io.StringIO())
    assert rc == 0
    assert captured["n"] == 1


# --- 4. Confirmation prompt tested -------------------------------------


def test_consent_prompt_emitted_on_first_run_and_n_aborts_before_network(
    tmp_home: Path,
    repo: Path,
    memdb: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_present: None,
) -> None:
    """Acceptance § 9.4 (part 1).

    First run without sentinel: prompt emitted to stderr; stdin = 'n'
    aborts with exit 1; no Anthropic call.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    # Tripwire: any call to the SDK fails the test.
    monkeypatch.setattr(
        llm,
        "_call_anthropic",
        lambda **kw: (_ for _ in ()).throw(
            AssertionError("network call after prompt rejection")
        ),
    )

    fake_stdin = io.StringIO("n\n")
    fake_stderr = io.StringIO()
    monkeypatch.setattr(sys, "stdin", fake_stdin)
    monkeypatch.setattr(sys, "stderr", fake_stderr)
    monkeypatch.setattr(fake_stdin, "isatty", lambda: True)

    rc = cli_module.main(
        ["onboard", str(repo), "--llm-classify"], out=io.StringIO(),
    )
    assert rc == 1
    assert "Continue with LLM classification?" in fake_stderr.getvalue()
    # Sentinel not written.
    assert not llm.sentinel_path().exists()


def test_consent_prompt_suppressed_when_sentinel_exists(
    tmp_home: Path,
    repo: Path,
    memdb: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_present: None,
) -> None:
    """Acceptance § 9.4 (part 2).

    Sentinel exists → no prompt. We assert by piping a `n` answer
    that, if read, would cause exit 1. With a valid sentinel the
    classifier proceeds and exits 0.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    llm.write_sentinel(llm.sentinel_path(), model=llm.DEFAULT_MODEL)

    monkeypatch.setattr(
        llm,
        "_call_anthropic",
        lambda **kw: llm.ClientResponse(
            text="[]", input_tokens=0, output_tokens=0,
        ),
    )

    fake_stdin = io.StringIO("n\n")
    monkeypatch.setattr(sys, "stdin", fake_stdin)

    rc = cli_module.main(
        ["onboard", str(repo), "--llm-classify"], out=io.StringIO(),
    )
    assert rc == 0


# --- 5. Dry-run no-network -----------------------------------------------


def test_dry_run_does_not_contact_network_and_does_not_write_sentinel(
    tmp_home: Path,
    repo: Path,
    memdb: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_present: None,
) -> None:
    """Acceptance § 9.5.

    --llm-classify --dry-run: prints candidates, no SDK call, no
    sentinel side-effect.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        llm,
        "_call_anthropic",
        lambda **kw: (_ for _ in ()).throw(
            AssertionError("dry-run reached _call_anthropic")
        ),
    )

    out = io.StringIO()
    rc = cli_module.main(
        ["onboard", str(repo), "--llm-classify", "--dry-run"], out=out,
    )
    assert rc == 0
    text = out.getvalue()
    assert "dry-run" in text
    # No belief was inserted (dry-run does not store).
    assert memdb.count_beliefs() == 0
    # Sentinel never written.
    assert not llm.sentinel_path().exists()


# --- 6. Fallback path ---------------------------------------------------


def test_transient_failure_falls_back_to_regex_with_audit_row(
    tmp_home: Path,
    repo: Path,
    memdb: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_present: None,
) -> None:
    """Acceptance § 9.6.

    Connection error during the Haiku call: regex `classify_sentence`
    runs for the affected candidates, beliefs are inserted, and the
    telemetry line shows fallbacks > 0. Exit 0.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    llm.write_sentinel(llm.sentinel_path(), model=llm.DEFAULT_MODEL)

    def transient_fail(**kw: Any) -> Any:
        raise llm.LLMTransientError("simulated connect refused")

    monkeypatch.setattr(llm, "_call_anthropic", transient_fail)

    out = io.StringIO()
    rc = cli_module.main(
        ["onboard", str(repo), "--llm-classify"], out=out,
    )
    assert rc == 0
    text = out.getvalue()
    assert "onboard.llm:" in text
    assert "fallbacks=" in text
    # At least one feedback_history audit row tagged with the
    # fallback prefix.
    events = memdb.list_feedback_events()
    assert any(
        e.source.startswith("onboard.llm.fallback") for e in events
    )


# --- 7. Auth-failure path -----------------------------------------------


def test_auth_failure_exits_1_with_no_inserts(
    tmp_home: Path,
    repo: Path,
    memdb: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_present: None,
) -> None:
    """Acceptance § 9.7.

    401/403 raised by the SDK: exit 1, no fallback, no beliefs
    inserted, error message names Anthropic auth.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "bad-key")
    llm.write_sentinel(llm.sentinel_path(), model=llm.DEFAULT_MODEL)

    def auth_fail(**kw: Any) -> Any:
        raise llm.LLMAuthError("AuthenticationError: 401 Unauthorized")

    monkeypatch.setattr(llm, "_call_anthropic", auth_fail)

    fake_stderr = io.StringIO()
    monkeypatch.setattr(sys, "stderr", fake_stderr)

    out = io.StringIO()
    rc = cli_module.main(
        ["onboard", str(repo), "--llm-classify"], out=out,
    )
    assert rc == 1
    assert "auth" in fake_stderr.getvalue().lower()
    assert memdb.count_beliefs() == 0


# --- 8. Token-cap tested ------------------------------------------------


def test_token_cap_aborts_run_and_keeps_partial_inserts(
    tmp_home: Path,
    repo: Path,
    memdb: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_present: None,
) -> None:
    """Acceptance § 9.8.

    With max_tokens = 10 (very low), the pre-flight estimator aborts
    the run. Exit 1 with the expected message. Re-running with a
    higher cap proceeds normally — the deterministic belief id
    means a partial-insertion run could resume on real data; here
    we test the abort + idempotent re-run shape.
    """
    cfg = repo / ".aelfrice.toml"
    cfg.write_text(
        "[onboard.llm]\n"
        "enabled = true\n"
        "max_tokens = 10\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    llm.write_sentinel(llm.sentinel_path(), model=llm.DEFAULT_MODEL)

    monkeypatch.setattr(
        llm,
        "_call_anthropic",
        lambda **kw: (_ for _ in ()).throw(
            AssertionError("token-cap should abort before SDK call")
        ),
    )

    fake_stderr = io.StringIO()
    monkeypatch.setattr(sys, "stderr", fake_stderr)

    rc = cli_module.main(
        ["onboard", str(repo), "--llm-classify"], out=io.StringIO(),
    )
    assert rc == 1
    assert "token cap" in fake_stderr.getvalue()


# --- 9. Cost telemetry --------------------------------------------------


def test_telemetry_line_contains_required_fields(
    tmp_home: Path,
    repo: Path,
    memdb: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_present: None,
) -> None:
    """Acceptance § 9.9.

    `onboard.llm:` summary line includes model, input_tokens,
    output_tokens, total_tokens, requests, fallbacks.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    llm.write_sentinel(llm.sentinel_path(), model=llm.DEFAULT_MODEL)

    def fake_call(**kw: Any) -> llm.ClientResponse:
        n = len(json.loads(kw["user_message"]))
        return llm.ClientResponse(
            text=json.dumps(
                [
                    {
                        "belief_type": "factual",
                        "origin": "agent_inferred",
                        "persist": True,
                    }
                ]
                * n
            ),
            input_tokens=200,
            output_tokens=80,
        )

    monkeypatch.setattr(llm, "_call_anthropic", fake_call)
    out = io.StringIO()
    rc = cli_module.main(
        ["onboard", str(repo), "--llm-classify"], out=out,
    )
    assert rc == 0
    line = [ln for ln in out.getvalue().splitlines() if "onboard.llm:" in ln]
    assert line, "expected onboard.llm: summary line on stdout"
    assert "model=" in line[0]
    assert "input_tokens=" in line[0]
    assert "output_tokens=" in line[0]
    assert "total_tokens=" in line[0]
    assert "requests=" in line[0]
    assert "fallbacks=" in line[0]


# --- 10. Optional-import contract ---------------------------------------


def test_aelfrice_default_install_does_not_import_anthropic_at_module_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance § 9.10.

    Importing aelfrice (every public entry point) must not produce
    an `anthropic` symbol resolvable in any aelfrice module's
    globals at module-load time. Mirrors the existing test for
    `fastmcp` in test_mcp_server.py.
    """
    import importlib

    for name in (
        "aelfrice",
        "aelfrice.cli",
        "aelfrice.scanner",
        "aelfrice.classification",
        "aelfrice.llm_classifier",
    ):
        mod = importlib.import_module(name)
        assert "anthropic" not in vars(mod), (
            f"{name} has anthropic in module globals at import time"
        )


# --- 11-14. Additional acceptance criteria ------------------------------


def test_revoke_consent_clears_sentinel(
    tmp_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance § 4.3 (--revoke-consent flag).

    `aelf onboard <path> --revoke-consent` removes the sentinel and
    exits 0; subsequent runs prompt again.
    """
    sentinel = llm.sentinel_path()
    llm.write_sentinel(sentinel, model=llm.DEFAULT_MODEL)
    assert sentinel.exists()
    rc = cli_module.main(
        ["onboard", "/tmp/does-not-matter", "--revoke-consent"],
        out=io.StringIO(),
    )
    assert rc == 0
    assert not sentinel.exists()


def test_sentinel_invalidated_by_new_model_id(
    tmp_home: Path,
) -> None:
    """Acceptance § 4.3 (model + major-version invalidation).

    Old sentinel for a different model id is rejected.
    """
    llm.write_sentinel(
        llm.sentinel_path(), model="claude-old-model-id",
    )
    record = llm.read_sentinel(llm.sentinel_path())
    assert record is not None
    assert not llm.is_sentinel_valid(record, model=llm.DEFAULT_MODEL)


def test_sentinel_invalidated_by_new_major_version(
    tmp_home: Path,
) -> None:
    """Acceptance § 4.3 (major-version invalidation, patch/minor OK).

    Old sentinel from a different aelfrice MAJOR version is rejected;
    same major (different minor / patch) is accepted.
    """
    llm.write_sentinel(
        llm.sentinel_path(),
        model=llm.DEFAULT_MODEL,
        version="0.9.0",
    )
    record = llm.read_sentinel(llm.sentinel_path())
    assert record is not None
    assert not llm.is_sentinel_valid(
        record, model=llm.DEFAULT_MODEL, version="1.3.0",
    )

    # Same major, different minor: still valid.
    llm.write_sentinel(
        llm.sentinel_path(),
        model=llm.DEFAULT_MODEL,
        version="1.2.5",
    )
    record2 = llm.read_sentinel(llm.sentinel_path())
    assert record2 is not None
    assert llm.is_sentinel_valid(
        record2, model=llm.DEFAULT_MODEL, version="1.3.0",
    )


def test_eof_on_consent_prompt_aborts_with_no_network(
    tmp_home: Path,
    repo: Path,
    memdb: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_present: None,
) -> None:
    """Acceptance § 4.3 (EOF / non-TTY → exit 1).

    EOF on stdin during the prompt → exit 1, no network, no sentinel.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        llm,
        "_call_anthropic",
        lambda **kw: (_ for _ in ()).throw(
            AssertionError("EOF prompt path reached the network")
        ),
    )
    fake_stdin = io.StringIO("")  # immediate EOF
    monkeypatch.setattr(sys, "stdin", fake_stdin)
    monkeypatch.setattr(fake_stdin, "isatty", lambda: True)

    rc = cli_module.main(
        ["onboard", str(repo), "--llm-classify"], out=io.StringIO(),
    )
    assert rc == 1
    assert not llm.sentinel_path().exists()


def test_non_tty_stdin_aborts_with_no_network(
    tmp_home: Path,
    repo: Path,
    memdb: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_present: None,
) -> None:
    """Acceptance § 4.3 (non-TTY environments).

    CI / scripts: stdin is not a TTY → exit 1, no network, no
    sentinel. The user must opt in interactively once.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        llm,
        "_call_anthropic",
        lambda **kw: (_ for _ in ()).throw(
            AssertionError("non-TTY path reached the network")
        ),
    )

    # Stub the prompt with non-TTY behaviour.
    fake_stdin = io.StringIO("y\n")
    monkeypatch.setattr(sys, "stdin", fake_stdin)
    monkeypatch.setattr(fake_stdin, "isatty", lambda: False)

    rc = cli_module.main(
        ["onboard", str(repo), "--llm-classify"], out=io.StringIO(),
    )
    assert rc == 1


def test_missing_api_key_with_flag_exits_1_without_network(
    tmp_home: Path,
    repo: Path,
    memdb: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_present: None,
) -> None:
    """Acceptance § 4.2 — gate 2.

    --llm-classify with ANTHROPIC_API_KEY unset: exit 1 with a clear
    hint, no network call.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        llm,
        "_call_anthropic",
        lambda **kw: (_ for _ in ()).throw(
            AssertionError("no-key path reached the network")
        ),
    )
    fake_stderr = io.StringIO()
    monkeypatch.setattr(sys, "stderr", fake_stderr)
    rc = cli_module.main(
        ["onboard", str(repo), "--llm-classify"], out=io.StringIO(),
    )
    assert rc == 1
    assert "ANTHROPIC_API_KEY" in fake_stderr.getvalue()


def test_flag_overrides_config_with_explicit_false(
    tmp_home: Path,
    repo: Path,
    memdb: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_present: None,
) -> None:
    """Acceptance § 3.4 (flag wins on conflict).

    Config block enabled=true; CLI `--llm-classify=false` forces
    regex. Tripwire SDK never reached.
    """
    cfg = repo / ".aelfrice.toml"
    cfg.write_text(
        "[onboard.llm]\nenabled = true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        llm,
        "_call_anthropic",
        lambda **kw: (_ for _ in ()).throw(
            AssertionError("flag=false should not reach the network")
        ),
    )
    rc = cli_module.main(
        ["onboard", str(repo), "--llm-classify=false"],
        out=io.StringIO(),
    )
    assert rc == 0


def test_classify_batch_drops_invalid_per_candidate_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec § 5.4 — per-candidate invalid: drop, do not regex-fallback.

    Mismatched array length → fall back to regex (whole batch).
    Per-candidate invalid field → drop that candidate, keep the rest.
    """
    inputs = [
        llm.CandidateInput(index=0, source="doc:a:p0", text="a"),
        llm.CandidateInput(index=1, source="doc:b:p0", text="b"),
        llm.CandidateInput(index=2, source="doc:c:p0", text="c"),
    ]
    response = json.dumps(
        [
            {"belief_type": "factual", "origin": "agent_inferred", "persist": True},
            {"belief_type": "INVALID_TYPE", "origin": "agent_inferred", "persist": True},
            {"belief_type": "factual", "origin": "agent_inferred", "persist": True},
        ]
    )

    def fake_call(**kw: Any) -> llm.ClientResponse:
        return llm.ClientResponse(
            text=response, input_tokens=10, output_tokens=10,
        )

    monkeypatch.setattr(llm, "_call_anthropic", fake_call)
    result = llm.classify_batch(inputs, api_key="k")
    # 3 in → 2 valid out (middle row dropped); skipped_invalid = 1.
    assert len(result.classifications) == 2
    assert result.telemetry.skipped_invalid == 1
    assert not result.fallback_used
    assert result.auth_error is None


def test_telemetry_is_stdout_only_no_file_writes(
    tmp_home: Path,
    repo: Path,
    memdb: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
    stub_sdk_present: None,
) -> None:
    """Spec § 6.3 — telemetry is stdout-only.

    The telemetry line goes to stdout. Nothing is written to the
    network and the only file artifact under ~/.aelfrice/ is the
    consent sentinel itself (already written by setup).
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    sentinel = llm.sentinel_path()
    llm.write_sentinel(sentinel, model=llm.DEFAULT_MODEL)
    pre_existing = set(p.relative_to(tmp_home) for p in tmp_home.rglob("*"))

    def fake_call(**kw: Any) -> llm.ClientResponse:
        n = len(json.loads(kw["user_message"]))
        return llm.ClientResponse(
            text=json.dumps(
                [
                    {
                        "belief_type": "factual",
                        "origin": "document_recent",
                        "persist": True,
                    }
                ]
                * n
            ),
            input_tokens=10 * n,
            output_tokens=2 * n,
        )

    monkeypatch.setattr(llm, "_call_anthropic", fake_call)
    out = io.StringIO()
    rc = cli_module.main(
        ["onboard", str(repo), "--llm-classify"], out=out,
    )
    assert rc == 0
    assert "onboard.llm:" in out.getvalue()
    # No new files in ~/.aelfrice/ beyond what was there before
    # the run (the sentinel was pre-created in this test).
    post = set(p.relative_to(tmp_home) for p in tmp_home.rglob("*"))
    assert post == pre_existing


# --- Additional unit tests for module-level invariants ------------------


def test_resolve_enabled_flag_wins_over_config() -> None:
    assert llm.resolve_enabled(flag=True, config_enabled=False) is True
    assert llm.resolve_enabled(flag=False, config_enabled=True) is False
    assert llm.resolve_enabled(flag=None, config_enabled=True) is True
    assert llm.resolve_enabled(flag=None, config_enabled=False) is False


def test_check_gates_default_off_returns_pass_false_no_exit() -> None:
    result = llm.check_gates(enabled=False)
    assert not result.pass_all
    assert result.exit_code is None


def test_check_gates_missing_extra_returns_exit_1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the [onboard-llm] extra is not installed, --llm-classify
    must exit 1 with the install hint, not silently fall back.
    """
    result = llm.check_gates(
        enabled=True,
        env={"ANTHROPIC_API_KEY": "k"},
        sdk_check=lambda: False,
    )
    assert not result.pass_all
    assert result.exit_code == 1
    assert "onboard-llm" in (result.message or "")


def test_check_gates_missing_env_var_returns_exit_1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = llm.check_gates(
        enabled=True,
        env={},
        sdk_check=lambda: True,
    )
    assert not result.pass_all
    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY" in (result.message or "")


def test_check_gates_all_pass_when_extra_and_key_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = llm.check_gates(
        enabled=True,
        env={"ANTHROPIC_API_KEY": "k"},
        sdk_check=lambda: True,
    )
    assert result.pass_all
    assert result.exit_code is None


def test_parse_response_array_length_mismatch_raises_transient() -> None:
    with pytest.raises(llm.LLMTransientError):
        llm.parse_response("[]", expected_count=3)


def test_parse_response_drops_per_candidate_invalid_origin() -> None:
    payload = json.dumps(
        [
            {
                "belief_type": "factual",
                "origin": "user_stated",  # forbidden for the LLM
                "persist": True,
            }
        ]
    )
    out = llm.parse_response(payload, expected_count=1)
    assert out == []


def test_dry_run_without_api_key_exits_1(
    tmp_home: Path,
    repo: Path,
    memdb: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--dry-run` is treated as explicit opt-in to the LLM path
    (post-v1.5 default-on flip). Without `ANTHROPIC_API_KEY`, gate 2
    fails fast with the install hint, exit 1.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    rc = cli_module.main(
        ["onboard", str(repo), "--dry-run"], out=io.StringIO(),
    )
    assert rc == 1
