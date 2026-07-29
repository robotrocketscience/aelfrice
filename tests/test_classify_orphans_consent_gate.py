"""Regression: `aelf doctor --classify-orphans` must not transmit without consent.

The documented boundary is four gates (`llm_classifier` module docstring):
the extra installed, the API key present, `enabled` resolved true, and a
recorded consent sentinel. `check_gates` only covers the first three and
punts gate 4 to the caller. `aelf onboard` wired it; `--classify-orphans`
did not — it hardcoded `enabled=True`, never read the sentinel, and sent
the content of stored beliefs to the vendor API.

Two distinct defects, both covered here:

1. **No consent check at all.** A user with the extra installed and
   `ANTHROPIC_API_KEY` exported could run a *diagnostic* subcommand and
   ship their memory store outbound with no prompt and no sentinel write
   — so `aelf doctor revoke-llm-consent` could not prevent it either.

2. **Disclosure gap.** The onboard prompt enumerates document sentences,
   commit subjects and docstrings, and promises "nothing outside the
   extracted candidate text". `--classify-orphans` sends stored belief
   content, which includes transcript-captured statements the user typed.
   An onboard sentinel therefore cannot authorise it, which is why
   consent is scoped (#1172) rather than a single boolean.

Falsifiable hypothesis: no outbound call is constructed unless a sentinel
recording the `stored_beliefs` scope is present and valid.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

import aelfrice.cli as cli_module
import aelfrice.llm_classifier as llm
from aelfrice.models import LOCK_NONE, ORIGIN_UNKNOWN, Belief
from aelfrice.store import MemoryStore


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryStore:
    db = tmp_path / "orphans.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    s = MemoryStore(str(db))
    import hashlib
    for i in range(3):
        content = f"orphan belief {i}"
        s.insert_belief(Belief(
            id=f"o{i}", content=content,
            content_hash=hashlib.sha256(content.encode()).hexdigest(),
            alpha=1.0, beta=1.0, type="unknown", lock_level=LOCK_NONE,
            locked_at=None, created_at="2026-01-01T00:00:00Z",
            last_retrieved_at=None, session_id=None, origin=ORIGIN_UNKNOWN,
        ))
    s.close()
    return s


@pytest.fixture
def sentinel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the sentinel away from the real `~/.aelfrice/`."""
    path = tmp_path / "llm-classify-consented"
    monkeypatch.setattr(cli_module, "_llm_sentinel_path", lambda: path)
    return path


@pytest.fixture
def tripwire(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Fail loudly if anything reaches the network seam."""
    calls = {"n": 0}

    def _boom(**_kwargs: Any) -> None:
        calls["n"] += 1
        raise AssertionError(
            "outbound call constructed without stored_beliefs consent",
        )

    monkeypatch.setattr(llm, "_call_anthropic", _boom)
    monkeypatch.setattr(llm, "_anthropic_importable", lambda _c: True)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return calls


def _run(argv: list[str]) -> tuple[int, str]:
    out = io.StringIO()
    return cli_module.main(argv, out=out), out.getvalue()


# --- Defect 1: no consent check --------------------------------------------


def test_no_sentinel_makes_no_outbound_call(
    store: MemoryStore, sentinel: Path, tripwire: dict[str, int],
) -> None:
    """The core assertion: absent consent, nothing is transmitted."""
    assert not sentinel.exists()
    rc, _ = _run(["doctor", "--classify-orphans"])

    assert rc == 1
    assert tripwire["n"] == 0


def test_refusal_does_not_write_a_sentinel(
    store: MemoryStore, sentinel: Path, tripwire: dict[str, int],
) -> None:
    """A declined prompt must not record consent as a side effect."""
    _run(["doctor", "--classify-orphans"])
    assert not sentinel.exists()


def test_dry_run_needs_no_consent_and_calls_nothing(
    store: MemoryStore, sentinel: Path, tripwire: dict[str, int],
) -> None:
    """--dry-run previews the candidate set with no gate and no network."""
    rc, text = _run(["doctor", "--classify-orphans", "--dry-run"])

    assert rc == 0
    assert tripwire["n"] == 0
    assert not sentinel.exists()
    assert "3" in text  # the three orphans are reported


# --- Defect 2: an onboard sentinel must not authorise belief content -------


def test_pre_1172_sentinel_does_not_authorise_belief_content(
    store: MemoryStore, sentinel: Path, tripwire: dict[str, int],
) -> None:
    """A sentinel written before scopes existed has no `scopes` key.

    Those users consented to the onboard disclosure, which never mentions
    stored belief content. Reading the absent key as "all scopes" would
    silently grandfather exactly the population this fix protects.
    """
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(json.dumps({
        "consented_at": "2026-01-01T00:00:00Z",
        "model": llm.LLMConfig.default().model,
        "aelfrice_version": llm._AELFRICE_VERSION,
    }), encoding="utf-8")

    rc, _ = _run(["doctor", "--classify-orphans"])
    assert rc == 1
    assert tripwire["n"] == 0


def test_onboard_scoped_sentinel_does_not_authorise_belief_content(
    store: MemoryStore, sentinel: Path, tripwire: dict[str, int],
) -> None:
    llm.write_sentinel(
        sentinel, model=llm.LLMConfig.default().model,
        scopes=(llm.CONSENT_SCOPE_ONBOARD_CANDIDATES,),
    )
    rc, _ = _run(["doctor", "--classify-orphans"])
    assert rc == 1
    assert tripwire["n"] == 0


def test_stored_beliefs_scope_permits_the_call(
    store: MemoryStore, sentinel: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control: with the right scope the command runs to completion."""
    monkeypatch.setattr(llm, "_anthropic_importable", lambda _c: True)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def _fake(**kwargs: Any) -> llm.ClientResponse:
        cands = json.loads(kwargs.get("user_message", "[]"))
        return llm.ClientResponse(
            text=json.dumps([
                {"index": c["index"], "belief_type": "factual",
                 "origin": "agent_inferred", "persist": True}
                for c in cands
            ]),
            input_tokens=1, output_tokens=1,
        )

    monkeypatch.setattr(llm, "_call_anthropic", _fake)
    llm.write_sentinel(
        sentinel, model=llm.LLMConfig.default().model,
        scopes=(llm.CONSENT_SCOPE_STORED_BELIEFS,),
    )

    rc, text = _run(["doctor", "--classify-orphans"])
    assert rc == 0
    assert "classified: 3" in text


# --- Sentinel semantics ----------------------------------------------------


def test_read_sentinel_defaults_missing_scopes_to_onboard(
    tmp_path: Path,
) -> None:
    path = tmp_path / "s"
    path.write_text(json.dumps({
        "consented_at": "2026-01-01T00:00:00Z",
        "model": "m", "aelfrice_version": "4.1.0",
    }), encoding="utf-8")

    rec = llm.read_sentinel(path)
    assert rec is not None
    assert rec.scopes == (llm.CONSENT_SCOPE_ONBOARD_CANDIDATES,)


def test_write_then_read_round_trips_scopes(tmp_path: Path) -> None:
    path = tmp_path / "s"
    both = (
        llm.CONSENT_SCOPE_ONBOARD_CANDIDATES,
        llm.CONSENT_SCOPE_STORED_BELIEFS,
    )
    llm.write_sentinel(path, model="m", scopes=both)
    rec = llm.read_sentinel(path)
    assert rec is not None
    assert rec.scopes == both


def test_required_scope_is_enforced_by_is_sentinel_valid() -> None:
    rec = llm.Sentinel(
        consented_at="2026-01-01T00:00:00Z", model="m",
        aelfrice_version=llm._AELFRICE_VERSION,
        scopes=(llm.CONSENT_SCOPE_ONBOARD_CANDIDATES,),
    )
    assert llm.is_sentinel_valid(rec, model="m") is True
    assert llm.is_sentinel_valid(
        rec, model="m", required_scope=llm.CONSENT_SCOPE_ONBOARD_CANDIDATES,
    ) is True
    assert llm.is_sentinel_valid(
        rec, model="m", required_scope=llm.CONSENT_SCOPE_STORED_BELIEFS,
    ) is False


def test_granting_belief_scope_preserves_onboard_scope(
    store: MemoryStore,
    sentinel: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Consenting here must not silently revoke onboard consent."""
    monkeypatch.setattr(llm, "_anthropic_importable", lambda _c: True)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        llm, "_call_anthropic",
        lambda **_k: llm.ClientResponse(text="[]", input_tokens=1, output_tokens=1),
    )
    monkeypatch.setattr(
        cli_module, "_llm_prompt_for_consent",
        lambda **_k: llm.PromptResult(accepted=True, reason="y"),
    )
    llm.write_sentinel(
        sentinel, model=llm.LLMConfig.default().model,
        scopes=(llm.CONSENT_SCOPE_ONBOARD_CANDIDATES,),
    )

    _run(["doctor", "--classify-orphans"])

    rec = llm.read_sentinel(sentinel)
    assert rec is not None
    assert llm.CONSENT_SCOPE_ONBOARD_CANDIDATES in rec.scopes
    assert llm.CONSENT_SCOPE_STORED_BELIEFS in rec.scopes


# --- Disclosure accuracy ---------------------------------------------------


def test_stored_beliefs_prompt_names_the_data_class() -> None:
    """The prompt must say what it sends; the onboard text does not.

    Reusing the onboard disclosure here would be a misstatement — it
    promises "nothing outside the extracted candidate text".
    """
    err = io.StringIO()
    llm.prompt_for_consent(
        stdin=io.StringIO("n\n"), stderr=err, is_tty=True,
        scope=llm.CONSENT_SCOPE_STORED_BELIEFS,
    )
    shown = err.getvalue().lower()

    assert "stored belief" in shown or "belief" in shown
    assert "memory store" in shown
    assert "transcript" in shown
    assert "--dry-run" in shown


def test_onboard_prompt_is_still_the_default_disclosure() -> None:
    err = io.StringIO()
    llm.prompt_for_consent(
        stdin=io.StringIO("n\n"), stderr=err, is_tty=True,
    )
    assert "aelf onboard" in err.getvalue()
