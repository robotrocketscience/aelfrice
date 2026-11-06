"""An SDK call that can't run stops with an error instead of falling back (#1846).

Each test runs the real `_call_anthropic` against a fake SDK module. The
CLI tests drive `main`, so they cover what the user sees: exit code, stderr,
the telemetry line, and the store.
"""
from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

import aelfrice.cli as cli_module
import aelfrice.llm_classifier as llm
from aelfrice.models import ORIGIN_UNKNOWN, Belief
from aelfrice.store import MemoryStore


class _FakeSdk:
    """Answers any attribute with the client factory, as `sdk_module`."""

    def __init__(self, factory: Callable[..., Any]) -> None:
        self._factory = factory

    def __getattr__(self, _name: str) -> Callable[..., Any]:
        return self._factory


def _client_with(create: Callable[..., Any]) -> _FakeSdk:
    def _factory(**_kwargs: Any) -> Any:
        return SimpleNamespace(messages=SimpleNamespace(create=create))

    return _FakeSdk(_factory)


def _raising(exc: BaseException) -> _FakeSdk:
    def _create(**_kwargs: Any) -> Any:
        raise exc

    return _client_with(_create)


def _create_without_temperature(
    *, model: str, max_tokens: int, system: str, messages: list[dict[str, str]],
) -> Any:
    """The keyword-only shape of `create` on an SDK that dropped the keyword."""
    raise AssertionError("the call must raise TypeError before reaching the body")


# The SDK's own class names, which `_falls_back` reads off the MRO.
class SdkBaseError(Exception):
    pass


class APIError(SdkBaseError):
    pass


class APIConnectionError(APIError):
    pass


class _StatusError(APIError):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


_CANDIDATES = [llm.CandidateInput(index=0, text="a sentence", source="doc:a.md:p1")]


def _batch(sdk: Any) -> llm.BatchResult:
    return llm.classify_batch(_CANDIDATES, api_key="k", sdk_module=sdk)


@pytest.mark.parametrize(
    "exc",
    [
        TypeError("got an unexpected keyword argument 'temperature'"),
        AttributeError("no attribute 'messages'"),
        ImportError("cannot import name"),
        ValueError("bad value"),
        KeyError("missing"),
        SdkBaseError("an SDK error outside the APIError family"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_exception_outside_the_network_family_is_a_mismatch(exc: Exception) -> None:
    result = _batch(_raising(exc))
    assert result.sdk_mismatch_error is not None
    assert result.sdk_mismatch_error.startswith(type(exc).__name__)
    assert not result.fallback_used
    assert result.telemetry.fallbacks == 0


def test_client_constructor_failure_is_a_mismatch() -> None:
    def _factory(**_kwargs: Any) -> Any:
        raise TypeError("unexpected keyword argument 'timeout'")

    result = _batch(_FakeSdk(_factory))
    assert result.sdk_mismatch_error is not None
    assert "timeout" in result.sdk_mismatch_error


def test_real_keyword_rejection_is_a_mismatch() -> None:
    result = _batch(_client_with(_create_without_temperature))
    assert result.sdk_mismatch_error is not None
    assert "TypeError" in result.sdk_mismatch_error
    assert "temperature" in result.sdk_mismatch_error


@pytest.mark.parametrize(
    "exc",
    [
        OSError("network unreachable"),
        ConnectionRefusedError("refused"),
        TimeoutError("timed out"),
        APIError("bare"),
        APIConnectionError("Connection error."),
        _StatusError(429),
        _StatusError(500),
        _StatusError(529),
        _StatusError(400),
        _StatusError(404),
        _StatusError(413),
        _StatusError(422),
    ],
    ids=lambda e: f"{type(e).__name__}-{getattr(e, 'status_code', '')}",
)
def test_network_fault_or_api_response_falls_back(exc: Exception) -> None:
    result = _batch(_raising(exc))
    assert result.sdk_mismatch_error is None
    assert result.auth_error is None
    assert result.fallback_used
    assert result.telemetry.fallbacks == len(_CANDIDATES)


def test_status_code_alone_falls_back() -> None:
    """A status error whose class isn't in the `APIError` family still falls back."""
    exc = RuntimeError("status 503")
    exc.status_code = 503  # type: ignore[attr-defined]
    result = _batch(_raising(exc))
    assert result.sdk_mismatch_error is None
    assert result.fallback_used


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failure_stays_an_auth_failure(status: int) -> None:
    result = _batch(_raising(_StatusError(status)))
    assert result.auth_error is not None
    assert result.sdk_mismatch_error is None


# --- Through `main` -------------------------------------------------------


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tiny project with a throwaway HOME, store, key, and full consent."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "memory.db"))
    monkeypatch.setenv("AELFRICE_USER_DB", str(tmp_path / "user.db"))
    monkeypatch.setenv(llm.ENV_API_KEY, "dummy-not-a-key")
    llm.write_sentinel(
        llm.sentinel_path(),
        model=llm.DEFAULT_MODEL,
        scopes=(llm.CONSENT_SCOPE_ONBOARD_CANDIDATES, llm.CONSENT_SCOPE_STORED_BELIEFS),
    )
    root = tmp_path / "proj"
    root.mkdir()
    (root / "README.md").write_text(
        "the publish workflow blocks on green CI\n\n"
        "we always use uv for python work in this repo\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(root)
    return root


def _use_sdk(monkeypatch: pytest.MonkeyPatch, sdk: Any) -> None:
    """Keep the real `_call_anthropic` in the path, with `sdk` as its module."""
    real = llm._call_anthropic  # pyright: ignore[reportPrivateUsage]

    def _call(**kwargs: Any) -> Any:
        return real(**{**kwargs, "sdk_module": sdk})

    monkeypatch.setattr(llm, "_call_anthropic", _call)
    monkeypatch.setattr(llm, "_anthropic_importable", lambda _check: True)


def _open_db(tmp_path: Path) -> MemoryStore:
    return MemoryStore(str(tmp_path / "memory.db"))


@pytest.mark.parametrize(
    "argv",
    [["--llm-classify"], []],
    ids=["explicit-flag", "default-on"],
)
def test_onboard_stops_and_names_the_exception(
    project: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
) -> None:
    _use_sdk(monkeypatch, _client_with(_create_without_temperature))

    out = io.StringIO()
    rc = cli_module.main(["onboard", str(project), *argv], out=out)

    err = capsys.readouterr().err
    assert rc == 1
    assert "TypeError" in err and "temperature" in err
    assert "--llm-classify=false" in err
    assert "onboard.llm:" not in out.getvalue()
    store = _open_db(tmp_path)
    try:
        assert store.count_beliefs() == 0
        assert store.list_feedback_events() == []
    finally:
        store.close()


def test_doctor_classify_orphans_stops_and_names_the_exception(
    project: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = _open_db(tmp_path)
    store.insert_belief(
        Belief(
            id="orphan1846",
            content="an orphan belief with no type",
            content_hash="orphan1846-hash",
            alpha=1.0,
            beta=1.0,
            type="unknown",
            lock_level="none",
            locked_at=None,
            created_at="2025-01-01T00:00:00Z",
            last_retrieved_at=None,
            session_id=None,
            origin=ORIGIN_UNKNOWN,
        )
    )
    store.close()
    _use_sdk(monkeypatch, _client_with(_create_without_temperature))

    out = io.StringIO()
    rc = cli_module.main(["doctor", "--classify-orphans"], out=out)

    err = capsys.readouterr().err
    assert rc == 1
    assert "TypeError" in err and "temperature" in err
    assert "No beliefs updated" in err
    assert "requests=" not in out.getvalue()
    store = _open_db(tmp_path)
    try:
        belief = store.get_belief("orphan1846")
        assert belief is not None and belief.type == "unknown"
    finally:
        store.close()


def test_onboard_still_falls_back_on_a_network_fault(
    project: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _use_sdk(monkeypatch, _raising(APIConnectionError("Connection error.")))

    out = io.StringIO()
    rc = cli_module.main(["onboard", str(project), "--llm-classify"], out=out)

    assert rc == 0
    assert "SDK" not in capsys.readouterr().err
    assert "onboard.llm:" in out.getvalue()
    store = _open_db(tmp_path)
    try:
        assert store.count_beliefs() > 0
        assert any(
            e.source.startswith("onboard.llm.fallback")
            for e in store.list_feedback_events()
        )
    finally:
        store.close()
