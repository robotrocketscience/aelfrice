"""An SDK call that can't run stops with an error instead of falling back (#1846).

Each test runs the real `_call_anthropic` against a fake SDK module. The
CLI tests drive `main`, so they cover what the user sees: exit code, stderr,
the telemetry line, and the store.
"""
from __future__ import annotations

import io
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
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


# --- The request count ----------------------------------------------------


class ConnectError(Exception):
    pass


class ConnectTimeout(Exception):
    pass


class ProxyError(Exception):
    pass


class UnsupportedProtocol(Exception):
    pass


class ReadError(Exception):
    pass


class ReadTimeout(Exception):
    pass


def _connection_error(cause: type[Exception]) -> APIConnectionError:
    exc = APIConnectionError("Connection error.")
    exc.__cause__ = cause("from the HTTP library")
    return exc


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (_connection_error(ConnectError), 0),
        (_connection_error(ConnectTimeout), 0),
        (_connection_error(ProxyError), 0),
        (_connection_error(UnsupportedProtocol), 0),
        (OSError("raised outside the SDK's send"), 0),
        (_connection_error(ReadError), 1),
        (_connection_error(ReadTimeout), 1),
        (_StatusError(429), 1),
        (_StatusError(500), 1),
        (_StatusError(400), 1),
    ],
    ids=[
        "ConnectError", "ConnectTimeout", "ProxyError", "UnsupportedProtocol",
        "raw-OSError", "ReadError", "ReadTimeout", "429", "500", "400",
    ],
)
def test_fallback_counts_only_a_request_that_was_sent(
    exc: Exception, expected: int,
) -> None:
    result = _batch(_raising(exc))
    assert result.fallback_used
    assert result.telemetry.requests == expected


def test_mismatch_counts_no_request() -> None:
    assert _batch(_raising(TypeError("nope"))).telemetry.requests == 0


def _responding(resp: Any) -> _FakeSdk:
    def _create(**_kwargs: Any) -> Any:
        return resp

    return _client_with(_create)


@pytest.mark.parametrize(
    "resp",
    [
        SimpleNamespace(usage=None),
        SimpleNamespace(content=[], usage=None),
        SimpleNamespace(content=[SimpleNamespace(text="not json")], usage=None),
    ],
    ids=["no-content", "no-text-block", "malformed-json"],
)
def test_unusable_response_counts_the_request(resp: Any) -> None:
    result = _batch(_responding(resp))
    assert result.fallback_used
    assert result.telemetry.requests == 1


def test_transient_error_defaults_to_no_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The request-time import failure raises with no flag, before any send."""
    def _fail(**_kwargs: Any) -> Any:
        raise llm.LLMTransientError("SDK not importable at request time")

    monkeypatch.setattr(llm, "_call_anthropic", _fail)
    result = _batch(None)
    assert result.fallback_used
    assert result.telemetry.requests == 0


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


def _seed_orphan(tmp_path: Path) -> None:
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
    _seed_orphan(tmp_path)
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


def test_onboard_prints_requests_0_when_nothing_was_sent(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_sdk(monkeypatch, _raising(_connection_error(ConnectError)))

    out = io.StringIO()
    rc = cli_module.main(["onboard", str(project), "--llm-classify"], out=out)

    assert rc == 0
    assert "requests=0 " in out.getvalue()


def test_doctor_prints_requests_0_when_nothing_was_sent(
    project: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_orphan(tmp_path)
    _use_sdk(monkeypatch, _raising(_connection_error(ConnectError)))

    out = io.StringIO()
    rc = cli_module.main(["doctor", "--classify-orphans"], out=out)

    assert rc == 0
    assert "requests=0 " in out.getvalue()


# --- The installed SDK, against a server on 127.0.0.1 --------------------
#
# These prove the real SDK's exceptions land where `_falls_back` and
# `_request_was_sent` expect, by class name and by `__cause__`. Nothing leaves
# the machine: the base URL is a closed local port or the server below.


class _RateLimited(BaseHTTPRequestHandler):
    hits = 0

    def do_POST(self) -> None:
        type(self).hits += 1
        self.rfile.read(int(self.headers.get("content-length", "0")))
        body = b'{"type":"error","error":{"type":"rate_limit_error","message":"x"}}'
        self.send_response(429)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        # The SDK waits this long before its retries.
        self.send_header("retry-after-ms", "1")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        del format, args


@pytest.fixture
def installed_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    if not llm._anthropic_importable(None):  # pyright: ignore[reportPrivateUsage]
        pytest.skip("requires the onboard-llm extra")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)


def test_installed_sdk_refused_connection_falls_back_with_requests_0(
    installed_sdk: None,
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{port}")

    out = io.StringIO()
    rc = cli_module.main(["onboard", str(project), "--llm-classify"], out=out)

    assert rc == 0
    assert "requests=0 " in out.getvalue()


def test_installed_sdk_rate_limit_falls_back_with_requests_1(
    installed_sdk: None,
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _RateLimited.hits = 0
    server = HTTPServer(("127.0.0.1", 0), _RateLimited)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    try:
        out = io.StringIO()
        rc = cli_module.main(["onboard", str(project), "--llm-classify"], out=out)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert rc == 0
    assert _RateLimited.hits >= 1
    assert "requests=1 " in out.getvalue()


# --- A rejected request warns once and carries on ------------------------


_REJECTED = [400, 404, 413, 422]
_NOT_REJECTED = [401, 403, 408, 409, 429, 500, 503, 529]


@pytest.mark.parametrize("status", _REJECTED)
def test_rejected_status_names_itself_and_counts_the_request(status: int) -> None:
    result = _batch(_raising(_StatusError(status)))
    assert result.fallback_used
    assert result.rejection is not None
    assert result.rejection.startswith(f"status {status}: ")
    assert result.telemetry.requests == 1


@pytest.mark.parametrize("status", _NOT_REJECTED)
def test_other_status_is_not_a_rejection(status: int) -> None:
    assert _batch(_raising(_StatusError(status))).rejection is None


@pytest.mark.parametrize(
    "exc",
    [
        _connection_error(ConnectError),
        _connection_error(ReadTimeout),
        OSError("network unreachable"),
    ],
    ids=["refused", "read-timeout", "raw-OSError"],
)
def test_network_fault_is_not_a_rejection(exc: Exception) -> None:
    assert _batch(_raising(exc)).rejection is None


def test_rejection_text_is_one_bounded_line() -> None:
    exc = _StatusError(413)
    exc.args = ("<html>\n" + "<p>too large</p>\n" * 200,)
    rejection = _batch(_raising(exc)).rejection
    assert rejection is not None
    assert "\n" not in rejection
    detail = rejection.removeprefix("status 413: ")
    assert detail.startswith("<html> <p>too large</p>")
    assert len(detail) == 200


def test_router_keeps_the_first_rejection_across_batches() -> None:
    router = llm.ScannerRouter(api_key="k", sdk_module=_raising(_StatusError(400)))
    candidate = SimpleNamespace(text="a sentence", source="doc:a.md:p1")
    router.classify([candidate])
    first = router.rejection
    assert first is not None and first.startswith("status 400: ")
    router._sdk_module = _raising(_StatusError(404))  # pyright: ignore[reportPrivateUsage]
    router.classify([candidate])
    assert router.rejection == first


def _warnings(err: str) -> list[str]:
    return [ln for ln in err.splitlines() if "rejected the classifier request" in ln]


@pytest.mark.parametrize("status", _REJECTED)
@pytest.mark.parametrize(
    "argv",
    [["--llm-classify"], []],
    ids=["explicit-flag", "default-on"],
)
def test_onboard_warns_once_on_a_rejection_and_finishes_on_regex(
    project: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    status: int,
) -> None:
    _use_sdk(monkeypatch, _raising(_StatusError(status)))

    out = io.StringIO()
    rc = cli_module.main(["onboard", str(project), *argv], out=out)

    assert rc == 0
    warnings = _warnings(capsys.readouterr().err)
    assert len(warnings) == 1
    assert f"status {status}: " in warnings[0]
    assert "regex classifier" in warnings[0]
    assert "requests=1 " in out.getvalue()
    store = _open_db(tmp_path)
    try:
        assert store.count_beliefs() > 0
        assert any(
            e.source.startswith("onboard.llm.fallback")
            for e in store.list_feedback_events()
        )
    finally:
        store.close()


@pytest.mark.parametrize("status", _REJECTED)
def test_doctor_warns_once_on_a_rejection_and_skips_the_orphans(
    project: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: int,
) -> None:
    _seed_orphan(tmp_path)
    _use_sdk(monkeypatch, _raising(_StatusError(status)))

    out = io.StringIO()
    rc = cli_module.main(["doctor", "--classify-orphans"], out=out)

    assert rc == 0
    warnings = _warnings(capsys.readouterr().err)
    assert len(warnings) == 1
    assert f"status {status}: " in warnings[0]
    assert "No beliefs updated" in warnings[0]
    assert "requests=1 " in out.getvalue()
    store = _open_db(tmp_path)
    try:
        belief = store.get_belief("orphan1846")
        assert belief is not None and belief.type == "unknown"
    finally:
        store.close()


@pytest.mark.parametrize("status", [429, 500, 529])
@pytest.mark.parametrize(
    "argv",
    [["onboard", "@P", "--llm-classify"], ["doctor", "--classify-orphans"]],
    ids=["onboard", "doctor"],
)
def test_other_fallback_status_stays_silent(
    project: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    status: int,
) -> None:
    _seed_orphan(tmp_path)
    _use_sdk(monkeypatch, _raising(_StatusError(status)))

    args = [str(project) if a == "@P" else a for a in argv]
    rc = cli_module.main(args, out=io.StringIO())

    assert rc == 0
    assert capsys.readouterr().err == ""


def test_successful_run_prints_no_warning(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _create(**kwargs: Any) -> Any:
        count = kwargs["messages"][0]["content"].count('"index"')
        row = '{"belief_type":"factual","origin":"agent_inferred","persist":true}'
        text = "[" + ",".join([row] * count) + "]"
        return SimpleNamespace(content=[SimpleNamespace(text=text)], usage=None)

    _use_sdk(monkeypatch, _client_with(_create))

    rc = cli_module.main(["onboard", str(project), "--llm-classify"], out=io.StringIO())

    assert rc == 0
    assert capsys.readouterr().err == ""
