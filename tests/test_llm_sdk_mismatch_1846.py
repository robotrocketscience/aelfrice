"""Classifier failures: which stop the run, which warn, which stay silent (#1846).

A call that doesn't fit the installed SDK stops the run. Everything else falls
back to regex: a network fault or rate limit in silence, a rejected request, a
malformed response, or a config failure before a send with one stderr warning.

The fake-SDK tests run the real `_call_anthropic`. The CLI tests drive `main`, so
they cover what the user sees: exit code, stderr, the telemetry line, and the
store. The installed-SDK tests use a closed port or a server on 127.0.0.1, so
nothing leaves the machine.
"""
from __future__ import annotations

import importlib.util
import io
import json
import socket
import sys
import threading
from collections.abc import Iterator
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


def _ctor_raising(exc: BaseException) -> _FakeSdk:
    def _factory(**_kwargs: Any) -> Any:
        raise exc

    return _FakeSdk(_factory)


def _create_without_temperature(
    *, model: str, max_tokens: int, system: str, messages: list[dict[str, str]],
) -> Any:
    """The keyword-only shape of `create` on an SDK that dropped the keyword."""
    raise AssertionError("the bind check must refuse this call before the body")


# The SDK's own class names, which `_fault_from_call` reads off the MRO.
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


# The HTTP library's class names, which it reads off `__cause__`.
class ConnectError(Exception):
    pass


class ConnectTimeout(Exception):
    pass


class ProxyError(Exception):
    pass


class UnsupportedProtocol(Exception):
    pass


class LocalProtocolError(Exception):
    pass


class ReadError(Exception):
    pass


class ReadTimeout(Exception):
    pass


class DecodingError(Exception):
    pass


class InvalidURL(Exception):
    pass


def _connection_error(cause: type[Exception]) -> APIConnectionError:
    exc = APIConnectionError("Connection error.")
    exc.__cause__ = cause("from the HTTP library")
    return exc


_CANDIDATES = [llm.CandidateInput(index=0, text="a sentence", source="doc:a.md:p1")]
_SECRET = "PROXYSECRET"


def _batch(sdk: Any) -> llm.BatchResult:
    return llm.classify_batch(_CANDIDATES, api_key="test-key-1846", sdk_module=sdk)


# --- What stops the run ---------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [TypeError("unexpected keyword argument 'proxies'"), AttributeError("no Client")],
    ids=["TypeError", "AttributeError"],
)
def test_constructor_type_or_attribute_error_is_a_mismatch(exc: Exception) -> None:
    result = _batch(_ctor_raising(exc))
    assert result.sdk_mismatch_error == f"{type(exc).__name__}: {exc}"
    assert not result.fallback_used
    assert result.telemetry.requests == 0


def test_client_without_messages_is_a_mismatch() -> None:
    result = _batch(_FakeSdk(lambda **_kwargs: SimpleNamespace()))
    assert result.sdk_mismatch_error is not None
    assert result.sdk_mismatch_error.startswith("AttributeError")


def test_messages_without_create_is_a_mismatch() -> None:
    client = SimpleNamespace(messages=SimpleNamespace())
    result = _batch(_FakeSdk(lambda **_kwargs: client))
    assert result.sdk_mismatch_error is not None
    assert result.sdk_mismatch_error.startswith("AttributeError")


def test_keyword_create_does_not_take_is_a_mismatch() -> None:
    result = _batch(_client_with(_create_without_temperature))
    assert result.sdk_mismatch_error is not None
    assert "TypeError" in result.sdk_mismatch_error
    assert "temperature" in result.sdk_mismatch_error
    assert result.telemetry.requests == 0


def test_sdk_import_failure_at_request_time_is_not_a_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The SDK's module name, without spelling it.
    module = llm.ENV_API_KEY.partition("_")[0].lower()
    monkeypatch.setitem(sys.modules, module, None)
    result = llm.classify_batch(_CANDIDATES, api_key="k")
    assert result.sdk_mismatch_error is None
    assert result.fallback_used
    assert result.telemetry.requests == 0


# --- What falls back with a warning ---------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        ImportError(f"socksio is missing for socks5://u:{_SECRET}@h"),
        ValueError(f"Unknown scheme for proxy URL socks://u:{_SECRET}@h"),
        InvalidURL(f"Invalid port: '{_SECRET}'"),
        OSError(f"no such cert {_SECRET}"),
        KeyError(_SECRET),
    ],
    ids=lambda e: type(e).__name__,
)
def test_constructor_config_failure_warns_by_class_only(exc: Exception) -> None:
    result = _batch(_ctor_raising(exc))
    assert result.sdk_mismatch_error is None
    assert result.fallback_used
    assert result.telemetry.requests == 0
    assert result.warning == f"the classifier didn't run ({type(exc).__name__})"


@pytest.mark.parametrize(
    "exc",
    [
        TypeError(_SECRET),
        AttributeError(_SECRET),
        ImportError(_SECRET),
        ValueError(_SECRET),
        UnicodeEncodeError("ascii", _SECRET, 0, 1, "nope"),
        OSError(_SECRET),
        ConnectionRefusedError(_SECRET),
        SdkBaseError(_SECRET),
    ],
    ids=lambda e: type(e).__name__,
)
def test_raw_exception_inside_create_warns_by_class_only(exc: Exception) -> None:
    """Past the bind check, a `TypeError` doesn't prove the call is at fault."""
    result = _batch(_raising(exc))
    assert result.sdk_mismatch_error is None
    assert result.auth_error is None
    assert result.fallback_used
    assert result.telemetry.requests == 0
    assert result.warning == f"the classifier didn't run ({type(exc).__name__})"


def _raising_in_response_handling(exc: BaseException) -> _FakeSdk:
    def _process_response() -> Any:
        raise exc

    def _create(**_kwargs: Any) -> Any:
        return _process_response()

    return _client_with(_create)


@pytest.mark.parametrize(
    "exc",
    [
        json.JSONDecodeError("Expecting value", _SECRET, 0),
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
        TypeError(_SECRET),
        AttributeError(_SECRET),
    ],
    ids=lambda e: type(e).__name__,
)
def test_raw_exception_in_response_handling_is_unusable_and_counts(
    exc: Exception,
) -> None:
    result = _batch(_raising_in_response_handling(exc))
    assert result.sdk_mismatch_error is None
    assert result.fallback_used
    assert result.telemetry.requests == 1
    assert result.warning == (
        f"the classifier response was unusable ({type(exc).__name__})"
    )


def test_undecodable_body_cause_is_unusable_and_counts() -> None:
    result = _batch(_raising(_connection_error(DecodingError)))
    assert result.telemetry.requests == 1
    assert result.warning == "the classifier response was unusable (DecodingError)"


@pytest.mark.parametrize(
    "cause", [UnsupportedProtocol, LocalProtocolError], ids=lambda c: c.__name__,
)
def test_config_cause_warns_and_counts_nothing(cause: type[Exception]) -> None:
    result = _batch(_raising(_connection_error(cause)))
    assert result.telemetry.requests == 0
    assert result.warning == f"the classifier didn't run ({cause.__name__})"


def _responding(resp: Any) -> _FakeSdk:
    def _create(**_kwargs: Any) -> Any:
        return resp

    return _client_with(_create)


@pytest.mark.parametrize(
    "resp",
    [
        None,
        "text the SDK returned for a non-JSON content type",
        [],
        SimpleNamespace(usage=None),
        SimpleNamespace(content=None, usage=None),
        SimpleNamespace(content=5, usage=None),
        SimpleNamespace(content="abc", usage=None),
        SimpleNamespace(content=[], usage=None),
        SimpleNamespace(content=[5], usage=None),
        SimpleNamespace(content=[SimpleNamespace(text="not json")], usage=None),
        SimpleNamespace(content=[SimpleNamespace(text="[]")], usage=None),
    ],
    ids=[
        "none", "str", "list", "no-content", "content-none", "content-int",
        "content-str", "content-empty", "content-int-block", "malformed-json",
        "wrong-length",
    ],
)
def test_unusable_response_warns_and_counts(resp: Any) -> None:
    result = _batch(_responding(resp))
    assert result.fallback_used
    assert result.telemetry.requests == 1
    assert result.warning is not None
    assert result.warning.startswith("the classifier response was unusable (")


def test_response_with_no_text_block_says_so() -> None:
    warning = _batch(_responding(SimpleNamespace(content=[], usage=None))).warning
    assert warning == (
        "the classifier response was unusable (response contains no text block)"
    )


def _text_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(content=[SimpleNamespace(text=text)], usage=None)


@pytest.mark.parametrize(
    ("resp", "reason"),
    [
        (_text_response("[" * 200_000), "malformed JSON response: RecursionError"),
        (_text_response("1" * 5_001), "malformed JSON response: ValueError"),
        (
            SimpleNamespace(
                content=[SimpleNamespace(text="[]")],
                usage=SimpleNamespace(input_tokens=float("inf"), output_tokens=1),
            ),
            "OverflowError",
        ),
    ],
    ids=["deep-brackets", "huge-int", "usage-inf"],
)
def test_response_that_crashes_a_parser_warns_and_counts(
    resp: SimpleNamespace, reason: str,
) -> None:
    result = _batch(_responding(resp))
    assert result.fallback_used
    assert result.telemetry.requests == 1
    assert result.warning == f"the classifier response was unusable ({reason})"


@pytest.mark.parametrize(
    "count", [float("nan"), "many", ["x"]], ids=["nan", "str", "list"],
)
def test_usage_that_isn_t_a_number_still_counts_as_zero(count: object) -> None:
    usage = SimpleNamespace(input_tokens=count, output_tokens=1)
    resp = SimpleNamespace(content=[SimpleNamespace(text="[]")], usage=usage)
    result = _batch(_responding(resp))
    assert result.telemetry.input_tokens == 0
    assert result.telemetry.output_tokens == 0
    assert result.warning is not None and "response length" in result.warning


@pytest.mark.parametrize(
    "status", [[400], {"code": 400}, "400", 400.0, None],
    ids=["list", "dict", "str", "float", "none"],
)
def test_non_integer_status_falls_back_without_crashing(status: object) -> None:
    class _OddStatus(Exception):
        status_code = status

    result = _batch(_raising(_OddStatus("odd")))
    assert result.fallback_used
    assert result.telemetry.requests == 0
    assert result.warning == "the classifier didn't run (_OddStatus)"


def test_parse_failure_prints_its_whole_reason() -> None:
    resp = SimpleNamespace(content=[SimpleNamespace(text="not json")], usage=None)
    warning = _batch(_responding(resp)).warning
    assert warning == (
        "the classifier response was unusable (malformed JSON response: "
        "Expecting value: line 1 column 1 (char 0))"
    )


class _BodyError(_StatusError):
    def __init__(self, status_code: int, body: object) -> None:
        super().__init__(status_code)
        self.body = body
        self.args = (f"Error code: {status_code} - {body} {_SECRET}",)


_CHECK = "Check the model name, billing, and the base URL"


@pytest.mark.parametrize("status", [400, 404, 413, 422])
def test_rejected_status_names_itself_and_counts_the_request(status: int) -> None:
    result = _batch(_raising(_StatusError(status)))
    assert result.fallback_used
    assert result.warning == (
        f"the API rejected the classifier request (status {status}). {_CHECK}"
    )
    assert result.telemetry.requests == 1


@pytest.mark.parametrize(
    "kind",
    ["not_found_error", "invalid_request_error", "billing_error", "request_too_large", "api_error"],
)
def test_rejection_adds_a_short_error_type(kind: str) -> None:
    body = {"type": "error", "error": {"type": kind, "message": _SECRET}}
    warning = _batch(_raising(_BodyError(404, body))).warning
    assert warning == (
        f"the API rejected the classifier request (status 404, {kind}). {_CHECK}"
    )


@pytest.mark.parametrize(
    "body",
    [
        None,
        f"Cannot POST /t/{_SECRET}/v1/messages",
        ["not_found_error"],
        {},
        {"error": "not_found_error"},
        {"error": {}},
        {"error": {"message": _SECRET}},
        {"error": {"type": _SECRET}},
        {"error": {"type": "gwtoken9f3a7c"}},
        {"error": {"type": "tok_9f3a7c"}},
        {"error": {"type": "hunter"}},
        {"error": {"type": "sk-ant-dummy"}},
        {"error": {"type": "Not_Found_Error"}},
        {"error": {"type": "not_found_error\n"}},
        {"error": {"type": "not found_error"}},
        {"error": {"type": "\x1b[31m_error"}},
        {"error": {"type": "_error"}},
        {"error": {"type": "error_"}},
        {"error": {"type": "not_found_err\u00f6r"}},
        {"error": {"type": "a_" + "b" * 39}},
        {"error": {"type": 5}},
        {"error": {"type": ["not_found_error"]}},
    ],
    ids=lambda b: repr(b)[:32],
)
def test_rejection_prints_nothing_else_from_the_body(body: object) -> None:
    warning = _batch(_raising(_BodyError(400, body))).warning
    assert warning == f"the API rejected the classifier request (status 400). {_CHECK}"


def test_error_type_at_the_length_limit_prints() -> None:
    kind = "a_" + "b" * 38
    assert len(kind) == 40
    warning = _batch(_raising(_BodyError(400, {"error": {"type": kind}}))).warning
    assert warning is not None and f"(status 400, {kind})" in warning


def _mismatch(text: str, key: str = "test-key-1846") -> str | None:
    sdk = _ctor_raising(TypeError(text))
    return llm.classify_batch(_CANDIDATES, api_key=key, sdk_module=sdk).sdk_mismatch_error


@pytest.mark.parametrize(
    "key", ["sk-FAKE1234", " sk-FAKE1234\r"], ids=["plain", "with-whitespace"],
)
def test_mismatch_text_replaces_the_key(key: str) -> None:
    assert _mismatch("bad sk-FAKE1234 here", key) == "TypeError: bad [key] here"


def test_mismatch_text_replaces_a_key_across_the_cap() -> None:
    text = "y" * 195 + "sk-FAKE1234"
    assert _mismatch(text, "sk-FAKE1234") == "TypeError: " + "y" * 195 + "[key]"


def test_empty_key_leaves_the_mismatch_text_alone() -> None:
    assert _mismatch("no key here", "") == "TypeError: no key here"


def test_mismatch_text_collapses_whitespace_runs() -> None:
    assert _mismatch("first\n\n  \t second") == "TypeError: first second"


def test_mismatch_text_is_bounded() -> None:
    assert _mismatch("x" * 500) == "TypeError: " + "x" * 200


@pytest.mark.parametrize(
    "char", ["\x1b", "\x00", "\x07", "\x7f", "\x9b", "\r", "\n", "\t"],
    ids=["esc", "nul", "bel", "del", "csi", "cr", "lf", "tab"],
)
def test_mismatch_text_strips_control_characters(char: str) -> None:
    assert _mismatch(f"{char}[31mred{char}text") == "TypeError: [31mred text"


# --- What falls back in silence -------------------------------------------


@pytest.mark.parametrize(
    ("exc", "requests"),
    [
        (_connection_error(ConnectError), 0),
        (_connection_error(ConnectTimeout), 0),
        (_connection_error(ProxyError), 0),
        (_connection_error(ReadError), 1),
        (_connection_error(ReadTimeout), 1),
        (APIError("bare"), 1),
        (_StatusError(408), 1),
        (_StatusError(409), 1),
        (_StatusError(429), 1),
        (_StatusError(500), 1),
        (_StatusError(503), 1),
        (_StatusError(529), 1),
    ],
    ids=[
        "ConnectError", "ConnectTimeout", "ProxyError", "ReadError", "ReadTimeout",
        "bare-APIError", "408", "409", "429", "500", "503", "529",
    ],
)
def test_network_fault_or_passing_status_stays_silent(
    exc: Exception, requests: int,
) -> None:
    result = _batch(_raising(exc))
    assert result.sdk_mismatch_error is None
    assert result.auth_error is None
    assert result.fallback_used
    assert result.telemetry.fallbacks == len(_CANDIDATES)
    assert result.telemetry.requests == requests
    assert result.warning is None


def test_status_code_outside_the_family_counts_as_a_response() -> None:
    exc = RuntimeError("status 503")
    exc.status_code = 503  # type: ignore[attr-defined]
    result = _batch(_raising(exc))
    assert result.fallback_used
    assert result.warning is None
    assert result.telemetry.requests == 1


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failure_stays_an_auth_failure(status: int) -> None:
    result = _batch(_raising(_StatusError(status)))
    assert result.auth_error is not None
    assert result.sdk_mismatch_error is None
    assert result.warning is None


def test_transient_error_defaults_to_no_request_and_no_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail(**_kwargs: Any) -> Any:
        raise llm.LLMTransientError("a fault with no details")

    monkeypatch.setattr(llm, "_call_anthropic", _fail)
    result = _batch(None)
    assert result.fallback_used
    assert result.telemetry.requests == 0
    assert result.warning is None


def test_router_keeps_the_first_warning_across_batches() -> None:
    router = llm.ScannerRouter(api_key="k", sdk_module=_raising(_StatusError(400)))
    candidate = SimpleNamespace(text="a sentence", source="doc:a.md:p1")
    router.classify([candidate])
    first = router.warning
    assert first is not None and "(status 400)" in first
    router._sdk_module = _raising(_StatusError(404))  # pyright: ignore[reportPrivateUsage]
    router.classify([candidate])
    assert router.warning == first


# --- Through `main`, with a fake SDK --------------------------------------


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


def _aelf_lines(err: str) -> list[str]:
    """The lines aelf wrote. An SDK's own `__del__` noise can sit beside them."""
    return [ln for ln in err.splitlines() if ln.startswith("aelf: ")]


def _assert_onboard_fell_back(tmp_path: Path) -> None:
    store = _open_db(tmp_path)
    try:
        assert store.count_beliefs() > 0
        assert any(
            e.source.startswith("onboard.llm.fallback")
            for e in store.list_feedback_events()
        )
    finally:
        store.close()


def _assert_orphan_untouched(tmp_path: Path) -> None:
    store = _open_db(tmp_path)
    try:
        belief = store.get_belief("orphan1846")
        assert belief is not None and belief.type == "unknown"
    finally:
        store.close()


_FLAG_AND_DEFAULT = pytest.mark.parametrize(
    "argv", [["--llm-classify"], []], ids=["explicit-flag", "default-on"],
)


@_FLAG_AND_DEFAULT
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


def test_doctor_stops_and_names_the_exception(
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
    _assert_orphan_untouched(tmp_path)


@pytest.mark.parametrize(
    ("sdk", "expected", "requests"),
    [
        (_raising(_StatusError(400)), "(status 400)", 1),
        (_raising(_StatusError(404)), "(status 404)", 1),
        (_raising(_StatusError(413)), "(status 413)", 1),
        (_raising(_StatusError(422)), "(status 422)", 1),
        (_responding(SimpleNamespace(content=5, usage=None)), "unusable", 1),
        (_ctor_raising(InvalidURL(_SECRET)), "didn't run (InvalidURL)", 0),
    ],
    ids=["400", "404", "413", "422", "unusable", "config"],
)
@_FLAG_AND_DEFAULT
def test_onboard_warns_once_and_finishes_on_regex(
    project: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    sdk: Any,
    expected: str,
    requests: int,
) -> None:
    _use_sdk(monkeypatch, sdk)

    out = io.StringIO()
    rc = cli_module.main(["onboard", str(project), *argv], out=out)

    assert rc == 0
    err = capsys.readouterr().err
    assert _SECRET not in err
    lines = err.splitlines()
    assert len(lines) == 1
    assert expected in lines[0]
    assert "regex classifier" in lines[0]
    assert f"requests={requests} " in out.getvalue()
    _assert_onboard_fell_back(tmp_path)


@pytest.mark.parametrize(
    ("sdk", "expected", "requests"),
    [
        (_raising(_StatusError(400)), "(status 400)", 1),
        (_raising(_StatusError(404)), "(status 404)", 1),
        (_raising(_StatusError(413)), "(status 413)", 1),
        (_raising(_StatusError(422)), "(status 422)", 1),
        (_responding(SimpleNamespace(content=5, usage=None)), "unusable", 1),
        (_ctor_raising(InvalidURL(_SECRET)), "didn't run (InvalidURL)", 0),
    ],
    ids=["400", "404", "413", "422", "unusable", "config"],
)
def test_doctor_warns_once_and_skips_the_orphans(
    project: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sdk: Any,
    expected: str,
    requests: int,
) -> None:
    _seed_orphan(tmp_path)
    _use_sdk(monkeypatch, sdk)

    out = io.StringIO()
    rc = cli_module.main(["doctor", "--classify-orphans"], out=out)

    assert rc == 0
    err = capsys.readouterr().err
    assert _SECRET not in err
    lines = err.splitlines()
    assert len(lines) == 1
    assert expected in lines[0]
    assert "No beliefs updated" in lines[0]
    assert f"requests={requests} " in out.getvalue()
    _assert_orphan_untouched(tmp_path)


@pytest.mark.parametrize(
    "exc",
    [
        _StatusError(429),
        _StatusError(500),
        _StatusError(529),
        _connection_error(ConnectError),
        _connection_error(ReadTimeout),
    ],
    ids=["429", "500", "529", "ConnectError", "ReadTimeout"],
)
@pytest.mark.parametrize(
    "argv",
    [["onboard", "@P", "--llm-classify"], ["onboard", "@P"], ["doctor", "--classify-orphans"]],
    ids=["onboard-flag", "onboard-default", "doctor"],
)
def test_network_fault_and_passing_status_stay_silent_through_main(
    project: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    exc: Exception,
) -> None:
    _seed_orphan(tmp_path)
    _use_sdk(monkeypatch, _raising(exc))

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


# --- Through `main`, with the installed SDK ------------------------------
#
# These prove the real SDK's exceptions land where `_call_anthropic` expects, by
# class name, by `__cause__`, and by frame name, on whichever SDK is installed.
#
# A failed client constructor leaves the SDK's HTTP wrapper half built, and its
# `__del__` raises when collected, so those tests ignore the unraisable warning.


class _Api(BaseHTTPRequestHandler):
    reply: tuple[int, dict[str, str], bytes] = (200, {}, b"")
    hits = 0

    def do_POST(self) -> None:
        cls = type(self)
        cls.hits += 1
        self.rfile.read(int(self.headers.get("content-length", "0")))
        status, headers, body = cls.reply
        self.send_response(status)
        self.send_header("content-length", str(len(body)))
        # The SDK waits this long before its retries.
        self.send_header("retry-after-ms", "1")
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        del format, args


@pytest.fixture
def installed_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    if not llm._anthropic_importable(None):  # pyright: ignore[reportPrivateUsage]
        pytest.skip("requires the onboard-llm extra")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    for name in ("ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)


@pytest.fixture
def local_api(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    _Api.hits = 0
    server = HTTPServer(("127.0.0.1", 0), _Api)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    try:
        yield
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _closed_port_url() -> str:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{probe.getsockname()[1]}"


_JSON = {"content-type": "application/json"}


def _message_body(text: str, input_tokens: str = "1") -> bytes:
    usage = f'{{"input_tokens":{input_tokens},"output_tokens":1}}'
    content = json.dumps([{"type": "text", "text": text}])
    return (
        '{"id":"m","type":"message","role":"assistant","model":"m",'
        f'"stop_reason":"end_turn","content":{content},"usage":{usage}}}'
    ).encode()


_MALFORMED = pytest.mark.parametrize(
    ("reply", "name"),
    [
        ((200, _JSON, b"not json"), "JSONDecodeError"),
        ((200, _JSON, b'{"content": [{"type":"text","text":"['), "JSONDecodeError"),
        ((200, _JSON, bytes([0xff, 0xfe, 0x00])), "UnicodeDecodeError"),
        ((204, _JSON, b""), "JSONDecodeError"),
        ((200, _JSON, b'{"content": 5}'), "no content list"),
        ((200, _JSON, b"null"), "no content list"),
        ((200, {"content-type": "text/html"}, b"<html>login</html>"), "no content list"),
        ((200, {**_JSON, "content-encoding": "gzip"}, b"not gzip"), "DecodingError"),
        ((200, _JSON, _message_body("[]", "1e999")), "OverflowError"),
        ((200, _JSON, _message_body("[" * 200_000)), "RecursionError"),
        ((200, _JSON, _message_body("1" * 5_001)), "ValueError"),
    ],
    ids=[
        "not-json", "truncated", "non-utf8", "204-empty", "content-int", "null",
        "html", "bad-gzip", "usage-inf", "deep-brackets", "huge-int",
    ],
)


@_MALFORMED
@_FLAG_AND_DEFAULT
def test_installed_sdk_malformed_response_warns_and_counts(
    installed_sdk: None,
    local_api: None,
    project: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    reply: tuple[int, dict[str, str], bytes],
    name: str,
) -> None:
    _Api.reply = reply

    out = io.StringIO()
    rc = cli_module.main(["onboard", str(project), *argv], out=out)

    assert rc == 0
    lines = _aelf_lines(capsys.readouterr().err)
    assert len(lines) == 1
    assert "response was unusable" in lines[0] and name in lines[0]
    assert _Api.hits >= 1
    assert "requests=1 " in out.getvalue()
    _assert_onboard_fell_back(tmp_path)


@_MALFORMED
def test_installed_sdk_malformed_response_warns_on_doctor(
    installed_sdk: None,
    local_api: None,
    project: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    reply: tuple[int, dict[str, str], bytes],
    name: str,
) -> None:
    _seed_orphan(tmp_path)
    _Api.reply = reply

    out = io.StringIO()
    rc = cli_module.main(["doctor", "--classify-orphans"], out=out)

    assert rc == 0
    lines = _aelf_lines(capsys.readouterr().err)
    assert len(lines) == 1
    assert "response was unusable" in lines[0] and name in lines[0]
    assert "requests=1 " in out.getvalue()
    _assert_orphan_untouched(tmp_path)


@pytest.mark.parametrize(
    ("env", "name"),
    [
        ({"ALL_PROXY": f"socks://u:{_SECRET}@127.0.0.1:1"}, "ValueError"),
        ({"ALL_PROXY": f"http://u:{_SECRET}/x@127.0.0.1:80"}, "InvalidURL"),
        ({"ANTHROPIC_BASE_URL": f"http://u:{_SECRET}/x@127.0.0.1:80"}, "InvalidURL"),
        ({"ANTHROPIC_BASE_URL": "http://127.0.0.1:notaport"}, "InvalidURL"),
        ({"ANTHROPIC_BASE_URL": "ftp://127.0.0.1"}, "UnsupportedProtocol"),
        ({"SSL_CERT_FILE": "/nonexistent/ca.pem"}, "FileNotFoundError"),
    ],
    ids=["socks-scheme", "proxy-bad-port", "url-bad-port", "url-not-a-port", "ftp", "cert"],
)
@pytest.mark.parametrize(
    "argv",
    [["onboard", "@P", "--llm-classify"], ["onboard", "@P"], ["doctor", "--classify-orphans"]],
    ids=["onboard-flag", "onboard-default", "doctor"],
)
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_installed_sdk_config_failure_warns_by_class_only(
    installed_sdk: None,
    project: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    env: dict[str, str],
    name: str,
) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", _closed_port_url())
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    if argv[0] == "doctor":
        _seed_orphan(tmp_path)

    out = io.StringIO()
    args = [str(project) if a == "@P" else a for a in argv]
    rc = cli_module.main(args, out=out)

    assert rc == 0
    err = capsys.readouterr().err
    assert _SECRET not in err
    assert "notaport" not in err
    lines = _aelf_lines(err)
    assert len(lines) == 1
    assert f"the classifier didn't run ({name})" in lines[0]
    assert "requests=0 " in out.getvalue()
    if argv[0] == "doctor":
        _assert_orphan_untouched(tmp_path)
    else:
        _assert_onboard_fell_back(tmp_path)


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_installed_sdk_socks_proxy_without_socksio_warns(
    installed_sdk: None,
    project: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if importlib.util.find_spec("socksio") is not None:
        pytest.skip("needs an environment without socksio")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", _closed_port_url())
    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:1")

    out = io.StringIO()
    rc = cli_module.main(["onboard", str(project)], out=out)

    assert rc == 0
    lines = _aelf_lines(capsys.readouterr().err)
    assert len(lines) == 1
    assert "the classifier didn't run (ImportError)" in lines[0]
    assert "requests=0 " in out.getvalue()
    _assert_onboard_fell_back(tmp_path)


def test_installed_sdk_non_ascii_key_falls_back(
    installed_sdk: None,
    project: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """0.97.0 raises `UnicodeEncodeError`; 2.3.0 wraps it, so it reads as a network fault."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", _closed_port_url())
    monkeypatch.setenv(llm.ENV_API_KEY, f"{_SECRET}’x")

    out = io.StringIO()
    rc = cli_module.main(["onboard", str(project)], out=out)

    assert rc == 0
    assert _SECRET not in capsys.readouterr().err
    assert "requests=0 " in out.getvalue()
    _assert_onboard_fell_back(tmp_path)


@pytest.mark.parametrize(
    "argv",
    [["onboard", "@P", "--llm-classify"], ["onboard", "@P"], ["doctor", "--classify-orphans"]],
    ids=["onboard-flag", "onboard-default", "doctor"],
)
def test_installed_sdk_key_ending_in_cr_counts_nothing(
    installed_sdk: None,
    local_api: None,
    project: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
) -> None:
    """The connection opens, and the header is refused before a byte is written."""
    monkeypatch.setenv(llm.ENV_API_KEY, f"{_SECRET}\r")
    if argv[0] == "doctor":
        _seed_orphan(tmp_path)

    out = io.StringIO()
    args = [str(project) if a == "@P" else a for a in argv]
    rc = cli_module.main(args, out=out)

    assert rc == 0
    err = capsys.readouterr().err
    assert _SECRET not in err
    lines = _aelf_lines(err)
    assert len(lines) == 1
    assert "the classifier didn't run (LocalProtocolError)" in lines[0]
    assert _Api.hits == 0
    assert "requests=0 " in out.getvalue()


def _error_reply(kind: str, message: str) -> bytes:
    return json.dumps({"type": "error", "error": {"type": kind, "message": message}}).encode()


@pytest.mark.parametrize(
    ("reply", "label"),
    [
        ((404, _JSON, _error_reply("not_found_error", f"bad {_SECRET}")), "(status 404, not_found_error)"),
        ((400, _JSON, _error_reply(_SECRET, "bad")), "(status 400)"),
        ((400, _JSON, _error_reply("\x1b[31m_error", "bad")), "(status 400)"),
        ((404, {"content-type": "text/plain"}, f"Cannot POST /t/{_SECRET}/".encode()), "(status 404)"),
        ((413, {"content-type": "text/html"}, b"\x1b[31m<html>" + _SECRET.encode()), "(status 413)"),
        ((422, _JSON, b"\x1b[31m not json " + _SECRET.encode()), "(status 422)"),
    ],
    ids=["type-and-echo", "type-is-secret", "type-has-esc", "text-404", "html-413", "broken-json-422"],
)
@pytest.mark.parametrize(
    "argv",
    [["onboard", "@P", "--llm-classify"], ["onboard", "@P"], ["doctor", "--classify-orphans"]],
    ids=["onboard-flag", "onboard-default", "doctor"],
)
def test_installed_sdk_warning_prints_no_server_text(
    installed_sdk: None,
    local_api: None,
    project: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    reply: tuple[int, dict[str, str], bytes],
    label: str,
) -> None:
    monkeypatch.setenv(llm.ENV_API_KEY, _SECRET)
    _Api.reply = reply
    if argv[0] == "doctor":
        _seed_orphan(tmp_path)
    args = [str(project) if a == "@P" else a for a in argv]

    rc = cli_module.main(args, out=io.StringIO())

    assert rc == 0
    err = capsys.readouterr().err
    assert _SECRET not in err
    assert "\x1b" not in err
    assert "Error code" not in err
    assert label in err
    assert len(err.splitlines()) == 1


def test_installed_sdk_refused_connection_stays_silent_with_requests_0(
    installed_sdk: None,
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", _closed_port_url())

    out = io.StringIO()
    rc = cli_module.main(["onboard", str(project), "--llm-classify"], out=out)

    assert rc == 0
    assert capsys.readouterr().err == ""
    assert "requests=0 " in out.getvalue()


def test_installed_sdk_rate_limit_stays_silent_with_requests_1(
    installed_sdk: None,
    local_api: None,
    project: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _Api.reply = (429, _JSON, b'{"type":"error","error":{"type":"x","message":"x"}}')

    out = io.StringIO()
    rc = cli_module.main(["onboard", str(project), "--llm-classify"], out=out)

    assert rc == 0
    assert capsys.readouterr().err == ""
    assert _Api.hits == 3
    assert "requests=1 " in out.getvalue()


@pytest.mark.parametrize("status", [400, 404, 413, 422])
def test_installed_sdk_rejection_warns_with_requests_1(
    installed_sdk: None,
    local_api: None,
    project: Path,
    capsys: pytest.CaptureFixture[str],
    status: int,
) -> None:
    body = {"type": "error", "error": {"type": "x", "message": "no"}}
    _Api.reply = (status, _JSON, json.dumps(body).encode())

    out = io.StringIO()
    rc = cli_module.main(["onboard", str(project)], out=out)

    assert rc == 0
    lines = _aelf_lines(capsys.readouterr().err)
    assert len(lines) == 1
    assert f"(status {status})" in lines[0]
    assert _Api.hits == 1
    assert "requests=1 " in out.getvalue()
