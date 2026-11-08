"""The classifier on the installed SDK 2.x, against a server on 127.0.0.1 (#1845).

What a gateway set through the base URL sees: the path, the credential header,
the version header, and the body. What comes back through `main`: usage counts,
parsed labels, retries, a timeout, a call that can't match the SDK, and a server
or debug log that repeats the key. No test here reaches a real API.
"""
from __future__ import annotations

import email.utils
import importlib
import io
import json
import logging
import os
import re
import subprocess
import sys
import threading
import tomllib
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import aelfrice.cli as cli_module
import aelfrice.llm_classifier as llm
from aelfrice.models import ORIGIN_UNKNOWN, Belief
from aelfrice.store import MemoryStore

REPO = Path(__file__).resolve().parents[1]
_KEY = "gateway-key-not-real-1845"
# The SDK's module name, which also heads its version header, without spelling it.
_SDK = llm.ENV_API_KEY.partition("_")[0].lower()
_ROW = {"belief_type": "preference", "origin": "document_recent", "persist": True}
_JSON = {"content-type": "application/json"}


@pytest.fixture
def waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Every wait the classifier asks for, with none of them slept."""
    asked: list[float] = []
    monkeypatch.setattr(llm, "_sleep", asked.append)
    return asked


class _FakeSdk:
    """Answers any attribute with the client factory, as `sdk_module`."""

    def __init__(self, factory: Any) -> None:
        self._factory = factory

    def __getattr__(self, _name: str) -> Any:
        return self._factory


def _message(rows: int, row: dict[str, object] = _ROW) -> bytes:
    """A reply whose JSON array spans two text blocks, with a block that has no text."""
    text = json.dumps([row] * rows)
    return json.dumps(
        {
            "id": "m",
            "type": "message",
            "role": "assistant",
            "model": "m",
            "stop_reason": "end_turn",
            "content": [
                {"type": "thinking", "thinking": "t", "signature": "s"},
                {"type": "text", "text": text[:1]},
                {"type": "text", "text": text[1:]},
            ],
            "usage": {"input_tokens": 7, "output_tokens": 3},
        }
    ).encode()


class _Gateway(BaseHTTPRequestHandler):
    """Records each request. It answers from `queue` first, then `reply`, then
    with one row per candidate."""

    server: ThreadingHTTPServer | None = None
    # Logger names whose level the handler records while a request is open.
    watched: tuple[str, ...] = ()
    levels_seen: list[list[int]] = []
    requests: list[tuple[str, dict[str, str], dict[str, Any]]] = []
    queue: list[tuple[int, dict[str, str], bytes]] = []
    reply: tuple[int, dict[str, str], bytes] | None = None
    hang = False
    release = threading.Event()

    def do_POST(self) -> None:
        cls = type(self)
        raw = self.rfile.read(int(self.headers.get("content-length", "0")))
        body = json.loads(raw)
        headers = {name.lower(): value for name, value in self.headers.items()}
        cls.requests.append((self.path, headers, body))
        cls.levels_seen.append([logging.getLogger(name).level for name in cls.watched])
        if cls.hang:
            cls.release.wait(timeout=300)
            return
        rows = body["messages"][0]["content"].count('"index"')
        queued = cls.queue.pop(0) if cls.queue else None
        status, extra, reply = queued or cls.reply or (200, _JSON, _message(rows))
        self.send_response(status)
        self.send_header("content-length", str(len(reply)))
        # The classifier waits this long before its retries, unless the reply
        # names its own wait or asks for none.
        if not any(name.startswith(("retry-after", "x-no-wait")) for name in extra):
            self.send_header("retry-after-ms", "1")
        for name, value in extra.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, format: str, *args: Any) -> None:
        """Keep request lines out of the test output."""


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tiny project with a throwaway HOME, store, key, and full consent."""
    if not llm._anthropic_importable(None):  # pyright: ignore[reportPrivateUsage]
        pytest.skip("requires the onboard-llm extra")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_CUSTOM_HEADERS", raising=False)
    for name in ("ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "memory.db"))
    monkeypatch.setenv("AELFRICE_USER_DB", str(tmp_path / "user.db"))
    monkeypatch.setenv(llm.ENV_API_KEY, _KEY)
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


@pytest.fixture
def gateway(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    _Gateway.requests = []
    _Gateway.queue = []
    _Gateway.reply = None
    _Gateway.watched = ()
    _Gateway.levels_seen = []
    _Gateway.hang = False
    _Gateway.release = threading.Event()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Gateway)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv(llm.ENV_BASE_URL, url)
    _Gateway.server = server
    try:
        yield url
    finally:
        _Gateway.release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _seed_orphan(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "memory.db"))
    store.insert_belief(
        Belief(
            id="orphan1845",
            content="an orphan belief with no type",
            content_hash="orphan1845-hash",
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


def _belief_count(tmp_path: Path) -> int:
    store = MemoryStore(str(tmp_path / "memory.db"))
    try:
        return store.count_beliefs()
    finally:
        store.close()


def _orphan_type(tmp_path: Path) -> str:
    store = MemoryStore(str(tmp_path / "memory.db"))
    try:
        belief = store.get_belief("orphan1845")
        assert belief is not None
        return belief.type
    finally:
        store.close()


def _main(argv: list[str], project: Path, tmp_path: Path) -> tuple[int, str]:
    if argv[0] == "doctor":
        _seed_orphan(tmp_path)
    out = io.StringIO()
    args = [str(project) if a == "@P" else a for a in argv]
    return cli_module.main(args, out=out), out.getvalue()


_EVERY_COMMAND = pytest.mark.parametrize(
    "argv",
    [["onboard", "@P", "--llm-classify"], ["onboard", "@P"], ["doctor", "--classify-orphans"]],
    ids=["onboard-flag", "onboard-default", "doctor"],
)


# --- What a gateway sees --------------------------------------------------


@_EVERY_COMMAND
def test_the_key_goes_out_as_a_bearer_header_and_nothing_else(
    project: Path, gateway: str, tmp_path: Path, argv: list[str],
) -> None:
    rc, _ = _main(argv, project, tmp_path)

    assert rc == 0
    assert len(_Gateway.requests) == 1
    path, headers, _ = _Gateway.requests[0]
    assert path == "/v1/messages"
    assert headers["authorization"] == f"Bearer {_KEY}"
    assert "x-api-key" not in headers


def test_the_version_header_is_the_sdk_s_own_and_not_the_old_one(
    project: Path, gateway: str, tmp_path: Path,
) -> None:
    constants = importlib.import_module(f"{_SDK}._constants")

    _main(["onboard", "@P"], project, tmp_path)

    sent = _Gateway.requests[0][1][f"{_SDK}-version"]
    assert sent == constants.ANTHROPIC_API_VERSION
    assert sent != "2023-06-01"


@_EVERY_COMMAND
def test_the_body_holds_only_what_the_classifier_sends(
    project: Path, gateway: str, tmp_path: Path, argv: list[str],
) -> None:
    _main(argv, project, tmp_path)

    body = _Gateway.requests[0][2]
    assert sorted(body) == ["max_tokens", "messages", "model", "system"]


def test_a_base_url_path_prefix_is_kept(
    project: Path, gateway: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(llm.ENV_BASE_URL, f"{gateway}/gw")

    rc, _ = _main(["onboard", "@P"], project, tmp_path)

    assert rc == 0
    assert _Gateway.requests[0][0] == "/gw/v1/messages"


def test_the_key_wins_over_an_auth_token_in_the_environment(
    project: Path, gateway: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "token-not-real-1845")

    _main(["onboard", "@P"], project, tmp_path)

    headers = _Gateway.requests[0][1]
    assert headers["authorization"] == f"Bearer {_KEY}"
    assert "token-not-real-1845" not in json.dumps(headers)


def test_a_custom_header_can_keep_a_gateway_that_reads_x_api_key_working(
    project: Path, gateway: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", f"X-Api-Key: {_KEY}")

    rc, _ = _main(["onboard", "@P"], project, tmp_path)

    assert rc == 0
    headers = _Gateway.requests[0][1]
    assert headers["x-api-key"] == _KEY
    assert headers["authorization"] == f"Bearer {_KEY}"


# --- What the classifier reads back ---------------------------------------


def test_usage_counts_reach_the_telemetry_line(
    project: Path, gateway: str, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    rc, out = _main(["onboard", "@P"], project, tmp_path)

    assert rc == 0
    assert capsys.readouterr().err == ""
    assert "input_tokens=7 output_tokens=3 " in out
    assert "requests=1 " in out
    assert "fallbacks=0" in out


def test_labels_come_from_text_blocks_joined_past_a_block_with_no_text(
    project: Path, gateway: str, tmp_path: Path,
) -> None:
    _main(["onboard", "@P"], project, tmp_path)

    store = MemoryStore(str(tmp_path / "memory.db"))
    try:
        ids = [row[0] for row in store.list_beliefs_for_indexing()]
        assert ids
        beliefs = [store.get_belief(belief_id) for belief_id in ids]
        assert {b.type for b in beliefs if b} == {"preference"}
        assert {b.origin for b in beliefs if b} == {"document_recent"}
        assert store.list_feedback_events() == []
    finally:
        store.close()


def test_onboarding_again_keeps_the_label_a_belief_already_has(
    project: Path, gateway: str, tmp_path: Path,
) -> None:
    """Two runs can disagree, and the store keeps the first run's label."""
    _main(["onboard", "@P"], project, tmp_path)
    rows = _Gateway.requests[0][2]["messages"][0]["content"].count('"index"')
    relabeled = {**_ROW, "belief_type": "factual", "origin": "agent_inferred"}
    _Gateway.reply = (200, _JSON, _message(rows, relabeled))

    rc, _ = _main(["onboard", "@P"], project, tmp_path)

    assert rc == 0
    store = MemoryStore(str(tmp_path / "memory.db"))
    try:
        ids = [row[0] for row in store.list_beliefs_for_indexing()]
        beliefs = [store.get_belief(belief_id) for belief_id in ids]
        assert len(beliefs) == rows
        assert {b.type for b in beliefs if b} == {"preference"}
    finally:
        store.close()


def test_doctor_writes_the_label_it_read(
    project: Path, gateway: str, tmp_path: Path,
) -> None:
    rc, _ = _main(["doctor", "--classify-orphans"], project, tmp_path)

    assert rc == 0
    assert _orphan_type(tmp_path) == "preference"


@pytest.mark.parametrize("status", [408, 409, 429, 500, 529])
def test_a_retried_status_reaches_the_server_3_times_and_counts_3(
    project: Path,
    gateway: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    waits: list[float],
    status: int,
) -> None:
    _Gateway.reply = (status, _JSON, b'{"type":"error","error":{"type":"x_y","message":"m"}}')

    rc, out = _main(["onboard", "@P"], project, tmp_path)

    assert rc == 0
    assert capsys.readouterr().err == ""
    assert len(_Gateway.requests) == 3
    assert "requests=3 " in out
    assert waits == [0.001, 0.001]


@pytest.mark.timeout(10)
def test_a_hung_server_times_out_and_falls_back_in_silence(
    project: Path,
    gateway: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(llm, "_REQUEST_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(llm, "_sleep", lambda _seconds: None)
    _Gateway.hang = True

    rc, out = _main(["onboard", "@P"], project, tmp_path)

    assert rc == 0
    assert capsys.readouterr().err == ""
    assert len(_Gateway.requests) == 3
    assert "requests=3 " in out
    assert _belief_count(tmp_path) > 0


_RATE_LIMITED = b'{"type":"error","error":{"type":"rate_limit_error","message":"m"}}'


@pytest.mark.parametrize(
    ("headers", "expected", "line"),
    [
        ({"retry-after": "60"}, [60.0], "waiting 60.0 s before it tries the request again"),
        ({"retry-after": "61"}, [0.5], None),
        ({"retry-after": "86400"}, [0.5], None),
        ({"retry-after": "Wed, 21 Oct 2099 07:28:00 GMT"}, [0.5], None),
        ({"retry-after": "Mon, 01 Jan 2001 00:00:00 GMT"}, [0.5], None),
        ({"retry-after": "Mon, 01 Jan 99999 00:00:00 GMT"}, [0.5], None),
        ({"retry-after": "-5"}, [0.5], None),
        ({"retry-after": "nan"}, [0.5], None),
        ({"retry-after-ms": "2500"}, [2.5], "waiting 2.5 s before it tries the request again"),
        ({"retry-after-ms": "1400"}, [1.4], "waiting 1.4 s before it tries the request again"),
        ({"retry-after-ms": "1000"}, [1.0], None),
        ({"retry-after-ms": "1"}, [0.001], None),
        ({"retry-after-ms": "2000", "retry-after": "30"}, [2.0], "waiting 2.0 s before it tries the request again"),
        ({"x-no-wait": "1"}, [0.5], None),
    ],
    ids=[
        "60-s", "61-s", "a-day", "a-future-date", "a-past-date", "a-date-out-of-range",
        "negative", "nan", "2.5-s-in-ms", "1.4-s-in-ms", "1-s-in-ms", "1-ms",
        "ms-before-seconds", "none-named",
    ],
)
def test_a_429_obeys_a_wait_up_to_60_seconds_and_says_so_past_1(
    project: Path,
    gateway: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    waits: list[float],
    headers: dict[str, str],
    expected: list[float],
    line: str | None,
) -> None:
    _Gateway.queue = [(429, {**_JSON, **headers}, _RATE_LIMITED)]

    rc, out = _main(["onboard", "@P"], project, tmp_path)

    err = capsys.readouterr().err
    assert rc == 0
    assert waits == expected
    assert len(_Gateway.requests) == 2
    assert "requests=2 " in out
    assert "fallbacks=0" in out
    if line is None:
        assert err == ""
    else:
        assert err.splitlines() == [f"aelf: the classifier is {line}"]


def test_the_backoff_is_half_a_second_then_one_second(
    project: Path, gateway: str, tmp_path: Path, waits: list[float],
) -> None:
    _Gateway.reply = (500, {**_JSON, "x-no-wait": "1"}, b"{}")

    rc, out = _main(["onboard", "@P"], project, tmp_path)

    assert rc == 0
    assert waits == [0.5, 1.0]
    assert len(_Gateway.requests) == 3


@pytest.mark.parametrize(
    ("status", "hits", "rc"),
    [(400, 3, 0), (404, 3, 0), (401, 1, 1), (403, 1, 1)],
)
def test_x_should_retry_true_retries_any_status_but_401_and_403(
    project: Path,
    gateway: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    waits: list[float],
    status: int,
    hits: int,
    rc: int,
) -> None:
    _Gateway.reply = (status, {**_JSON, "x-should-retry": "true"}, b"{}")

    code, out = _main(["onboard", "@P"], project, tmp_path)

    assert code == rc
    assert len(_Gateway.requests) == hits
    if rc == 0:
        assert f"requests={hits} " in out
        assert capsys.readouterr().err.count(f"(status {status})") == 1


def test_a_server_that_says_not_to_retry_gets_one_request(
    project: Path, gateway: str, tmp_path: Path, waits: list[float],
) -> None:
    _Gateway.reply = (500, {**_JSON, "x-should-retry": "false"}, b"{}")

    rc, out = _main(["onboard", "@P"], project, tmp_path)

    assert rc == 0
    assert waits == []
    assert len(_Gateway.requests) == 1
    assert "requests=1 " in out


def test_a_500_then_a_refused_connection_counts_the_one_request_sent(
    project: Path,
    gateway: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _Gateway.reply = (500, _JSON, b"{}")
    server = _Gateway.server
    assert server is not None

    def _close_the_server(_seconds: float) -> None:
        server.shutdown()
        server.server_close()

    monkeypatch.setattr(llm, "_sleep", _close_the_server)

    rc, out = _main(["onboard", "@P"], project, tmp_path)

    assert rc == 0
    assert capsys.readouterr().err == ""
    assert len(_Gateway.requests) == 1
    assert "requests=1 " in out


class _Redirect(BaseHTTPRequestHandler):
    """Answers every request with a 307 to `target`, another origin."""

    target = ""

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("content-length", "0")))
        self.send_response(307)
        self.send_header("location", type(self).target + self.path)
        self.send_header("content-length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        """Keep request lines out of the test output."""


def test_a_cross_origin_redirect_carries_a_custom_x_api_key_and_drops_bearer(
    project: Path, gateway: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The risk the docs state for the `ANTHROPIC_CUSTOM_HEADERS` route."""
    _Redirect.target = gateway
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Redirect)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv(llm.ENV_BASE_URL, f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", f"X-Api-Key: {_KEY}")
    try:
        rc, _ = _main(["onboard", "@P"], project, tmp_path)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert rc == 0
    headers = _Gateway.requests[0][1]
    assert headers["x-api-key"] == _KEY
    assert "authorization" not in headers


class _RedirectToFtp(_Redirect):
    target = "ftp://127.0.0.1"


@pytest.mark.parametrize(
    ("handler", "requests", "warning"),
    [
        (_Redirect, "requests=3 ", ""),
        (_RedirectToFtp, "requests=1 ", "aelf: the classifier response was unusable (UnsupportedProtocol)"),
    ],
    ids=["to-a-closed-port", "to-ftp"],
)
def test_a_redirect_that_fails_still_counts_the_request_the_server_saw(
    project: Path,
    gateway: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    waits: list[float],
    handler: type[_Redirect],
    requests: str,
    warning: str,
) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    saved_target = _Redirect.target
    if handler is _Redirect:
        _Redirect.target = _closed_port_url()
    monkeypatch.setenv(llm.ENV_BASE_URL, f"http://127.0.0.1:{server.server_port}")
    try:
        rc, out = _main(["onboard", "@P"], project, tmp_path)
    finally:
        _Redirect.target = saved_target
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert rc == 0
    assert requests in out
    assert capsys.readouterr().err.strip() == (
        f"{warning}. Classified with the regex classifier." if warning else ""
    )


def _closed_port_url() -> str:
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{probe.getsockname()[1]}"


# --- A call that can't match the SDK stops the run ------------------------


@pytest.mark.parametrize(
    ("argv", "scope"),
    [
        (["onboard", "@P"], "No beliefs inserted"),
        (["onboard", "@P", "--llm-classify"], "No beliefs inserted"),
        (["doctor", "--classify-orphans"], "No beliefs updated"),
    ],
    ids=["onboard-default", "onboard-flag", "doctor"],
)
def test_a_keyword_the_installed_sdk_refuses_stops_the_run(
    project: Path,
    gateway: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    scope: str,
) -> None:
    real = llm._request_args  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(
        llm, "_request_args", lambda **kw: {**real(**kw), "temperature": 0.0},
    )

    rc, out = _main(argv, project, tmp_path)

    err = capsys.readouterr().err
    assert rc == 1
    assert "TypeError" in err and "temperature" in err
    assert scope in err
    assert _Gateway.requests == []
    assert "requests=" not in out
    if argv[0] == "doctor":
        assert _orphan_type(tmp_path) == "unknown"
    else:
        assert _belief_count(tmp_path) == 0


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_a_constructor_keyword_the_installed_sdk_refuses_stops_the_run(
    project: Path,
    gateway: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sdk = importlib.import_module(_SDK)
    client_class = getattr(sdk, _SDK.capitalize())
    # Only the client class, so no HTTP client is built ahead of it.
    wrapped = type("Sdk", (), {
        _SDK.capitalize(): staticmethod(lambda **kw: client_class(**kw, no_such_keyword=1)),
    })()
    real = llm._call_anthropic  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(
        llm, "_call_anthropic", lambda **kw: real(**{**kw, "sdk_module": wrapped}),
    )

    rc, _ = _main(["onboard", "@P"], project, tmp_path)

    err = capsys.readouterr().err
    assert rc == 1
    assert "TypeError" in err and "no_such_keyword" in err
    assert _Gateway.requests == []
    assert _belief_count(tmp_path) == 0


# --- The key stays out of what gets printed -------------------------------


@pytest.mark.parametrize(
    ("status", "reply", "label"),
    [
        (
            401,
            (_JSON, json.dumps({"error": {"type": "authentication_error", "message": f"bad {_KEY}"}}).encode()),
            "status 401, authentication_error",
        ),
        (
            403,
            (_JSON, json.dumps({"error": {"type": "permission_error", "message": f"bad {_KEY}"}}).encode()),
            "status 403, permission_error",
        ),
        (
            401,
            (_JSON, json.dumps({"error": {"type": _KEY, "message": "bad"}}).encode()),
            "status 401;",
        ),
        (
            401,
            ({"content-type": "text/plain"}, f"Bearer {_KEY} refused".encode()),
            "status 401;",
        ),
    ],
    ids=["401-echo", "403-echo", "401-type-is-key", "401-plain-text"],
)
@_EVERY_COMMAND
def test_an_auth_reply_that_repeats_the_key_stops_without_printing_it(
    project: Path,
    gateway: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    status: int,
    reply: tuple[dict[str, str], bytes],
    label: str,
) -> None:
    _Gateway.reply = (status, reply[0], reply[1])

    rc, out = _main(argv, project, tmp_path)

    captured = capsys.readouterr()
    assert rc == 1
    assert _KEY not in captured.err
    assert _KEY not in captured.out
    assert _KEY not in out
    assert label in captured.err
    assert f"{llm.ENV_BASE_URL} must accept `Authorization: Bearer`" in captured.err
    assert len(_Gateway.requests) == 1
    if argv[0] == "doctor":
        assert _orphan_type(tmp_path) == "unknown"
    else:
        assert _belief_count(tmp_path) == 0


class AuthenticationError(Exception):
    """The SDK's class name, read off the exception with no status code on it."""


class _Status401(Exception):
    status_code = 401


def _sdk_raising(exc: Exception) -> Any:
    def _create(**_kwargs: Any) -> Any:
        raise exc

    client = SimpleNamespace(messages=SimpleNamespace(create=_create))
    return _FakeSdk(lambda **_kwargs: client)


def _auth_text(exc: Exception) -> str | None:
    candidates = [llm.CandidateInput(index=0, text="a sentence", source="doc:a.md:p1")]
    return llm.classify_batch(candidates, api_key=_KEY, sdk_module=_sdk_raising(exc)).auth_error


@_EVERY_COMMAND
def test_a_key_shaped_error_type_never_prints(
    project: Path,
    gateway: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
) -> None:
    key = "invented_review_key"
    monkeypatch.setenv(llm.ENV_API_KEY, key)
    body = json.dumps({"error": {"type": key, "message": "m"}}).encode()
    _Gateway.reply = (401, _JSON, body)

    rc, _ = _main(argv, project, tmp_path)

    err = capsys.readouterr().err
    assert rc == 1
    assert key not in err
    assert "status 401;" in err


_HELD = (_SDK, f"{_SDK}._base_client", "httpx2", "httpcore2", "httpcore2.http11")


@pytest.fixture
def host_levels() -> Iterator[list[int]]:
    """Levels a host process set on the SDK's loggers, put back after the test."""
    loggers = [logging.getLogger(name) for name in _HELD]
    saved = [logger.level for logger in loggers]
    levels = [logging.ERROR, logging.DEBUG, logging.DEBUG, logging.INFO, logging.NOTSET]
    for logger, level in zip(loggers, levels):
        logger.setLevel(level)
    try:
        yield levels
    finally:
        for logger, level in zip(loggers, saved):
            logger.setLevel(level)


@pytest.mark.parametrize(
    "reply",
    [None, (401, _JSON, b"{}"), (400, _JSON, b"{}")],
    ids=["success", "auth-stop", "rejected"],
)
def test_logger_levels_come_back_exactly_on_every_exit(
    project: Path,
    gateway: str,
    tmp_path: Path,
    host_levels: list[int],
    reply: tuple[int, dict[str, str], bytes] | None,
) -> None:
    _Gateway.reply = reply
    _Gateway.watched = _HELD

    _main(["onboard", "@P"], project, tmp_path)

    assert _Gateway.levels_seen[0] == [max(level, logging.WARNING) for level in host_levels]
    assert [logging.getLogger(name).level for name in _HELD] == host_levels


@pytest.mark.parametrize(
    ("key", "reply"),
    [
        (f"{_KEY}\r", None),
        (_KEY, (200, {**_JSON, "x-echo": f"Bearer {_KEY}"}, b"{}")),
    ],
    ids=["key-the-http-library-refuses", "echo-in-a-reply-header"],
)
def test_a_child_logger_set_to_debug_never_prints_the_key(
    project: Path,
    gateway: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    host_levels: list[int],
    key: str,
    reply: tuple[int, dict[str, str], bytes] | None,
) -> None:
    monkeypatch.setenv(llm.ENV_API_KEY, key)
    _Gateway.reply = reply
    captured = io.StringIO()
    handler = logging.StreamHandler(captured)
    root = logging.getLogger()
    saved_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        _main(["onboard", "@P"], project, tmp_path)
    finally:
        root.removeHandler(handler)
        root.setLevel(saved_level)

    assert _KEY not in captured.getvalue()


def test_auth_text_names_the_class_when_there_is_no_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(llm.ENV_BASE_URL, raising=False)
    assert _auth_text(AuthenticationError(f"bad {_KEY}")) == "AuthenticationError"


def test_auth_text_names_no_gateway_when_no_base_url_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(llm.ENV_BASE_URL, raising=False)
    assert _auth_text(_Status401(f"bad {_KEY}")) == "status 401"


def test_auth_text_never_prints_the_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(llm.ENV_BASE_URL, "https://u:gw-password@gateway.invalid")
    text = _auth_text(_Status401("bad"))
    assert text is not None
    assert "gw-password" not in text
    assert "gateway.invalid" not in text


# Every level turned down the way `ANTHROPIC_LOG=debug` and a root DEBUG config do.
_DEBUG_RUN = """
import logging, os
logging.basicConfig(level=logging.DEBUG)
from aelfrice import llm_classifier as llm
candidates = [llm.CandidateInput(index=0, text="a sentence", source="doc:a.md:p1")]
result = llm.classify_batch(candidates, api_key=os.environ[llm.ENV_API_KEY])
print(result.warning, result.auth_error, result.telemetry.requests)
"""


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    ("key", "reply", "expected"),
    [
        (f"{_KEY}\r", None, "LocalProtocolError"),
        (_KEY, (200, {**_JSON, "x-echo": f"Bearer {_KEY}"}, _message(1)), "None None 1"),
        (_KEY, (401, {**_JSON, "x-echo": _KEY}, b'{"error":{"type":"x_y","message":"m"}}'), "status 401"),
    ],
    ids=["key-the-http-library-refuses", "echo-in-a-reply-header", "echo-on-a-401"],
)
def test_sdk_debug_logging_never_prints_the_key(
    project: Path,
    gateway: str,
    key: str,
    reply: tuple[int, dict[str, str], bytes] | None,
    expected: str,
) -> None:
    _Gateway.reply = reply
    env = {**os.environ, "ANTHROPIC_LOG": "debug", llm.ENV_API_KEY: key}

    run = subprocess.run(
        [sys.executable, "-c", _DEBUG_RUN],
        env=env,
        capture_output=True,
        text=True,
        timeout=25,
        check=False,
    )

    assert run.returncode == 0, run.stderr[-400:]
    assert expected in run.stdout
    assert _KEY not in run.stdout
    assert _KEY not in run.stderr
    # The HTTP library's INFO line, which carries the server's reason phrase.
    assert "HTTP Request" not in run.stderr


# --- The range the extra declares -----------------------------------------


def _extra_spec() -> str:
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    specs: list[str] = pyproject["project"]["optional-dependencies"]["onboard-llm"]
    assert len(specs) == 1
    return specs[0]


def test_the_extra_declares_the_2_x_range_and_the_installed_sdk_is_in_it(
    project: Path,
) -> None:
    assert _extra_spec() == f"{_SDK}>=2,<3"
    assert re.fullmatch(r"2\.\d+\.\d+", importlib.import_module(_SDK).__version__)


def test_the_consent_prompt_says_labels_vary_and_cites_no_figure() -> None:
    text = " ".join(llm._PROMPT_TEXT.split())  # pyright: ignore[reportPrivateUsage]
    assert "two runs can label a sentence differently" in text
    assert "macro-F1" not in text


class _ClosingSdk:
    """A fake SDK whose HTTP client counts its closes."""

    def __init__(self, create: Any) -> None:
        self.closed = 0
        self._create = create

    def DefaultHttpxClient(self, **_kwargs: Any) -> Any:  # noqa: N802
        def _close() -> None:
            self.closed += 1

        return SimpleNamespace(close=_close)

    def __getattr__(self, _name: str) -> Any:
        return lambda **_kwargs: SimpleNamespace(messages=SimpleNamespace(create=self._create))


def _row_reply(**_kwargs: Any) -> Any:
    text = json.dumps([_ROW])
    return SimpleNamespace(content=[SimpleNamespace(text=text)], usage=None)


def _raise_value_error(**_kwargs: Any) -> Any:
    raise ValueError("not sent")


@pytest.mark.parametrize("create", [_row_reply, _raise_value_error], ids=["success", "failure"])
def test_the_http_client_the_classifier_builds_is_closed(create: Any) -> None:
    sdk = _ClosingSdk(create)
    candidates = [llm.CandidateInput(index=0, text="a sentence", source="doc:a.md:p1")]

    llm.classify_batch(candidates, api_key=_KEY, sdk_module=sdk)

    assert sdk.closed == 1


class _FailingBuildSdk(_ClosingSdk):
    """A fake SDK whose HTTP client is built and whose client class then fails."""

    def __init__(self, failure: BaseException | None, create: Any = None) -> None:
        super().__init__(create)
        self._failure = failure

    def __getattr__(self, _name: str) -> Any:
        def _client(**_kwargs: Any) -> Any:
            if self._failure is not None:
                raise self._failure
            return SimpleNamespace(messages=SimpleNamespace(create=self._create))

        return _client


def _create_without_system(*, model: str, max_tokens: int, messages: Any) -> Any:
    raise AssertionError("the bind check refuses the call first")


@pytest.mark.parametrize(
    ("sdk", "field"),
    [
        (_FailingBuildSdk(TypeError("no such keyword")), "sdk_mismatch_error"),
        (_FailingBuildSdk(ValueError("bad proxy")), "warning"),
        (_FailingBuildSdk(None, _create_without_system), "sdk_mismatch_error"),
    ],
    ids=["constructor-type-error", "constructor-config-error", "bind-mismatch"],
)
def test_the_http_client_is_closed_when_the_build_fails(sdk: _FailingBuildSdk, field: str) -> None:
    candidates = [llm.CandidateInput(index=0, text="a sentence", source="doc:a.md:p1")]

    result = llm.classify_batch(candidates, api_key=_KEY, sdk_module=sdk)

    assert getattr(result, field) is not None
    assert sdk.closed == 1


def test_overlapping_holds_restore_the_level_once_the_last_one_ends(
    host_levels: list[int],
) -> None:
    hold = llm._hold_sdk_logs_at_warning  # pyright: ignore[reportPrivateUsage]
    restore = llm._restore_log_levels  # pyright: ignore[reportPrivateUsage]
    child = logging.getLogger(f"{_SDK}._base_client")

    first = hold(_SDK)
    second = hold(_SDK)
    restore(first)
    during = child.level
    restore(second)

    assert during == logging.WARNING
    assert child.level == logging.DEBUG


def test_a_retry_after_date_inside_60_seconds_is_obeyed(
    project: Path,
    gateway: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    waits: list[float],
) -> None:
    """The clock is pinned, so the date is exactly 30 seconds away."""
    now = 1_900_000_000.0
    monkeypatch.setattr(llm.time, "time", lambda: now)
    date = email.utils.formatdate(now + 30, usegmt=True)
    _Gateway.queue = [(429, {**_JSON, "retry-after": date}, _RATE_LIMITED)]

    rc, _ = _main(["onboard", "@P"], project, tmp_path)

    assert rc == 0
    assert waits == [30.0]
