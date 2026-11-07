"""The classifier on the installed SDK 2.x, against a server on 127.0.0.1 (#1845).

What a gateway set through the base URL sees: the path, the credential header,
the version header, and the body. What comes back through `main`: usage counts,
parsed labels, retries, a timeout, a call that can't match the SDK, and a server
or debug log that repeats the key. No test here reaches a real API.
"""
from __future__ import annotations

import importlib
import io
import json
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
    """Records each request. With no `reply` set, it answers with one row per candidate."""

    requests: list[tuple[str, dict[str, str], dict[str, Any]]] = []
    reply: tuple[int, dict[str, str], bytes] | None = None
    hang = False
    release = threading.Event()

    def do_POST(self) -> None:
        cls = type(self)
        raw = self.rfile.read(int(self.headers.get("content-length", "0")))
        body = json.loads(raw)
        headers = {name.lower(): value for name, value in self.headers.items()}
        cls.requests.append((self.path, headers, body))
        if cls.hang:
            cls.release.wait(timeout=300)
            return
        rows = body["messages"][0]["content"].count('"index"')
        status, extra, reply = cls.reply or (200, _JSON, _message(rows))
        self.send_response(status)
        self.send_header("content-length", str(len(reply)))
        # The SDK waits this long before its retries.
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
    _Gateway.reply = None
    _Gateway.hang = False
    _Gateway.release = threading.Event()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Gateway)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv(llm.ENV_BASE_URL, url)
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


@pytest.mark.parametrize("status", [408, 409, 500, 529])
def test_a_retried_status_reaches_the_server_3_times_and_counts_once(
    project: Path,
    gateway: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    status: int,
) -> None:
    _Gateway.reply = (status, _JSON, b'{"type":"error","error":{"type":"x_y","message":"m"}}')

    rc, out = _main(["onboard", "@P"], project, tmp_path)

    assert rc == 0
    assert capsys.readouterr().err == ""
    assert len(_Gateway.requests) == 3
    assert "requests=1 " in out


@pytest.mark.timeout(10)
def test_a_hung_server_times_out_and_falls_back_in_silence(
    project: Path,
    gateway: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(llm, "_REQUEST_TIMEOUT_SECONDS", 0.05)
    _Gateway.hang = True

    rc, out = _main(["onboard", "@P"], project, tmp_path)

    assert rc == 0
    assert capsys.readouterr().err == ""
    assert len(_Gateway.requests) == 3
    assert "requests=1 " in out
    assert _belief_count(tmp_path) > 0


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
    wrapped = _FakeSdk(lambda **kw: client_class(**kw, no_such_keyword=1))
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
