"""The installed anthropic SDK must take every keyword the classifier sends (#1844).

The classifier's other tests replace the SDK with stubs that accept any
keyword, so an SDK that drops a parameter still passes them. anthropic 1.0
removed `temperature` from `Messages.create`. On 1.0 and later
`_call_anthropic` raised `TypeError` before any request left, the classifier
treated it as a transient error, and onboard fell back to the regex
classifier with no message. Dependabot's bump to 2.3.0 (#1827) was green for
that reason.

Each test takes the keywords from the classifier's own call, not from a
hand-copied list. The first records the call on a fake client and checks
each keyword against the installed `Messages.create` signature. The second
checks that the recorded call sends no `temperature`, since SDK 2.x takes none
(#1845). The third runs the real SDK client over a mock transport from the
SDK's own HTTP package, so the installed SDK builds and serializes the request
and nothing reaches the network. That package is `httpx2` on SDK 2.x, and the
test reads it off the SDK's `DefaultHttpxClient`.

The module skips when `anthropic` is not importable. CI's pytest job installs
the `[onboard-llm]` extra and asserts `import anthropic` before it runs the
suite, so the skip cannot happen there.
"""
from __future__ import annotations

import inspect
import json
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from aelfrice import llm_classifier as llm

anthropic = pytest.importorskip(
    "anthropic",
    reason=(
        "requires the 'onboard-llm' extra: uv sync --all-groups --all-extras "
        "(or --extra onboard-llm). These tests check the installed SDK "
        "against the classifier's call - see #1844"
    ),
)
from anthropic.resources.messages import Messages  # noqa: E402

# The HTTP package the installed SDK builds its client on.
_HTTP_BASE = next(
    b for b in anthropic.DefaultHttpxClient.__mro__
    if not b.__module__.startswith("anthropic")
)
_http = sys.modules[_HTTP_BASE.__module__.split(".")[0]]

_RESPONSE_JSON: dict[str, Any] = {
    "id": "msg_1844",
    "type": "message",
    "role": "assistant",
    "model": llm.DEFAULT_MODEL,
    "content": [{"type": "text", "text": "[]"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 7, "output_tokens": 3},
}


def _call(sdk_module: Any) -> llm.ClientResponse:
    return llm._call_anthropic(  # pyright: ignore[reportPrivateUsage]
        api_key="test-key",
        model=llm.DEFAULT_MODEL,
        system_prompt="system prompt",
        user_message="user message",
        max_output_tokens=16,
        sdk_module=sdk_module,
    )


def _recorded_create_kwargs() -> dict[str, Any]:
    """The keywords `_call_anthropic` passes to `client.messages.create`."""
    calls: list[dict[str, Any]] = []

    class _Messages:
        def create(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(text="[]")], usage=None)

    class _Client:
        def __init__(self, **_kwargs: Any) -> None:
            self.messages = _Messages()

    _call(SimpleNamespace(Anthropic=_Client))
    assert len(calls) == 1, f"expected one messages.create call, got {len(calls)}"
    return calls[0]


def test_installed_messages_create_takes_every_classifier_keyword() -> None:
    sent = _recorded_create_kwargs()
    params = inspect.signature(Messages.create).parameters
    # A `**kwargs` parameter would accept any keyword, and the check below
    # would pass on an SDK that ignores what it is given.
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()), (
        f"anthropic {anthropic.__version__} Messages.create takes **kwargs, "
        "so this signature check proves nothing"
    )
    missing = sorted(set(sent) - set(params))
    assert not missing, (
        f"anthropic {anthropic.__version__} Messages.create has no parameter "
        f"{missing}; the classifier sends {sorted(sent)}. The call raises "
        "TypeError and onboard falls back to the regex classifier (#1844)."
    )


def test_the_classifier_sends_no_temperature() -> None:
    """The operator ruled that classification is no longer pinned at 0 (#1845)."""
    sent = _recorded_create_kwargs()
    assert "temperature" not in sent, (
        f"the classifier sends temperature={sent['temperature']!r}"
    )


def test_the_installed_sdk_sends_the_classifier_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    requests: list[Any] = []
    clients: list[Any] = []

    def _handler(request: Any) -> Any:
        requests.append(request)
        return _http.Response(200, json=_RESPONSE_JSON)

    def _client(**kwargs: Any) -> Any:
        http_client = _http.Client(transport=_http.MockTransport(_handler))
        clients.append(http_client)
        return anthropic.Anthropic(**kwargs, http_client=http_client, max_retries=0)

    try:
        resp = _call(SimpleNamespace(Anthropic=_client))
    finally:
        for c in clients:
            c.close()

    assert len(requests) == 1, f"expected one request, got {len(requests)}"
    body = json.loads(requests[0].content)
    # The recorded keywords as JSON, the form the SDK serializes them into.
    sent = json.loads(json.dumps(_recorded_create_kwargs()))
    assert body == sent, (
        f"the request body is {body!r}, the classifier passed {sent!r}"
    )
    assert resp.text == "[]"
    assert (resp.input_tokens, resp.output_tokens) == (7, 3)
