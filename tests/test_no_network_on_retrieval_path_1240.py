"""The retrieval path makes no outbound network call (#1240).

`README.md` and `docs/user/PRIVACY.md` both state that the only outbound
call the default install makes is the TTL-gated PyPI update notifier, and
that it sits off the retrieval path. That was true and unenforced: nothing
in `tests/` patched the network, so a fetch added inside `retrieve_v2`, a
lane resolver, or a store migration would have shipped green while
falsifying a documented privacy claim.

Two design choices carry these tests, and both are load-bearing:

**The guard records; it does not rely on raising.** Almost every lane in
this package is deliberately fail-soft — `_fetch_pypi_json` swallows
`OSError`, `_coverage_inputs` swallows a bare `Exception`, the hook wraps
the notifier in `except Exception: pass`. A guard that only raised would be
caught by the very code it is meant to police and the test would pass while
the call went out. So the assertion is on the recorded attempts, and the
raise is only there to stop real egress.

**The guard is armed at the socket layer, not at `urlopen`.** Patching
`urllib.request.urlopen` would gate exactly one library and miss
`http.client`, `requests`, `httpx`, or a raw socket. `socket.getaddrinfo`
plus `socket.socket.connect` sits underneath all of them. `getaddrinfo` is
included because DNS resolution is the *first* network touch — without it
these tests would pass vacuously on a sandboxed runner where the resolver
fails before any connect is reached.

`test_the_guard_detects_a_real_outbound_call` is the distinguishing assert:
it drives the one function in the package that genuinely does reach the
network and proves the guard sees it. Without that, the zero-attempt
assertions below cannot tell "nothing called out" from "the guard is not
wired up".
"""
from __future__ import annotations

import io
import json
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from aelfrice.hook import user_prompt_submit
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.retrieval import retrieve, retrieve_v2
from aelfrice.store import MemoryStore

# Loopback is not egress. Nothing on the retrieval path should resolve
# anything at all, but a runner-local helper binding 127.0.0.1 must not be
# reported as a privacy violation.
_LOOPBACK: frozenset[str] = frozenset(
    {"localhost", "127.0.0.1", "::1", "0.0.0.0", ""}
)


class _NetworkAttempted(OSError):
    """Raised after recording, to stop the call actually leaving.

    Deliberately an `OSError`, which is what `socket.gaierror` and
    `ConnectionRefusedError` are. Fail-soft callers already handle that
    family, so the guard reproduces a machine with no route to the host
    rather than injecting a novel exception type they would let escape.
    The recording is the evidence either way.
    """


def _host_of(address: object) -> str:
    if isinstance(address, tuple) and address:
        return str(address[0])
    return str(address)


@contextmanager
def _network_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[list[tuple[str, str]]]:
    """Record every non-loopback DNS lookup and TCP connect.

    Yields the recording. `(kind, host)` pairs, in call order.
    """
    attempts: list[tuple[str, str]] = []
    real_getaddrinfo = socket.getaddrinfo
    real_connect = socket.socket.connect

    def guarded_getaddrinfo(host, *args, **kwargs):  # type: ignore[no-untyped-def]
        if str(host) not in _LOOPBACK:
            attempts.append(("getaddrinfo", str(host)))
            raise _NetworkAttempted(f"getaddrinfo({host!r})")
        return real_getaddrinfo(host, *args, **kwargs)

    def guarded_connect(self, address):  # type: ignore[no-untyped-def]
        host = _host_of(address)
        if self.family in (socket.AF_INET, socket.AF_INET6):
            if host not in _LOOPBACK:
                attempts.append(("connect", host))
                raise _NetworkAttempted(f"connect({address!r})")
        return real_connect(self, address)

    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    yield attempts


def _mk(bid: str, content: str, lock_level: str = LOCK_NONE) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at="2026-07-31T00:00:00Z" if lock_level == LOCK_USER else None,
        created_at="2026-07-31T00:00:00Z",
        last_retrieved_at=None,
    )


@pytest.fixture()
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    """A real on-disk store, seeded so retrieval has something to rank.

    Every lane must actually run: a store that returns nothing exercises
    the early-out path and would not notice a fetch further in.
    """
    s = MemoryStore(str(tmp_path / "memory.db"))
    for i, text in enumerate(
        [
            "the kitchen is full of bananas and the bananas are ripe",
            "the harbour ferry leaves at noon every weekday in winter",
            "the retrieval path is local-first and reads only sqlite",
            "bananas ripen faster next to apples in a paper bag",
        ]
    ):
        s.insert_belief(_mk(f"B{i}", text))
    s.insert_belief(_mk("L0", "the operator prefers atomic commits", LOCK_USER))
    try:
        yield s
    finally:
        s.close()


@pytest.fixture(autouse=True)
def _pin_ambient(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep the live store and the real notifier cache out of these tests.

    Without this the ambient `AELFRICE_DB` leaks in and the update-check
    TTL is read from the developer's own cache, which makes the notifier
    tests pass or fail depending on when that machine last checked.
    """
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "memory.db"))
    monkeypatch.setenv("AELFRICE_DOTDIR", str(tmp_path / "dotdir"))
    monkeypatch.delenv("AELF_NO_UPDATE_CHECK", raising=False)


# ---------------------------------------------------------------------------
# The guard itself must be able to fail. Everything below is vacuous if
# this test does not pass.
# ---------------------------------------------------------------------------

def test_the_guard_detects_a_real_outbound_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_fetch_pypi_json` is the package's one genuine network call.

    Driving it under the guard proves the guard is wired to something real.
    It also pins the documented destination: if the notifier is ever
    repointed, this test names the new host out loud.
    """
    from aelfrice.lifecycle import PYPI_JSON_URL, _fetch_pypi_json

    with _network_guard(monkeypatch) as attempts:
        # Fail-soft by contract: it swallows the guard's raise and returns
        # None. The recording, not the return value, is the evidence.
        assert _fetch_pypi_json() is None

    assert attempts, "guard recorded nothing for a call that does reach out"
    assert all(host == "pypi.org" for _kind, host in attempts), attempts
    # Compare the parsed hostname, not a substring of the URL:
    # `"pypi.org" in url` also accepts `https://evil.example/?x=pypi.org`,
    # which is the whole point of pinning the destination here.
    assert urlsplit(PYPI_JSON_URL).hostname == "pypi.org"


# ---------------------------------------------------------------------------
# The property under test.
# ---------------------------------------------------------------------------

def test_retrieve_makes_no_network_call(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _network_guard(monkeypatch) as attempts:
        beliefs = retrieve(store, "which bananas are ripe in the kitchen")

    assert beliefs, "retrieval returned nothing; the lanes never ran"
    assert attempts == []


def test_retrieve_v2_makes_no_network_call(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _network_guard(monkeypatch) as attempts:
        result = retrieve_v2(store, "which bananas are ripe in the kitchen")

    assert result.beliefs, "retrieval returned nothing; the lanes never ran"
    assert attempts == []


def test_user_prompt_submit_hook_makes_no_in_process_network_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hook body is the surface a user actually runs every turn.

    The notifier it fires is out-of-process by construction, so the
    in-process guard cannot see it. Recording `Popen` closes that hole by
    allowlisting what the hook may spawn: local `git` reads for the
    recent-work sub-block, and the documented notifier. A `curl`, a
    `wget`, or a telemetry POST fails here even though no socket is
    opened in this process.

    The allowlist is deliberately by argv[0] rather than a count. The
    number of `git` calls is an implementation detail of the recent-work
    block and would make this test fail on unrelated changes; *which
    programs run* is the property worth pinning.
    """
    import subprocess

    db = tmp_path / "memory.db"
    s = MemoryStore(str(db))
    s.insert_belief(_mk("B0", "the kitchen is full of ripe bananas"))
    s.close()
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.chdir(tmp_path)

    spawns: list[list[str]] = []

    def recording_popen(argv, *args, **kwargs):  # type: ignore[no-untyped-def]
        spawns.append(list(argv))
        raise OSError("spawn suppressed under test")

    monkeypatch.setattr(subprocess, "Popen", recording_popen)

    payload = json.dumps(
        {
            "session_id": "s1",
            "transcript_path": "/dev/null",
            "cwd": str(tmp_path),
            "hook_event_name": "UserPromptSubmit",
            "prompt": "how many bananas are in the kitchen",
        }
    )
    sout = io.StringIO()
    with _network_guard(monkeypatch) as attempts:
        rc = user_prompt_submit(stdin=io.StringIO(payload), stdout=sout)

    assert rc == 0
    assert attempts == []
    for argv in spawns:
        joined = " ".join(argv)
        is_git = Path(argv[0]).name == "git"
        is_notifier = (
            "aelfrice.lifecycle" in joined and "check_for_update" in joined
        )
        assert is_git or is_notifier, (
            f"unexpected subprocess off the hook path: {argv}"
        )


def test_notifier_does_not_spawn_when_opted_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`AELF_NO_UPDATE_CHECK=1` removes the one documented call entirely."""
    import subprocess

    from aelfrice.lifecycle import maybe_check_for_update_async

    spawns: list[list[str]] = []

    def recording_popen(argv, *args, **kwargs):  # type: ignore[no-untyped-def]
        spawns.append(list(argv))
        raise OSError("spawn suppressed under test")

    monkeypatch.setattr(subprocess, "Popen", recording_popen)
    monkeypatch.setenv("AELF_NO_UPDATE_CHECK", "1")

    with _network_guard(monkeypatch) as attempts:
        assert maybe_check_for_update_async() is False

    assert spawns == []
    assert attempts == []
