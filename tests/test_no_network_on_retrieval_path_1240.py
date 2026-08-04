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
`http.client`, `requests`, and `httpx`. Those three all reach the network
through `socket.create_connection`, so the four `socket` entry points below
sit underneath all of them:

  * `getaddrinfo` and `gethostbyname` — resolution. `getaddrinfo` is
    included because DNS is the *first* network touch, without which these
    tests would pass vacuously on a sandboxed runner whose resolver fails
    before any connect is reached. `gethostbyname` is the legacy door to
    the same place and is not reached through `getaddrinfo`.
  * `socket.connect` and `socket.connect_ex` — connection. These are
    sibling methods, not one wrapping the other, so patching only
    `connect` left a `connect_ex` to a literal IP unrecorded *and
    unblocked* until #1247.

Each of the four has a test that reddens when its arm is removed; an arm
with nothing to prove it can fire is not covered.

**What this guard does NOT cover, stated rather than implied.** It enforces
the **TCP connect path and those two resolvers** — not "any raw socket". At
least three doors bypass all four arms, measured against this guard:

```
gethostbyname_ex         recorded=[]  -> gaierror (real resolver reached)
getnameinfo              recorded=[]  -> ('198.51.100.7', 'http')
UDP sendto (SOCK_DGRAM)  recorded=[]  -> returned 1   (a byte left)
TCP connect (control)    recorded=[('connect', '198.51.100.7')]  -> blocked
```

A `SOCK_DGRAM` socket never calls `connect` or `connect_ex` — `sendto`
takes the address directly — so UDP to a literal IP is neither recorded nor
blocked. `gethostbyname_ex` and `getnameinfo` are further resolver doors on
the same footing as `gethostbyname`.

These are deliberately left open. Nothing in `src/` speaks UDP or reaches
for the legacy resolvers, the guard is test-only, and an arms race here has
no natural stopping point. The point of writing them down is that the
previous wording — "or a raw socket" — promised coverage the arms do not
deliver, which is the same defect #1247 was filed about. Widen the arms
only if something on the retrieval path starts using one of these; do not
widen the claim without widening the arms.

`test_the_guard_detects_a_real_outbound_call` is the distinguishing assert:
it drives the one function in the package that genuinely does reach the
network and proves the guard sees it. Without that, the zero-attempt
assertions below cannot tell "nothing called out" from "the guard is not
wired up".
"""
from __future__ import annotations

import errno
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

    Four entry points are patched, and the pairs matter (#1247). Resolution
    has two doors — `getaddrinfo` and the legacy `gethostbyname` — and
    connection has two — `connect` and `connect_ex`. `connect_ex` is a
    separate method rather than a wrapper around `connect`, so patching one
    leaves the other live; before #1247 a `connect_ex` to a literal IP was
    neither recorded nor blocked while this guard was armed.

    Nothing in `src/` reaches for either of the legacy spellings today. They
    are covered because the module docstring's claim should be true of what
    is enforced rather than of what was intended.

    Scope, so this is not read as wider than it is: these four cover the TCP
    connect path and two of the resolvers. UDP `sendto`, `gethostbyname_ex`
    and `getnameinfo` bypass all four and are deliberately not armed — see
    the module docstring for the measurements and the reasoning.
    """
    attempts: list[tuple[str, str]] = []
    real_getaddrinfo = socket.getaddrinfo
    real_gethostbyname = socket.gethostbyname
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def guarded_getaddrinfo(host, *args, **kwargs):  # type: ignore[no-untyped-def]
        if str(host) not in _LOOPBACK:
            attempts.append(("getaddrinfo", str(host)))
            raise _NetworkAttempted(f"getaddrinfo({host!r})")
        return real_getaddrinfo(host, *args, **kwargs)

    def guarded_gethostbyname(host, *args, **kwargs):  # type: ignore[no-untyped-def]
        if str(host) not in _LOOPBACK:
            attempts.append(("gethostbyname", str(host)))
            raise _NetworkAttempted(f"gethostbyname({host!r})")
        return real_gethostbyname(host, *args, **kwargs)

    def guarded_connect(self, address):  # type: ignore[no-untyped-def]
        host = _host_of(address)
        if self.family in (socket.AF_INET, socket.AF_INET6):
            if host not in _LOOPBACK:
                attempts.append(("connect", host))
                raise _NetworkAttempted(f"connect({address!r})")
        return real_connect(self, address)

    def guarded_connect_ex(self, address):  # type: ignore[no-untyped-def]
        """Record, then report the host as unreachable.

        `connect_ex` reports failure by *returning* an errno rather than
        raising, so returning `ENETUNREACH` is what a machine with no route
        actually does — the same reasoning that makes `_NetworkAttempted` an
        `OSError` elsewhere in this file. Raising here would be a behaviour
        no real `connect_ex` exhibits, which is exactly the kind of novel
        failure a fail-soft caller would not handle. The recording is the
        evidence either way.
        """
        host = _host_of(address)
        if self.family in (socket.AF_INET, socket.AF_INET6):
            if host not in _LOOPBACK:
                attempts.append(("connect_ex", host))
                return errno.ENETUNREACH
        return real_connect_ex(self, address)

    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket, "gethostbyname", guarded_gethostbyname)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
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
    """Keep the live store out of these tests.

    Without this the ambient `AELFRICE_DB` leaks in and retrieval ranks the
    developer's real beliefs instead of the seeded ones.

    The dotdir pin that used to live here was a no-op: nothing in
    `src/aelfrice` reads `AELFRICE_DOTDIR` from the environment (#1320).
    The session-autouse `_sandbox_real_home` fixture in `conftest.py`
    pins it for real, by `setattr` on the module constants.

    `AELF_NO_UPDATE_CHECK` is deliberately unset here — this module's
    whole point is to prove the network guard fires — which also undoes
    the conftest fixture's only defence against the *detached* update
    check. Tests that need a controlled cache pass `cache_path`
    explicitly; none here trigger a spawn.
    """
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "memory.db"))
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
    import aelfrice.lifecycle as lifecycle

    with _network_guard(monkeypatch) as attempts:
        # Fail-soft by contract: it swallows the guard's raise and returns
        # None. The recording, not the return value, is the evidence.
        assert lifecycle._fetch_pypi_json() is None

    assert attempts, "guard recorded nothing for a call that does reach out"
    assert all(host == "pypi.org" for _kind, host in attempts), attempts
    # Compare the parsed hostname, not a substring of the URL:
    # `"pypi.org" in url` also accepts `https://evil.example/?x=pypi.org`,
    # which is the whole point of pinning the destination here.
    assert urlsplit(lifecycle.PYPI_JSON_URL).hostname == "pypi.org"


# ---------------------------------------------------------------------------
# One test per guard arm (#1247). `_fetch_pypi_json` above exercises the
# `getaddrinfo` arm against real code; the three below are the only way the
# other arms can be shown to fire, because nothing in `src/` reaches for
# them. An arm with no test that reddens it is not covered.
#
# Addresses here are reserved by RFC and are never real hosts:
# `192.0.2.0/24` and `198.51.100.0/24` are TEST-NET-1 and TEST-NET-2
# (RFC 5737), `.invalid` is reserved by RFC 6761. With the arms in place the
# guard intercepts before the syscall, so no packet leaves the machine; the
# reserved ranges are what keeps that true under a mutation run too.
# ---------------------------------------------------------------------------

def test_the_guard_records_and_blocks_connect_ex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`connect_ex` is a sibling of `connect`, not a wrapper around it.

    Patching only `connect` left this path both unrecorded and *unblocked*.
    Blocking is half the guard's job, so this asserts the return value too:
    `ENETUNREACH` is what a machine with no route reports, and `connect_ex`
    reports by returning rather than raising.
    """
    sock = socket.socket()
    # Only reached if the arm is removed; bounds the mutation run rather
    # than letting it block on a TEST-NET address that never answers.
    sock.settimeout(0.05)
    try:
        with _network_guard(monkeypatch) as attempts:
            rc = sock.connect_ex(("192.0.2.1", 80))
    finally:
        sock.close()

    assert attempts == [("connect_ex", "192.0.2.1")], attempts
    assert rc == errno.ENETUNREACH


def test_the_guard_records_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    """The `connect` arm had no test of its own until #1247.

    Every other test in this file reaches the network through a hostname, so
    they all trip `getaddrinfo` first and none of them exercises `connect`.
    Removing the `connect` arm therefore left the whole file green — the same
    uncovered-guard-arm shape #1247 was filed about, one method over.
    """
    sock = socket.socket()
    sock.settimeout(0.05)
    try:
        with _network_guard(monkeypatch) as attempts:
            with pytest.raises(_NetworkAttempted):
                sock.connect(("198.51.100.7", 80))
    finally:
        sock.close()

    assert attempts == [("connect", "198.51.100.7")], attempts


def test_the_guard_records_gethostbyname(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The legacy resolver is a second door to the same place.

    `gethostbyname` does not route through `getaddrinfo`, so the "DNS is the
    first network touch" argument needs both patched to hold.
    """
    with _network_guard(monkeypatch) as attempts:
        with pytest.raises(_NetworkAttempted):
            socket.gethostbyname("nonexistent.invalid")

    assert attempts == [("gethostbyname", "nonexistent.invalid")], attempts


def test_the_guard_lets_loopback_through_on_every_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A runner-local helper is not egress, and must not be recorded.

    Without this an arm that recorded unconditionally would pass every
    assertion above while making the whole guard useless — it would report a
    privacy violation for a test that binds `127.0.0.1`.

    All four arms are driven, which is what makes "every arm" literal. The
    arm-removal tests above pin each arm's *recording*; only a loopback call
    through the same arm pins its *exemption*. Driving two of the four left
    dropping the `_LOOPBACK` check from `getaddrinfo` or `connect` green
    across the whole file.
    """
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(2)
    port = server.getsockname()[1]
    client_ex = socket.socket()
    client = socket.socket()
    try:
        with _network_guard(monkeypatch) as attempts:
            assert client_ex.connect_ex(("127.0.0.1", port)) == 0
            client.connect(("127.0.0.1", port))
            assert socket.gethostbyname("127.0.0.1") == "127.0.0.1"
            assert socket.getaddrinfo("127.0.0.1", port)
    finally:
        client.close()
        client_ex.close()
        server.close()

    assert attempts == []


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

    The hook calls `maybe_check_for_update_async()` with no arguments, so
    it always consults the real home-derived cache. On a machine that
    checked recently the notifier never spawns and the allowlist's
    notifier branch is never reached — the test would then only be pinning
    `git`. `read_cache` is stubbed stale so the notifier spawns on every
    machine and both branches of the allowlist are exercised.
    """
    import subprocess

    import aelfrice.lifecycle as lifecycle

    monkeypatch.setattr(
        lifecycle,
        "read_cache",
        lambda *_a, **_k: lifecycle.UpdateStatus(
            update_available=False, installed="0", latest="0", checked=0.0
        ),
    )

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

    def _is_notifier(argv: list[str]) -> bool:
        joined = " ".join(argv)
        return "aelfrice.lifecycle" in joined and "check_for_update" in joined

    for argv in spawns:
        is_git = Path(argv[0]).name == "git"
        assert is_git or _is_notifier(argv), (
            f"unexpected subprocess off the hook path: {argv}"
        )

    # Without this the allowlist above is satisfied by a run that spawned
    # nothing but `git`, and the notifier branch would be unexercised code.
    assert any(_is_notifier(argv) for argv in spawns), spawns


def test_the_opt_out_is_what_removes_the_notifier_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`AELF_NO_UPDATE_CHECK=1` removes the one documented call entirely.

    Both halves are asserted, because the opt-out half alone does not
    discriminate. `maybe_check_for_update_async` consults the cache before
    it would ever spawn, so on any machine with a fresh
    `~/.cache/aelfrice/update_check.json` nothing was going to spawn either
    way and the assertion never observes the opt-out doing anything.

    `cache_path` is therefore passed explicitly: `CACHE_FILE` is
    home-derived and bound as an import-time default, so no fixture can
    redirect it. Pointing it at an empty `tmp_path` makes the cache stale,
    which is what leaves the opt-out as the only reason nothing spawns.
    """
    import subprocess

    import aelfrice.lifecycle as lifecycle

    cache_path = tmp_path / "update_check.json"  # absent, therefore stale
    spawns: list[list[str]] = []

    def recording_popen(argv, *args, **kwargs):  # type: ignore[no-untyped-def]
        spawns.append(list(argv))
        raise OSError("spawn suppressed under test")

    monkeypatch.setattr(subprocess, "Popen", recording_popen)

    monkeypatch.setenv("AELF_NO_UPDATE_CHECK", "1")
    with _network_guard(monkeypatch) as attempts:
        assert lifecycle.maybe_check_for_update_async(cache_path=cache_path) is False

    assert spawns == []
    assert attempts == []

    # The distinguishing half: same call, same stale cache, opt-out cleared.
    # The return value is deliberately not asserted here — it is True only
    # when `Popen` succeeds, and this recorder raises `OSError`, which the
    # function catches and reports as False. The spawn record is the evidence.
    monkeypatch.delenv("AELF_NO_UPDATE_CHECK", raising=False)
    with _network_guard(monkeypatch) as attempts:
        lifecycle.maybe_check_for_update_async(cache_path=cache_path)

    assert len(spawns) == 1, spawns
    joined = " ".join(spawns[0])
    assert "aelfrice.lifecycle" in joined and "check_for_update" in joined
    # Still nothing in-process: the notifier reaches the network from the
    # child, which is exactly why the hook test allowlists it by argv.
    assert attempts == []
