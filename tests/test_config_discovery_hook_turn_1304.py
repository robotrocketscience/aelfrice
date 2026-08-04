"""#1304: one UserPromptSubmit turn shares one discovery walk.

A turn runs `retrieve()` more than once, and before #1304 each call
opened its own `config_discovery_scope` — so the memo was built and
thrown away per retrieval, on top of the three private walk loops each
retrieval already paid for. Decorating `user_prompt_submit` with the
scope makes the whole turn one memo, and the nested per-retrieval scopes
become free.

What this deliberately does NOT assert is "one walk per turn". That is
unachievable, and asserting it would have meant deleting a real
behaviour: some config on this path is resolved from the *payload's*
cwd rather than the hook process's cwd (#909/#887, and the #1279
exploration slot), because the agent's project is not necessarily the
directory the hook process happens to be in. Those are two different
questions with two different answers, so they are two legitimate memo
keys and the floor is two walks whenever the two directories differ.
The second test below pins that as a fact rather than leaving it as a
missed optimisation.

Counts, never wall clock: the perf gate here is load-sensitive and a
latency number does not reproduce.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pytest

import importlib
import pkgutil

import aelfrice
import aelfrice.retrieval as retrieval
from aelfrice import hook
from aelfrice.config_discovery import CONFIG_FILENAME, discover_config


def _converted_readers() -> list[tuple[str, object, str]]:
    """Every module that resolves config through the shared walk.

    Enumerated by importing `aelfrice.*` and looking for the name, not
    from a literal list. The whole point of #1304 is that the set grows
    as private walk loops are converted, and a literal list would have
    silently stopped covering the ones added after it was written —
    which is exactly what happened to the three-module version of this
    harness once the remaining eleven were converted: an unspied module
    warmed the memo first and the spied ones then measured zero.

    `retrieval` re-exports the helper under a private name for its own
    callers, so it is named explicitly rather than found by the scan.
    """
    found: list[tuple[str, object, str]] = [
        ("retrieval", retrieval, "_discover_config"),
    ]
    for info in pkgutil.iter_modules(aelfrice.__path__):
        if info.name in {"retrieval", "config_discovery"}:
            continue
        try:
            module = importlib.import_module(f"aelfrice.{info.name}")
        except Exception:  # noqa: BLE001 - an unimportable extra is not ours
            continue
        if hasattr(module, "discover_config"):
            found.append((info.name, module, "discover_config"))
    return found

_PROMPT = "which widget beliefs rank highest in the unit"


@pytest.fixture()
def pinned_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Pin everything that would otherwise drop a walker or a store.

    `expansion_gate`'s read is short-circuited by either force flag and
    `deferred_feedback`'s by an explicit enqueue value, so an ambient
    export silently removes a reader from the count and the assertion
    passes for the wrong reason. The dotdir is pinned by the
    session-autouse `_sandbox_real_home` fixture in `conftest.py`; the
    `AELFRICE_DOTDIR` setenv that used to stand in for it here was dead,
    since no module reads that variable (#1320).
    """
    for name in (
        "AELFRICE_FORCE_EXPANSION",
        "AELFRICE_NO_EXPANSION_GATE",
        "AELFRICE_IMPLICIT_FEEDBACK_ENQUEUE",
    ):
        monkeypatch.delenv(name, raising=False)
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    _seed(db)
    return tmp_path


def _seed(db: Path) -> None:
    from aelfrice.derivation import DerivationInput, derive
    from aelfrice.models import INGEST_SOURCE_FILESYSTEM
    from aelfrice.store import MemoryStore

    store = MemoryStore(str(db))
    for i, word in enumerate(("alpha", "beta", "gamma")):
        out = derive(
            DerivationInput(
                source_kind=INGEST_SOURCE_FILESYSTEM,
                raw_text=f"the widget {word} unit ranks beliefs",
                source_path=f"doc{i}.md",
                session_id=None,
                ts="2026-01-01T00:00:00+00:00",
            ),
        )
        assert out.belief is not None
        store.insert_or_corroborate(out.belief, source_type="filesystem_ingest")
    store.close()


def _run_turn(payload_cwd: Path) -> None:
    payload = json.dumps({
        "session_id": "sess-1304",
        "transcript_path": "/dev/null",
        "cwd": str(payload_cwd),
        "hook_event_name": "UserPromptSubmit",
        "prompt": _PROMPT,
    })
    rc = hook.user_prompt_submit(
        stdin=io.StringIO(payload),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert rc == 0


class _Attribution:
    """Probes charged to the converted readers, and who reached them.

    Attribution is per reader rather than per probe, and the reader set
    is enumerated rather than listed — see `_converted_readers`. With
    every module converted (#1304) there are no private walk loops left
    to exclude, so the charged count is the turn's whole config cost.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.probes: list[str] = []
        self.reached: list[str] = []
        self.charged = 0
        real_is_file = Path.is_file

        def counting_is_file(self_: Path) -> bool:
            if self_.name == CONFIG_FILENAME:
                self.probes.append(str(self_))
            return real_is_file(self_)

        monkeypatch.setattr(Path, "is_file", counting_is_file)

        for name, module, attr in _converted_readers():
            self._spy(monkeypatch, name, module, attr)

    def _spy(
        self,
        monkeypatch: pytest.MonkeyPatch,
        name: str,
        module: object,
        attr: str,
    ) -> None:
        real = getattr(module, attr)

        def wrapper(*args: object, **kwargs: object) -> object:
            self.reached.append(name)
            before = len(self.probes)
            try:
                return real(*args, **kwargs)
            finally:
                self.charged += len(self.probes) - before

        monkeypatch.setattr(module, attr, wrapper)

    def walk_cost(self, start: Path | None = None) -> int:
        """One unmemoized walk from `start`, measured in this process."""
        before = len(self.probes)
        discover_config(start)
        return len(self.probes) - before

    def reset(self) -> None:
        self.charged = 0
        self.reached.clear()


def test_a_turn_from_the_process_cwd_costs_one_walk(
    pinned_env: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every converted reader on the turn shares a single walk.

    The turn resolves config dozens of times across several `retrieve()`
    calls. Without the turn-wide scope each `retrieve()` builds its own
    memo, so the same walk is repeated once per retrieval.
    """
    here = Path.cwd()
    _run_turn(here)  # warm every non-config cache

    att = _Attribution(monkeypatch)
    one_walk = att.walk_cost()
    assert one_walk > 0, "baseline measured no filesystem probes"

    att.reset()
    _run_turn(here)

    assert att.reached.count("retrieval") > 1, (
        "the turn resolved retrieval config fewer than twice; the count "
        "cannot distinguish a turn-wide memo from a per-retrieval one"
    )
    assert att.charged == one_walk, (
        f"the converted readers cost {att.charged} probes across "
        f"{len(att.reached)} calls; one walk is {one_walk}"
    )


def test_a_payload_cwd_elsewhere_costs_exactly_two_walks(
    pinned_env: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The floor is two walks, and this is why — not a missed memo.

    Issue #1304's acceptance asked for one walk per hook turn. It cannot
    be one: config resolved from the payload's cwd is a different
    question from config resolved from the hook process's cwd, and
    answering it with the process cwd's answer would silently
    reintroduce the bug #909/#887 fixed. Two distinct starts, two memo
    keys, two walks — and no more than two, which is the part worth
    pinning.
    """
    _run_turn(pinned_env)  # warm

    att = _Attribution(monkeypatch)
    process_walk = att.walk_cost()
    payload_walk = att.walk_cost(pinned_env)
    assert process_walk > 0 and payload_walk > 0
    assert pinned_env.resolve() != Path.cwd().resolve(), (
        "fixture cwd coincides with the process cwd; this test needs "
        "them distinct or it degenerates into the one-walk case"
    )

    att.reset()
    _run_turn(pinned_env)

    assert att.charged == process_walk + payload_walk, (
        f"the converted readers cost {att.charged} probes; two walks "
        f"from the two distinct starts are "
        f"{process_walk} + {payload_walk}"
    )


def test_no_module_carries_a_private_config_walk() -> None:
    """The census #1304 was filed on, as a standing assertion.

    Fourteen modules each carried their own `cwd`-to-root loop, so N
    readers cost N walks even inside a shared scope — the memo can only
    collapse walks that go through it. Converting them is only half the
    fix; without this arm the next reader copies the same twelve lines
    from a neighbour, exactly as the first fourteen did.

    Keyed on the loop's *shape* by regex, not on one spelling and not on
    a module list, so a new private walk is caught wherever it lands and
    whatever it names its loop variable. An earlier version of this guard
    matched the literal `while current not in seen`; the two loops in
    `cli.py` spell it `while candidate not in seen` and sailed past it,
    so the guard was green while the regression it names was present.
    Renaming the loop variable is not a defence.

    `rglob` rather than `glob`: 15 modules live under `wonder/`,
    `slash_commands/` and `query_understanding/` and were never scanned.

    The second clause matches the config filename *value*, not the token
    `CONFIG_FILENAME`: `project_warm` defines its own `_CONFIG_FILENAME`
    (`"config.json"`) and walks for a sentinel directory, so a token
    match reports it as an offender when its loop has nothing to do with
    `.aelfrice.toml`.

    `config_discovery` itself is the one legitimate implementor.
    """
    walk = re.compile(r"while\s+\w+\s+not\s+in\s+seen")
    src = Path(__file__).resolve().parents[1] / "src" / "aelfrice"
    offenders = sorted(
        str(module.relative_to(src))
        for module in src.rglob("*.py")
        if module.name != "config_discovery.py"
        and "__pycache__" not in module.parts
        and walk.search(text := module.read_text(encoding="utf-8"))
        and CONFIG_FILENAME in text
    )
    assert offenders == [], (
        f"these modules walk to find .aelfrice.toml themselves: {offenders}. "
        "Use `config_discovery.discover_config`, which is memoized inside a "
        "`config_discovery_scope` — a private loop is invisible to the memo "
        "and costs a full walk per reader."
    )
