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
from pathlib import Path

import pytest

import aelfrice.deferred_feedback as deferred_feedback
import aelfrice.expansion_gate as expansion_gate
import aelfrice.retrieval as retrieval
from aelfrice import hook
from aelfrice.config_discovery import CONFIG_FILENAME, discover_config

_PROMPT = "which widget beliefs rank highest in the unit"


@pytest.fixture()
def pinned_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Pin everything that would otherwise drop a walker or a store.

    `expansion_gate`'s read is short-circuited by either force flag and
    `deferred_feedback`'s by an explicit enqueue value, so an ambient
    export silently removes a reader from the count and the assertion
    passes for the wrong reason. `AELFRICE_DOTDIR` is pinned so the turn
    cannot touch the real one.
    """
    for name in (
        "AELFRICE_FORCE_EXPANSION",
        "AELFRICE_NO_EXPANSION_GATE",
        "AELFRICE_IMPLICIT_FEEDBACK_ENQUEUE",
    ):
        monkeypatch.delenv(name, raising=False)
    dotdir = tmp_path / "dotdir"
    dotdir.mkdir()
    monkeypatch.setenv("AELFRICE_DOTDIR", str(dotdir))
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

    Attribution is per reader rather than per probe so the count ignores
    the modules that still carry private walk loops; those are follow-up
    work and their probes must not decide this test.
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

        for name, module, attr in (
            ("retrieval", retrieval, "_discover_config"),
            ("expansion_gate", expansion_gate, "discover_config"),
            ("deferred_feedback", deferred_feedback, "discover_config"),
        ):
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
