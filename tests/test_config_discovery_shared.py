"""#1304: one `.aelfrice.toml` walk per `retrieve()`, not three.

#1289 / PR #1298 memoised discovery *inside* `retrieval.py`. Two sibling
modules reached during every retrieval — `expansion_gate` and
`deferred_feedback` — carried their own copy-pasted walk loops, so a
`retrieve()` still cost three walks. This pins the shared module and the
two conversions.

Counts, never wall clock. The defect is a number of `stat` calls; a
latency budget re-flakes on a loaded machine (see the #1289 discussion),
and the baseline is measured in the same process so nothing encodes this
checkout's path depth.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import aelfrice.deferred_feedback as deferred_feedback
import aelfrice.expansion_gate as expansion_gate
import aelfrice.retrieval as retrieval
from aelfrice.config_discovery import (
    CONFIG_FILENAME,
    config_discovery_scope,
    discover_config,
)

# Ambient env that would silently remove a walker from the measurement.
#
# `expansion_gate._read_toml_flag` is short-circuited by either force
# flag, and `deferred_feedback._read_toml_value` is short-circuited by an
# explicit enqueue env value. With any of them set on the developer's
# shell the "before" count is 2 walks or 1, and the assertion below
# passes for the wrong reason.
_MUST_BE_UNSET = (
    "AELFRICE_FORCE_EXPANSION",
    "AELFRICE_NO_EXPANSION_GATE",
    "AELFRICE_IMPLICIT_FEEDBACK_ENQUEUE",
)


@pytest.fixture()
def pinned_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _MUST_BE_UNSET:
        monkeypatch.delenv(name, raising=False)


def _probe_counter(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record one entry per `.aelfrice.toml` existence probe."""
    probes: list[str] = []
    real_is_file = Path.is_file

    def counting_is_file(self: Path) -> bool:
        if self.name == CONFIG_FILENAME:
            probes.append(str(self))
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", counting_is_file)
    return probes


def _hitting_store():
    """A store whose `retrieve()` returns beliefs.

    Load-bearing, not incidental: `deferred_feedback`'s config read sits
    inside `retrieve()`'s `if out:` branch, so a query that returns zero
    beliefs never reaches it and the three-walk baseline silently
    becomes two.
    """
    from aelfrice.derivation import DerivationInput, derive
    from aelfrice.models import INGEST_SOURCE_FILESYSTEM
    from aelfrice.store import MemoryStore

    store = MemoryStore(":memory:")
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
    return store


def test_retrieve_costs_exactly_one_walk(
    pinned_env: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The #1304 acceptance, as a count.

    Before the conversion this was three walks: `retrieval`'s own
    resolvers, `expansion_gate._read_toml_flag`, and
    `deferred_feedback._read_toml_value`. Reverting either caller to a
    private walk loop makes this 2x or 3x the baseline.
    """
    store = _hitting_store()
    hits = retrieval.retrieve(store, "widget")
    assert hits, (
        "fixture query returned no beliefs; deferred_feedback's config "
        "read is inside `if out:` and would never fire"
    )

    probes = _probe_counter(monkeypatch)
    discover_config()
    one_walk = len(probes)
    assert one_walk > 0, "baseline measured no filesystem probes"

    probes.clear()
    retrieval.retrieve(store, "widget")
    store.close()
    assert len(probes) == one_walk, (
        f"{len(probes)} probes for one retrieve(); one walk costs "
        f"{one_walk}. Probed: {probes}"
    )


def test_all_three_readers_are_actually_reached(
    pinned_env: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keeps the count above from passing for the wrong reason.

    A change that simply stopped calling `expansion_gate` or
    `deferred_feedback` during retrieval would also drive the probe
    count to one walk, while delivering none of the fix. This asserts
    all three readers still run and all three go through the shared
    module.
    """
    store = _hitting_store()
    retrieval.retrieve(store, "widget")  # warm

    seen: list[str] = []

    def spy(name: str, module: object, attr: str) -> None:
        real = getattr(module, attr)

        def wrapper(*args: object, **kwargs: object) -> object:
            seen.append(name)
            return real(*args, **kwargs)

        monkeypatch.setattr(module, attr, wrapper)

    spy("retrieval", retrieval, "_discover_config")
    spy("expansion_gate", expansion_gate, "discover_config")
    spy("deferred_feedback", deferred_feedback, "discover_config")

    retrieval.retrieve(store, "widget")
    store.close()

    for name in ("retrieval", "expansion_gate", "deferred_feedback"):
        assert name in seen, (
            f"{name} never reached shared discovery during retrieve(); "
            "the one-walk count is measuring a reader that no longer runs"
        )


def test_converted_readers_share_the_retrieval_scope(
    pinned_env: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mechanism, isolated from `retrieve()`.

    Three readers, three different sections (`[retrieval]`,
    `[retrieval] expansion_gate_enabled`, `[implicit_feedback]`), one
    walk — because discovery is section-independent.
    """
    probes = _probe_counter(monkeypatch)
    discover_config()
    one_walk = len(probes)

    probes.clear()
    with config_discovery_scope():
        retrieval._read_toml_flag_for("use_bfs")
        expansion_gate._read_toml_flag()
        deferred_feedback._read_toml_value("enqueue_on_retrieve")
    assert len(probes) == one_walk, (
        f"{len(probes)} probes for 3 readers in one scope; one walk "
        f"costs {one_walk}"
    )


def test_outside_a_scope_each_converted_reader_still_walks(
    pinned_env: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No process-lifetime cache: unscoped callers keep old behaviour.

    Without this, a memo that leaked past its scope would satisfy every
    count assertion above while making a `.aelfrice.toml` created after
    the first call invisible until restart — the shape #1289 explicitly
    rejected.
    """
    probes = _probe_counter(monkeypatch)
    discover_config()
    one_walk = len(probes)

    probes.clear()
    expansion_gate._read_toml_flag()
    deferred_feedback._read_toml_value("enqueue_on_retrieve")
    assert len(probes) == 2 * one_walk, (
        f"{len(probes)} probes for 2 unscoped readers; expected two "
        f"independent walks of {one_walk}"
    )


def test_converted_readers_see_a_config_written_between_scopes(
    tmp_path: Path, pinned_env: None,
) -> None:
    """Staleness contract, exercised through the converted callers.

    `expansion_gate` and `deferred_feedback` now consult a memo they did
    not consult before, so the freshness property has to be re-asserted
    at *their* surface, not only at `retrieval`'s.
    """
    with config_discovery_scope():
        assert expansion_gate._read_toml_flag(tmp_path) is None
        assert deferred_feedback._read_toml_value(
            "enqueue_on_retrieve", start=tmp_path,
        ) is None

    (tmp_path / CONFIG_FILENAME).write_text(
        "[retrieval]\nexpansion_gate_enabled = false\n"
        "[implicit_feedback]\nenqueue_on_retrieve = true\n",
    )

    with config_discovery_scope():
        assert expansion_gate._read_toml_flag(tmp_path) is False
        assert deferred_feedback._read_toml_value(
            "enqueue_on_retrieve", start=tmp_path,
        ) is True


def test_distinct_start_dirs_are_distinct_memo_keys(tmp_path: Path) -> None:
    """A caller resolving from a *different* directory gets that answer.

    The hook resolves some config from the payload's cwd rather than the
    hook process's cwd (#909/#887). Those are different questions and
    the memo must not conflate them — which is also why one hook turn
    cannot be reduced below two walks.
    """
    near = tmp_path / "near"
    near.mkdir()
    (near / CONFIG_FILENAME).write_text("[retrieval]\nuse_bfs = true\n")
    far = tmp_path / "far"
    far.mkdir()

    with config_discovery_scope():
        assert discover_config(near) == near / CONFIG_FILENAME
        assert discover_config(far) != near / CONFIG_FILENAME
