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

import importlib
import os
import pkgutil
import sys
from pathlib import Path

import pytest

import aelfrice.config_discovery as config_discovery
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


def _config_filename_reexports() -> dict[str, str]:
    """Every `aelfrice.*` module still binding a config-filename name.

    Enumerated, not listed. The three-module list this replaces was
    written when three modules had been converted and then silently
    stopped describing the set — the same failure mode #1304's own perf
    harness had, and the reason the shared constant exists at all.

    Both spellings are collected. The public `CONFIG_FILENAME` is a
    documented back-compat surface; the private one is scanned so that a
    module reintroducing `_CONFIG_FILENAME` as a second literal is caught
    the same way, rather than being invisible for being underscored.
    """
    package = sys.modules[config_discovery.__package__]
    found: dict[str, str] = {}
    for info in pkgutil.iter_modules(package.__path__):
        if info.name == "config_discovery":
            continue
        try:
            module = importlib.import_module(f"aelfrice.{info.name}")
        except Exception:  # noqa: BLE001 - an unimportable extra is not ours
            continue
        for attr in ("CONFIG_FILENAME", "_CONFIG_FILENAME"):
            value = getattr(module, attr, None)
            if isinstance(value, str) and value.endswith(".toml"):
                found[f"{info.name}.{attr}"] = value
    return found


def test_back_compat_config_filename_reexports() -> None:
    """No module re-declares the config filename as a second literal.

    #1304 deleted the local `Final` constant from `expansion_gate` and
    `deferred_feedback` and moved `retrieval`'s to the shared module. The
    names stay bound so `from aelfrice.expansion_gate import
    CONFIG_FILENAME` keeps resolving, and they are bound by assignment
    rather than by a bare import so a dead-import check cannot delete
    the re-export and silently break those callers. Same object in every
    case, so the walk and the filename it walks for cannot drift.

    The drift this pins is not hypothetical in one direction only: a
    module that keeps its own literal is written under one name by a test
    fixture and looked for under another by the loader the moment the
    shared constant changes, and nothing else reports it.
    """
    reexports = _config_filename_reexports()
    # A scan that finds nothing passes every assertion below, so the
    # known-present names are the floor. The first three are the
    # documented back-compat surface; `cadence` is bound because twelve
    # test modules build their fixture from it.
    for expected in (
        "retrieval.CONFIG_FILENAME",
        "expansion_gate.CONFIG_FILENAME",
        "deferred_feedback.CONFIG_FILENAME",
        "cadence.CONFIG_FILENAME",
    ):
        assert expected in reexports, (
            f"{expected} is no longer bound; callers import it, and the "
            "enumeration below is vacuous without it"
        )
    for name, value in sorted(reexports.items()):
        assert value == CONFIG_FILENAME, (
            f"{name} drifted from the shared constant "
            f"({value!r} != {CONFIG_FILENAME!r})"
        )
    assert not hasattr(retrieval, "_CONFIG_DISCOVERY_MEMO"), (
        "retrieval re-exports the memo ContextVar again. It is private to "
        "config_discovery and unread here, so the re-export is dead weight "
        "a static analyser correctly flags; read it from config_discovery."
    )


def test_onboard_config_resolves_the_same_whether_the_target_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`aelf onboard <path>` reads one config, existing target or not.

    This is a real behaviour change from #1304 and not a no-op, so it is
    pinned rather than left to be rediscovered. `cli._load_llm_config`
    used to start its private walk at `root.resolve() if root.exists()
    else root` — an unresolved relative target made the walk start at
    `Path(".")`, whose `.parent` is itself, so it stopped immediately and
    every ancestor config was invisible. An existing target resolved and
    walked normally. `discover_config` resolves unconditionally, so the
    two cases now agree.

    They agree in the direction that keeps a project's config honoured:
    `[onboard.llm] enabled = false` in an ancestor governs `aelf onboard
    typo-dir` the same way it governs `aelf onboard real-dir`. The reverse
    reading — that the old lexical start was load-bearing — does not hold,
    because it was already not what an existing path did.
    """
    from aelfrice import cli

    project = tmp_path / "project"
    (project / "child").mkdir(parents=True)
    (project / ".aelfrice.toml").write_text(
        "[onboard.llm]\nenabled = false\nmodel = \"from-ancestor\"\n",
    )
    monkeypatch.chdir(project / "child")
    (project / "child" / "real-dir").mkdir()

    existing = cli._load_llm_config(Path("real-dir"))
    missing = cli._load_llm_config(Path("typo-dir"))

    assert existing.enabled is False, (
        "an existing relative target stopped reading the ancestor config"
    )
    assert (missing.enabled, missing.model) == (existing.enabled, existing.model), (
        "a target that does not exist resolves a different config than one "
        f"that does: {missing} vs {existing}"
    )


def test_scope_binds_start_none_to_the_cwd_at_first_call(
    tmp_path: Path,
) -> None:
    """The documented invariant: no caller may `os.chdir` inside a scope.

    `start=None` is memoized on the cwd read at the scope's first such
    call and is deliberately not re-read, because re-reading costs
    `Path.cwd().resolve()` — O(path depth) in `lstat` — on each of the
    ~26 `start=None` calls in one `retrieve()`. This pins both halves:
    inside a scope the first answer is reused, and outside one the new
    directory is seen immediately, so the staleness cannot outlive the
    operation.
    """
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (b / CONFIG_FILENAME).write_text("[retrieval]\nuse_bfs = true\n")

    saved = os.getcwd()
    try:
        os.chdir(a)
        with config_discovery_scope():
            assert discover_config() is None
            os.chdir(b)
            # Bound to `a` — this is the invariant, and why callers must
            # not chdir inside a scope.
            assert discover_config() is None
        # Outside the scope the memo is gone and `b` is seen at once.
        assert discover_config() == b / CONFIG_FILENAME
    finally:
        os.chdir(saved)


def test_no_aelfrice_module_starts_a_thread_or_task() -> None:
    """Keeps the memo's concurrency caveat latent rather than live.

    A `ContextVar` set inside a scope is *copied* into an
    `asyncio.create_task` child, which then keeps using the memo after
    the scope exits. `threading.Thread` gets a fresh context and is
    safe. Nothing in `aelfrice` creates either today, which is what
    makes the asyncio case a documented caveat instead of a defect —
    so assert it, rather than leaving the claim in a comment where it
    can quietly go false.
    """
    import ast

    src = Path(retrieval.__file__).parent
    offenders: list[str] = []
    for path in sorted(src.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                root = name.split(".")[0]
                if root in {"asyncio"} or name.startswith("concurrent.futures"):
                    offenders.append(f"{path.name}: imports {name}")
    assert not offenders, (
        "a module now creates async tasks; a task started inside a "
        "config_discovery_scope inherits the memo and outlives it. "
        "Enter a fresh scope in the task instead of inheriting: "
        + "; ".join(offenders)
    )
