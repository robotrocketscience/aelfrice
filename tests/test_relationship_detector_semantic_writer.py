"""Unit tests for the #988 CONTRADICTS edge writer + ingest wiring.

Covers ``write_semantic_edges`` (high-confidence CONTRADICTS edges, the
complement of the sub-confidence ``write_potentially_stale_edges`` set),
the default-off ``auto_detect`` flag resolver, the determinism guarantee,
the per-belief write-gate, and the byte-identical off-path through
``ingest_turn``.

All tests use a real ``MemoryStore(":memory:")`` — no mocks.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CONTRADICTS,
    LOCK_NONE,
    Belief,
)
from aelfrice.relationship_detector import (
    DEFAULT_MAX_EDGES_PER_BELIEF,
    ENV_AUTO_RELATIONSHIPS,
    is_auto_relationship_detection_enabled,
    write_semantic_edges,
)
from aelfrice.store import MemoryStore

# High-confidence contradiction: "always X" vs "never X" — universal
# affirmation vs negation over identical residual content → score 1.0.
_ALWAYS = "the deployment script always runs the database migration step"
_NEVER = "the deployment script never runs the database migration step"
# Lexically distant filler that relates to nothing else.
_UNRELATED = "the harbor seals bask on the warm rocks at noon each day"
# Sub-confidence contradiction: "always X" vs "rarely X" — adjacent
# quantifier axes over identical residual content -> score 0.4, i.e.
# below the module default confidence_min (0.5) and above 0.3.
_RARELY = "the deployment script rarely runs the database migration step"


def _make_belief(
    store: MemoryStore,
    *,
    belief_id: str,
    content: str,
    created_at: str = "2026-01-01T00:00:00Z",
) -> Belief:
    b = Belief(
        id=belief_id,
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at=created_at,
        last_retrieved_at=None,
    )
    store.insert_belief(b)
    return b


def _contradicts_edges(store: MemoryStore) -> list[tuple[str, str]]:
    """Return all CONTRADICTS edges as sorted (src, dst) tuples."""
    rows = store._conn.execute(  # type: ignore[attr-defined]
        "SELECT src, dst FROM edges WHERE type = ? ORDER BY src, dst",
        (EDGE_CONTRADICTS,),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(":memory:")


# ---------------------------------------------------------------------------
# Flag resolver precedence
# ---------------------------------------------------------------------------


def test_flag_defaults_off(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.delenv(ENV_AUTO_RELATIONSHIPS, raising=False)
    # start at an empty dir so no repo .aelfrice.toml is found
    assert is_auto_relationship_detection_enabled(start=tmp_path) is False


def test_flag_env_wins_over_kwarg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_AUTO_RELATIONSHIPS, "off")
    assert is_auto_relationship_detection_enabled(explicit=True) is False
    monkeypatch.setenv(ENV_AUTO_RELATIONSHIPS, "on")
    assert is_auto_relationship_detection_enabled(explicit=False) is True


def test_flag_unrecognised_env_is_not_decisive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_AUTO_RELATIONSHIPS, "maybe")
    assert is_auto_relationship_detection_enabled(explicit=True) is True


def test_flag_toml_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.delenv(ENV_AUTO_RELATIONSHIPS, raising=False)
    (tmp_path / ".aelfrice.toml").write_text(
        "[relationship_detector]\nauto_detect = true\n"
    )
    assert is_auto_relationship_detection_enabled(start=tmp_path) is True


# ---------------------------------------------------------------------------
# write_semantic_edges — write / scope / idempotency
# ---------------------------------------------------------------------------


def test_writes_contradicts_edge_for_high_confidence_pair(
    store: MemoryStore,
) -> None:
    _make_belief(store, belief_id="b1", content=_ALWAYS)
    _make_belief(store, belief_id="b2", content=_NEVER)
    report = write_semantic_edges(store)
    assert report.n_contradicts_high == 1
    assert report.n_edges_written == 1
    # Canonical direction: src = min(id), dst = max(id).
    assert _contradicts_edges(store) == [("b1", "b2")]


def test_idempotent_second_run_writes_nothing(store: MemoryStore) -> None:
    _make_belief(store, belief_id="b1", content=_ALWAYS)
    _make_belief(store, belief_id="b2", content=_NEVER)
    write_semantic_edges(store)
    report = write_semantic_edges(store)
    assert report.n_edges_written == 0
    assert report.n_edges_skipped_existing == 1
    assert _contradicts_edges(store) == [("b1", "b2")]


def test_unrelated_pair_writes_no_edge(store: MemoryStore) -> None:
    _make_belief(store, belief_id="b1", content=_ALWAYS)
    _make_belief(store, belief_id="b2", content=_UNRELATED)
    report = write_semantic_edges(store)
    assert report.n_edges_written == 0
    assert _contradicts_edges(store) == []


def test_agreeing_modality_is_refines_not_contradicts(
    store: MemoryStore,
) -> None:
    # Two "never" statements over the same residual content agree in
    # modality → refines, which this CONTRADICTS-only writer ignores.
    _make_belief(store, belief_id="b1", content=_NEVER)
    _make_belief(
        store,
        belief_id="b2",
        content=_NEVER + " before launch",
    )
    report = write_semantic_edges(store)
    assert report.n_edges_written == 0
    assert _contradicts_edges(store) == []


# ---------------------------------------------------------------------------
# Determinism — byte-equal edge sets across two fresh stores
# ---------------------------------------------------------------------------


def test_determinism_byte_equal_edges() -> None:
    def build() -> list[tuple[str, str]]:
        s = MemoryStore(":memory:")
        _make_belief(s, belief_id="b1", content=_ALWAYS)
        _make_belief(s, belief_id="b2", content=_NEVER)
        _make_belief(s, belief_id="b3", content=_UNRELATED)
        write_semantic_edges(s)
        return _contradicts_edges(s)

    assert build() == build()


# ---------------------------------------------------------------------------
# Write-gate — per-belief edge cap (Exp-48 dilution guard)
# ---------------------------------------------------------------------------


def test_write_gate_caps_per_belief_edges(store: MemoryStore) -> None:
    # b1 contradicts both b2 and b3 (both negate b1's universal claim).
    # b2 and b3 agree with each other (both "never") → no edge between them.
    _make_belief(store, belief_id="b1", content=_ALWAYS)
    _make_belief(store, belief_id="b2", content=_NEVER)
    _make_belief(
        store, belief_id="b3", content=_NEVER + " before each launch"
    )
    report = write_semantic_edges(store, max_edges_per_belief=1)
    # b1 may only accrue one CONTRADICTS edge; the second pair is gated.
    assert report.n_edges_written == 1
    assert report.n_edges_skipped_gated == 1
    edges = _contradicts_edges(store)
    assert len(edges) == 1
    # The surviving edge is the first in deterministic (a_id, b_id) order.
    assert edges == [("b1", "b2")]


def test_default_cap_is_positive() -> None:
    assert DEFAULT_MAX_EDGES_PER_BELIEF >= 1


# ---------------------------------------------------------------------------
# Ingest wiring — off-path byte-identical, on-path writes
# ---------------------------------------------------------------------------


def test_ingest_off_path_writes_no_contradicts_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aelfrice.ingest import ingest_turn

    monkeypatch.delenv(ENV_AUTO_RELATIONSHIPS, raising=False)
    s = MemoryStore(":memory:")
    ingest_turn(s, _ALWAYS, source="t", session_id="sess")
    ingest_turn(s, _NEVER, source="t", session_id="sess")
    assert _contradicts_edges(s) == []


def test_ingest_on_path_writes_contradicts_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aelfrice.ingest import ingest_turn

    monkeypatch.setenv(ENV_AUTO_RELATIONSHIPS, "1")
    s = MemoryStore(":memory:")
    ingest_turn(s, _ALWAYS, source="t", session_id="sess")
    ingest_turn(s, _NEVER, source="t", session_id="sess")
    assert len(_contradicts_edges(s)) == 1


# ---------------------------------------------------------------------------
# #1299 — [relationship_detector] thresholds reach the ingest write path
# ---------------------------------------------------------------------------


def _ingest_pair_under_toml(toml_body: str, tmp_path: Path) -> list[
    tuple[str, str]
]:
    """Ingest the ALWAYS/NEVER pair with `tmp_path/.aelfrice.toml` in scope."""
    from aelfrice.ingest import ingest_turn

    (tmp_path / ".aelfrice.toml").write_text(toml_body)
    s = MemoryStore(":memory:")
    ingest_turn(s, _ALWAYS, source="t", session_id="sess")
    ingest_turn(s, _NEVER, source="t", session_id="sess")
    return _contradicts_edges(s)


def test_ingest_honours_toml_jaccard_min(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """`jaccard_min` from TOML gates the ingest write path (#1299).

    The pair's token Jaccard is ~0.78, so a threshold above it must
    suppress the edge and a threshold below it must let it through.
    Before #1299 ingest called ``write_semantic_edges`` with no threshold
    arguments, so both arms wrote the edge and the key was inert here.
    """
    monkeypatch.delenv(ENV_AUTO_RELATIONSHIPS, raising=False)
    monkeypatch.chdir(tmp_path)

    # Threshold ABOVE the pair's overlap -> prefilter drops it, no edge.
    assert _ingest_pair_under_toml(
        "[relationship_detector]\nauto_detect = true\njaccard_min = 0.95\n",
        tmp_path,
    ) == []

    # Positive control: same corpus, threshold BELOW the overlap -> edge.
    # Without this arm the assertion above would pass vacuously.
    assert len(_ingest_pair_under_toml(
        "[relationship_detector]\nauto_detect = true\njaccard_min = 0.1\n",
        tmp_path,
    )) == 1


def test_ingest_honours_toml_confidence_min(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """`confidence_min` from TOML gates the ingest write path (#1299)."""
    monkeypatch.delenv(ENV_AUTO_RELATIONSHIPS, raising=False)
    monkeypatch.chdir(tmp_path)

    # ALWAYS vs RARELY: quantifier axes are near, so the pair scores 0.4 —
    # below the module default 0.5 and above a configured 0.3.
    def edges_at(floor: str) -> list[tuple[str, str]]:
        from aelfrice.ingest import ingest_turn

        (tmp_path / ".aelfrice.toml").write_text(
            "[relationship_detector]\n"
            "auto_detect = true\n"
            f"confidence_min = {floor}\n"
        )
        s = MemoryStore(":memory:")
        ingest_turn(s, _ALWAYS, source="t", session_id="sess")
        ingest_turn(s, _RARELY, source="t", session_id="sess")
        return _contradicts_edges(s)

    # At the module default the pair is sub-confidence -> no edge.
    assert edges_at("0.5") == []
    # Lowering the floor in TOML must let it through. Before #1299 ingest
    # passed no confidence_min, so this arm stayed empty.
    assert len(edges_at("0.3")) == 1


def test_ingest_threads_max_candidate_pairs_from_toml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """`max_candidate_pairs` reaches the writer's call site (#1299).

    Unlike the two threshold keys this one has no small-corpus outcome to
    observe — a two-belief store never approaches any sane pair budget —
    so assert the value that arrives at ``write_semantic_edges`` instead.
    """
    from aelfrice import relationship_detector as rd
    from aelfrice.ingest import ingest_turn

    monkeypatch.delenv(ENV_AUTO_RELATIONSHIPS, raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".aelfrice.toml").write_text(
        "[relationship_detector]\n"
        "auto_detect = true\n"
        "jaccard_min = 0.55\n"
        "confidence_min = 0.65\n"
        "max_candidate_pairs = 17\n"
    )

    seen: list[dict[str, object]] = []
    real_writer = rd.write_semantic_edges

    def spy(store: MemoryStore, **kwargs: object) -> object:
        seen.append(dict(kwargs))
        return real_writer(store, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(rd, "write_semantic_edges", spy)
    s = MemoryStore(":memory:")
    ingest_turn(s, _ALWAYS, source="t", session_id="sess")
    ingest_turn(s, _NEVER, source="t", session_id="sess")

    assert seen, "write_semantic_edges was never called on the on-path"
    for call in seen:
        assert call["jaccard_min"] == 0.55
        assert call["confidence_min"] == 0.65
        assert call["max_candidate_pairs"] == 17


# ---------------------------------------------------------------------------
# #1299 — the fix must not add a second config walk to the ingest hot path
# ---------------------------------------------------------------------------


def _count_config_probes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Install a counting `Path.is_file` and return its mutable counter."""
    counter = [0]
    real_is_file = Path.is_file

    def counting_is_file(self: Path) -> bool:
        if self.name == ".aelfrice.toml":
            counter[0] += 1
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", counting_is_file)
    return counter


def test_resolve_ingest_config_walks_the_tree_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """One `.aelfrice.toml` walk per resolve, not two (#1289/#1298).

    ``resolve_ingest_relationship_config`` runs on every ingested turn.
    Calling ``load_relationship_detector_config()`` and
    ``is_auto_relationship_detection_enabled()`` independently would probe
    each ancestor directory twice; the flag must be handed over from the
    config read instead.
    """
    from aelfrice.relationship_detector import (
        load_relationship_detector_config,
        resolve_ingest_relationship_config,
    )

    monkeypatch.delenv(ENV_AUTO_RELATIONSHIPS, raising=False)
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (tmp_path / ".aelfrice.toml").write_text(
        "[relationship_detector]\nauto_detect = true\njaccard_min = 0.55\n"
    )

    counter = _count_config_probes(monkeypatch)
    load_relationship_detector_config(start=deep)
    baseline = counter[0]
    assert baseline > 0

    counter[0] = 0
    enabled, config = resolve_ingest_relationship_config(start=deep)
    assert counter[0] == baseline
    assert enabled is True
    assert config.jaccard_min == 0.55


def test_resolve_ingest_config_env_off_probes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """`AELFRICE_AUTO_RELATIONSHIPS=0` costs zero config probes (#1289).

    The flag-only resolver this call site replaced checked env before
    touching the filesystem, so an env-disabled install walked no
    directories at all per ingested turn. Loading the config first would
    silently move that to a full walk to root — on a path that runs every
    turn, for a config whose only consumer is a writer that will not run.

    Both arms are asserted so neither passes vacuously: env-off must be
    0 while env-unset over the *same* tree is non-zero.
    """
    from aelfrice.relationship_detector import (
        resolve_ingest_relationship_config,
    )

    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    counter = _count_config_probes(monkeypatch)

    monkeypatch.setenv(ENV_AUTO_RELATIONSHIPS, "0")
    counter[0] = 0
    assert resolve_ingest_relationship_config(start=deep)[0] is False
    assert counter[0] == 0

    monkeypatch.delenv(ENV_AUTO_RELATIONSHIPS, raising=False)
    counter[0] = 0
    assert resolve_ingest_relationship_config(start=deep)[0] is False
    assert counter[0] > 0


def test_resolve_ingest_config_env_still_wins_over_toml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Env keeps precedence over `auto_detect` in both directions (#1299)."""
    from aelfrice.relationship_detector import (
        DEFAULT_JACCARD_MIN,
        resolve_ingest_relationship_config,
    )

    (tmp_path / ".aelfrice.toml").write_text(
        "[relationship_detector]\nauto_detect = true\njaccard_min = 0.55\n"
    )
    monkeypatch.setenv(ENV_AUTO_RELATIONSHIPS, "off")
    enabled, config = resolve_ingest_relationship_config(start=tmp_path)
    assert enabled is False
    # Env-off short-circuits before the config walk, so the returned
    # config is the module defaults, not the file's 0.55. Documented
    # contract: the config is meaningful only when `enabled` is True.
    assert config.jaccard_min == DEFAULT_JACCARD_MIN

    monkeypatch.setenv(ENV_AUTO_RELATIONSHIPS, "on")
    (tmp_path / ".aelfrice.toml").write_text(
        "[relationship_detector]\nauto_detect = false\n"
    )
    enabled, _ = resolve_ingest_relationship_config(start=tmp_path)
    assert enabled is True


def test_config_loader_reads_auto_detect(tmp_path: Path) -> None:
    """`auto_detect` is parsed onto the config object (#1299)."""
    from aelfrice.relationship_detector import (
        load_relationship_detector_config,
    )

    assert load_relationship_detector_config(start=tmp_path).auto_detect is False
    (tmp_path / ".aelfrice.toml").write_text(
        "[relationship_detector]\nauto_detect = true\n"
    )
    assert load_relationship_detector_config(start=tmp_path).auto_detect is True
    (tmp_path / ".aelfrice.toml").write_text(
        "[relationship_detector]\nauto_detect = \"yes\"\n"
    )
    assert load_relationship_detector_config(start=tmp_path).auto_detect is False
