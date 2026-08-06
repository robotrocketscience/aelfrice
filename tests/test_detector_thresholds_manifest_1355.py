"""The #1355 threshold manifest must track live source, and cover every writer.

Three independent failure modes, three independent arms:

1. **A constant drifts.** Someone edits ``DEFAULT_JACCARD_MIN`` and the
   manifest still records the old value.
   → ``test_pinned_value_matches_live_source``.

2. **The manifest is edited without a version bump.** Someone updates the
   manifest to match their new constant and ships, leaving
   ``DETECTOR_THRESHOLDS_VERSION`` at 1 — so two different edge-producing
   behaviours claim the same version.
   → ``test_manifest_digest_is_pinned`` +
   ``test_digest_history_is_contiguous_and_complete``, which together make
   the repair an append to ``DIGEST_HISTORY`` rather than an edit to one
   literal. Not airtight — a historical row can still be overwritten — and
   the module says so rather than overclaiming.

3. **A new writer lands unpinned.** Someone adds a module that writes a
   non-spine edge and never touches the manifest.
   → ``test_every_edge_writer_is_classified``.

The arms are deliberately not collapsible into one. #1353's lesson, and the
defect this issue was filed on, is the same: a test coupled to the *symbol*
rather than to the shipped *value* survives any change to what the symbol
resolves to. ``tests/test_relationship_detector.py`` asserts
``cfg.jaccard_min == DEFAULT_JACCARD_MIN``, which compares the constant to
itself and would stay green if the value became 0.9.
"""
from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

import aelfrice
from aelfrice.detector_thresholds import (
    COVERED_WRITER_MODULES,
    DETECTOR_THRESHOLDS_VERSION,
    EXCLUDED_WRITERS,
    KINDS,
    DIGEST_HISTORY,
    MANIFEST_DIGEST,
    PinnedThreshold,
    THRESHOLDS,
    manifest_digest,
    pin_value,
    size_of,
)

_PKG_ROOT = Path(aelfrice.__file__).resolve().parent

# Only real call sites: a leading dot means it is invoked on a store
# instance. The prose references in store.py and retrieval.py name the
# method without one, so they are not swept up.
_CALL_RE = re.compile(r"\.insert_edge\s*\(")
# A writer that reaches the table directly would bypass the call-site
# sweep entirely, so it is checked separately.
#
# `REPLACE INTO` has to be here, not as a nicety: `MemoryStore.insert_edge`
# issues a bare INSERT against a `PRIMARY KEY (src, dst, type)` table, so a
# module wanting an idempotent write reaches for REPLACE first — the one
# spelling most likely to be used is the one the first pattern could not see.
# The table name is also matched through a schema qualifier and the three
# quoting styles SQLite accepts; `store.py` already ships a qualified write
# (`INSERT INTO temp.fts ...`), so that form is house style rather than
# hypothetical. The trailing `\b` additionally stops `edges_backup` and
# `edge_versions` from reading as writes to `edges`.
_RAW_SQL_RE = re.compile(
    r"\b(?:INSERT(?:\s+OR\s+\w+)?|REPLACE)\s+INTO\s+(?:\w+\s*\.\s*)?[\"'`\[]?edges\b",
    re.IGNORECASE,
)


def _module_name(path: Path) -> str:
    rel = path.relative_to(_PKG_ROOT).with_suffix("")
    return "aelfrice." + ".".join(rel.parts)


def _source_files() -> list[Path]:
    return sorted(p for p in _PKG_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _edge_writer_modules() -> set[str]:
    """Every module calling ``insert_edge`` on a store, from source text."""
    found: set[str] = set()
    for path in _source_files():
        if _CALL_RE.search(path.read_text(encoding="utf-8")):
            found.add(_module_name(path))
    return found


# --- Arm 1: the manifest tracks live source ---------------------------


@pytest.mark.parametrize(
    "entry", THRESHOLDS, ids=[f"{t.module}.{t.name}" for t in THRESHOLDS],
)
def test_pinned_value_matches_live_source(entry: PinnedThreshold) -> None:
    """Every pinned value is re-derived from the live constant, not asserted
    against itself.

    The manifest holds hand-written literals and imports nothing from
    ``aelfrice``; this test does the importing. That asymmetry is the whole
    point — if the manifest imported the constants it describes, this
    comparison would be ``x == x``.

    Verified by mutation: changing ``DEFAULT_JACCARD_MIN`` from 0.4 to 0.9
    in source turns this arm red. It leaves arm 2 green — the digest covers
    the manifest, not live source, and the two arms are meant to catch
    different things. The same mutation leaves
    ``test_config_loader_overrides_and_falls_back`` green, which is the
    defect this file exists to close.
    """
    live = getattr(importlib.import_module(entry.module), entry.name)
    assert pin_value(live) == entry.value, (
        f"{entry.module}.{entry.name} moved: manifest records {entry.value}, "
        f"source now yields {pin_value(live)}. Update the manifest AND bump "
        f"DETECTOR_THRESHOLDS_VERSION."
    )
    assert size_of(live) == entry.size, (
        f"{entry.module}.{entry.name} changed size: manifest records "
        f"{entry.size}, source has {size_of(live)}."
    )


def test_scalar_entries_pin_a_literal_not_a_digest() -> None:
    """A scalar must stay readable as its literal.

    Guards the lazy repair path: a digest satisfies arm 1 just as well as a
    literal, so a failing scalar could be "fixed" by converting it to
    ``sha256:...`` — which would technically pass while destroying the
    reviewability the issue asked for ("a test asserting the literal
    shipped values").
    """
    for entry in THRESHOLDS:
        if entry.size is None and entry.kind in {"numeric_cutoff", "cap", "weight", "literal"}:
            assert not entry.value.startswith("sha256:"), (
                f"{entry.module}.{entry.name} is a scalar and must pin its "
                f"literal, not a digest"
            )


def test_pin_value_and_size_of_agree_on_what_a_scalar_is() -> None:
    """The two functions must not disagree, or a whole type becomes unpinnable.

    ``test_scalar_entries_pin_a_literal_not_a_digest`` above selects scalars
    by ``entry.size is None`` — i.e. by :func:`size_of` — and then demands
    :func:`pin_value` produced a literal. So the two have to classify the
    same values the same way. ``bool`` is the case where they can drift
    apart: it is the one type that is both a scalar and an ``int`` subclass,
    and excluding it from ``pin_value``'s scalar branch (as the first
    revision of this module did) made a hypothetical pinned boolean carry
    ``size=None`` and a ``sha256:`` value simultaneously — the exact pair
    that guard rejects. Digesting it bought nothing either: ``true`` and
    ``1`` are already distinct JSON.

    No boolean is pinned today, which is why this is asserted directly
    rather than left to the manifest to demonstrate.
    """
    for scalar in (True, False, 0, 1, 0.4, "v2"):
        assert size_of(scalar) is None, f"size_of says {scalar!r} is not a scalar"
        assert not pin_value(scalar).startswith("sha256:"), (
            f"pin_value digested {scalar!r}, but size_of classifies it as a "
            f"scalar — a manifest entry for it would fail "
            f"test_scalar_entries_pin_a_literal_not_a_digest with nothing wrong"
        )

    assert pin_value(True) == "true"
    assert pin_value(1) == "1", "bools and ints must still pin distinguishably"


def test_entries_are_wellformed_and_unique() -> None:
    seen: set[tuple[str, str]] = set()
    for entry in THRESHOLDS:
        key = (entry.module, entry.name)
        assert key not in seen, f"duplicate manifest entry: {key}"
        seen.add(key)
        assert entry.kind in KINDS, f"{key}: unknown kind {entry.kind!r}"
        assert entry.edge_types, f"{key}: must name the edge types it affects"
        assert entry.gates.strip(), f"{key}: must say what it gates"


# --- Arm 2: editing the manifest forces a version bump ----------------


def test_manifest_digest_is_pinned() -> None:
    """The pinned content must match the digest recorded for this version.

    The earlier shape of this test held a single hand-written
    ``MANIFEST_DIGEST`` literal, and it did NOT close failure mode 2 above
    — it only announced it. Editing a constant, its manifest entry and the
    digest literal returned the suite to green with the version untouched,
    which is precisely two behaviours shipping as version 1. The failure
    message even instructed that repair.

    Keyed by version, the cheap repair is gone: the ways back to green are
    to revert, or to bump the version and append a row to
    ``DIGEST_HISTORY``. Overwriting a historical row still works and is
    the honest limit of this mechanism — see the constant's comment. Only
    a merge-base check in CI is fully mechanical, and that is not built.
    """
    # Ordered deliberately. The module resolves MANIFEST_DIGEST with
    # `DIGEST_HISTORY.get(VERSION, "")` so that a missing row does not take
    # the import down with a KeyError — but subscripting DIGEST_HISTORY
    # here would reintroduce exactly that KeyError and report the contract
    # breach as a crash. Naming it first is what the `.get` is for.
    assert MANIFEST_DIGEST, (
        f"MANIFEST_DIGEST resolved to the empty-string fallback: "
        f"DIGEST_HISTORY has no row for version "
        f"{DETECTOR_THRESHOLDS_VERSION}. APPEND one — do not rewrite an "
        f"existing row."
    )
    assert manifest_digest() == MANIFEST_DIGEST, (
        f"pinned content does not match the digest recorded for version "
        f"{DETECTOR_THRESHOLDS_VERSION}. Revert the change, or bump "
        f"DETECTOR_THRESHOLDS_VERSION and APPEND a row to DIGEST_HISTORY. "
        f"Do not rewrite the existing row."
    )


def test_digest_history_is_contiguous_and_complete() -> None:
    """Every version from 1 to the current one has exactly one digest.

    Without this, bumping the version without appending a row raises
    KeyError in a place that reads like a crash rather than a contract
    breach, and a gap in the history would let a version be skipped to
    dodge a row.
    """
    assert set(DIGEST_HISTORY) == set(range(1, DETECTOR_THRESHOLDS_VERSION + 1)), (
        f"DIGEST_HISTORY keys {sorted(DIGEST_HISTORY)} are not exactly "
        f"1..{DETECTOR_THRESHOLDS_VERSION}"
    )
    assert len(set(DIGEST_HISTORY.values())) == len(DIGEST_HISTORY), (
        "two versions record the same content digest — one of them did "
        "not need a bump"
    )
    for version, digest in DIGEST_HISTORY.items():
        assert len(digest) == 64, f"version {version}: not a sha256 hex digest"


def test_version_is_a_positive_int() -> None:
    assert isinstance(DETECTOR_THRESHOLDS_VERSION, int)
    assert DETECTOR_THRESHOLDS_VERSION >= 1


# --- Arm 3: every writer is classified --------------------------------


def test_every_edge_writer_is_classified() -> None:
    """No module may write an edge without being either covered or excluded.

    Swept from source text rather than from a hand-list, so a new writer
    landing tomorrow fails here instead of silently sitting outside the
    manifest. This is the acceptance criterion the issue puts last and the
    easiest one to under-deliver: pinning ``relationship_detector`` alone
    would look complete while leaving the explicit-relation surface
    (``triple_extractor``) and the wonder lane unpinned.

    A writer belongs in ``EXCLUDED_WRITERS`` only with a stated reason —
    it writes a spine or ``DERIVED_FROM`` edge, it relays edges decided
    elsewhere, or it is a fixture builder rather than a detector.
    """
    excluded = {m for m, _ in EXCLUDED_WRITERS}
    classified = COVERED_WRITER_MODULES | excluded
    actual = _edge_writer_modules()

    unclassified = actual - classified
    assert not unclassified, (
        f"these modules call insert_edge but are neither pinned in "
        f"COVERED_WRITER_MODULES nor listed in EXCLUDED_WRITERS: "
        f"{sorted(unclassified)}"
    )

    vanished = classified - actual
    assert not vanished, (
        f"these modules are classified but no longer call insert_edge — "
        f"drop them from the manifest: {sorted(vanished)}"
    )


def test_covered_modules_all_have_entries() -> None:
    """A module in the covered set must actually contribute a threshold.

    Match is on the exact module, not a package prefix. Prefix matching is
    what let ``wonder.lifecycle`` look covered because ``wonder.evaluator``
    had entries — and ``evaluator`` is research-only, imported solely by
    the bake-off runner against an in-memory store, so it decides no edge
    in any user's store. Exact matching makes that substitution impossible.
    """
    pinned = {t.module for t in THRESHOLDS}
    for module in COVERED_WRITER_MODULES:
        assert module in pinned, (
            f"{module} is in COVERED_WRITER_MODULES but no manifest entry "
            f"names it. Matching by package prefix instead would let a "
            f"sibling module's entry stand in for it — which is how "
            f"wonder.lifecycle first looked covered by wonder.evaluator, a "
            f"research-only module that writes to no live store."
        )


def test_covered_and_excluded_do_not_overlap() -> None:
    excluded = {m for m, _ in EXCLUDED_WRITERS}
    assert not (COVERED_WRITER_MODULES & excluded)
    for module, reason in EXCLUDED_WRITERS:
        assert reason.strip(), f"{module} excluded without a reason"

    # `manifest_digest()` covers the version and THRESHOLDS, not the two
    # coverage lists, so moving a module from covered to excluded moves no
    # digest and no version. Without this, that move is silent AND leaves the
    # manifest self-contradictory: `test_covered_modules_all_have_entries`
    # stops applying to the module while its entries still sit in THRESHOLDS
    # claiming to gate edges the exclusion says it does not decide.
    pinned = {t.module for t in THRESHOLDS}
    assert not (pinned & excluded), (
        f"these modules are excluded as making no detection decision, yet "
        f"carry pinned thresholds that say otherwise: {sorted(pinned & excluded)}"
    )


def test_only_the_store_writes_the_edges_table_directly() -> None:
    """Raw SQL would bypass the call-site sweep arm 3 depends on.

    ``insert_edge`` is also where the #1254 ownership gate lives, so a
    module reaching the table directly would evade that too. Keeping the
    check here means arm 3 cannot be quietly defeated.
    """
    offenders = [
        _module_name(p)
        for p in _source_files()
        if _RAW_SQL_RE.search(p.read_text(encoding="utf-8"))
    ]
    assert offenders == ["aelfrice.store"], (
        f"only the store may write the edges table directly; found "
        f"{offenders}"
    )
