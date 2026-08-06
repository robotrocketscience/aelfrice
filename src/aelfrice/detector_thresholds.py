"""Frozen, versioned record of the constants that decide the non-spine edge set.

#1283 restated AC2 in two halves. The recompute half — the deterministic
replay of the ``TEMPORAL_NEXT`` spine keyed on ``(created_at, ingest_log
ULID)`` — shipped in #1336. This module is the other half (#1355).

**Why it exists.** The edges that are *not* ``TEMPORAL_NEXT`` and not
``DERIVED_FROM`` (2.8% of the table: ``CONTRADICTS``, ``SUPERSEDES``,
``RELATES_TO``, ``TESTS``, ``CITES``, ``SUPPORTS``, ``IMPLEMENTS``,
``RESOLVES``, ``POTENTIALLY_STALE``) are a function of the belief set
**and** of detector thresholds. "Edges are recomputable" is therefore only
true if those thresholds are pinned and versioned. Before this module they
were bare module constants that could drift silently: the tests that
nominally covered them compared a constant to itself
(``assert cfg.jaccard_min == DEFAULT_JACCARD_MIN``), which survives any
change to what the symbol resolves to.

**What this module is.** A hand-written record of the shipped values, held
as *literals*. It deliberately imports nothing from ``aelfrice`` — if it
imported the constants it would describe, it would be tautological in
exactly the way the tests it replaces were, and it would drag
``store``/``bm25`` into the import graph of anything that reads the
manifest. The comparison against live source is done by
``tests/test_detector_thresholds_manifest_1355.py``, which imports each
constant by name and re-derives its pinned form with :func:`pin_value`.

**Forward-only.** Pinning forward does not make the past reproducible: the
``edges`` table carries no version and no ``created_at``, so a historical
edge cannot be attributed to the thresholds that produced it. Adding those
columns is an ``edges``-table migration — the operation that left stores
unopenable-forever in #1161 — and historical reproduction is explicitly
out of scope here. What this buys is that from ``DETECTOR_THRESHOLDS_VERSION
= 1`` onward, a change to any listed value makes :data:`DIGEST_HISTORY`
disagree with the pinned content, so landing it means either bumping the
version and appending a row or visibly rewriting a historical one. See
that constant for what this does and does not enforce.

**Honesty about overrides.** Several entries are defaults that a config
file or environment variable can override at runtime — ``jaccard_min``,
``confidence_min`` and ``max_candidate_pairs`` are readable from
``[relationship_detector]`` in ``.aelfrice.toml``. For those, the manifest
pins the *shipped default*, not the value a given store actually ran with.
That is a real limit and the ``overridable`` field on every entry names the
mechanism, so a reader can tell the two apart rather than over-trusting the
record.

**What this cannot pin.** Entries are resolved by ``(module, name)``, so a
value with no name is out of reach. Three writers stamp an edge weight as a
literal inside the ``Edge(...)`` constructor — ``relationship_detector``
(both writers), ``triple_extractor`` and ``wonder.lifecycle`` all pass
``weight=1.0`` inline. ``contradiction`` is the one that names it
(``SUPERSEDES_WEIGHT``), which is why it appears below and the others do
not. Naming those three would be a behaviour-preserving refactor and would
close the gap; it is deliberately not bundled into this change. Constants
reached only transitively are a different case and *are* covered:
``_QUANTIFIER_TOKENS`` is derived from ``QUANT_AXIS``, and the noun-phrase
regex fragments are compiled into ``_PATTERNS``, so both move their
digests already.

**Writers versus suppliers.** The coverage test sweeps ``insert_edge`` call
sites, so it can only see modules that *write*. A module that merely
*supplies the decision* is invisible to it and has to be added by hand.
Three such are pinned below — ``bm25._TOKEN_PATTERN``,
``models.ANCHOR_TEXT_MAX_LEN`` and ``wonder_consolidation._TOKENIZER_DROP``
— because each moves edges in every writer downstream of it. Two are known
and deliberately left out: ``dedup`` supplies the candidate-pair prefilter
whose semantics differ from the detector's own (empty-versus-empty scores
1.0 there and 0.0 here, and blank-content beliefs are skipped before
scoring), and ``config_discovery`` decides *which* ``.aelfrice.toml``
supplies the overridable values above. Both are behavioural surfaces
rather than constants; pinning them means pinning functions, which this
mechanism does not do. They are named here so their absence is a recorded
decision rather than an oversight. So is ``--axes-budget`` (default 24 on
``analyze_gaps`` / ``build_dispatch_payload``), which caps the anchor tuple
and therefore how many ``RELATES_TO`` edges each persisted phantom writes:
a bigger lever than several constants pinned below, but a signature default
rather than a module constant, so it is out of reach of the ``(module,
name)`` scheme for the same reason the inline weights are.

**Pinning these is necessary, not sufficient.** Two stores with identical
belief sets and every value below at its manifest reading can still hold
different ``CONTRADICTS`` edges, because ``DEFAULT_MAX_EDGES_PER_BELIEF`` is
consumed in sorted pair order and the shipped path is incremental — the cap
is spent on whichever pairs arrived first. A full-store
``write_semantic_edges()`` and the incremental
``write_semantic_edges(new_belief_ids=[…])`` can therefore disagree on the
same beliefs. So belief arrival ORDER is a third input alongside the belief
set and these thresholds, and anyone re-deriving edges from beliefs plus
this manifest will see a false mismatch on any incrementally built store —
which is every real one.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, fields, is_dataclass
from typing import TYPE_CHECKING, Any, Final, cast

if TYPE_CHECKING:  # pragma: no cover - typing only
    # Used only inside a string-literal `cast`, so the name is never
    # evaluated at runtime and importing it unconditionally reads as dead.
    from collections.abc import Sequence

# Bump when any pinned value below changes, and append the new content
# digest to DIGEST_HISTORY at the bottom of this module. The guard in
# tests/test_detector_thresholds_manifest_1355.py fails until you do.
DETECTOR_THRESHOLDS_VERSION: Final[int] = 1

# --- Kinds -------------------------------------------------------------

KIND_CUTOFF: Final[str] = "numeric_cutoff"
KIND_CAP: Final[str] = "cap"
KIND_WEIGHT: Final[str] = "weight"
KIND_PATTERN_TABLE: Final[str] = "pattern_table"
KIND_TOKEN_SET: Final[str] = "token_set"
# A bare string constant that is neither a cutoff nor a collection.
KIND_LITERAL: Final[str] = "literal"

KINDS: Final[frozenset[str]] = frozenset({
    KIND_CUTOFF,
    KIND_CAP,
    KIND_WEIGHT,
    KIND_PATTERN_TABLE,
    KIND_TOKEN_SET,
    KIND_LITERAL,
})

# Override mechanisms, as strings so the manifest stays literal.
OVERRIDE_NONE: Final[str] = "no"
OVERRIDE_TOML: Final[str] = "toml:[relationship_detector]"
OVERRIDE_KWARG: Final[str] = "kwarg"


# --- Canonicalisation --------------------------------------------------


def _canonical(obj: Any) -> Any:
    """Reduce a constant to a JSON-able form with a stable ordering.

    Sets and dicts are order-unstable across builds, so both are sorted.
    Compiled patterns reduce to ``(pattern, flags)`` — the flags matter:
    dropping ``re.IGNORECASE`` changes which triples match, and therefore
    which edges get written, without touching the pattern text.
    """
    if isinstance(obj, re.Pattern):
        pattern = cast("re.Pattern[str]", obj)
        return {"regex": pattern.pattern, "flags": int(pattern.flags)}
    if is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: _canonical(getattr(obj, f.name))
            for f in sorted(fields(obj), key=lambda f: f.name)
        }
    if isinstance(obj, (frozenset, set)):
        members = cast("frozenset[Any]", obj)
        return sorted(_canonical(x) for x in members)
    if isinstance(obj, dict):
        mapping = cast("dict[str, Any]", obj)
        return [[k, _canonical(v)] for k, v in sorted(mapping.items())]
    if isinstance(obj, (tuple, list)):
        seq = cast("Sequence[Any]", obj)
        return [_canonical(x) for x in seq]
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    raise TypeError(f"not canonicalisable: {type(obj).__name__}")


def _dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def pin_value(obj: Any) -> str:
    """Return the pinned form of a live constant.

    Scalars pin as their literal, so the manifest stays readable and a
    reviewer can check ``0.4`` against the source line by eye. Collections
    pin as a digest of their canonical form — a stopword set is not
    reviewable inline, but its digest changes the moment a token is added
    or removed, which is the property the manifest needs.

    ``bool`` counts as a scalar here, and deliberately so. It is not
    ambiguous with ``int`` — ``json.dumps`` renders ``True`` as ``true``
    and ``1`` as ``1`` — so digesting it buys no discrimination and costs
    readability. Excluding it also put this function at odds with
    :func:`size_of`, which reports ``None`` (scalar) for a bool: a pinned
    boolean constant would have carried ``size=None`` and a ``sha256:``
    value at once, which is precisely the combination
    ``test_scalar_entries_pin_a_literal_not_a_digest`` rejects. No boolean
    is pinned today; the point is that one now can be.
    """
    if not isinstance(obj, (int, float, str)):
        return "sha256:" + hashlib.sha256(
            _dumps(_canonical(obj)).encode("utf-8")
        ).hexdigest()
    return _dumps(obj)


def size_of(obj: Any) -> int | None:
    """Element count for a collection; ``None`` for a scalar.

    Recorded alongside the digest because a digest mismatch alone says
    "something moved" — the count says whether the set grew or shrank,
    which is usually the first thing a reviewer wants to know.
    """
    if isinstance(obj, (str, bytes)) or isinstance(obj, re.Pattern):
        return None
    if isinstance(obj, bool) or isinstance(obj, (int, float)):
        return None
    try:
        return len(obj)
    except TypeError:
        return None


# --- The manifest ------------------------------------------------------


@dataclass(frozen=True)
class PinnedThreshold:
    """One pinned constant behind a non-spine edge write.

    ``module``/``name``
        Import path and attribute, resolved by the test to reach the live
        object. Private names (leading underscore) are included: the
        detector's behaviour does not care about the convention, and
        ``_STOPWORDS`` moves the residual-overlap score as surely as
        ``DEFAULT_RESIDUAL_OVERLAP_MIN`` does.

    ``value``
        Output of :func:`pin_value` — a literal for scalars, a
        ``sha256:`` digest for collections.

    ``edge_types``
        Which edge types this constant can change. Informational; the
        coverage test keys on the writer set, not on this field.

    ``gates``
        What observably changes about the written edge set when the value
        moves. A constant that cannot answer this does not belong here.
    """

    module: str
    name: str
    kind: str
    value: str
    size: int | None
    edge_types: tuple[str, ...]
    overridable: str
    gates: str


THRESHOLDS: Final[tuple[PinnedThreshold, ...]] = (
    # --- relationship_detector: CONTRADICTS + POTENTIALLY_STALE --------
    #
    # Two writers share this threshold set and partition the contradicting
    # pairs by score: `write_semantic_edges` takes score >= confidence_min
    # and emits CONTRADICTS, `write_potentially_stale_edges` takes the
    # sub-confidence half and emits POTENTIALLY_STALE. A change to
    # confidence_min therefore does not just resize one edge population,
    # it moves pairs from one edge type to the other.
    PinnedThreshold(
        module="aelfrice.relationship_detector",
        name="DEFAULT_JACCARD_MIN",
        kind=KIND_CUTOFF,
        value="0.4",
        size=None,
        edge_types=("CONTRADICTS", "POTENTIALLY_STALE"),
        overridable=OVERRIDE_TOML,
        gates=(
            "Token-overlap floor a pair must clear to enter the "
            "classifier at all. Lowering it enlarges the candidate pool "
            "monotonically, but the written edge set does NOT move "
            "monotonically with it: `DEFAULT_MAX_EDGES_PER_BELIEF` is "
            "consumed in sorted pair order, so a newly-admitted pair can "
            "evict one that was previously written."
        ),
    ),
    PinnedThreshold(
        module="aelfrice.relationship_detector",
        name="DEFAULT_RESIDUAL_OVERLAP_MIN",
        kind=KIND_CUTOFF,
        value="0.4",
        size=None,
        edge_types=("CONTRADICTS", "POTENTIALLY_STALE"),
        overridable=OVERRIDE_KWARG,
        gates=(
            "Residual-content overlap floor. Below it `analyze` returns "
            "`unrelated` with score 0.0 regardless of modality signals, "
            "so no edge of either type is written for the pair."
        ),
    ),
    PinnedThreshold(
        module="aelfrice.relationship_detector",
        name="DEFAULT_CONFIDENCE_MIN",
        kind=KIND_CUTOFF,
        value="0.5",
        size=None,
        edge_types=("CONTRADICTS", "POTENTIALLY_STALE"),
        overridable=OVERRIDE_TOML,
        gates=(
            "The split point between the two writers, not a simple "
            "on/off: a contradicting pair at or above it becomes a "
            "CONTRADICTS edge, below it a POTENTIALLY_STALE edge."
        ),
    ),
    PinnedThreshold(
        module="aelfrice.relationship_detector",
        name="DEFAULT_MAX_CANDIDATE_PAIRS",
        kind=KIND_CAP,
        value="5000",
        size=None,
        edge_types=("CONTRADICTS", "POTENTIALLY_STALE"),
        overridable=OVERRIDE_TOML,
        gates=(
            "Truncates the candidate-pair pool. Pairs past the cap are "
            "never classified, so on a large store this silently bounds "
            "which edges can exist at all."
        ),
    ),
    PinnedThreshold(
        module="aelfrice.relationship_detector",
        name="DEFAULT_MAX_EDGES_PER_BELIEF",
        kind=KIND_CAP,
        value="8",
        size=None,
        edge_types=("CONTRADICTS",),
        overridable=OVERRIDE_KWARG,
        gates=(
            "Per-belief write-gate bounding the Exp-48 coverage-dilution "
            "failure mode. Pairs are processed in deterministic audit "
            "order, so which edges survive the cap is deterministic too."
        ),
    ),
    PinnedThreshold(
        module="aelfrice.relationship_detector",
        name="QUANT_AXIS",
        kind=KIND_PATTERN_TABLE,
        value="sha256:fba89a73d04492ecb3bf51ee6925fa66e40b5b524e8a735360179212c509b7ff",
        size=13,
        edge_types=("CONTRADICTS", "POTENTIALLY_STALE"),
        overridable=OVERRIDE_NONE,
        gates=(
            "Quantifier positions on the frequency axis. The axis "
            "distance is halved into `q_term` and the score halves it "
            "again, so a pure quantifier disagreement scores a QUARTER "
            "of the distance: `always` vs `sometimes` is 1.0 apart and "
            "scores 0.25 — below `confidence_min`, so it lands as "
            "POTENTIALLY_STALE, not CONTRADICTS. Size any edit here "
            "against 4x the axis gap, not 2x."
        ),
    ),
    PinnedThreshold(
        module="aelfrice.relationship_detector",
        name="_NEGATION_TOKENS",
        kind=KIND_TOKEN_SET,
        value="sha256:1bfa98c280a92274a3caf35607bf41a56e9e55da3ab0f3fdf1d57e7c8609a7d9",
        size=20,
        edge_types=("CONTRADICTS", "POTENTIALLY_STALE"),
        overridable=OVERRIDE_NONE,
        gates=(
            "Drives the negation term, which is 1.0 when exactly one "
            "side is negated. Adding a token can flip a pair from "
            "`refines` (no edge) to `contradicts` (edge)."
        ),
    ),
    PinnedThreshold(
        module="aelfrice.relationship_detector",
        name="_CONTRACTION_NEGATION_RE",
        kind=KIND_PATTERN_TABLE,
        value="sha256:2410353c94c8a1dcd6aa85bc838af043a7feb45f5c34e07c8f5122537b571f21",
        size=None,
        edge_types=("CONTRADICTS", "POTENTIALLY_STALE"),
        overridable=OVERRIDE_NONE,
        gates=(
            "Second negation pass over raw content, catching contracted "
            "forms the tokenizer splits. Same effect on the negation "
            "term as the token set."
        ),
    ),
    PinnedThreshold(
        module="aelfrice.relationship_detector",
        name="_STOPWORDS",
        kind=KIND_TOKEN_SET,
        value="sha256:b913860f62f01f806cf7bf2f18a68c4630c7feb2e082898c3a59c17d74f3890c",
        size=125,
        edge_types=("CONTRADICTS", "POTENTIALLY_STALE"),
        overridable=OVERRIDE_NONE,
        gates=(
            "Subtracted before the residual-overlap check, so it sets "
            "the denominator of that score. Note it is a union with the "
            "negation and quantifier vocabularies — editing either of "
            "those moves this digest too."
        ),
    ),
    # --- contradiction: SUPERSEDES ------------------------------------
    PinnedThreshold(
        module="aelfrice.contradiction",
        name="SUPERSEDES_WEIGHT",
        kind=KIND_WEIGHT,
        value="1.0",
        size=None,
        edge_types=("SUPERSEDES",),
        overridable=OVERRIDE_NONE,
        gates=(
            "Weight stamped on every SUPERSEDES edge. Does not change "
            "which edges are written. It does decide whether they are "
            "visible downstream: `clustering` drops any edge below "
            "`DEFAULT_CLUSTER_EDGE_FLOOR` (0.4), so at 1.0 these clear "
            "the floor and any value under 0.4 silently removes every "
            "SUPERSEDES edge from candidate clustering. It is also a BFS "
            "sort key. Note this is `Edge.weight`, not `EDGE_VALENCE` — "
            "valence is keyed on edge TYPE and is untouched by this."
        ),
    ),
    PinnedThreshold(
        module="aelfrice.contradiction",
        name="CLASS_NAMES",
        kind=KIND_PATTERN_TABLE,
        value="sha256:4a55a5bd080b914f9d77b3b7c58c85dca6eaf48a05c0ea9ff06b91a4ed365749",
        size=6,
        edge_types=("SUPERSEDES",),
        overridable=OVERRIDE_NONE,
        gates=(
            "Keyed on the PRECEDENCE_* integers, so this one digest pins "
            "the whole precedence ordering. `_pick_winner` compares those "
            "integers to choose the winner, and the winner becomes the "
            "edge's `src` — reordering them does not resize the edge set, "
            "it REVERSES edges that are already there, which no count-based "
            "check would notice."
        ),
    ),
    # --- triple_extractor: the explicit-relation surface ---------------
    PinnedThreshold(
        module="aelfrice.triple_extractor",
        name="_PATTERNS",
        kind=KIND_PATTERN_TABLE,
        value="sha256:db6c470a051a084c83abdc0ca03e38948ff228b77c906799271b834d164b47cf",
        size=25,
        edge_types=(
            "SUPPORTS", "CITES", "CONTRADICTS", "SUPERSEDES",
            "RELATES_TO", "DERIVED_FROM", "IMPLEMENTS", "TESTS",
            "TEMPORAL_NEXT",
        ),
        overridable=OVERRIDE_NONE,
        gates=(
            "The phrase-to-edge-type table. This is the only writer that "
            "chooses among most edge types, so it decides both whether "
            "an edge exists and which type it is. Note it includes four "
            "TEMPORAL_NEXT patterns (`follows`, `comes after`, `is "
            "after`, `succeeds`), so the spine is NOT the only producer "
            "of that type — see EXCLUDED_WRITERS."
        ),
    ),
    PinnedThreshold(
        module="aelfrice.triple_extractor",
        name="ANCHOR_CONTEXT_TARGET",
        kind=KIND_CAP,
        value="80",
        size=None,
        edge_types=("SUPPORTS", "CITES", "RELATES_TO", "TESTS"),
        overridable=OVERRIDE_NONE,
        gates=(
            "Target width of the anchor text stored on the edge. Changes "
            "the persisted `anchor_text` column, which a recompute "
            "comparing whole edge rows will read as divergence."
        ),
    ),
    # --- value_compare: the typed-slot contradiction gate --------------
    #
    # DORMANT, and pinned deliberately anyway. The gate is reached only
    # from `analyze(use_value_comparison=True)`; nothing shipped passes
    # that flag — it defaults False at relationship_detector.py:263 and
    # :349, and `relationships_audit` does not thread it — so today these
    # four constants change no edge in any store. They are recorded
    # because the flag is the only thing between them and the write path:
    # flipping it makes a slot conflict short-circuit the
    # residual-overlap floor at a fixed score of 1.0, minting CONTRADICTS
    # edges for pairs the token path calls unrelated. Pinning now means
    # the flip is a one-line change against a known baseline rather than
    # a change to four unpinned constants at once. The `gates` text below
    # describes what happens WHEN reached; read it in that mood.
    PinnedThreshold(
        module="aelfrice.value_compare",
        name="DEFAULT_NUMERIC_REL_TOL",
        kind=KIND_CUTOFF,
        value="0.01",
        size=None,
        edge_types=("CONTRADICTS",),
        overridable=OVERRIDE_KWARG,
        gates=(
            "Relative tolerance below which two numbers for the same "
            "slot count as agreeing. Widening it suppresses conflicts "
            "and removes edges; narrowing it mints them."
        ),
    ),
    PinnedThreshold(
        module="aelfrice.value_compare",
        name="ENUM_VOCAB",
        kind=KIND_PATTERN_TABLE,
        value="sha256:75f2af93073693b6657e7718e252c11bed07eff5a7c33bc9202d1a71a97675b1",
        size=9,
        edge_types=("CONTRADICTS",),
        overridable=OVERRIDE_NONE,
        gates=(
            "Mutual-exclusion groups per category. Two beliefs landing "
            "in different groups of one category is a conflict, which "
            "forces `contradicts` at score 1.0."
        ),
    ),
    PinnedThreshold(
        module="aelfrice.value_compare",
        name="_NUMERIC_KEY_DROP",
        kind=KIND_TOKEN_SET,
        value="sha256:64bbd8a03443c804b2814e92200a44b1bf4328c85cc892de39f565633bf4e4ff",
        size=24,
        edge_types=("CONTRADICTS",),
        overridable=OVERRIDE_NONE,
        gates=(
            "Keys discarded before numeric slots are compared. A key "
            "added here can no longer produce a conflict, so edges "
            "disappear silently."
        ),
    ),
    PinnedThreshold(
        module="aelfrice.value_compare",
        name="_NUMERIC_RE",
        kind=KIND_PATTERN_TABLE,
        value="sha256:b6152502c407f836d8ac933a84564fb08a421ae6068aa0232c68ac74c74b6dd8",
        size=None,
        edge_types=("CONTRADICTS",),
        overridable=OVERRIDE_NONE,
        gates=(
            "Extracts the numeric slots that are compared at all. A "
            "pattern change alters which values are even eligible for a "
            "conflict."
        ),
    ),
    # --- wonder: RELATES_TO from phantom to constituents ---------------
    #
    # `wonder.lifecycle` writes one RELATES_TO edge per constituent of an
    # ingested phantom, so what decides the edge set is what decides
    # which phantoms reach it. There are two such paths, and NEITHER is
    # the bake-off:
    #
    #   1. `cli.py` / `mcp_server.py` rank BFS hops by
    #      `wonder_consolidation.score` and keep `[: --top]`.
    #   2. `wonder.skill_integration` turns research-agent documents into
    #      phantoms over an anchor tuple, seeded by `wonder.dispatch`.
    #
    # `wonder.{strategies,evaluator,simulator,runner}` are deliberately
    # NOT pinned. The package docstring states they are research-only and
    # do not write to a live store, and `runner` — their sole importer —
    # builds against `MemoryStore(":memory:")`. Their thresholds
    # (JACCARD_REDUNDANCY, JUNK_RATE_DEFER, TC_EDGE_TYPES, …) are real
    # knobs on the #228 bake-off and change no edge in any user's store,
    # so pinning them here would pad the manifest with entries whose
    # `gates` text could not be true.
    PinnedThreshold(
        module="aelfrice.wonder.lifecycle",
        name="_CONSTITUENT_KEY_VERSION",
        kind=KIND_LITERAL,
        value="\"v2\"",
        size=None,
        edge_types=("RELATES_TO",),
        overridable=OVERRIDE_NONE,
        gates=(
            "Prefix of the phantom idempotency key. Bumping it makes every "
            "existing phantom miss its own dedup guard and re-ingest as a "
            "new belief, minting a second full set of RELATES_TO edges — "
            "so this string decides edge *duplication*, not just naming."
        ),
    ),
    PinnedThreshold(
        module="aelfrice.wonder.dispatch",
        name="UNCERTAINTY_THRESHOLD",
        kind=KIND_CUTOFF,
        value="0.7",
        size=None,
        edge_types=("RELATES_TO",),
        overridable=OVERRIDE_NONE,
        gates=(
            "Posterior-uncertainty floor selecting "
            "`high_uncertainty_beliefs`, which decides whether an "
            "`uncertainty_deep_dive` research axis is emitted at all. "
            "That axis is a document, and documents become phantoms with "
            "RELATES_TO edges, so the floor gates a whole class of them. "
            "It does NOT filter the anchor tuple — `anchors` is built "
            "from the unfiltered `known_beliefs` — and it does not touch "
            "the BFS path in item 1 above."
        ),
    ),
    PinnedThreshold(
        module="aelfrice.wonder_consolidation",
        name="_TOKENIZER_DROP",
        kind=KIND_TOKEN_SET,
        value="sha256:e5e1d1b663e88634d05ab0cd7349e2c7f0e165d0ea66d20f9aaae6d45bdb2661",
        size=14,
        edge_types=("RELATES_TO",),
        overridable=OVERRIDE_NONE,
        gates=(
            "Punctuation stripped before the relatedness score that ranks "
            "BFS hops. That ranking, then a `[: --top]` slice, is what "
            "selects the phantoms the BFS path persists — so this set "
            "reorders the candidate list and changes which RELATES_TO "
            "edges survive the cut."
        ),
    ),
    # --- shared upstream: the token universe and the anchor column -----
    #
    # Neither module writes an edge, so the call-site sweep cannot see
    # them; both are pinned because a change to either moves edges in
    # every writer downstream of it.
    PinnedThreshold(
        module="aelfrice.bm25",
        name="_TOKEN_PATTERN",
        kind=KIND_PATTERN_TABLE,
        value="sha256:0194ef069cb572f77ccace10a4d389d81c0aee13e7cb8757ff2f7155d886adbb",
        size=None,
        edge_types=("CONTRADICTS", "POTENTIALLY_STALE"),
        overridable=OVERRIDE_NONE,
        gates=(
            "`tokenize` is the token universe for the Jaccard prefilter, "
            "the negation and quantifier membership tests, and the "
            "residual-overlap set. Widening or narrowing this one pattern "
            "moves every CONTRADICTS and POTENTIALLY_STALE edge in the "
            "store, which is why it outranks any single cutoff below."
        ),
    ),
    PinnedThreshold(
        module="aelfrice.models",
        name="ANCHOR_TEXT_MAX_LEN",
        kind=KIND_CAP,
        value="1000",
        size=None,
        edge_types=("SUPPORTS", "CITES", "RELATES_TO", "TESTS"),
        overridable=OVERRIDE_NONE,
        gates=(
            "`Edge.__post_init__` truncates `anchor_text` to this length "
            "on construction, so it alters the persisted row of every "
            "edge that carries one. Does not change which edges exist; "
            "does change what a row-level recompute comparison sees."
        ),
    ),
)


# --- Digest guard ------------------------------------------------------

# Content digest of THRESHOLDS at each version. **Append a row when you
# bump the version; never rewrite one.**
#
# This is keyed by version rather than held as a single literal for a
# specific reason. A lone `MANIFEST_DIGEST = "<hex>"` sitting beside the
# thing it digests does not force anything: edit a constant, edit its
# manifest entry, edit the digest, and the suite is green again with the
# version untouched — two different edge-producing behaviours both
# shipping as version 1. That is failure mode 2 in the test file's own
# docstring, and a bare literal invites exactly the repair that causes it.
#
# Keyed by version, the cheap repair is gone: a content change makes
# `manifest_digest()` disagree with `DIGEST_HISTORY[VERSION]`, and the
# ways back to green are to revert, or to bump the version and append a
# row. Overwriting a historical row still works, but it is a visibly
# dishonest edit in the diff rather than the obvious one.
#
# Honest limit: this raises the cost of the wrong move, it does not make
# it impossible. Only a check against the merge-base — if the THRESHOLDS
# digest differs from `main`'s, require VERSION to have increased — is
# truly mechanical. That belongs in CI and is deliberately not built here.
DIGEST_HISTORY: Final[dict[int, str]] = {
    1: "ffaaca91fa8e74cb9d79d9a9322cf8ce0d3d3d41ac628694609e7c1fbbaeec74",
}

# The digest the current version must produce. Derived, never hand-edited.
# `.get` rather than `[...]`: bumping the version without appending a row
# is a contract breach the tests should NAME, and an import-time KeyError
# would instead take the whole module down and report as a collection
# error in every unrelated test that imports it.
MANIFEST_DIGEST: Final[str] = DIGEST_HISTORY.get(
    DETECTOR_THRESHOLDS_VERSION, ""
)


def manifest_digest() -> str:
    """Digest the pinned content, independent of the version it ships under.

    Content-only on purpose: the version is the key into
    :data:`DIGEST_HISTORY`, so folding it into the digested payload would
    make every version bump change the digest for reasons unrelated to
    what the manifest says.
    """
    payload = {"thresholds": [_canonical(t) for t in THRESHOLDS]}
    return hashlib.sha256(_dumps(payload).encode("utf-8")).hexdigest()


# Every module that writes a non-TEMPORAL_NEXT, non-DERIVED_FROM edge.
# The coverage test asserts this equals the set of modules reaching
# `MemoryStore.insert_edge`, minus the documented exclusions below, so a
# new writer landing without a manifest entry fails rather than passing
# unnoticed.
COVERED_WRITER_MODULES: Final[frozenset[str]] = frozenset({
    "aelfrice.relationship_detector",
    "aelfrice.contradiction",
    "aelfrice.triple_extractor",
    "aelfrice.wonder.lifecycle",
})

# Call sites of `insert_edge` that are deliberately unpinned, with the
# reason. Kept here rather than in the test so the exclusion list is part
# of the record a reviewer reads.
EXCLUDED_WRITERS: Final[tuple[tuple[str, str], ...]] = (
    (
        "aelfrice.temporal_spine",
        "writes TEMPORAL_NEXT only — the spine, recomputed by #1336. "
        "Read as a statement about THIS module, not about the type: "
        "`triple_extractor` also mints TEMPORAL_NEXT from four prose "
        "patterns, so the spine recompute does not account for the "
        "whole TEMPORAL_NEXT population",
    ),
    (
        "aelfrice.ingest",
        "writes DERIVED_FROM only; population is #1354's scope",
    ),
    (
        "aelfrice.derivation_worker",
        "relays edges built by derive(), which returns none today "
        "(#1354); it applies no threshold of its own",
    ),
    (
        "aelfrice.migrate",
        "copies existing edge rows between stores; makes no detection "
        "decision, so there is no threshold to pin",
    ),
    (
        "aelfrice.benchmark",
        "builds a fixed synthetic multi-hop fixture for benchmarks, not "
        "a detector over user beliefs",
    ),
    (
        "aelfrice.wonder.simulator",
        "seeds the #228 synthetic bake-off corpus from a seed; not a "
        "detector over user beliefs",
    ),
)


__all__ = [
    "COVERED_WRITER_MODULES",
    "DETECTOR_THRESHOLDS_VERSION",
    "EXCLUDED_WRITERS",
    "KINDS",
    "DIGEST_HISTORY",
    "MANIFEST_DIGEST",
    "PinnedThreshold",
    "THRESHOLDS",
    "manifest_digest",
    "pin_value",
    "size_of",
]
