"""Minimal v0.9.0-rc benchmark harness.

Runs aelfrice retrieval against a small deterministic synthetic
corpus (16 beliefs across 4 topics) and a deterministic query set
(16 queries with one known correct belief each). Produces a single
JSON document containing hit@1 / hit@3 / hit@5, MRR, and latency
percentiles.

This harness is the **measurement instrument**, not a proof of the
central claim. It does not currently differentiate
"with-feedback" vs "no-feedback" because retrieval ranking in
v1.0 is BM25-only (posterior alpha/beta does not factor into
search_beliefs ordering — see store.py:244). A v1.x retrieval
upgrade that consumes posterior is the precondition for using this
harness to claim feedback drives accuracy.

What this harness DOES validate at v0.9.0-rc:
- The full ingest -> retrieve pipeline runs end-to-end against a
  fresh on-disk store.
- A reproducible score is produced. Re-running the harness against
  an empty store yields identical numbers.
- Latency stays below a sane ceiling for the size-16 corpus
  (regression test asserts p99 < 100ms).

Score floor (regression-asserted):
- hit_at_5 >= 0.75 — at least 12 of 16 queries find their correct
  belief in the top 5 results. Anything below means BM25 escaping
  or scoring has regressed.

The corpus, queries, and expected-correct mappings are all
hand-authored. They live in this module rather than a separate
data file because (a) we want the harness to ship in the wheel
without needing package_data plumbing, and (b) review of changes
is just a regular code-review of this file.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Final

from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CITES,
    EDGE_SUPPORTS,
    LOCK_NONE,
    ORIGIN_UNKNOWN,
    Belief,
    Edge,
)
from aelfrice.retrieval import retrieve
from aelfrice.store import MemoryStore

BENCHMARK_NAME: Final[str] = "aelfrice-bench-v1"
"""Stable identifier for this benchmark configuration.

Bumped (v2, v3, ...) only when corpus / queries / expected mapping
change in a way that breaks score comparability across releases.
Within a v1 series, hit@K numbers are directly comparable.
"""

DEFAULT_TOP_K: Final[int] = 5
DEFAULT_TOKEN_BUDGET: Final[int] = 4000
"""Higher than retrieval's default 2000 so the budget doesn't
cap the harness at fewer than top_k results when belief contents
are long."""


# --- Synthetic corpus -------------------------------------------------


@dataclass(frozen=True)
class _CorpusEntry:
    id: str
    content: str


# 4 topics * 4 beliefs each = 16 beliefs. Topic key embedded in id
# prefix so reviewers can sanity-check the expected-mapping table
# without rerunning the harness.
_CORPUS: Final[tuple[_CorpusEntry, ...]] = (
    # Cooking
    _CorpusEntry("cook_01", "the kitchen knife should be sharpened weekly to maintain a clean cutting edge"),
    _CorpusEntry("cook_02", "boiling water for pasta needs at least one tablespoon of salt per liter"),
    _CorpusEntry("cook_03", "cast iron skillets must be seasoned with oil after every wash to prevent rust"),
    _CorpusEntry("cook_04", "a sourdough starter ferments overnight at room temperature before baking"),
    # Gardening
    _CorpusEntry("gard_01", "tomato plants should be staked when the stem reaches knee height to prevent breakage"),
    _CorpusEntry("gard_02", "compost piles need turning every two weeks to aerate and accelerate decomposition"),
    _CorpusEntry("gard_03", "drip irrigation delivers water directly to plant roots with minimal evaporation loss"),
    _CorpusEntry("gard_04", "raised beds drain better than ground-level plots in clay-heavy soil"),
    # Networking
    _CorpusEntry("net_01", "TCP retransmits lost packets after a timeout based on the round-trip estimate"),
    _CorpusEntry("net_02", "BGP announces routes between autonomous systems across the public internet"),
    _CorpusEntry("net_03", "TLS handshake negotiates a symmetric key using asymmetric certificate exchange"),
    _CorpusEntry("net_04", "subnet masks partition an IP address space into smaller administrative groups"),
    # Programming
    _CorpusEntry("prog_01", "garbage collection reclaims memory from objects no longer reachable from a root set"),
    _CorpusEntry("prog_02", "B-trees keep database indexes balanced so disk page reads stay logarithmic in row count"),
    _CorpusEntry("prog_03", "compilers lower high-level code into machine instructions through a sequence of passes"),
    _CorpusEntry("prog_04", "merge sort recursively splits and combines lists to achieve guaranteed n log n time"),
)


# --- Queries with expected-correct mapping ----------------------------


@dataclass(frozen=True)
class _QueryEntry:
    query: str
    correct_id: str


# Queries are deliberately constrained to terms present (surface-form,
# case-insensitive) in their target belief, because FTS5 with the
# simple tokenizer does not stem and store._escape_fts5_query joins
# whitespace tokens with implicit AND. Single-keyword queries would
# match too many beliefs; 2–4 specific terms gives a sharp signal.
_QUERIES: Final[tuple[_QueryEntry, ...]] = (
    _QueryEntry("kitchen knife sharpened", "cook_01"),
    _QueryEntry("salt pasta water", "cook_02"),
    _QueryEntry("cast iron seasoned", "cook_03"),
    _QueryEntry("sourdough starter overnight", "cook_04"),
    _QueryEntry("tomato plants staked", "gard_01"),
    _QueryEntry("compost turning weeks", "gard_02"),
    _QueryEntry("drip irrigation roots", "gard_03"),
    _QueryEntry("raised beds clay", "gard_04"),
    _QueryEntry("TCP packets timeout", "net_01"),
    _QueryEntry("BGP routes autonomous", "net_02"),
    _QueryEntry("TLS handshake certificate", "net_03"),
    _QueryEntry("subnet masks IP", "net_04"),
    _QueryEntry("garbage collection memory", "prog_01"),
    _QueryEntry("database indexes balanced", "prog_02"),
    _QueryEntry("compilers machine instructions", "prog_03"),
    _QueryEntry("merge sort recursively", "prog_04"),
)


# --- Multi-hop corpus -------------------------------------------------
#
# 3 chains of 4 hops each (12 beliefs total). Each chain anchors on a
# developer-style identifier that appears in the source belief AND in
# the edge's anchor_text, but NOT in the terminal belief. That gap is
# exactly the L2.5 entity-index / edge-traversal gap: L1 BM25 cannot
# bridge it; L2.5 can.
#
# Chain layout (A -> B -> C -> D):
#   A mentions identifier_X in body.
#   Edge A->B has anchor_text = "identifier_X".
#   B does NOT contain identifier_X.
#   Queries reference identifier_X and ask about D.
#
# Bridge identifiers per chain:
#   Chain 1 (auth):   auth_service.py / AuthController / E_AUTH_001
#   Chain 2 (ingest): ingest_pipeline.py / DataNormalizer / E_PIPE_002
#   Chain 3 (config): config_loader.py / SettingsManager / E_CFG_003


@dataclass(frozen=True)
class _MultiHopEntry:
    id: str
    content: str


_MULTIHOP_CORPUS: Final[tuple[_MultiHopEntry, ...]] = (
    # --- Chain 1: auth ---
    # A: auth_service.py is the bridge identifier into B
    _MultiHopEntry(
        "mh_auth_01",
        "auth_service.py validates JWT tokens by checking signature and expiry "
        "on every inbound API request",
    ),
    # B: bridge is referenced only in edge anchor_text, not here
    _MultiHopEntry(
        "mh_auth_02",
        "AuthController delegates session creation to the token factory and "
        "stores the resulting session identifier in a short-lived cookie",
    ),
    # C: AuthController is the bridge identifier into D
    _MultiHopEntry(
        "mh_auth_03",
        "AuthController raises E_AUTH_001 when the token factory reports an "
        "expired refresh token that cannot be silently renewed",
    ),
    # D: target — does NOT contain auth_service.py
    _MultiHopEntry(
        "mh_auth_04",
        "E_AUTH_001 triggers a forced re-login flow in the client that clears "
        "the local credential cache and redirects to the sign-in page",
    ),
    # --- Chain 2: ingest ---
    # A: ingest_pipeline.py is the bridge identifier into B
    _MultiHopEntry(
        "mh_pipe_01",
        "ingest_pipeline.py reads raw events from the message queue and passes "
        "each batch to a configurable chain of transformer stages",
    ),
    # B: bridge is referenced only in edge anchor_text
    _MultiHopEntry(
        "mh_pipe_02",
        "DataNormalizer converts ISO-8601 timestamps to Unix epoch integers "
        "and strips null fields before the batch reaches the sink stage",
    ),
    # C: DataNormalizer is the bridge identifier into D
    _MultiHopEntry(
        "mh_pipe_03",
        "DataNormalizer raises E_PIPE_002 when a batch contains a record whose "
        "timestamp field is missing entirely rather than null",
    ),
    # D: target — does NOT contain ingest_pipeline.py
    _MultiHopEntry(
        "mh_pipe_04",
        "E_PIPE_002 causes the entire batch to be moved to a dead-letter queue "
        "and an alert is sent to the on-call rotation",
    ),
    # --- Chain 3: config ---
    # A: config_loader.py is the bridge identifier into B
    _MultiHopEntry(
        "mh_cfg_01",
        "config_loader.py reads YAML files from the config directory and merges "
        "them with environment variable overrides at startup",
    ),
    # B: bridge is referenced only in edge anchor_text
    _MultiHopEntry(
        "mh_cfg_02",
        "SettingsManager caches the merged configuration in memory and exposes "
        "a typed accessor interface to all subsystems that need runtime config",
    ),
    # C: SettingsManager is the bridge identifier into D
    _MultiHopEntry(
        "mh_cfg_03",
        "SettingsManager raises E_CFG_003 when a required key is absent from "
        "both the YAML file and the environment variable overrides",
    ),
    # D: target — does NOT contain config_loader.py
    _MultiHopEntry(
        "mh_cfg_04",
        "E_CFG_003 causes the application to exit with a non-zero status code "
        "and logs the missing key name to help operators diagnose startup failures",
    ),
)


@dataclass(frozen=True)
class _MultiHopEdgeSpec:
    """Lightweight spec for a CITES/SUPPORTS edge in the multi-hop corpus."""

    src: str
    dst: str
    edge_type: str
    anchor: str


# Each edge carries the bridge identifier as anchor_text.
# Chain 1: auth_service.py bridges A->B; AuthController bridges B->C; E_AUTH_001 bridges C->D.
# Chain 2: ingest_pipeline.py bridges A->B; DataNormalizer bridges B->C; E_PIPE_002 bridges C->D.
# Chain 3: config_loader.py bridges A->B; SettingsManager bridges B->C; E_CFG_003 bridges C->D.
_MULTIHOP_EDGES: Final[tuple[_MultiHopEdgeSpec, ...]] = (
    # Chain 1 auth
    _MultiHopEdgeSpec("mh_auth_01", "mh_auth_02", EDGE_CITES, "auth_service.py"),
    _MultiHopEdgeSpec("mh_auth_02", "mh_auth_03", EDGE_SUPPORTS, "AuthController"),
    _MultiHopEdgeSpec("mh_auth_03", "mh_auth_04", EDGE_CITES, "E_AUTH_001"),
    # Chain 2 ingest
    _MultiHopEdgeSpec("mh_pipe_01", "mh_pipe_02", EDGE_CITES, "ingest_pipeline.py"),
    _MultiHopEdgeSpec("mh_pipe_02", "mh_pipe_03", EDGE_SUPPORTS, "DataNormalizer"),
    _MultiHopEdgeSpec("mh_pipe_03", "mh_pipe_04", EDGE_CITES, "E_PIPE_002"),
    # Chain 3 config
    _MultiHopEdgeSpec("mh_cfg_01", "mh_cfg_02", EDGE_CITES, "config_loader.py"),
    _MultiHopEdgeSpec("mh_cfg_02", "mh_cfg_03", EDGE_SUPPORTS, "SettingsManager"),
    _MultiHopEdgeSpec("mh_cfg_03", "mh_cfg_04", EDGE_CITES, "E_CFG_003"),
)


# 8 multi-hop queries. Surface terms in each query do NOT appear in the
# target belief. The path from query to target requires traversing at least
# one CITES/SUPPORTS edge whose anchor_text contains the bridge identifier.
@dataclass(frozen=True)
class _MultiHopQuery:
    query: str
    correct_id: str


_MULTIHOP_QUERIES: Final[tuple[_MultiHopQuery, ...]] = (
    # Chain 1 — query names auth_service.py, target is mh_auth_04 (E_AUTH_001 behavior)
    _MultiHopQuery("auth_service.py expired token forced re-login", "mh_auth_04"),
    _MultiHopQuery("auth_service.py E_AUTH_001 credential cache redirect", "mh_auth_04"),
    # Chain 1 — query names AuthController, target is mh_auth_04
    _MultiHopQuery("AuthController session creation E_AUTH_001 behavior", "mh_auth_04"),
    # Chain 2 — query names ingest_pipeline.py, target is mh_pipe_04 (dead-letter)
    _MultiHopQuery("ingest_pipeline.py batch processing dead-letter", "mh_pipe_04"),
    _MultiHopQuery("ingest_pipeline.py E_PIPE_002 alert on-call", "mh_pipe_04"),
    # Chain 3 — query names config_loader.py, target is mh_cfg_04 (exit behavior)
    _MultiHopQuery("config_loader.py startup failure exit status", "mh_cfg_04"),
    _MultiHopQuery("config_loader.py E_CFG_003 missing key operator", "mh_cfg_04"),
    # Chain 3 — query names SettingsManager, target is mh_cfg_04
    _MultiHopQuery("SettingsManager E_CFG_003 application exit non-zero", "mh_cfg_04"),
)


# --- Scored output ----------------------------------------------------


@dataclass(frozen=True)
class BenchmarkReport:
    """Reproducible single-run summary."""

    benchmark_name: str
    aelfrice_version: str
    corpus_size: int
    query_count: int
    top_k: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    p50_latency_ms: float
    p99_latency_ms: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# --- Harness ----------------------------------------------------------


def seed_corpus(store: MemoryStore, *, created_at: str = "2026-04-26T00:00:00Z") -> int:
    """Insert the deterministic benchmark corpus into `store`.

    Inserts each `_CorpusEntry` as an unlocked factual belief with
    Jeffreys prior (alpha=1, beta=1). Returns the count inserted.
    """
    inserted = 0
    for entry in _CORPUS:
        store.insert_belief(
            Belief(
                id=entry.id,
                content=entry.content,
                content_hash=f"bench_{entry.id}",
                alpha=1.0,
                beta=1.0,
                type=BELIEF_FACTUAL,
                lock_level=LOCK_NONE,
                locked_at=None,
                demotion_pressure=0,
                created_at=created_at,
                last_retrieved_at=None,
                origin=ORIGIN_UNKNOWN,
            )
        )
        inserted += 1
    return inserted


def seed_multihop_corpus(
    store: MemoryStore, *, created_at: str = "2026-04-26T00:00:00Z"
) -> int:
    """Insert the multi-hop benchmark corpus and edges into ``store``.

    Inserts the 12 beliefs from ``_MULTIHOP_CORPUS`` followed by all 9 edges
    from ``_MULTIHOP_EDGES``. Returns the number of beliefs inserted (12).
    The caller is responsible for creating the store; this function does not
    seed the original 16-belief corpus.
    """
    inserted = 0
    for entry in _MULTIHOP_CORPUS:
        store.insert_belief(
            Belief(
                id=entry.id,
                content=entry.content,
                content_hash=f"bench_{entry.id}",
                alpha=1.0,
                beta=1.0,
                type=BELIEF_FACTUAL,
                lock_level=LOCK_NONE,
                locked_at=None,
                demotion_pressure=0,
                created_at=created_at,
                last_retrieved_at=None,
                origin=ORIGIN_UNKNOWN,
            )
        )
        inserted += 1
    for spec in _MULTIHOP_EDGES:
        store.insert_edge(
            Edge(
                src=spec.src,
                dst=spec.dst,
                type=spec.edge_type,
                weight=1.0,
                anchor_text=spec.anchor,
            )
        )
    return inserted


def run_benchmark(
    store: MemoryStore,
    *,
    aelfrice_version: str,
    top_k: int = DEFAULT_TOP_K,
) -> BenchmarkReport:
    """Run all queries against `store` and return scored results.

    Caller must have seeded the corpus (e.g. via `seed_corpus`) into
    `store` before calling. The harness does not seed implicitly to
    keep store ownership explicit and to let benchmark variants
    layer additional state (e.g. pre-locked beliefs) on top of the
    base corpus.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    reciprocal_ranks: list[float] = []
    latencies_ms: list[float] = []

    for q in _QUERIES:
        t0 = time.perf_counter()
        results = retrieve(
            store, q.query, token_budget=DEFAULT_TOKEN_BUDGET, l1_limit=top_k
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        rank = _rank_of(q.correct_id, results)
        if rank is not None:
            reciprocal_ranks.append(1.0 / rank)
            if rank == 1:
                hits_at_1 += 1
            if rank <= 3:
                hits_at_3 += 1
            if rank <= 5:
                hits_at_5 += 1
        else:
            reciprocal_ranks.append(0.0)

    n = len(_QUERIES)
    return BenchmarkReport(
        benchmark_name=BENCHMARK_NAME,
        aelfrice_version=aelfrice_version,
        corpus_size=len(_CORPUS),
        query_count=n,
        top_k=top_k,
        hit_at_1=hits_at_1 / n,
        hit_at_3=hits_at_3 / n,
        hit_at_5=hits_at_5 / n,
        mrr=sum(reciprocal_ranks) / n,
        p50_latency_ms=_percentile(latencies_ms, 0.50),
        p99_latency_ms=_percentile(latencies_ms, 0.99),
    )


def _rank_of(correct_id: str, results: list[Belief]) -> int | None:
    """1-indexed position of `correct_id` in `results`, or None."""
    for i, b in enumerate(results, start=1):
        if b.id == correct_id:
            return i
    return None


def _percentile(values: list[float], q: float) -> float:
    """Inclusive percentile with linear interpolation. q in [0, 1]."""
    if not values:
        return 0.0
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac
