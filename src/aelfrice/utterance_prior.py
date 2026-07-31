"""Utterance-vs-knowledge document prior (#1174 item 3).

A query-independent term for the L1 rerank, in the spirit of the entry-page
prior of Kraaij, Westerveld & Hiemstra (2002) — the ranking function is asked
to prefer documents that *are* knowledge over documents that merely *look
like* the query.

The failure it targets is not synonymy. This store contains its own query log:
user turns are ingested as beliefs, so the nearest lexical neighbour of a query
is frequently a prior query. Measured on the live store, **30.8% of ranked
top-10 hits on real user prompts are token-set near-copies (cos > 0.9) of the
query itself**. BM25 is behaving correctly; the corpus is the problem. Every
query-expansion mechanism makes this worse, because expansion seeded on
query-shaped beliefs produces query-shaped expansion terms.

Mechanism: a per-document prior estimated as a naive-Bayes log-odds over the
document's own stems, trained on two classes that are *observed* in the
append-only ingest log rather than hand-labelled:

    Q (utterance) <- source_kind in CLASS_Q_SOURCE_KINDS
    K (knowledge) <- source_kind in CLASS_K_SOURCE_KINDS

**The class definition is load-bearing and is pinned here deliberately.** An
earlier draft trained Q on all transcript-*derived beliefs* rather than on the
logged turns themselves. That variant separates its own classes at AUC 0.908 —
it looks like a success — but what it learns is "which corpus did this text
come from", because the K side is AST-derived docstrings whose vocabulary is
code identifiers. Applied as a prior it inverts: on held-out prose it ranked
question-shaped text *below* statement-shaped text (AUC 0.350), i.e. it would
have promoted exactly the echoes it exists to demote. See
`test_utterance_prior.py::test_class_sources_are_pinned`.

Determinism: `logodds` is a table of (stem, log-odds) derived by counting rows
in an append-only log. No sampling, no RNG, no embeddings. The audit answer to
"why was B demoted" is "B scored +0.42 because it contains {stems} with
log-odds {values}, estimated from N_Q logged turns and N_K file/commit rows".

Ships inert: `resolve_utterance_prior_weight` defaults to 0.0 and
`utterance_prior_penalty` returns 0.0 at weight 0, so the rerank is
byte-identical with the lane off and nothing reads the ingest log.
"""
from __future__ import annotations

import math
import os
from collections import Counter
from typing import TYPE_CHECKING, Final

from .bm25 import tokenize_stemmed

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .store import MemoryStore

__all__ = [
    "CLASS_K_SOURCE_KINDS",
    "CLASS_Q_SOURCE_KINDS",
    "ENV_UTTERANCE_PRIOR_WEIGHT",
    "UtterancePrior",
    "resolve_utterance_prior_weight",
    "utterance_logodds",
    "utterance_prior_penalty",
]

# Class sources. Pinned, not configurable — see the module docstring for the
# measured consequence of getting this wrong.
CLASS_Q_SOURCE_KINDS: Final[tuple[str, ...]] = ("transcript",)
CLASS_K_SOURCE_KINDS: Final[tuple[str, ...]] = ("filesystem", "git")

# Laplace smoothing on both class counts. 1.0 = add-one; keeps a stem seen in
# one class only from producing an infinite log-odds.
SMOOTHING_ALPHA: Final[float] = 1.0

# A stem must appear in at least this many documents across both classes to
# enter the table. Below it the log-odds is dominated by smoothing and the
# stem is noise — on the live store this drops the table from ~16.8k to a few
# thousand entries without moving the measured separation.
MIN_DOCUMENT_FREQUENCY: Final[int] = 5

ENV_UTTERANCE_PRIOR_WEIGHT: Final[str] = "AELFRICE_UTTERANCE_PRIOR_WEIGHT"


class UtterancePrior:
    """A built log-odds table plus its per-document scorer.

    Construct via :func:`utterance_logodds`. Immutable in practice; the
    caller is expected to build once per store and reuse.
    """

    __slots__ = ("logodds", "n_knowledge", "n_utterance")

    def __init__(
        self,
        logodds: dict[str, float],
        *,
        n_utterance: int,
        n_knowledge: int,
    ) -> None:
        self.logodds = logodds
        self.n_utterance = n_utterance
        self.n_knowledge = n_knowledge

    def score(self, text: str) -> float:
        """Mean log-odds over the document's known stems.

        Positive = utterance-like, negative = knowledge-like. A mean (not a
        sum) so the term does not scale with document length — length is
        already handled by BM25F's per-field normalisation, and summing here
        would double-count it.

        Returns 0.0 for a document with no stem in the table, which is the
        correct neutral: absence of evidence is not evidence of utterance.
        """
        stems = set(tokenize_stemmed(text))
        vals = [self.logodds[s] for s in stems if s in self.logodds]
        if not vals:
            return 0.0
        return sum(vals) / len(vals)


def utterance_logodds(
    store: MemoryStore, *, min_df: int = MIN_DOCUMENT_FREQUENCY
) -> UtterancePrior:
    """Build the log-odds table from ``store``'s ingest log.

    One pass over `ingest_log`, counting *document* frequency per stem (a
    stem occurring twice in one row counts once) so a single verbose row
    cannot dominate its class.

    `min_df` is exposed so the W-sweep can vary it and so small fixtures can
    build a non-empty table; production callers should take the default.

    Returns a prior with an empty table if either class is unpopulated — a
    store with no ingest history scores every document 0.0 rather than
    raising, so enabling the lane on a fresh store degrades to a no-op.
    """
    cq: Counter[str] = Counter()
    ck: Counter[str] = Counter()
    n_q = n_k = 0
    kinds = CLASS_Q_SOURCE_KINDS + CLASS_K_SOURCE_KINDS
    placeholders = ",".join("?" for _ in kinds)
    # `placeholders` is a fixed-length run of `?` derived from the pinned
    # class tuples, never from input; the kinds themselves are bound.
    cur = store._conn.execute(
        f"SELECT source_kind, raw_text FROM ingest_log "
        f"WHERE source_kind IN ({placeholders})",
        kinds,
    )
    for row in cur:
        text = row["raw_text"] or ""
        if not text.strip():
            continue
        stems = set(tokenize_stemmed(text))
        if not stems:
            continue
        if row["source_kind"] in CLASS_Q_SOURCE_KINDS:
            cq.update(stems)
            n_q += 1
        else:
            ck.update(stems)
            n_k += 1

    if n_q == 0 or n_k == 0:
        return UtterancePrior({}, n_utterance=n_q, n_knowledge=n_k)

    a = SMOOTHING_ALPHA
    denom_q = math.log(n_q + 2 * a)
    denom_k = math.log(n_k + 2 * a)
    logodds = {
        stem: (math.log(cq[stem] + a) - denom_q) - (math.log(ck[stem] + a) - denom_k)
        for stem in set(cq) | set(ck)
        if cq[stem] + ck[stem] >= min_df
    }
    return UtterancePrior(logodds, n_utterance=n_q, n_knowledge=n_k)


def utterance_prior_penalty(
    prior: UtterancePrior | None, content: str, weight: float
) -> float:
    """Log-additive demotion for one belief.

    Returns 0.0 when the lane is off (`prior is None` or `weight == 0`).
    Otherwise `-weight * max(0, score)`: clamped at 0 so this is a pure
    demotion of utterance-shaped content, never a promotion of
    knowledge-shaped content. The clamp matters — the rerank score is in the
    log domain and negative, so an unclamped term would promote every
    knowledge document rather than leaving it neutral, silently changing the
    ranking of documents the lane has no opinion about.

    Log-additive rather than multiplicative for the same reason as
    `_entity_persist_penalty`: a multiplicative factor against a negative
    log-domain score inverts a demotion into a promotion.
    """
    if prior is None or weight == 0.0:
        return 0.0
    return -weight * max(0.0, prior.score(content))


def _env_weight_override() -> float | None:
    raw = os.environ.get(ENV_UTTERANCE_PRIOR_WEIGHT)
    if raw is None or not raw.strip():
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if not math.isfinite(value) or value < 0.0:
        return None
    return value


def resolve_utterance_prior_weight(explicit: float | None = None) -> float:
    """Resolve the prior's weight W.

    Precedence: env var, then the explicit kwarg, then 0.0.

    Default 0.0 — the lane ships inert. The W-sweep that would justify a
    non-zero default needs a relevance gold set, and the store's own
    observed-utility signal cannot supply one (5 positive references in
    16,055 resolved `injection_events` rows). Flipping this default is a
    separate operator call, not a side effect of building the mechanism.
    """
    env = _env_weight_override()
    if env is not None:
        return env
    if explicit is not None and math.isfinite(explicit) and explicit >= 0.0:
        return explicit
    return 0.0
