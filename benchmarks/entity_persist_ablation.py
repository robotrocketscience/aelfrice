"""#1096 entity-persistence demotion — offline ranking ablation (G3).

Builds a SYNTHETIC labeled corpus (durable technical beliefs grounded to
file paths vs ephemeral coordination grounded to bare PR numbers, all at
the same Beta posterior) and reports the AUC for ranking durable above
ephemeral under three priors:

  * posterior_mean only (the current prior),
  * entity-persistence S1 only,
  * posterior + log(S1) (the demotion prior).

Reproduces, on synthetic data, the live-store finding recorded on #1096
(posterior-only AUC ~0.5; the demotion prior lifts it well above). No
live-store content — deterministic and CI-safe.

Run: `python benchmarks/entity_persist_ablation.py`
"""
from __future__ import annotations

import math

from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.store import MemoryStore

N_PER_CLASS = 60


def _mk(bid: str, content: str) -> Belief:
    return Belief(
        id=bid, content=content, content_hash=f"h_{bid}",
        alpha=1.0, beta=1.0, type=BELIEF_FACTUAL, lock_level=LOCK_NONE,
        locked_at=None, created_at="2026-06-01T00:00:00Z",
        last_retrieved_at=None,
    )


def _add(store: MemoryStore, bid: str, lower: str, kind: str) -> None:
    store._conn.execute(
        "INSERT INTO belief_entities(belief_id, entity_lower, entity_raw, "
        "kind, span_start, span_end) VALUES (?,?,?,?,0,0)",
        (bid, lower, lower, kind),
    )


def _auc(scores: list[tuple[str, float]]) -> float:
    dur = [s for lab, s in scores if lab == "durable"]
    eph = [s for lab, s in scores if lab == "ephemeral"]
    wins = ties = 0
    for d in dur:
        for e in eph:
            if d > e:
                wins += 1
            elif d == e:
                ties += 1
    return (wins + 0.5 * ties) / (len(dur) * len(eph))


def main() -> None:
    s = MemoryStore(":memory:")
    labels: dict[str, str] = {}
    for i in range(N_PER_CLASS):
        d, e = f"dur{i}", f"eph{i}"
        s.insert_belief(_mk(d, f"module topic number {i} implementation"))
        s.insert_belief(_mk(e, f"rebased and pushed item number {i} now"))
        for bid in (d, e):
            s._conn.execute(
                "DELETE FROM belief_entities WHERE belief_id=?", (bid,)
            )
        _add(s, d, f"src/mod_{i}.py", "file_path")     # durable grounding
        _add(s, e, f"#{100 + i}", "identifier")        # transient grounding
        labels[d] = "durable"
        labels[e] = "ephemeral"
    s._conn.commit()

    ep = s.entity_persistence_scores(list(labels))
    s.close()

    # All synthetic beliefs share the same Beta(1,1) posterior (mu=0.5),
    # so posterior alone cannot separate the classes — the demotion prior
    # supplies all the ranking signal.
    mu = 0.5
    post = [(labels[i], mu) for i in labels]
    s1 = [(labels[i], ep.get(i, 0.0)) for i in labels]
    demote = [
        (labels[i], math.log(mu + 1e-6) + math.log(ep.get(i, 0.0) + 1e-3))
        for i in labels
    ]
    print(f"synthetic corpus: {N_PER_CLASS} durable + {N_PER_CLASS} ephemeral")
    print(f"  AUC posterior_mean only : {_auc(post):.3f}")
    print(f"  AUC entity-persistence  : {_auc(s1):.3f}")
    print(f"  AUC posterior + log(S1) : {_auc(demote):.3f}")


if __name__ == "__main__":
    main()
