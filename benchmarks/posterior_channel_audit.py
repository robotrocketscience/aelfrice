"""#1267 posterior-channel audit — which channels move a belief posterior,
in which direction, at default settings.

#1267 states an asymmetry: *"The system has an automatic, evidence-driven
channel that moves posteriors up, and none that moves them down."* This
enumerates every `apply_feedback` route on the production path and drives
each one against a fresh in-memory store, so the claim is settled by
observed alpha/beta rather than by reading constants.

The distinction that matters is **resolver default vs. call-site
behaviour**. #1267 quotes `HOOK_RETRIEVAL_VALENCE = 0.1` and the
`apply_feedback` call in `record_retrieval`, which are both real — but the
call passes `update_posterior=_exposure_updates_posterior()`, and that
resolver has defaulted to False since #1086. Quoting the constant without
the gate reads as an always-on channel.

Three channels are checked:

  1. **Hook retrieval exposure** (`hook_search.record_retrieval`,
     `HOOK_RETRIEVAL_VALENCE = +0.1`). Fires on the hot path for every
     belief in every pack. Gated by `AELFRICE_EXPOSURE_UPDATES_POSTERIOR`,
     default off (#1086) — audit row written, posterior untouched.

  2. **Sentiment-from-prose** (`sentiment_feedback.apply_sentiment_to_pending`,
     wired into `UserPromptSubmit` at #606). Regex-matches the user's prose
     and distributes the signal over the previous turn's pack. Emits
     **negative** valence, so this is an automatic down-channel. Opt-in via
     `[feedback] sentiment_from_prose`, default off.

  3. **Deferred-feedback sweeper** (`deferred_feedback.sweep_deferred_feedback`).
     Audit-only since #1162 — classifies what it *would* have applied and
     writes nothing. Its enqueue side is also default off.

No live store is read and no belief content is emitted: every belief here
is synthetic.

**Both configuration tiers are pinned, so a developer's own opt-ins cannot
change the reported defaults** (#1295). These resolvers are
env -> kwarg -> TOML -> default, and pinning only one tier is not enough:

  * **env** — every ambient `AELFRICE_*` variable is deleted before
    `aelfrice` is imported, since several resolvers read the environment
    at import time.
  * **TOML** — `_read_toml_flag_for` walks *up from the working
    directory* looking for `.aelfrice.toml`, so clearing the environment
    does nothing about it. Every call site that can take one is given a
    scratch `start=` / `config_start=`, and `_scratch_walk_hits` fails
    the run if anything above that scratch directory carries a config
    after all — the pin is verified, not assumed.

Only channel 3's resolvers have a TOML tier to pin. Channel 1's
`_exposure_updates_posterior` is env-only, and channel 2's `is_enabled`
reads a caller-supplied dict and never touches disk.

Run: `python benchmarks/posterior_channel_audit.py`
Exits non-zero if any channel's observed behaviour departs from the table
above, so this doubles as a regression guard on the defaults.
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Clear ambient opt-ins BEFORE importing aelfrice: several of these
# resolvers read the environment at import time or per call, and the
# whole point of this script is to report the *default* posture.
_CLEARED = sorted(k for k in os.environ if k.startswith("AELFRICE_"))
for _k in _CLEARED:
    del os.environ[_k]

from aelfrice.deferred_feedback import (  # noqa: E402
    CONFIG_FILENAME,
    enqueue_retrieval_exposures,
    is_enqueue_on_retrieve_enabled,
    resolve_epsilon,
    resolve_grace_seconds,
    sweep_deferred_feedback,
)
from aelfrice.hook_search import (  # noqa: E402
    HOOK_RETRIEVAL_VALENCE,
    record_retrieval,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief  # noqa: E402
from aelfrice.sentiment_feedback import (  # noqa: E402
    BASE_VALENCE,
    ESCALATED_NEGATIVE_VALENCE,
    apply_sentiment_to_pending,
    detect_sentiment,
    is_enabled as sentiment_is_enabled,
)
from aelfrice.store import MemoryStore  # noqa: E402

ENV_EXPOSURE = "AELFRICE_EXPOSURE_UPDATES_POSTERIOR"


def _belief(bid: str) -> Belief:
    return Belief(
        id=bid,
        content=f"synthetic belief {bid}",
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-07-31T00:00:00Z",
        last_retrieved_at=None,
    )


def _seed(bid: str) -> MemoryStore:
    store = MemoryStore(":memory:")
    store.insert_belief(_belief(bid))
    return store


def _ab(store: MemoryStore, bid: str) -> tuple[float, float]:
    belief = store.get_belief(bid)
    return (belief.alpha, belief.beta)


def channel_1_exposure() -> list[str]:
    """Hook retrieval exposure. Default: audit row, no posterior move."""
    failures: list[str] = []
    print("=" * 72)
    print(f"CHANNEL 1 — hook retrieval exposure (valence +{HOOK_RETRIEVAL_VALENCE})")
    print("=" * 72)

    for env_value in (None, "1"):
        os.environ.pop(ENV_EXPOSURE, None)
        if env_value is not None:
            os.environ[ENV_EXPOSURE] = env_value
        store = _seed("b1")
        before = _ab(store, "b1")
        written = record_retrieval(store, [store.get_belief("b1")])
        after = _ab(store, "b1")
        audit_rows = int(store.count_feedback_events("b1"))
        store.close()

        label = "DEFAULT (unset)" if env_value is None else f"{ENV_EXPOSURE}=1"
        moved = before != after
        print(f"  {label:38s} rows={written} audit={audit_rows} "
              f"a/b {before} -> {after}  moved={moved}")

        if env_value is None:
            if moved:
                failures.append("exposure moved the posterior at DEFAULT")
            if audit_rows != 1:
                failures.append("exposure wrote no audit row at DEFAULT")
        elif after[0] != before[0] + HOOK_RETRIEVAL_VALENCE:
            failures.append("exposure did not apply +valence when opted in")

    os.environ.pop(ENV_EXPOSURE, None)
    print("  => automatic UP channel is OFF by default (#1086); "
          "opting in restores it.")
    return failures


def channel_2_sentiment() -> list[str]:
    """Sentiment-from-prose. An automatic DOWN channel, opt-in."""
    failures: list[str] = []
    print()
    print("=" * 72)
    print("CHANNEL 2 — sentiment-from-prose (UserPromptSubmit, #193/#606)")
    print("=" * 72)

    enabled = sentiment_is_enabled({})
    print(f"  is_enabled(default config) = {enabled}")
    if enabled:
        failures.append("sentiment-from-prose is enabled by default")

    signal = detect_sentiment("no that's wrong")
    if signal is None or signal.valence >= 0:
        failures.append("no negative sentiment signal detected for a correction")
        print("  detect_sentiment('no that's wrong') -> None")
        return failures
    print(f"  detect_sentiment(\"no that's wrong\") -> "
          f"({signal.sentiment}, {signal.valence})")

    store = _seed("b2")
    before = _ab(store, "b2")
    apply_sentiment_to_pending(
        store=store, signal=signal, pending_belief_ids=["b2"],
    )
    plain = _ab(store, "b2")
    apply_sentiment_to_pending(
        store=store, signal=signal, pending_belief_ids=["b2"], escalated=True,
    )
    escalated = _ab(store, "b2")
    store.close()

    print(f"  negative signal    a/b {before} -> {plain}   "
          f"moved_down={plain[1] > before[1]}")
    print(f"  escalated negative a/b {plain} -> {escalated}")
    print(f"  magnitude vs exposure: "
          f"{BASE_VALENCE / HOOK_RETRIEVAL_VALENCE:.0f}x .. "
          f"{ESCALATED_NEGATIVE_VALENCE / HOOK_RETRIEVAL_VALENCE:.0f}x")

    if plain[1] <= before[1]:
        failures.append("negative sentiment did not move the posterior down")
    if escalated[1] <= plain[1]:
        failures.append("escalated negative did not exceed the base negative")

    print("  => an automatic DOWN channel EXISTS and is wired to the hot "
          "path; it is opt-in, exactly as the UP channel is.")
    return failures


def _scratch_walk_hits(scratch: Path) -> list[str]:
    """Config files an upward walk from `scratch` would still find.

    Clearing the environment pins only the env tier. These resolvers are
    env -> kwarg -> TOML -> default, and `_read_toml_flag_for` walks up
    from its `start` looking for `.aelfrice.toml`. Passing a scratch
    directory bounds that walk, but a scratch directory is only clean if
    nothing above it carries a config either — so verify rather than
    assume, and report instead of silently measuring someone's config.
    """
    return [
        str(parent / CONFIG_FILENAME)
        for parent in (scratch, *scratch.parents)
        if (parent / CONFIG_FILENAME).exists()
    ]


def channel_3_sweeper(scratch: Path) -> list[str]:
    """Deferred-feedback sweeper. Audit-only since #1162.

    `scratch` pins the TOML tier: every resolver reached here accepts a
    `start` (or `config_start`), so the walk can be bounded to a
    directory with no `.aelfrice.toml`. Without it a developer with
    `[implicit_feedback] enqueue_on_retrieve = true` at or above the
    repo gets "retrieval enqueues exposures by default" on a tree where
    no default moved (#1295).
    """
    failures: list[str] = []
    print()
    print("=" * 72)
    print("CHANNEL 3 — deferred-feedback sweeper (#191, audit-only since #1162)")
    print("=" * 72)

    stray = _scratch_walk_hits(scratch)
    if stray:
        failures.append(
            "the scratch walk is not clean, so the TOML tier is unpinned: "
            + ", ".join(stray)
        )

    enqueue_on = is_enqueue_on_retrieve_enabled(start=scratch)
    print(f"  is_enqueue_on_retrieve_enabled(default) = {enqueue_on}")
    if enqueue_on:
        failures.append("retrieval enqueues exposures by default")

    store = _seed("b3")
    # The queue has to be non-empty for this to test anything. A sweep
    # over an empty queue never enters its classification loop, so it is
    # a no-op for the audit-only sweeper and for the pre-#1162 mutating
    # one alike — the two are indistinguishable and the check below
    # passes either way. Bank a real row and backdate it past the grace
    # window so the row is eligible and the loop actually runs on it.
    grace = resolve_grace_seconds(start=scratch)
    epsilon = resolve_epsilon(start=scratch)
    enqueued_at = (
        datetime.now(UTC) - timedelta(seconds=grace + 60)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    enqueue_retrieval_exposures(store, ["b3"], now=enqueued_at)

    before = _ab(store, "b3")
    result = sweep_deferred_feedback(store, config_start=scratch)
    after = _ab(store, "b3")
    audit_rows = int(store.count_feedback_events("b3"))
    store.close()

    print(f"  enqueued 1 row, backdated {grace + 60}s (grace={grace}s) "
          f"-> eligible")
    print(f"  sweep would_apply={result.would_apply} "
          f"alpha_withheld={result.alpha_withheld} "
          f"epsilon={result.epsilon_used}")
    print(f"  a/b {before} -> {after}  moved={before != after}  "
          f"feedback_history rows={audit_rows}")

    # `would_apply == 1` is what makes the rest of this load-bearing: it
    # proves the eligibility ladder ran and elected to apply. The pre-
    # #1162 sweeper would have moved alpha by exactly `epsilon` on this
    # row and written a `feedback_history` entry; audit-only does neither.
    if result.would_apply != 1:
        failures.append(
            f"the eligible row was not classified would_apply "
            f"(got {result.would_apply}) — this check proves nothing"
        )
    if before != after:
        failures.append("the sweeper mutated a posterior")
    if audit_rows != 0:
        failures.append("the sweeper wrote a feedback_history row")
    if result.alpha_withheld != round(epsilon, 6):
        failures.append(
            f"alpha_withheld {result.alpha_withheld} does not account for "
            f"the one withheld epsilon ({epsilon})"
        )

    print("  => no residual exposure-as-evidence path: an eligible row is "
          "classified and then withheld, not applied.")
    return failures


def main() -> int:
    if _CLEARED:
        print(f"(cleared ambient: {', '.join(_CLEARED)})\n")

    failures: list[str] = []
    failures += channel_1_exposure()
    failures += channel_2_sentiment()
    with tempfile.TemporaryDirectory() as _tmp:
        failures += channel_3_sweeper(Path(_tmp))

    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    print("  At default settings NO automatic channel moves a belief")
    print("  posterior in EITHER direction. Both automatic channels are")
    print("  opt-in: exposure (up) and sentiment-from-prose (down).")

    if failures:
        print()
        for failure in failures:
            print(f"  FAIL: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
