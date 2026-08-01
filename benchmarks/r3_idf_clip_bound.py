"""Reachability and blast-radius bound for the R3 IDF-clip (#1158 §4, #1174 `19`).

Two umbrellas carry the same claim: that the R3 half of the ratified
``stack-r1-r3`` query strategy is **inert**, because ``transform_query``
tokenises with :func:`aelfrice.bm25.tokenize` (unstemmed) and then looks the
terms up in ``BM25Index.vocabulary``, whose keys come from
:func:`aelfrice.bm25.tokenize_stemmed`. Both prescribe the same one-line fix:
stem before the lookup.

The evidence on file for "inert" is a single hand-built query. This module
measures it, and the measurement does not support the claim. It reports four
quantities, each of which is a decision input for that fix:

1. **How much of R3 actually fires today.** Per query term: resolves in the
   vocabulary as spelled / resolves only after stemming / out-of-vocabulary
   under both spellings.

2. **Whether the boost arm is reachable at all.** This is a property of the
   IDF *distribution*, not of any query. ``high_threshold`` is the 0.75
   quantile of the vocabulary IDF vector; the shipped IDF is Robertson's
   smoothed form, monotone decreasing in document frequency, so its maximum
   is attained at ``df == 1``. If hapax terms are more than
   ``1 - high_quantile`` of the vocabulary — which Zipf makes the ordinary
   case, not the corner case — the 0.75 quantile *equals* ``max(idf)`` and
   ``term_idf > high_threshold`` is unsatisfiable for every term in the
   vocabulary.

3. **What the proposed fix would change.** Not the miss rate.
   ``clip_with_quantile_thresholds`` emits an unresolved term once
   (pass-through) and emits an in-band term once — byte-identical outcomes.
   The fix only changes output for terms that currently miss, resolve after
   stemming, *and* land strictly outside the ``[low, high]`` band. With the
   boost arm unreachable, that set is exactly the newly-**dropped** terms.

4. **Whether the fix empties the query.** ``context_rebuilder`` feeds the
   rewritten string straight into ``retrieve()`` with no empty check, and
   ``_query_for_recent_turns``'s own docstring records what an empty query
   means: ``retrieve()`` returns L0 only. A fix that strips the query is a
   fix that turns a rebuild into a locks-only injection.

Input shape matters and is measured both ways
---------------------------------------------
The production input to ``transform_query`` is **not** a raw prompt. The
rebuilder passes ``_query_for_recent_turns(recent_turns)`` — entity and
triple extraction over the concatenated turn window, deduplicated — so term
rarity there differs from prose. Arm A runs raw prompts (diagnostic only);
arm B reconstructs the production shape through the real extractor.

Determinism
-----------
No sampling, no seed, no model call. Given the same store and the same audit
files the output is byte-identical: ``BM25Index.build`` is deterministic,
the quantiles are a closed form over its IDF vector, and prompt selection is
a fixed filter applied in file order.

Privacy
-------
Aggregate counts only. No prompt text, no belief content, and no query
string is emitted to stdout or to ``--json-out``.

Usage::

    uv run python benchmarks/r3_idf_clip_bound.py \\
        --store .git/aelfrice/memory.db \\
        --audit .git/aelfrice/hook_audit.jsonl \\
        --json-out r3_bound.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Final

import numpy as np

from aelfrice.bm25 import BM25Index, tokenize, tokenize_stemmed
from aelfrice.context_rebuilder import RecentTurn, _query_for_recent_turns
from aelfrice.query_understanding.entity_expand import (
    expand_with_capitalised_entities,
)
from aelfrice.query_understanding.idf_clip import (
    DEFAULT_BOOST_QF,
    DEFAULT_HIGH_QUANTILE,
    DEFAULT_LOW_QUANTILE,
    clip_with_quantile_thresholds,
    compute_idf_quantile_thresholds,
)
from aelfrice.store import MemoryStore

# UserPromptSubmit fires on every harness-injected payload, not only on
# turns a person typed. Tool plumbing is the majority of the log and its
# term distribution is nothing like a query, so it is filtered out.
HARNESS_PREFIXES: Final[tuple[str, ...]] = (
    "<task-notification>",
    "<system-reminder>",
    "<local-command",
    "<command-name>",
    "<command-message>",
    "<user-prompt-submit-hook>",
    "<bash-input>",
    "<bash-stdout>",
    "Caveat: The messages below",
)
_TAG_ONLY: Final[re.Pattern[str]] = re.compile(r"^\s*<[a-z-]+>")

DEFAULT_WINDOW: Final[int] = 8


def is_user_turn(prompt: str) -> bool:
    """True if `prompt` looks like a turn a person typed."""
    stripped = prompt.strip()
    if not stripped or stripped.startswith(HARNESS_PREFIXES):
        return False
    return not _TAG_ONLY.match(stripped)


def load_prompts(paths: list[Path]) -> list[str]:
    """Return de-duplicated user-turn prompts from hook-audit JSONL files.

    Order is file order then line order, so the result is stable. Note the
    audit stores ``prompt_prefix``, which is truncated; term counts per
    prompt are therefore a lower bound on the untruncated turn.
    """
    seen: set[str] = set()
    out: list[str] = []
    for path in paths:
        if not path.exists():
            # Skipping silently would report numbers over a partial corpus
            # with no signal that one of several --audit paths was misspelt.
            print(f"warning: audit path not found, skipping: {path}",
                  file=sys.stderr)
            continue
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("hook") != "user_prompt_submit":
                    continue
                prompt = row.get("prompt_prefix") or ""
                if is_user_turn(prompt) and prompt not in seen:
                    seen.add(prompt)
                    out.append(prompt)
    return out


def clip_stemmed(
    terms: list[str],
    vocabulary: dict[str, int],
    idf: np.ndarray,
    low: float,
    high: float,
    *,
    boost_qf: int = DEFAULT_BOOST_QF,
) -> list[str]:
    """The fix as specified on #1158 §4 / #1174 `19`.

    Identical to `clip_with_quantile_thresholds` except the vocabulary
    lookup is keyed on the Porter stem while the emitted token stays the
    surface form.
    """
    out: list[str] = []
    for term in terms:
        stems = tokenize_stemmed(term)
        idx = vocabulary.get(stems[0]) if stems else None
        if idx is None:
            out.append(term)
            continue
        term_idf = float(idf[idx])
        if term_idf < low:
            continue
        if term_idf > high:
            out.extend([term] * boost_qf)
        else:
            out.append(term)
    return out


def _band(term_idf: float, low: float, high: float) -> str:
    if term_idf < low:
        return "drop"
    if term_idf > high:
        return "boost"
    return "keep"


def reachability(
    index: BM25Index,
    low: float,
    high: float,
) -> dict[str, Any]:
    """Report whether each clip arm can fire on this store's vocabulary."""
    idf = index.idf
    n_docs = len(index.belief_ids)
    idf_max = float(idf.max()) if idf.size else 0.0
    # Robertson smoothed IDF at df == 1, the maximum attainable value.
    hapax_idf = (
        float(np.log(1.0 + (n_docs - 1 + 0.5) / 1.5)) if n_docs else 0.0
    )
    n_hapax = int((idf >= idf_max - 1e-6).sum()) if idf.size else 0
    n_boostable = int((idf > high).sum()) if idf.size else 0
    n_droppable = int((idf < low).sum()) if idf.size else 0
    # Invert the IDF form to report the cutoff as a document frequency,
    # which is the interpretable unit: "drops any term seen in >= N beliefs".
    #
    # The shipped IDF is `log(1 + (N - df + 0.5) / (df + 0.5))`, so with
    # `E = exp(low) - 1`:
    #
    #     E * (df + 0.5) = N - df + 0.5
    #     df * (E + 1)   = N + 0.5 - 0.5 * E
    #     df             = (N + 0.5 - 0.5 * E) / (E + 1)
    #
    # Inverting `log(1 + (N + 0.5) / (df + 0.5))` instead — i.e. dropping the
    # `- df` from the numerator — agrees to 0.02% at this store's operating
    # point, where `E >> 1` and both forms are dominated by `N / E`. It
    # diverges badly as the cutoff falls: 1.9% at idf 4, 15.7% at idf 2,
    # 58.2% at idf 1. Re-runnability on a smaller or less Zipfian corpus is
    # the whole point of this harness, so it uses the exact inverse.
    if low > 0.0:
        exp_low_minus_1 = float(np.exp(low)) - 1.0
        df_at_low = (
            (n_docs + 0.5 - 0.5 * exp_low_minus_1) / (exp_low_minus_1 + 1.0)
        )
    else:
        df_at_low = float("nan")
    return {
        "n_docs": n_docs,
        "vocabulary": len(index.vocabulary),
        "idf_low": low,
        "idf_high": high,
        "idf_max": idf_max,
        "idf_at_df_1": hapax_idf,
        "hapax_terms": n_hapax,
        "hapax_share": n_hapax / len(index.vocabulary)
        if index.vocabulary
        else 0.0,
        "boostable_vocab_terms": n_boostable,
        "droppable_vocab_terms": n_droppable,
        "df_at_low_cutoff": df_at_low,
        "boost_arm_reachable": n_boostable > 0,
    }


def census(
    queries: list[str],
    index: BM25Index,
    low: float,
    high: float,
) -> dict[str, Any]:
    """Per-term census plus the blast radius of the stemming fix."""
    vocab, idf = index.vocabulary, index.idf
    counts: Counter[str] = Counter()
    newly: Counter[str] = Counter()
    changed = empty_now = empty_fixed = 0

    for query in queries:
        expanded = expand_with_capitalised_entities(query, tokenize(query))
        for term in expanded:
            counts["terms"] += 1
            hit = vocab.get(term)
            if hit is not None:
                counts["resolves_today"] += 1
                counts[f"today_{_band(float(idf[hit]), low, high)}"] += 1
                continue
            stems = tokenize_stemmed(term)
            hit = vocab.get(stems[0]) if stems else None
            if hit is None:
                counts["oov_both"] += 1
                continue
            counts["newly_resolves"] += 1
            newly[_band(float(idf[hit]), low, high)] += 1

        current = clip_with_quantile_thresholds(expanded, vocab, idf, low, high)
        fixed = clip_stemmed(expanded, vocab, idf, low, high)
        if expanded and not current:
            empty_now += 1
        if expanded and not fixed:
            empty_fixed += 1
        if current != fixed:
            changed += 1

    return {
        "queries": len(queries),
        "counts": dict(counts),
        "newly_resolving_band": dict(newly),
        "terms_changed_by_fix": newly["drop"] + newly["boost"],
        "queries_changed_by_fix": changed,
        "queries_empty_today": empty_now,
        "queries_empty_after_fix": empty_fixed,
    }


def _pct(numerator: int, denominator: int) -> str:
    return f"{100.0 * numerator / denominator:.1f}%" if denominator else "n/a"


def _report(label: str, result: dict[str, Any]) -> None:
    counts = result["counts"]
    newly = result["newly_resolving_band"]
    total = counts.get("terms", 0)
    n_q = result["queries"]
    print(f"\n=== {label}  (n={n_q} queries, {total} terms) ===")
    print(
        f"  resolve today            : {counts.get('resolves_today', 0):6d}"
        f"  {_pct(counts.get('resolves_today', 0), total)}"
        f"   drop={counts.get('today_drop', 0)}"
        f" boost={counts.get('today_boost', 0)}"
        f" keep={counts.get('today_keep', 0)}"
    )
    print(
        f"  miss now, hit if stemmed : {counts.get('newly_resolves', 0):6d}"
        f"  {_pct(counts.get('newly_resolves', 0), total)}"
        f"   would-drop={newly.get('drop', 0)}"
        f" would-boost={newly.get('boost', 0)}"
        f" in-band(no change)={newly.get('keep', 0)}"
    )
    print(
        f"  OOV under both spellings : {counts.get('oov_both', 0):6d}"
        f"  {_pct(counts.get('oov_both', 0), total)}"
    )
    print(
        f"  fix changes              : "
        f"{result['terms_changed_by_fix']} terms"
        f" ({_pct(result['terms_changed_by_fix'], total)})"
        f", {result['queries_changed_by_fix']}/{n_q} queries"
        f" ({_pct(result['queries_changed_by_fix'], n_q)})"
    )
    print(
        f"  empty rewritten query    : today"
        f" {result['queries_empty_today']}/{n_q}"
        f"  ->  after fix {result['queries_empty_after_fix']}/{n_q}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--audit", required=True, nargs="+", type=Path)
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help="turns per reconstructed rebuild window (arm B)",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    prompts = load_prompts(args.audit)
    if not prompts:
        print("no user-turn prompts in the audit files", file=sys.stderr)
        return 2

    store = MemoryStore(str(args.store))
    try:
        index = BM25Index.build(store)
    finally:
        store.close()
    low, high = compute_idf_quantile_thresholds(
        index.idf, DEFAULT_LOW_QUANTILE, DEFAULT_HIGH_QUANTILE,
    )

    reach = reachability(index, low, high)
    print("=== clip-arm reachability (a property of the store, not a query) ===")
    print(f"  beliefs {reach['n_docs']}, vocabulary {reach['vocabulary']}")
    print(f"  IDF band        : low={low:.4f}  high={high:.4f}")
    print(
        f"  max IDF in vocab: {reach['idf_max']:.4f}"
        f"   (analytic df=1: {reach['idf_at_df_1']:.4f})"
    )
    print(
        f"  hapax (df==1)   : {reach['hapax_terms']}"
        f" = {_pct(reach['hapax_terms'], reach['vocabulary'])} of vocabulary"
    )
    print(f"  vocab terms idf >  high (boostable): "
          f"{reach['boostable_vocab_terms']}")
    print(f"  vocab terms idf <  low  (droppable): "
          f"{reach['droppable_vocab_terms']}")
    if not reach["boost_arm_reachable"]:
        print(
            "  ** boost arm UNREACHABLE: no vocabulary term satisfies"
            " `term_idf > high_threshold` **"
        )
    print(
        f"  low cutoff drops any query term seen in >="
        f" {reach['df_at_low_cutoff']:.1f} beliefs"
    )

    arm_a = census(prompts, index, low, high)
    _report("A. raw user prompts (diagnostic; NOT the production input)", arm_a)

    windows: list[str] = []
    for start in range(0, len(prompts), args.window):
        turns = [
            RecentTurn(role="user", text=text, session_id="s", ts=None)
            for text in prompts[start:start + args.window]
        ]
        query = _query_for_recent_turns(turns)
        if query.strip():
            windows.append(query)
    arm_b = census(windows, index, low, high)
    _report(
        f"B. production shape: _query_for_recent_turns(window={args.window})",
        arm_b,
    )

    if args.json_out:
        # `df_at_low_cutoff` is NaN when `low == 0.0`, and `json.dumps` would
        # emit the bare token `NaN`, which RFC 8259 does not permit and strict
        # parsers reject. Serialise it as null instead.
        reach_json = dict(reach)
        df_cut = reach_json.get("df_at_low_cutoff")
        if isinstance(df_cut, float) and math.isnan(df_cut):
            reach_json["df_at_low_cutoff"] = None
        args.json_out.write_text(
            json.dumps(
                {
                    "reachability": reach_json,
                    "arm_a_raw_prompts": arm_a,
                    "arm_b_production_shape": arm_b,
                    "window": args.window,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
