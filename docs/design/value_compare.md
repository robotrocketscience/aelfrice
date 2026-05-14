# Value-comparison contradiction gate (#422)

`aelfrice.value_compare` is the typed-slot relatedness gate that
replaces the residual-overlap floor in `relationship_detector` for
contradiction detection. Stdlib-only, deterministic, no embeddings.

## Why it exists

The R2 detector from #201 used residual-Jaccard token overlap to
decide if two beliefs were "about the same subject" before checking
modality / quantifier disagreement. On the labeled adversarial
corpus this gate caught 2 of 60 (`recall = 0.033`) — the failure
mode is paraphrase: real natural-language contradictions almost
never share enough surface tokens to clear the overlap floor.

The v3 gate sidesteps this by extracting **typed slots** from each
belief and firing `contradicts` when the two beliefs disagree on a
slot value, regardless of token overlap. Slot match is the
relatedness signal.

## What gets extracted

`extract_values(text)` returns a `ValueSlots` with two tuples:

**Numeric slots** — `NumericSlot(key, value)`. The `key` is the
alphabetic token immediately preceding a number (with optional
``=`` / ``:`` / ``is`` / ``of`` / ``to`` / ``equals`` separator).
``value`` is parsed as float. Examples that match:

| input | extracted |
|---|---|
| ``alpha = 0.5`` | ``(alpha, 0.5)`` |
| ``timeout: 30`` | ``(timeout, 30.0)`` |
| ``set retries to 3`` | ``(retries, 3.0)`` |
| ``max_depth=4 depth=2`` | ``(max_depth, 4.0)`` and ``(depth, 2.0)`` |

Filler keys (``is``, ``of``, ``the``, ``a``, …) are dropped — see
`_NUMERIC_KEY_DROP`. Unit-aware comparison is **out of scope** for
v3: the regex's greedy capture of trailing tokens introduced too
many false negatives (e.g. ``alpha = 0.5 prior`` vs ``alpha = 1.0
in config`` produced different "units" — ``prior`` vs ``in`` — and
silently skipped the conflict). If unit-aware comparison becomes
needed, file a separate issue with a curated unit vocabulary.

**Enum slots** — `EnumSlot(category, group_id, member)`. Members
come from `ENUM_VOCAB`, a curated taxonomy of 9 categories grouped
by mutual exclusion:

| category | groups |
|---|---|
| `execution_mode` | `{sync, synchronous}`, `{async, asynchronous}` |
| `default_state` | `{default-on, enabled}`, `{default-off, disabled}` |
| `storage_mode` | `{indexed}`, `{scan, full-scan, table-scan}` |
| `completeness` | `{full}`, `{incremental}`, `{partial}` |
| `strictness` | `{strict}`, `{lax, permissive}` |
| `necessity` | `{required}`, `{optional}` |
| `visibility` | `{public}`, `{private}` |
| `access_mode` | `{readonly, read-only}`, `{writable, read-write}` |
| `determinism` | `{deterministic}`, `{non-deterministic, nondeterministic, stochastic}` |

Members within a group are aliases — they do **not** conflict with
each other. `group_id` is the alphabetically-first member of the
group, used as a stable cross-belief identifier. Adding a category
extends the contradiction surface; the dict in
``src/aelfrice/value_compare.py`` is the single source of truth.

## When the comparator fires

`find_conflicts(slots_a, slots_b)` returns a tuple of `SlotConflict`:

- **Numeric conflict**: same key, values outside relative tolerance
  (default 1%). The 0/0 case is silent (degenerate denominator).
- **Enum conflict**: same category, disjoint group_id sets. Aliases
  collapse to one group, so `sync` vs `synchronous` is silent.

When the comparator returns multiple distinct conflicts on the same
pair, all of them are surfaced — the integration layer decides how
many it takes to fire.

## Integration

`relationship_detector.analyze(a, b, use_value_comparison=True)`
runs the gate **before** the residual-overlap floor. If any
conflict is found:

```
RelationshipVerdict(
    label="contradicts",
    score=1.0,
    residual_overlap=0.0,
    rationale="value_comparison:numeric=alpha",
)
```

Score is pinned at 1.0 so the auto-emit policy (#422 acceptance #3)
can use a single threshold for "slot fired" vs "modality-only
fired" verdicts.

`use_value_comparison=False` (the default) preserves v1 behaviour
byte-for-byte — the v3 path adds zero overhead until flipped.

## Determinism

Same `(text_a, text_b)` produces byte-identical slots and identical
conflicts across runs. No embeddings, no learned classifiers, no
random seeds. This is load-bearing for replay-equality (#262 / #403)
and bench-gate reproducibility (per the v2.0 README rationale).

## Bench gate

`tests/bench_gate/test_contradiction_v3.py` runs the v3 path against
the labeled adversarial `contradiction` corpus and asserts:

- recall on `contradicts` ≥ 0.5
- precision on `contradicts` ≥ 0.7

Skip-on-no-corpus on public CI. Below-floor blocks the v3 default-on
flip; the operator decides whether to widen `ENUM_VOCAB`, tune
`DEFAULT_NUMERIC_REL_TOL`, or accept the precision/recall trade-off
under audit-only surface (`aelf resolve` style human-in-loop).

## Maintenance

Adding a category:
1. Append the entry to `ENUM_VOCAB` with mutually-exclusive groups.
2. Pin a unit test in `tests/test_value_compare.py` covering the
   new contradiction shape.
3. Re-run the bench gate. If recall/precision drop, the new
   category is too noisy — narrow it or revert.

Tuning numeric tolerance: `DEFAULT_NUMERIC_REL_TOL` is pinned at 1%.
Drift on this value silently changes the gate; a unit test in
`test_value_compare.py` pins it explicitly.
