"""The bench-gate verdict protocol: the vocabulary both ends share.

Two modules speak this vocabulary and neither owns it. The guards in
`tests/bench_gate/null_model.py` *write* a verdict through
`record_property`; the terminal summary in `tests/conftest.py` *reads*
it back off the reports and classifies the tier on it. Keeping the
strings in either module makes the other import it, and importing in
both directions is what produced the cycle this module exists to break:
`null_model` imported `conftest` for these keys while `conftest`
imported `null_model` for the registry, and the only reason that worked
was that one of the two imports was deferred into a function body. A
cycle that resolves by import timing fails late and confusingly — here,
under the summary hook, where the failure would be read as a broken
tier rather than a broken import.

So this module is a leaf. It imports nothing from `tests`, nothing from
`aelfrice`, and nothing that can import it back. Anything added here
has to keep that property: it is a shared vocabulary, not a shared
utility drawer.
"""
from __future__ import annotations

BENCH_MEASUREMENT_PROPERTY = "bench_measurement"
"""`record_property` key a bench-gate test uses to report its numbers.

A gate that only prints on failure records nothing on the runs that
matter most — the green ones at a release cut, which are the only
evidence that the measurement was taken at all and the only place a
drift between cuts would show. Inverted checks make this acute: a
tripwire is *expected* to be green, so a message built solely inside its
assertion is dead code in the shipped path.

The summary prints whatever tests attach under this key, so the numbers
land in the same block that already says which modules ran.
"""

BENCH_NULL_VERDICT_PROPERTY = "bench_null_verdict"
"""`record_property` key carrying one gate's null-model verdict (#1581).

Value shape: `"<corpus module>|<state>|<why>"`, where the state is one
of the three constants below.

Separate from `BENCH_MEASUREMENT_PROPERTY` because the summary has to
*classify* on it, not just print it: a corpus the null model defeated is
its own tier state, distinct from a gate that executed and from a module
that produced no verdict at all. Folding it into the free-text
measurement line would leave the summary parsing prose.
"""

BENCH_VERDICT_ACCEPT = "ACCEPT"
"""The null model ran, did not clear the bar, and the shipped arm scored."""

BENCH_VERDICT_REJECT = "REJECT"
"""The corpus does not count: its own null model defeats the gate."""

BENCH_VERDICT_UNVERIFIED = "UNVERIFIED"
"""The null model ran, but the shipped arm produced no score.

Distinct from `BENCH_VERDICT_REJECT`, which is a finding about the
corpus, and from `BENCH_VERDICT_ACCEPT`, which asserts a graded run
happened. A gate whose shipped arm raises has graded nothing, so it is
not an executed test — but the null model it did run is still evidence
and is still recorded.
"""

NO_VERDICT_RECORDED = (
    "the test ran against the corpus but recorded no null-model verdict "
    "— it calls no guard, or its guard call never executed"
)
"""Why a scored module's bench-gated test was not counted as executed.

The AST wiring check in `tests/test_bench_gate_null_model_1581.py` sees
a `guard_*_gate(...)` call that is present; it cannot see one that is
unreachable, wrapped in a false branch, or short-circuited by an earlier
return. Requiring the property at summary time is what closes that gap:
a guard that does not run leaves no verdict, and a report with no
verdict is not evidence.
"""
