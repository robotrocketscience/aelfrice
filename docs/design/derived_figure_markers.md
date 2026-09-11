# Derived-figure markers: where the seven #1469 instances stand

[#1469](https://github.com/robotrocketscience/aelfrice/issues/1469) requires
that each of the seven published figures it names is either annotated with a
marker the gate understands or explained as out of scope. This page is that
record. It is the ledger, not the specification: for the marker grammar, the
two figure classes, and what CI does with each, read the module docstring in
[`scripts/check_derived_figures.py`](../../scripts/check_derived_figures.py).

Read a row as the answer to one question: if this figure went stale tomorrow,
what would catch it?

## The rule this ledger applies

A marker is only worth adding when the figure behind it can be re-derived. A
marker that stamps a value nobody can recompute does not guard the figure; it
publishes a second unchecked number beside the first and makes the entry look
guarded. So a figure gets a marker when a producer can emit it, and gets a
written reason when it cannot. Three of the seven fall on the second side, and
saying so is the deliverable for them.

## The seven

### #1445 — the Stop-block bound and its reduction factor

**Annotated, both halves of the row.** The markers live in
`CHANGELOG/v4.md` and again in `src/aelfrice/hook.py`, because the figures
ship in both.

`benchmarks/published_constants.py#stop_prompt_max_items` and
`#stop_prompt_max_content` are store-free: they read the shipped constants, and
CI re-runs them. The two rendered-byte maxima are store-backed —
`benchmarks/stop_prompt_block_bounds.py` measures a rendered block against a
real belief store — so their markers carry the corpus label, the measurement
date, and a `producer-sha=` stamp, and CI holds them to self-consistency and to
code staleness rather than to a value it cannot recompute.

**The reduction factor is the half this row was missing.** The entry publishes
"a 299.7x reduction" beside the two maxima it divides, and the first pass left
it unmarked while calling the row annotated — `--list-unmarked` listed
`299.7x`, which is the gate reporting the gap the ledger denied. A ratio of two
measured values is arithmetic, not a third measurement, so
`stop_prompt_block_bounds.py` now emits it as
`post_1315.worst_case_reduction_factor` and the entry carries a third
store-backed marker. `tests/test_derived_figures_1469.py` reads all three
markers out of the shipped file and asserts the factor is the rounded ratio of
the other two, so the trio cannot drift apart without a test going red.

Adding that key changed the producer's bytes, so the stamps on the existing
markers were re-run through `--restamp`. That is the restamp the flag's
docstring permits: the new key is computed from two values the same function
already produced, and the edit provably cannot move them. Proven rather than
asserted — `measure()` from the pre-edit file and from the shipped one, run
over one synthetic store, agree on every key the pre-edit file emitted.

### #1446 — the rate pair "2.30x (8.69% versus 20.00%)"

**Out of scope, because the figure was withdrawn rather than corrected.**
`benchmarks/sidecar_rebuild_rate.py` now states that the pair is a worked
example on a constructed log and that no live ratio exists to cite: the
per-fire outcome field is written only by code that is not yet in the released
package, so every row on the real log predates it and the script reports
`NO MEASUREMENT YET` before printing any rate.

There is no value for a marker to bind. A marker here would name a producer
that deliberately emits nothing, which is a worse claim than the prose makes.
When the field has shipped long enough to produce a live rate, that
measurement is the one to mark.

### #1447 — "135 genuine pasted commands", "of 788 prompts", "762 of 805"

**Out of scope for this branch, and deferred rather than dismissed.** The
figures were corrected on `main`. They are measured over an archived
user-prompt corpus that does not live in this repository and must not be
copied to a public runner ([#1456](https://github.com/robotrocketscience/aelfrice/issues/1456)).

Marking them needs two things this repository does not record: which corpus
snapshot produced them and on what date. Both are required attributes of a
store-backed marker, and inventing either would ship exactly the defect #1469
exists to stop. The next person to re-run that measurement should stamp it
then, when both halves are known.

### #1449 — "44,683 active beliefs" against "44,687" in the same pull request

**Deferred; needs a belief store.** The branch's finding is that these are not
an arithmetic error: they are two real snapshots of one store taken five days
apart, and only a date distinguishes them. That finding is recorded at
`scripts/check_derived_figures.py`'s `CORPUS_RE`.

The gate's self-consistency check is precisely the check this instance needed,
and it runs without a store. What blocks annotation is narrower: nothing in the
repository records the date of the 44,683 snapshot, and a store-backed marker
must carry one. Marking the two figures under one key would also be wrong —
they are different measurements, so they need different keys, and inventing a
date to separate them is the defect wearing a marker.

To close this row, re-measure both counts against a named store, then stamp
each site with its own corpus label and date.

### #1450 — "the wrapper shipped in v1.3.0", later "v1.0.1"

**Corrected, and deliberately not marked.** `class RetrievalCache` is absent
from `src/` at `v1.0.1` and at `v1.0.2` and present at `v1.0.3`, added by
`0c27a937`. Every sentence in [`bfs_multihop.md`](bfs_multihop.md) and
[`bayesian_ranking.md`](bayesian_ranking.md) that dated the *wrapper* now reads
v1.0.3, except the one that quotes the spec's own v1.0.1 wording and says in
the next clause that the quotation is the target and not the ship date. No
count is published for how many sites that was: the honest way to see them is
to read them, so run

```
git grep -n 'RetrievalCache' docs/design/bfs_multihop.md docs/design/bayesian_ranking.md
```

To re-derive the ship version itself, run:

```
git grep -l 'class RetrievalCache' v1.0.1 -- src   # no output, exit 1
git grep -l 'class RetrievalCache' v1.0.2 -- src   # no output, exit 1
git grep -l 'class RetrievalCache' v1.0.3 -- src   # src/aelfrice/retrieval.py
```

A marker does not fit this figure. The grammar binds a number a producer
emits from the shipped code, and a first-shipped-at version is a property of
git history instead — it needs tags a shallow CI checkout may not have. The three
tag greps above are the re-derivation, and they are recorded in the prose at
each corrected site.

**The first pass of this sweep over-corrected, and that is the part worth
recording.** Not every `v1.0.1` near the cache dates the wrapper. The
store-level invalidation registry — the wipe-on-write policy — really did ship
at v1.0.1: `add_invalidation_callback` is present at that tag, and
`insert_edge`, `update_edge` and `delete_edge` all call `_fire_invalidation()`
there. A sentence about the policy therefore says v1.0.1 and is right. The
sites that do are `bfs_multihop.md`'s § Non-goals bullet and its
decision-table row, both of which the first pass changed and this one puts
back, and the reference in
[`src/aelfrice/retrieval.py`](../../src/aelfrice/retrieval.py), which it left
alone. § Cache invalidation in `bfs_multihop.md` now states the test — does
this sentence date the class or the policy? — beside both dates.

This class of error is invisible to the gate in this branch, and deliberately
so: `_MASKS` strips dotted version strings before figure extraction, because a
version is not a measured figure. A version literal is checked by reading it,
not by running the gate.

### #1451 — "thirteen mutations" over a list that enumerated twelve

**Corrected and annotated, and this is the one instance the gate now holds to
a producer.** The entry publishes `13`, in digits so a scanner can see it, and
carries a marker naming
`benchmarks/published_constants.py#ci_manual_dispatch_mutations`. That key
parses `tests/test_ci_manual_dispatch.py` and counts its top-level `test_`
functions, one per listed mutation.

Proven by mutation: appending a fourteenth top-level test makes the producer
emit 14 against the published 13, and `check_derived_figures.py --mode all`
exits 1 with `the figure is stale: re-derive it, or fix the producer`.

**What that does not cover, which is the half #1451 was actually about.** The
gate binds the published 13 to the producer and to a figure reading 13 in the
annotated text. It does not bind it to the length of the list the sentence
introduces. Delete two of the thirteen enumerated mutations from the prose and
leave the tests alone, and the producer still emits 13, the marker still finds
a 13 in the text, and `--mode text`, `--mode all` and
`tests/test_derived_figures_1469.py` all stay green — which is #1451's exact
shape, a published count over a shorter list. `check_derived_figures.py` has no
enumeration-length check anywhere, and this branch did not add one: counting
list items in prose means deciding which commas separate items, and the live
sentence contains a parenthesised triple that a comma counter reads as three
more mutations. A rule that mis-counts a legitimate sentence gets switched off,
and a rule that only works on one sentence shape reports green over every other
one — the #1160 failure this repository has already paid for.

So what this instance gained is a producer for the count, not coverage of the
enumeration. The list is held to the test file by reading, and
`tests/test_derived_figures_1469.py::test_the_gate_does_not_check_an_enumeration_against_its_own_list`
pins that boundary so the claim stays falsifiable.

### #1452 — "restoring `shlex.split` turns 2 red"

**Corrected, and not markable.** A mutation count has no producer: it is the
result of reverting a named line and running the suite, and nothing in the
repository emits it. What it can have is a site, which is what the original
sentence lacked and what made the count impossible to check.

Both readings were run with `uv run pytest -q -p no:randomly`. The suite size
they were taken over is deliberately not published: it moves on every merge,
and the figure this branch first wrote for it already disagreed with the tree
that shipped it. Replacing the body of `command_tokens` in
`src/aelfrice/launcher.py` with a bare `shlex.split` turns 29 red. Replacing
the single `launcher.command_tokens(stripped)` call in
`src/aelfrice/doctor.py` turns 1 red. Neither is the 2 the entry published or
the 3 the issue's table asserts, and the entry now publishes both counts with
their sites.

Three sibling counts in the same sentence recorded no site either, and their
figures are withdrawn rather than restated. The nearest reading of "reverting
the key derivation" turns 33 red against the 6 that shipped, which is the
evidence that a site-less mutation count cannot be recovered after the fact.

## What this ledger does not cover

Every other published figure in the repository is unmarked and stays legal.
The annotation backlog is the whole corpus, and a gate that failed on it would
be switched off within a day. Run
`python3 scripts/check_derived_figures.py --list-unmarked <path>` to enumerate
what is still unannotated in a file, so grandfathered is not the same as
invisible.

That claim was false when it was first written. Inline code was masked before
figures were extracted, so a figure in code font — this repo's house style for
a measured value — was in nothing the gate lists and in nothing it checks. The
live instance was the #1356 entry in `CHANGELOG/v4.md`, whose two headline
shares are published as `` `93.69%` `` and `` `94.86%` ``: `--list-unmarked`
named neither, and the hard overclaim rule could be satisfied by typing two
backticks. A code span whose whole content is a number is now a figure; a span
carrying anything else is still masked, because a false positive there makes
the overclaim rule unsatisfiable. Run
`python3 scripts/check_derived_figures.py --mask-delta` to price the three
candidate rules against the current tree rather than take a number for it.

**What the gate still cannot see, stated rather than implied.** A figure inside
a code span that also carries words — `` `93.69% of rows` `` — is masked, so
the two-backtick bypass is narrowed and not closed. A version literal is masked
outright (`_MASKS` strips dotted versions), which is how the over-correction
recorded under #1450 above got past every check in this branch. And an
enumerated count is not held to the length of the list it introduces; see
#1451.
