"""Schema validation for the v2.0 evaluation corpus (#307, #1580).

Walks every `*.jsonl` under `<corpus root>/<module>/` and enforces:

  1. Each line parses as JSON.
  2. Every row has the common envelope (`id`, `provenance`, `labeller_note`,
     `label`) with non-empty strings.
  3. `id` values are unique within a module.
  4. `label` is one of the module-specific allowed set, and — where the
     consuming detector exposes its own label constant — one the scorer
     can actually return.
  5. Module-specific extra fields exist with the right shape.

The corpus root comes from `AELFRICE_CORPUS_ROOT` when it is set, and
falls back to the public tree under `tests/corpus/v2_0/`. That fallback
holds `.gitkeep` files and a README and no rows at all, so before #1580
every assertion above skipped on every run and no corpus row had ever
been validated. Two things follow from that history:

* A skip names the root it inspected and says nothing was validated. A
  silent skip is what hid the problem for as long as it hid.
* A module that runs reports how many rows it validated, and fails if
  that count is zero, so "validated 0 rows" cannot read as a pass. The
  count is reported whether the module passes or fails — a failure
  with no count leaves the reader unable to tell whether validation
  read 3 rows or 500 before it stopped, which is the run where the
  number is most load-bearing.

Violations the corpus carries today are recorded in `KNOWN_FAILURES`
with a reason each rather than fixed here or waived: turning the
validator on and relabelling the corpus are different decisions, and
#1580 is only the first. The list is a ratchet in both directions — a
violation that is not recorded fails the module, and a recorded one
that stops reproducing fails it too, so the list cannot rot.

The ≥50/module v0.1 threshold is **not** asserted here — that flips on
once labelling is complete across the listed modules. See
`tests/corpus/v2_0/README.md` for the schema contract. `dedup`,
`enforcement`, and `promotion_trigger` are absent on purpose: #1579 retired
them because the code their bench gates graded does not exist.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from aelfrice.relationship_detector import VERDICT_LABELS
from conftest import CORPUS_SCHEMA_PROPERTY

CORPUS_ENV_VAR = "AELFRICE_CORPUS_ROOT"

PUBLIC_CORPUS_ROOT = Path(__file__).parent / "corpus" / "v2_0"
"""The in-repo scaffold. Holds directories and a README, never rows."""

# Synthetic provenance must follow `synthetic-vN.M` per the README carve-out
# (`tests/corpus/v2_0/README.md` § v0.1 acceptance). The format guarantees
# future re-labellers can pivot off the version suffix instead of free-text.
SYNTHETIC_PROVENANCE_RE = re.compile(r"^synthetic-v\d+\.\d+$")


# Module → (allowed labels, extra-required-fields-spec)
# Spec values: "str" = non-empty string, "list[str]" = non-empty list of strings.
MODULES: dict[str, tuple[set[str], dict[str, str]]] = {
    "contradiction": (
        # Spelt out rather than derived from
        # `relationship_detector.VERDICT_LABELS`, even though the two must
        # agree. Deriving it would make them agree by construction, and
        # `test_module_label_vocabulary_matches_detector_constant` — the
        # test whose whole job is to catch them drifting apart — could
        # then never fail. One side has to be a literal for the
        # comparison to mean anything.
        {"contradicts", "refines", "unrelated"},
        {"belief_a": "str", "belief_b": "str"},
    ),
    "wonder_consolidation": (
        {"1", "2", "3", "4", "5"},
        {"seed_belief": "str", "retrieved_neighbors": "list[str]"},
    ),
    "sentiment": (
        {"positive", "negative", "neutral"},
        {"user_message": "str"},
    ),
    "directive_detection": (
        {"directive", "not_directive"},
        {"prompt": "str"},
    ),
    "bfs_relates_to": (
        {"graded"},
        {
            "beliefs": "list[belief]",
            "edges": "list[edge]",
            "seed_ids": "list[str]",
            "expected_hit_ids": "list[str]",
            "k": "int",
        },
    ),
    "implements_edge": (
        {"graded"},
        {
            "beliefs": "list[belief]",
            "edges": "list[edge]",
            "seed_ids": "list[str]",
            "expected_hit_ids": "list[str]",
            "k": "int",
        },
    ),
    "derived_from_edge": (
        {"graded"},
        {
            "beliefs": "list[belief]",
            "edges": "list[edge]",
            "seed_ids": "list[str]",
            "expected_hit_ids": "list[str]",
            "k": "int",
        },
    ),
    "temporal_next_edge": (
        {"graded"},
        {
            "beliefs": "list[belief]",
            "edges": "list[edge]",
            "seed_ids": "list[str]",
            "expected_hit_ids": "list[str]",
            "k": "int",
        },
    ),
    "tests_edge": (
        {"graded"},
        {
            "beliefs": "list[belief]",
            "edges": "list[edge]",
            "seed_ids": "list[str]",
            "expected_hit_ids": "list[str]",
            "k": "int",
        },
    ),
    # #154 v1.7 default-on flip — per-flag NDCG@k uplift over a graded
    # ranking. Same beliefs+edges shape as the BFS modules; expected
    # output is an *ordered* list (top-relevant first) rather than a
    # set, since NDCG cares about position.
    "retrieve_uplift": (
        {"graded"},
        {
            "query": "str",
            "beliefs": "list[belief]",
            "edges": "list[edge]",
            "expected_top_k": "list[str]",
            "k": "int",
        },
    ),
    # #421 — edge-type-keyed rerank consumer. Same graded-row shape as
    # the Track A bench fixtures plus `stale_ids`: the subset of belief
    # ids that have ≥1 ``POTENTIALLY_STALE`` incoming edge in the row's
    # `edges` list. The bench gate measures ≥1pp@k drop in stale-tagged
    # retrieval after the rerank pass vs. before.
    "bfs_potentially_stale": (
        {"graded"},
        {
            "beliefs": "list[belief]",
            "edges": "list[edge]",
            "seed_ids": "list[str]",
            "expected_hit_ids": "list[str]",
            "stale_ids": "list[str]",
            "k": "int",
        },
    ),
    # #389 Track B — `aelf reason` ship gate. Same row structure as
    # bfs_relates_to (the gate measures hit@k uplift over a graph) plus
    # a `query` field for BM25 seed selection on the runtime path.
    "reasoning": (
        {"graded"},
        {
            "query": "str",
            "beliefs": "list[belief]",
            "edges": "list[edge]",
            "expected_hit_ids": "list[str]",
            "baseline_search_only_top_k": "list[str]",
            "k": "int",
        },
    ),
    # #389 Track B — `aelf wonder` ship gate. The on-line surface has a
    # narrower contract: row supplies a graph + a seed + the set of
    # belief ids the wonder pass should surface. No baseline arm — gate
    # is recall@10 on rows where ≥1 expected candidate is in top-10.
    "wonder_online": (
        {"graded"},
        {
            "beliefs": "list[belief]",
            "edges": "list[edge]",
            "seed_id": "str",
            "expected_candidate_ids": "list[str]",
        },
    ),
    # #436 intentional clustering. cluster_coverage@k uplift on the
    # multi-fact corpus; expected_clusters partitions expected_belief_ids
    # into the labeller's cluster groupings so the bench can distinguish
    # "found two beliefs from one cluster" from "found two clusters."
    "multi_fact": (
        {"graded"},
        {
            "query": "str",
            "expected_belief_ids": "list[str]",
            "expected_clusters": "list[list_str]",
            "n_clusters_required": "int",
            "tag": "str",
        },
    ),
    # #435 doc linker. NDCG@k uplift on a labelled query/belief/anchor
    # fixture. `expected_belief_ids` floors the parity check (top-k must
    # contain them in both runs); `expected_doc_uris` are the anchors
    # written against those beliefs in the populated arm only.
    "doc_linker": (
        {"graded"},
        {
            "query": "str",
            "beliefs": "list[belief]",
            "expected_belief_ids": "list[str]",
            "expected_doc_uris": "list[str]",
            "k": "int",
        },
    ),
    # #819 labeled rerank-relevance corpus — bench unblock for #769 / #724 /
    # #800 R5 / #817 flip-default. Per-row: a query, a candidate-belief
    # pool, and the labeller-judged gold relevant set at rank k. Optional
    # `gold_ordering` (not enforced here) carries a full preference order
    # when assignable; consumers that only need recall/precision@k read
    # `gold_top_k` and ignore `gold_ordering` when absent.
    "rerank_relevance": (
        {"graded"},
        {
            "query": "str",
            "beliefs": "list[belief]",
            "gold_top_k": "list[str]",
            "k": "int",
        },
    ),
}

COMMON_REQUIRED = ("id", "provenance", "labeller_note", "label")


DETECTOR_LABEL_CONSTANTS: dict[str, frozenset[str]] = {
    "contradiction": frozenset(VERDICT_LABELS),
}
"""Module -> the label constant its consuming detector exposes.

A corpus label the scorer can never return is not a near miss, it is a
row graded wrong whatever the detector does. Validating the module's
vocabulary against the runtime constant catches that at the corpus,
where it is one relabelling, instead of at the gate, where it surfaces
as unexplained accuracy loss.
"""

# Every violation the row validator can report. Rule ids are stable
# strings because `KNOWN_FAILURES` keys off them; a typo there would
# waive nothing while looking like it waived something, so
# `test_known_failures_name_rules_the_validator_can_emit` pins them.
RULE_PROVENANCE_FORMAT = "provenance-format"
RULE_LABEL_VOCABULARY = "label-vocabulary"
RULE_LABEL_UNSCOREABLE = "label-unscoreable"
RULE_DUPLICATE_ID = "duplicate-id"


def _envelope_rule(field: str) -> str:
    return f"envelope:{field}"


def _field_rule(field: str) -> str:
    return f"field:{field}"


KNOWN_FAILURES: dict[str, tuple[tuple[str, str], ...]] = {
    "directive_detection": (
        (
            _envelope_rule("provenance"),
            "The v0_1 batch (285 rows) predates the envelope and carries no "
            "`provenance` at all; the v0_2 batch (225 rows) carries a "
            "conforming `synthetic-v0.2`. Backfilling the older batch is a "
            "corpus decision, not a validator one (#1580, out of scope).",
        ),
        (
            RULE_LABEL_VOCABULARY,
            "Both batches spell the negative class `not-directive` (304 of "
            "510 rows) where this table declares `not_directive`. Either the "
            "corpus or the table is wrong and the consumer decides which; "
            "until then every one of those rows is unscoreable (#1580).",
        ),
    ),
    "contradiction": (
        (
            RULE_LABEL_VOCABULARY,
            "The labelled contradiction batch uses `compatible`, which this "
            "table does not declare: its vocabulary is the three verdicts "
            "`relationship_detector.VERDICT_LABELS` holds (#1580).",
        ),
        (
            RULE_LABEL_UNSCOREABLE,
            "`compatible` is not in `relationship_detector.VERDICT_LABELS` "
            "(`contradicts`, `refines`, `unrelated`), so `classify` can never "
            "return it and those rows score wrong however the detector "
            "behaves - a direct contributor to that gate's measured accuracy. "
            "Verified against a lab corpus copy of 79 rows, 27 of them "
            "labelled `compatible`; no `contradiction/` module is mounted "
            "under the canonical lab root today (#1580).",
        ),
    ),
}
"""Module -> the violations its rows carry today, with a reason each.

Recorded rather than fixed: #1580 turns the validator on, and
relabelling a corpus is the corpus owner's decision. Recorded rather
than waived, because the list is checked in both directions - an
unrecorded violation fails the module, and a recorded violation that
stops reproducing fails it too, so a fix forces the entry out instead
of leaving a permanent exemption behind.
"""

KNOWN_UNDECLARED_MODULES: dict[str, str] = {
    "compression_uplift": (
        "Rows are an unlabelled belief pool (`id`, `content`, "
        "`retention_class`, `lock_level`) read by "
        "`tests/bench_gate/test_compression_uplift.py`. They carry no "
        "envelope - no `label`, `provenance` or `labeller_note` - and the "
        "module has no public scaffold directory and no `MODULES` entry, so "
        "nothing has ever schema-validated it (#1580)."
    ),
    "query_strategy": (
        "Rows are retrieve-uplift shaped (`id`, `query`, `k`, `beliefs`, "
        "`edges`, `expected_top_k`) read by "
        "`tests/bench_gate/test_query_strategy.py`, with the same missing "
        "envelope and the same absent `MODULES` entry as "
        "`compression_uplift` (#1580)."
    ),
}
"""Corpus directories that hold rows but declare no schema here.

A module absent from `MODULES` is never parametrized, so turning the
validator on does not reach it - the row count stays zero and the
absence reads as a pass. These are listed so the gap is stated, and
`test_corpus_root_declares_every_module_holding_rows` fails on any
directory that is neither declared nor listed.
"""


def _resolve_corpus_root() -> tuple[Path, str]:
    """The root to validate, and where it came from.

    Env first, public tree second. The origin travels with the path
    because a skip naming only a directory leaves the reader unable to
    tell a corpus that is not mounted from one that is mounted and
    empty.
    """
    raw = os.environ.get(CORPUS_ENV_VAR, "").strip()
    if raw:
        return Path(raw).expanduser(), f"${CORPUS_ENV_VAR}"
    return PUBLIC_CORPUS_ROOT, f"the public tree, because ${CORPUS_ENV_VAR} is unset"


def _no_rows_skip(module: str, root: Path, origin: str) -> str:
    return (
        f"corpus-schema: module {module!r} holds no JSONL rows under {root} "
        f"(root resolved from {origin}). Nothing was validated."
    )


def _iter_module_files(root: Path, module: str) -> list[Path]:
    return sorted((root / module).glob("*.jsonl"))


def _check_field(row: dict, field: str, spec: str, where: str) -> None:
    val = row.get(field)
    if spec == "str":
        assert isinstance(val, str) and val, (
            f"{where}: field {field!r} must be non-empty string, got {val!r}"
        )
    elif spec == "list[str]":
        assert isinstance(val, list) and val, (
            f"{where}: field {field!r} must be non-empty list, got {val!r}"
        )
        assert all(isinstance(x, str) and x for x in val), (
            f"{where}: field {field!r} must contain only non-empty strings"
        )
    elif spec == "int":
        assert isinstance(val, int) and not isinstance(val, bool) and val >= 1, (
            f"{where}: field {field!r} must be int ≥ 1, got {val!r}"
        )
    elif spec == "list[belief]":
        assert isinstance(val, list) and val, (
            f"{where}: field {field!r} must be non-empty list of beliefs"
        )
        row_belief_ids: set[str] = set()
        for i, b in enumerate(val):
            assert isinstance(b, dict), (
                f"{where}: {field}[{i}] must be object, got {type(b).__name__}"
            )
            bid = b.get("id")
            btext = b.get("text")
            assert isinstance(bid, str) and bid, (
                f"{where}: {field}[{i}].id must be non-empty string"
            )
            assert isinstance(btext, str) and btext, (
                f"{where}: {field}[{i}].text must be non-empty string"
            )
            assert bid not in row_belief_ids, (
                f"{where}: {field}[{i}] duplicate belief id {bid!r}"
            )
            row_belief_ids.add(bid)
    elif spec == "list[list_str]":
        assert isinstance(val, list) and val, (
            f"{where}: field {field!r} must be non-empty list of lists"
        )
        for i, inner in enumerate(val):
            assert isinstance(inner, list) and inner, (
                f"{where}: {field}[{i}] must be a non-empty list, got {inner!r}"
            )
            assert all(isinstance(x, str) and x for x in inner), (
                f"{where}: {field}[{i}] must contain only non-empty strings"
            )
    elif spec == "list[edge]":
        assert isinstance(val, list) and val, (
            f"{where}: field {field!r} must be non-empty list of edges"
        )
        for i, e in enumerate(val):
            assert isinstance(e, dict), (
                f"{where}: {field}[{i}] must be object, got {type(e).__name__}"
            )
            for k in ("src", "dst", "type"):
                v = e.get(k)
                assert isinstance(v, str) and v, (
                    f"{where}: {field}[{i}].{k} must be non-empty string"
                )
            w = e.get("weight")
            assert isinstance(w, (int, float)) and not isinstance(w, bool), (
                f"{where}: {field}[{i}].weight must be number, got {w!r}"
            )
    else:  # pragma: no cover - guard against typos in the spec table
        raise AssertionError(f"unknown field spec {spec!r}")


def _violation(row: dict, field: str, spec: str, where: str) -> str | None:
    """`_check_field`'s message, or None when the field is well formed.

    The shape checks stay assertions because they are also the failure
    text a reader gets; this adapter only stops the first bad row
    ending the walk, so one run reports every rule a module breaks
    instead of the alphabetically first.
    """
    try:
        _check_field(row, field, spec, where)
    except AssertionError as exc:
        return str(exc)
    return None


def _row_violations(
    module: str,
    row: dict,
    allowed_labels: set[str],
    extra_spec: dict[str, str],
    where: str,
) -> dict[str, str]:
    """Rule id -> failure message for every rule this row breaks."""
    found: dict[str, str] = {}

    # Common envelope: `id`, `provenance`, `labeller_note`, `label`, each
    # a non-empty string.
    for field in COMMON_REQUIRED:
        message = _violation(row, field, "str", where)
        if message:
            found.setdefault(_envelope_rule(field), message)

    # Synthetic provenance must follow `synthetic-vN.M`. README carve-out
    # at tests/corpus/v2_0/README.md, § v0.1 acceptance.
    prov = row.get("provenance")
    if (
        isinstance(prov, str)
        and prov.startswith("synthetic")
        and not SYNTHETIC_PROVENANCE_RE.match(prov)
    ):
        found.setdefault(
            RULE_PROVENANCE_FORMAT,
            f"{where}: synthetic provenance {prov!r} must match "
            f"`synthetic-vN.M` (e.g. 'synthetic-v0.1')",
        )

    # Module-specific fields.
    for field, spec in extra_spec.items():
        message = _violation(row, field, spec, where)
        if message:
            found.setdefault(_field_rule(field), message)

    label = row.get("label")
    if isinstance(label, str) and label:
        if label not in allowed_labels:
            found.setdefault(
                RULE_LABEL_VOCABULARY,
                f"{where}: label {label!r} not in {sorted(allowed_labels)}",
            )
        constant = DETECTOR_LABEL_CONSTANTS.get(module)
        if constant is not None and label not in constant:
            found.setdefault(
                RULE_LABEL_UNSCOREABLE,
                f"{where}: label {label!r} is not in the consuming detector's "
                f"own constant {sorted(constant)}, so the scorer can never "
                f"return it and the row is graded wrong whatever it does",
            )

    # Optional `seed` boolean flag.
    if "seed" in row and not isinstance(row["seed"], bool):
        found.setdefault(
            _field_rule("seed"), f"{where}: optional field 'seed' must be bool"
        )

    return found


def _legal_rule_ids() -> set[str]:
    """Every rule id `_row_violations` can produce."""
    ids = {
        RULE_PROVENANCE_FORMAT,
        RULE_LABEL_VOCABULARY,
        RULE_LABEL_UNSCOREABLE,
        RULE_DUPLICATE_ID,
        _field_rule("seed"),
    }
    ids |= {_envelope_rule(field) for field in COMMON_REQUIRED}
    for _labels, extra_spec in MODULES.values():
        ids |= {_field_rule(field) for field in extra_spec}
    return ids


@dataclass
class _WalkStats:
    """What the walk has established so far: rows, rules, per-rule rows.

    A dataclass the caller owns, rather than a tuple the walk returns,
    because the walk can die partway through — an unparseable line, a
    row that is not an object, an undecodable byte — and the count is
    most load-bearing on exactly those runs. A return value is lost
    when the frame unwinds; an object the caller allocated survives,
    so the caller can report `validated` from a `finally` whatever
    happened inside.
    """

    validated: int = 0
    found: dict[str, str] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)


def _walk_module(
    module: str, files: list[Path], stats: _WalkStats | None = None
) -> _WalkStats:
    """Validate `files`, accumulating into `stats` as it goes.

    Separate from the test so `duplicate-id` is reachable from a unit
    test. That rule is the one violation no single row can carry — it
    needs two rows that agree on `id` — so it cannot live in
    `_row_violations`, and leaving it inline made it the one rule with
    no arm of its own.

    The per-rule row count exists because a `KNOWN_FAILURES` waiver is
    per module and per rule, not per row: a later batch carrying the
    same violation lands inside the existing entry and the ratchet
    stays quiet. The figure cannot be gated on, since it depends on
    which corpus is mounted, but printed next to the waiver it is the
    number that moves when the waiver's scope grows.

    `stats` is updated in place before each failure can be raised, so a
    caller that passes one in reads the partial figures back even when
    this function does not return. Callers that only care about a
    completed walk can let it allocate.
    """
    if stats is None:
        stats = _WalkStats()
    allowed_labels, extra_spec = MODULES[module]
    seen_ids: set[str] = set()
    found = stats.found
    counts = stats.counts
    for path in files:
        with path.open() as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                where = f"{path.name}:{lineno}"
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    pytest.fail(f"{where}: invalid JSON: {exc}")

                assert isinstance(row, dict), f"{where}: row must be object"
                stats.validated += 1

                for rule, message in _row_violations(
                    module, row, allowed_labels, extra_spec, where
                ).items():
                    found.setdefault(rule, message)
                    counts[rule] = counts.get(rule, 0) + 1

                # ID must be unique within the module.
                rid = row.get("id")
                if isinstance(rid, str) and rid:
                    if rid in seen_ids:
                        found.setdefault(
                            RULE_DUPLICATE_ID,
                            f"{where}: duplicate id {rid!r} within module {module!r}",
                        )
                        counts[RULE_DUPLICATE_ID] = (
                            counts.get(RULE_DUPLICATE_ID, 0) + 1
                        )
                    seen_ids.add(rid)
    return stats


@pytest.mark.parametrize("module", sorted(MODULES.keys()))
def test_corpus_module_files_valid(module: str, record_property) -> None:  # type: ignore[no-untyped-def]
    """Every JSONL row in this module conforms to the schema."""
    root, origin = _resolve_corpus_root()
    files = _iter_module_files(root, module)
    if not files:
        pytest.skip(_no_rows_skip(module, root, origin))

    # The count is reported from a `finally`, not after the walk, because
    # the walk raises on an unparseable line or a non-object row — and a
    # failure with no count is the one report where the reader most needs
    # it: it leaves them unable to tell whether validation read 3 rows or
    # 500 before it stopped. `finally` also covers the failures nobody
    # enumerated, such as an undecodable byte, which surface as `error`.
    stats = _WalkStats()
    completed = False
    try:
        _walk_module(module, files, stats)
        completed = True
    finally:
        source = f"{len(files)} file(s) under {root}"
        record_property(
            CORPUS_SCHEMA_PROPERTY,
            f"module {module!r}: {stats.validated} row(s) validated from "
            f"{source}"
            if completed
            else f"module {module!r}: {stats.validated} row(s) validated from "
            f"{source} before validation stopped on an unreadable row — the "
            f"walk did not finish, so this is a floor, not the module's size",
        )

    validated, found, counts = stats.validated, stats.found, stats.counts
    # A module reaching here has files, so zero rows means every file is
    # blank. Without this the run reads as a pass on an empty corpus,
    # which is the whole of #1580.
    assert validated > 0, (
        f"module {module!r}: {len(files)} JSONL file(s) under {root} hold no "
        f"rows at all; 0 rows validated is not a pass"
    )

    known = dict(KNOWN_FAILURES.get(module, ()))
    unexpected = sorted(set(found) - set(known))
    assert not unexpected, (
        f"module {module!r} ({validated} row(s) validated under {root}) "
        f"breaks {len(unexpected)} rule(s) that KNOWN_FAILURES does not "
        f"record:\n"
        + "\n".join(f"  [{rule}] {found[rule]}" for rule in unexpected)
    )
    repaired = sorted(set(known) - set(found))
    assert not repaired, (
        f"module {module!r} ({validated} row(s) validated under {root}) no "
        f"longer breaks {repaired}; drop the KNOWN_FAILURES entr"
        f"{'y' if len(repaired) == 1 else 'ies'} so the rule starts being "
        f"enforced instead of staying permanently waived"
    )
    # A recorded violation still has to be legible. Green plus a
    # KNOWN_FAILURES entry nobody reads is the quiet exemption #1580
    # asks not to create, so the reproducing rules print alongside the
    # row count on every run.
    if found:
        breakdown = ", ".join(
            f"{rule} ({counts[rule]} of {validated} row(s))"
            for rule in sorted(found)
        )
        record_property(
            CORPUS_SCHEMA_PROPERTY,
            f"module {module!r}: {len(found)} recorded violation(s) still "
            f"reproducing and NOT enforced — {breakdown}; see KNOWN_FAILURES "
            f"in tests/test_corpus_schema.py",
        )


def test_corpus_root_declares_every_module_holding_rows() -> None:
    """A corpus directory with rows and no `MODULES` entry is never validated.

    The per-module test parametrizes over `MODULES`, so a module the
    table does not name contributes no rows, no failure and no skip —
    it is simply invisible, which is indistinguishable from clean.
    """
    root, origin = _resolve_corpus_root()
    if not root.is_dir():
        pytest.skip(
            f"corpus-schema: {root} is not a directory (root resolved from "
            f"{origin}). Nothing was validated."
        )
    with_rows = {
        child.name
        for child in root.iterdir()
        if child.is_dir() and any(child.glob("*.jsonl"))
    }
    if not with_rows:
        pytest.skip(
            f"corpus-schema: no module under {root} holds JSONL rows (root "
            f"resolved from {origin}). Nothing was validated."
        )

    undeclared = with_rows - set(MODULES)
    unrecorded = sorted(undeclared - set(KNOWN_UNDECLARED_MODULES))
    assert not unrecorded, (
        f"corpus modules {unrecorded} under {root} hold rows but have no "
        f"MODULES entry, so nothing validates them. Add a schema, or record "
        f"the gap in KNOWN_UNDECLARED_MODULES with a reason"
    )
    declared_again = sorted(set(KNOWN_UNDECLARED_MODULES) & set(MODULES))
    assert not declared_again, (
        f"{declared_again} now have MODULES entries; drop them from "
        f"KNOWN_UNDECLARED_MODULES so the schema is what governs them"
    )


def test_known_failures_name_rules_the_validator_can_emit() -> None:
    """A waiver keyed off a rule id nothing emits waives nothing.

    It would also never fall out of the list, because the
    stopped-reproducing check compares the same misspelt id.
    """
    legal = _legal_rule_ids()
    for module, entries in KNOWN_FAILURES.items():
        assert module in MODULES, f"KNOWN_FAILURES names unknown module {module!r}"
        rules = [rule for rule, _reason in entries]
        assert len(rules) == len(set(rules)), (
            f"KNOWN_FAILURES[{module!r}] records a rule twice: {rules}"
        )
        for rule, reason in entries:
            assert rule in legal, (
                f"KNOWN_FAILURES[{module!r}] names rule {rule!r}, which "
                f"_row_violations never emits; legal ids are {sorted(legal)}"
            )
            assert reason.strip(), (
                f"KNOWN_FAILURES[{module!r}][{rule!r}] has no reason — a "
                f"waiver without one is a silent waiver"
            )
    for module, reason in KNOWN_UNDECLARED_MODULES.items():
        assert reason.strip(), (
            f"KNOWN_UNDECLARED_MODULES[{module!r}] has no reason"
        )


def _conforming_row() -> dict[str, object]:
    """A hand-written row that every rule accepts.

    Written here rather than lifted from the corpus: the lab rows do
    not enter this repository (#1456), and a fixture that has to stay
    valid is clearer as a literal anyway.
    """
    return {
        "id": "fixture-0001",
        "provenance": "synthetic-v0.1",
        "labeller_note": "hand-written fixture for the validator's own tests",
        "label": "contradicts",
        "belief_a": "the release tag is cut from main",
        "belief_b": "the release tag is cut from the release branch",
    }


def _violations_for(row: dict, module: str = "contradiction", **kwargs) -> dict[str, str]:  # type: ignore[no-untyped-def]
    allowed_labels, extra_spec = MODULES[module]
    return _row_violations(
        module,
        row,
        kwargs.get("allowed_labels", allowed_labels),
        extra_spec,
        kwargs.get("where", "fixture.jsonl:1"),
    )


def test_validator_accepts_a_conforming_row() -> None:
    """The negative control. Without it every rule below passes vacuously."""
    assert _violations_for(_conforming_row()) == {}


def test_validator_rejects_a_row_with_no_provenance() -> None:
    """`directive_detection`'s v0_1 batch in miniature."""
    row = _conforming_row()
    del row["provenance"]
    assert set(_violations_for(row)) == {_envelope_rule("provenance")}


def test_validator_rejects_an_empty_provenance() -> None:
    """Present but blank is the same defect as absent."""
    row = _conforming_row()
    row["provenance"] = ""
    assert set(_violations_for(row)) == {_envelope_rule("provenance")}


def test_validator_rejects_an_unversioned_synthetic_provenance() -> None:
    row = _conforming_row()
    row["provenance"] = "synthetic"
    assert set(_violations_for(row)) == {RULE_PROVENANCE_FORMAT}


def test_validator_rejects_an_empty_labeller_note() -> None:
    row = _conforming_row()
    row["labeller_note"] = ""
    assert set(_violations_for(row)) == {_envelope_rule("labeller_note")}


def test_validator_rejects_a_label_outside_the_modules_set() -> None:
    """Checked on a module with no detector constant, so one rule fires.

    `sentiment` isolates the vocabulary rule from the scoreability rule
    below; on `contradiction` the two coincide and neither arm would
    prove the other exists.
    """
    row = {
        "id": "fixture-0002",
        "provenance": "synthetic-v0.1",
        "labeller_note": "hand-written fixture",
        "label": "furious",
        "user_message": "this is the third time the deploy has rolled back",
    }
    assert set(_violations_for(row, module="sentiment")) == {RULE_LABEL_VOCABULARY}


def test_validator_rejects_a_label_the_detector_cannot_return() -> None:
    """The `contradiction`/`compatible` case, isolated.

    The module's declared set is widened to admit the label, so the
    vocabulary rule passes and only the scoreability rule can fire.
    That is the arm that proves rule 3 is a rule of its own rather than
    a restatement of rule 2.

    Widening is necessary, not a convenience. On the shipped table the
    two rules cannot disagree: rule 3 fires alone only where the module
    declares a label the detector cannot return, and
    `test_module_label_vocabulary_matches_detector_constant` exists to
    forbid exactly that. So rule 3 is unreachable on its own for as
    long as the table and the constant agree, which is the state the
    other test enforces — it is the guard that catches the interval
    between a detector losing a label and the table following, and this
    arm is where its behaviour in that interval is pinned.
    """
    row = _conforming_row()
    row["label"] = "compatible"
    declared, _extra = MODULES["contradiction"]
    widened = declared | {"compatible"}
    assert set(_violations_for(row, allowed_labels=widened)) == {
        RULE_LABEL_UNSCOREABLE
    }


def _write_module(tmp_path: Path, module: str, rows: list[dict]) -> list[Path]:
    """One JSONL file of hand-written rows, for the walk-level arms."""
    path = tmp_path / module / "fixture.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return [path]


def test_walk_accepts_two_rows_with_distinct_ids(tmp_path: Path) -> None:
    """The negative control for the duplicate-id arm below."""
    first = _conforming_row()
    second = _conforming_row()
    second["id"] = "fixture-0002"
    files = _write_module(tmp_path, "contradiction", [first, second])

    stats = _walk_module("contradiction", files)
    assert stats.validated == 2
    assert stats.found == {}
    assert stats.counts == {}


def test_walk_rejects_two_rows_sharing_an_id(tmp_path: Path) -> None:
    """`duplicate-id` is the one rule no single row can break.

    It compares a row against the rows before it, so it lives in the
    walk rather than in `_row_violations`, and it is the only rule the
    per-row arms above cannot reach.
    """
    first = _conforming_row()
    second = _conforming_row()
    second["belief_b"] = "the release tag is cut from a tag branch"
    assert first["id"] == second["id"]
    files = _write_module(tmp_path, "contradiction", [first, second])

    stats = _walk_module("contradiction", files)
    assert stats.validated == 2
    assert set(stats.found) == {RULE_DUPLICATE_ID}
    assert "fixture-0001" in stats.found[RULE_DUPLICATE_ID]
    assert stats.counts == {RULE_DUPLICATE_ID: 1}


def test_walk_counts_rows_across_files_and_skips_blank_lines(
    tmp_path: Path,
) -> None:
    """The row count is what the summary prints, so pin what it counts."""
    module_dir = tmp_path / "contradiction"
    module_dir.mkdir(parents=True)
    first = module_dir / "a.jsonl"
    first.write_text(json.dumps(_conforming_row()) + "\n\n\n")
    second_row = _conforming_row()
    second_row["id"] = "fixture-0002"
    second = module_dir / "b.jsonl"
    second.write_text(json.dumps(second_row) + "\n")

    stats = _walk_module("contradiction", [first, second])
    assert stats.validated == 2
    assert stats.found == {}


def test_walk_counts_how_many_rows_break_each_rule(tmp_path: Path) -> None:
    """A waiver is per module and per rule, so the row count is the scope.

    `KNOWN_FAILURES` cannot say "these 285 rows and no more", and a
    later batch carrying the same violation lands inside the existing
    entry with nothing in the tail changing. The count is what changes,
    so it prints next to the waiver.
    """
    rows = []
    for i, has_provenance in enumerate((True, False, False), start=1):
        row = _conforming_row()
        row["id"] = f"fixture-{i:04d}"
        if not has_provenance:
            del row["provenance"]
        rows.append(row)
    files = _write_module(tmp_path, "contradiction", rows)

    stats = _walk_module("contradiction", files)
    assert stats.validated == 3
    assert set(stats.found) == {_envelope_rule("provenance")}
    assert stats.counts == {_envelope_rule("provenance"): 2}


def _record_into(sink: list[tuple[str, str]]):  # type: ignore[no-untyped-def]
    """A stand-in for the `record_property` fixture that keeps the calls."""

    def record_property(key: str, value: object) -> None:
        sink.append((key, str(value)))

    return record_property


def test_a_module_that_validates_cleanly_records_its_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The negative control for the failing arm below.

    Without it, an arm asserting that a failure records a count passes
    just as well against a version that records one unconditionally
    and wrongly, and neither arm would show the two paths differ.
    `sentiment` rather than `contradiction` because it carries no
    `KNOWN_FAILURES` entry, so conforming rows leave the ratchet quiet.
    """
    module_dir = tmp_path / "sentiment"
    module_dir.mkdir(parents=True)
    rows = [
        {
            "id": f"fixture-{i:04d}",
            "provenance": "synthetic-v0.1",
            "labeller_note": "hand-written fixture",
            "label": "neutral",
            "user_message": "the deploy finished",
        }
        for i in (1, 2)
    ]
    (module_dir / "fixture.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    monkeypatch.setenv(CORPUS_ENV_VAR, str(tmp_path))

    recorded: list[tuple[str, str]] = []
    test_corpus_module_files_valid("sentiment", _record_into(recorded))

    assert [key for key, _value in recorded] == [CORPUS_SCHEMA_PROPERTY]
    assert "2 row(s) validated from 1 file(s)" in recorded[0][1]
    assert "did not finish" not in recorded[0][1]


def test_a_module_that_fails_to_parse_still_records_its_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure with no count is the report that needs one most.

    `_walk_module` raises at the unparseable line, which is before the
    point the count was recorded, so the terminal summary carried no
    `corpus_schema` line at all for the module that failed — the
    reader could not tell whether validation had read 2 rows or 500
    before it stopped. Recording from a `finally` is what fixes it,
    and 2 rather than 0 or 3 is what proves the figure is the walk's
    real progress and not a placeholder.
    """
    module_dir = tmp_path / "contradiction"
    module_dir.mkdir(parents=True)
    second = _conforming_row()
    second["id"] = "fixture-0002"
    truncated = '{"id": "fixture-0003", "provenance":'
    (module_dir / "fixture.jsonl").write_text(
        json.dumps(_conforming_row())
        + "\n"
        + json.dumps(second)
        + "\n"
        + truncated
        + "\n"
    )
    monkeypatch.setenv(CORPUS_ENV_VAR, str(tmp_path))

    recorded: list[tuple[str, str]] = []
    with pytest.raises(pytest.fail.Exception) as excinfo:
        test_corpus_module_files_valid("contradiction", _record_into(recorded))

    assert "fixture.jsonl:3" in str(excinfo.value)
    assert "invalid JSON" in str(excinfo.value)
    assert [key for key, _value in recorded] == [CORPUS_SCHEMA_PROPERTY]
    assert "2 row(s) validated" in recorded[0][1]
    # The count is a floor, and has to say so: the module's real size is
    # unknown once the walk stops early.
    assert "did not finish" in recorded[0][1]


def test_a_non_object_row_still_records_its_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The walk's other early exit, which raises `AssertionError`.

    Pinned separately because the two failures leave the walk by
    different exception types, and a fix that caught only the
    `pytest.fail` path would still drop the count here.
    """
    module_dir = tmp_path / "contradiction"
    module_dir.mkdir(parents=True)
    (module_dir / "fixture.jsonl").write_text(
        json.dumps(_conforming_row()) + "\n" + json.dumps(["not", "an", "object"]) + "\n"
    )
    monkeypatch.setenv(CORPUS_ENV_VAR, str(tmp_path))

    recorded: list[tuple[str, str]] = []
    with pytest.raises(AssertionError, match="row must be object"):
        test_corpus_module_files_valid("contradiction", _record_into(recorded))

    assert [key for key, _value in recorded] == [CORPUS_SCHEMA_PROPERTY]
    assert "1 row(s) validated" in recorded[0][1]


def test_validator_rejects_a_missing_module_field() -> None:
    row = _conforming_row()
    del row["belief_b"]
    assert set(_violations_for(row)) == {_field_rule("belief_b")}


def test_validator_reports_every_broken_rule_not_just_the_first() -> None:
    """One run has to name the whole repair, not one step of it."""
    row = _conforming_row()
    del row["provenance"]
    row["labeller_note"] = ""
    assert set(_violations_for(row)) == {
        _envelope_rule("provenance"),
        _envelope_rule("labeller_note"),
    }


def test_corpus_root_is_env_first_and_public_tree_second(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """AC1/AC2: the env var wins, and the fallback is named, not silent."""
    monkeypatch.delenv(CORPUS_ENV_VAR, raising=False)
    root, origin = _resolve_corpus_root()
    assert root == PUBLIC_CORPUS_ROOT
    assert CORPUS_ENV_VAR in origin and "public tree" in origin

    monkeypatch.setenv(CORPUS_ENV_VAR, str(tmp_path))
    root, origin = _resolve_corpus_root()
    assert root == tmp_path
    assert CORPUS_ENV_VAR in origin

    # A blank value is not a mount.
    monkeypatch.setenv(CORPUS_ENV_VAR, "   ")
    root, _origin = _resolve_corpus_root()
    assert root == PUBLIC_CORPUS_ROOT


def test_skip_message_names_the_root_it_inspected() -> None:
    """The silent skip is the defect #1580 is about, so pin its text."""
    message = _no_rows_skip("dedup", Path("/corpus/v2_0"), f"${CORPUS_ENV_VAR}")
    assert "dedup" in message
    assert "/corpus/v2_0" in message
    assert CORPUS_ENV_VAR in message
    assert "Nothing was validated" in message


def test_corpus_root_readme_present() -> None:
    """The schema contract README must exist alongside the corpus."""
    assert (PUBLIC_CORPUS_ROOT / "README.md").is_file(), (
        "tests/corpus/v2_0/README.md is required — it documents the schema"
    )


def test_synthetic_provenance_regex() -> None:
    """`synthetic-vN.M` must accept versioned, reject sloppy variants."""
    assert SYNTHETIC_PROVENANCE_RE.match("synthetic-v0.1")
    assert SYNTHETIC_PROVENANCE_RE.match("synthetic-v12.34")
    assert not SYNTHETIC_PROVENANCE_RE.match("synthetic")
    assert not SYNTHETIC_PROVENANCE_RE.match("synthetic-v1")
    assert not SYNTHETIC_PROVENANCE_RE.match("synthetic-foo")
    assert not SYNTHETIC_PROVENANCE_RE.match("synthetic-v0.1-extra")


@pytest.mark.parametrize("module", sorted(DETECTOR_LABEL_CONSTANTS))
def test_module_label_vocabulary_matches_detector_constant(module: str) -> None:
    """The declared set and the scorer's own constant must agree.

    A label the module declares but the detector cannot return produces
    rows that are graded wrong by construction; a label the detector can
    return but the module rejects means a correct row fails validation.
    Both directions are drift, so both are checked.

    This only has teeth because `MODULES[module]` spells its vocabulary
    out instead of deriving it from the constant. Derive it and the two
    sides become one object, the comparison holds for every possible
    value of the constant, and the test passes on a detector that has
    changed underneath the corpus.
    """
    declared, _extra = MODULES[module]
    constant = DETECTOR_LABEL_CONSTANTS[module]

    unscoreable = sorted(set(declared) - set(constant))
    assert not unscoreable, (
        f"MODULES[{module!r}] declares {unscoreable}, which the consuming "
        f"detector's own constant cannot return — rows labelled that way can "
        f"never be scored correct"
    )
    unrepresented = sorted(set(constant) - set(declared))
    assert not unrepresented, (
        f"the consuming detector can return {unrepresented}, which "
        f"MODULES[{module!r}] does not declare — a correct row would fail "
        f"validation"
    )
