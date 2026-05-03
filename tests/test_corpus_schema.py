"""Schema validation for the v2.0 evaluation corpus (#307).

Walks every `*.jsonl` under `tests/corpus/v2_0/<module>/` and enforces:

  1. Each line parses as JSON.
  2. Every row has the common envelope (`id`, `provenance`, `labeller_note`,
     `label`) with non-empty strings.
  3. `id` values are unique within a module.
  4. `label` is one of the module-specific allowed set.
  5. Module-specific extra fields exist with the right shape.

The ≥50/module v0.1 threshold is **not** asserted here — that flips on
once labelling is complete across all six modules. See
`tests/corpus/v2_0/README.md` for the schema contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

CORPUS_ROOT = Path(__file__).parent / "corpus" / "v2_0"


# Module → (allowed labels, extra-required-fields-spec)
# Spec values: "str" = non-empty string, "list[str]" = non-empty list of strings.
MODULES: dict[str, tuple[set[str], dict[str, str]]] = {
    "dedup": (
        {"duplicate", "near-duplicate", "distinct"},
        {"belief_a": "str", "belief_b": "str"},
    ),
    "enforcement": (
        {"compliant", "violated", "n/a"},
        {"user_directive": "str", "agent_output": "str"},
    ),
    "contradiction": (
        {"contradicts", "compatible", "unrelated"},
        {"belief_a": "str", "belief_b": "str"},
    ),
    "wonder_consolidation": (
        {"1", "2", "3", "4", "5"},
        {"seed_belief": "str", "retrieved_neighbors": "list[str]"},
    ),
    "promotion_trigger": (
        {"should_promote", "should_not"},
        {"belief_sequence": "list[str]"},
    ),
    "sentiment": (
        {"positive", "negative", "neutral"},
        {"user_message": "str"},
    ),
    "directive_detection": (
        {"directive", "not_directive"},
        {"prompt": "str"},
    ),
}

COMMON_REQUIRED = ("id", "provenance", "labeller_note", "label")


def _iter_module_files(module: str) -> list[Path]:
    return sorted((CORPUS_ROOT / module).glob("*.jsonl"))


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
    else:  # pragma: no cover - guard against typos in the spec table
        raise AssertionError(f"unknown field spec {spec!r}")


@pytest.mark.parametrize("module", sorted(MODULES.keys()))
def test_corpus_module_files_valid(module: str) -> None:
    """Every JSONL row in this module conforms to the schema."""
    allowed_labels, extra_spec = MODULES[module]
    files = _iter_module_files(module)
    if not files:
        pytest.skip(f"module {module!r} has no JSONL files yet")

    seen_ids: set[str] = set()
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

                # Common envelope.
                for field in COMMON_REQUIRED:
                    _check_field(row, field, "str", where)

                # Module-specific.
                for field, spec in extra_spec.items():
                    _check_field(row, field, spec, where)

                # Label value must be in the allowed set.
                assert row["label"] in allowed_labels, (
                    f"{where}: label {row['label']!r} not in {sorted(allowed_labels)}"
                )

                # Optional `seed` boolean flag.
                if "seed" in row:
                    assert isinstance(row["seed"], bool), (
                        f"{where}: optional field 'seed' must be bool"
                    )

                # ID must be unique within the module.
                rid = row["id"]
                assert rid not in seen_ids, (
                    f"{where}: duplicate id {rid!r} within module {module!r}"
                )
                seen_ids.add(rid)


def test_corpus_root_readme_present() -> None:
    """The schema contract README must exist alongside the corpus."""
    assert (CORPUS_ROOT / "README.md").is_file(), (
        "tests/corpus/v2_0/README.md is required — it documents the schema"
    )
