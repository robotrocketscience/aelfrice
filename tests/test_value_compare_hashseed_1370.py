"""`extract_values` is byte-identical across hash seeds (#1370 §8, #1157).

`extract_values` documents "Pure function. Same input → byte-identical
output." `_ENUM_MEMBER_INDEX` is built by iterating the frozensets in
`ENUM_VOCAB`, so without an explicit sort its insertion order — and
therefore the order `_extract_enums` emits slots in — is keyed on string
hash randomisation and differs from process to process.

`PYTHONHASHSEED` is read once at interpreter start, so this has to run
out of process. Each child prints a canonical rendering of the extracted
slots; the parent asserts every child printed the same bytes.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

# One interpreter start per seed. The suite's base timeout is sized for
# in-process unit tests, so under parallel load this would report as a
# hang rather than as slowness (#1307).
_SUBPROCESS_TIMEOUT_S = 120

# Every group below has two or more members, which is the only place the
# order can vary: categories are a dict and groups are a tuple (both
# already ordered), members are a frozenset.
#   storage_mode: scan / full-scan / table-scan
#   determinism:  non-deterministic / nondeterministic / stochastic
#   execution_mode: sync / synchronous, async / asynchronous
#   access_mode:  readonly / read-only
#   strictness:   lax / permissive
#   default_state: default-on / enabled, default-off / disabled
_TEXT = (
    "the scan lane is a full-scan today and a table-scan under load; "
    "ranking is non-deterministic, the writer is nondeterministic, "
    "the sampler is stochastic; the path is sync and synchronous while "
    "the flush is async and asynchronous; the peer opens readonly, "
    "strictly read-only; the parser is lax, arguably permissive; "
    "the lane ships default-on and enabled, the flag default-off "
    "and disabled"
)

# Seeds verified to produce distinct frozenset iteration orders for these
# groups on the interpreter this suite runs under. More than two so the
# test does not hinge on a single pair staying distinguishable.
_SEEDS = ("0", "1", "2", "3", "4", "5")

_CHILD = (
    "from aelfrice.value_compare import extract_values\n"
    "s = extract_values({text!r})\n"
    "print('|'.join(f'{{e.category}},{{e.group_id}},{{e.member}}' for e in s.enum))\n"
    "print('|'.join(f'{{n.key}},{{n.value!r}}' for n in s.numeric))\n"
).format(text=_TEXT)


def _render(seed: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=seed)
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD],
        capture_output=True,
        text=True,
        env=env,
        timeout=_SUBPROCESS_TIMEOUT_S,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


@pytest.mark.timeout(_SUBPROCESS_TIMEOUT_S)
def test_extract_values_output_is_hash_seed_independent() -> None:
    renders = {seed: _render(seed) for seed in _SEEDS}
    distinct = sorted(set(renders.values()))
    assert len(distinct) == 1, (
        "extract_values emitted "
        f"{len(distinct)} different renderings across PYTHONHASHSEED "
        f"{list(_SEEDS)}; first two:\n{distinct[0]!r}\n{distinct[1]!r}"
    )


def test_fixture_actually_exercises_multi_member_groups() -> None:
    """Guard the fixture: the text must hit groups with >1 member.

    A single-member group cannot expose the ordering bug, so if the text
    ever drifts to only such members the test above would pass for the
    wrong reason.
    """
    from aelfrice.value_compare import ENUM_VOCAB, extract_values

    slots = extract_values(_TEXT)
    hit = {(e.category, e.group_id) for e in slots.enum}
    multi = {
        (category, sorted(group)[0])
        for category, groups in ENUM_VOCAB.items()
        for group in groups
        if len(group) > 1
    }
    assert len(hit & multi) >= 4
