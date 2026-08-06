"""The §10 eigensolve must be bit-identical ACROSS processes, not just calls.

`test_compute_eigenbasis_is_bit_identical_across_calls` repeats the solve
four times inside one interpreter. That is the weaker of the two available
assertions, and it is weaker in a way that matters here: the defect it
guards against was ARPACK drawing `v0` from an internal RNG whose state
advances **per call**, so a within-process repetition is exactly the shape
a per-call RNG would break — but a *per-process* source of variation
(hash randomisation reaching a set or dict on the path from store to
Laplacian) would repeat identically four times in one interpreter and
still differ between runs.

#1157's determinism contract is about reproducing a ranking from the write
log on another machine, another day, in another process. That is the claim
this file tests. The sibling §8 test already goes out-of-process for the
same reason; §10 had no equivalent.

The seed derivation is content-addressed (`blake2b` over the CSR shape,
indptr, indices and data with explicit int64/float64 casts) and feeds
`np.random.default_rng`, so no hash randomisation is involved and the
property is expected to hold. It is asserted rather than assumed because
"expected to hold" is what the pre-#1370 code also looked like.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

_SUBPROCESS_TIMEOUT_S = 120

# Varied so the test cannot pass by every child happening to share one
# hash seed. Six mirrors the §8 test rather than inventing a new count.
_SEEDS = ("0", "1", "2", "3", "4", "5")

# Built in the child so the graph itself is constructed under the child's
# hash seed — constructing it in the parent and pickling it across would
# test the solver while hiding any set/dict ordering upstream of it.
_CHILD = """
import numpy as np, scipy.sparse as sp
from aelfrice.graph_spectral import compute_eigenbasis

n = 24
rows, cols, vals = [], [], []
for i in range(n):
    for j in (i + 1, i + 3, i + 7):
        if j < n:
            rows.append(i); cols.append(j); vals.append(0.5 + 0.1 * ((i * j) % 5))
A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)
A = ((A + A.T) / 2.0).tocsr()
deg = np.asarray(A.sum(axis=1)).ravel()
L = (sp.diags(deg) - A).tocsr()

ev, vc = compute_eigenbasis(L, k=4)
print(np.asarray(ev, dtype=np.float64).tobytes().hex())
print(np.asarray(vc, dtype=np.float64).tobytes().hex())
"""


def _solve(seed: str) -> str:
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
def test_eigenbasis_is_bit_identical_across_processes() -> None:
    """Six fresh interpreters, six hash seeds, one answer.

    Verified by mutation: reverting the `v0=` argument on the `eigsh`
    call makes this emit several distinct digests, the same way the
    within-process test fails — but this one also covers the per-process
    class that the within-process test cannot see.
    """
    outputs = {seed: _solve(seed) for seed in _SEEDS}
    distinct = sorted(set(outputs.values()))
    assert len(distinct) == 1, (
        f"the eigensolve produced {len(distinct)} different results across "
        f"PYTHONHASHSEED {', '.join(_SEEDS)} — the heat-kernel authority "
        f"ranking is not reproducible from the write log on another process. "
        f"Seeds by result: "
        + "; ".join(
            f"{d[:16]}... <- {[s for s, o in outputs.items() if o == d]}"
            for d in distinct
        )
    )

    # A digest of nothing would also be "one distinct result".
    only = distinct[0].strip().splitlines()
    assert len(only) == 2 and all(len(line) > 32 for line in only), (
        f"expected two non-trivial hex digests from the child, got {only!r}"
    )
