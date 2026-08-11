"""#1351: importing the hook must not import the numeric stack.

Every hook fire is a fresh process. Before this, `import aelfrice.hook` pulled
numpy, scipy (153 submodules) and snowballstemmer at module load — ~105 ms of
scipy alone — through two chains:

    aelfrice.hook -> context_rebuilder -> query_understanding -> store_cache -> bm25 -> scipy.sparse
    aelfrice.hook -> retrieval -> graph_spectral -> scipy.sparse.linalg

Half of UserPromptSubmit fires never reach the L1 lane at all (the shape gate
skips system-generated and trivial prompts), and no `Stop` / `PreToolUse` /
`PostToolUse` / `PreCompact` / `SessionStart` fire runs it, so those processes
paid the whole cost for nothing.

The assertion is on the *module set*, not on wall-clock milliseconds. A timing
budget here would be a flake generator under CI contention — #1307 is the
precedent, where exactly that shape was diagnosed as a deadlock twice.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

# Roots that must not appear in `sys.modules` after importing the hook.
_NUMERIC_ROOTS = ("numpy", "scipy", "snowballstemmer")

_PROBE = """
import sys
import aelfrice.hook
roots = {m.split(".")[0] for m in sys.modules}
print(",".join(sorted(roots & {%s})))
""" % ", ".join(repr(r) for r in _NUMERIC_ROOTS)


def _probe(statement: str) -> str:
    """Run `statement` in a clean interpreter and return its stdout."""
    proc = subprocess.run(
        [sys.executable, "-c", statement],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


@pytest.mark.timeout(30)
def test_importing_the_hook_does_not_import_the_numeric_stack() -> None:
    """The gate. Must run in a subprocess.

    An in-process assertion is worthless: by the time pytest reaches this
    module it has already imported most of the tree, so `sys.modules` would
    contain numpy either way and the test would pass on the unfixed code.
    """
    assert _probe(_PROBE) == "", (
        "importing aelfrice.hook pulled the numeric stack back into the import "
        "graph. Every hook process pays this, including the ones that never "
        "retrieve. Find the new eager edge with:\n"
        "  python -X importtime -c 'import aelfrice.hook'"
    )


@pytest.mark.timeout(30)
def test_the_probe_can_see_the_numeric_stack_when_it_is_there() -> None:
    """Guard the guard: a probe that can never observe numpy proves nothing.

    If `_NUMERIC_ROOTS` were misspelled, or the subprocess silently failed to
    import anything, the assertion above would pass vacuously and read exactly
    like a clean import graph.
    """
    observed = _probe(
        "import sys, numpy, scipy.sparse, snowballstemmer\n"
        'roots = {m.split(".")[0] for m in sys.modules}\n'
        'print(",".join(sorted(roots & {%s})))'
        % ", ".join(repr(r) for r in _NUMERIC_ROOTS)
    )
    assert observed == "numpy,scipy,snowballstemmer", observed


_RETRIEVAL_PROBE = '''
import os
import sys

# Strip every ambient aelfrice var BEFORE importing the package: the lane
# resolvers are env-first, so a stray AELFRICE_ENTITY_INDEX in the runner's
# environment silently changes which lanes execute and therefore which
# modules get imported.
for _k in [k for k in os.environ if k.startswith(("AELFRICE_", "AELF_"))]:
    del os.environ[_k]

_tmp = %(tmp)r
# HOME repoints the home-derived constants resolved at import time; the
# live ambient store is repo-local (.git/aelfrice/memory.db) and config
# discovery walks up from cwd, so leaving the repo is what pins those two.
os.environ["HOME"] = _tmp
os.environ["AELFRICE_DOTDIR"] = os.path.join(_tmp, ".aelfrice")
os.environ["AELFRICE_DB"] = os.path.join(_tmp, "memory.db")
os.environ["AELF_NO_UPDATE_CHECK"] = "1"
os.chdir(_tmp)

from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.retrieval import retrieve
from aelfrice.store import MemoryStore

_TARGETS = {%(roots)s}


def _roots():
    return ",".join(sorted({m.split(".")[0] for m in sys.modules} & _TARGETS))


def _belief(bid, content):
    return Belief(
        id=bid,
        content=content,
        content_hash="h_" + bid,
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    )


store = MemoryStore(os.environ["AELFRICE_DB"])
for _i, _text in enumerate((
    "the quokka telemetry pipeline flushes every 30 seconds",
    "quokka telemetry is sharded by tenant id",
    "the marmoset scheduler retries failed jobs twice",
    "telemetry dashboards are rendered with grafana",
    "the quokka ingest buffer is 4 megabytes",
)):
    store.insert_belief(_belief("P_%%d" %% _i, _text))

# Baseline. Without it the reading after the call proves nothing: it would
# hold whether retrieve() pulled the stack in or the bare imports above did.
print("before:" + _roots())
hits = retrieve(store, query="quokka telemetry", token_budget=10000)
print("after:" + _roots())
print("hits:%%d" %% len(hits))
store.close()
'''


@pytest.mark.timeout(90)
def test_the_numeric_stack_still_loads_when_retrieval_runs(tmp_path) -> None:
    """The imports moved; they were not removed.

    This drives a real `retrieve()` against a real SQLite store rather than
    importing `aelfrice.bm25` by hand. The distinction is the whole point:
    an `import aelfrice.bm25` in the probe satisfies the module-set
    assertion on its own, so the test would pass even if every deferred
    import inside `retrieve` had been deleted rather than moved. It could
    not fail, which is what CodeRabbit caught on PR #1352.

    Three things are asserted, and each catches a different regression:

    * **`before` is empty.** Importing `aelfrice.models` / `.retrieval` /
      `.store` must pull none of the numeric stack. This is the same
      property the first test asserts for the hook, re-checked here so the
      `after` reading is attributable to the `retrieve()` call and nothing
      else.
    * **`after` names all three.** The deferred imports still fire on the
      retrieval path. `bm25` is reached through `tokenize_stemmed` and the
      BM25F cache, and it imports numpy, scipy.sparse and snowballstemmer
      at module scope.
    * **`hits` is non-zero.** The lanes actually ran to completion. The
      store has no locked beliefs, so every hit is a relevance-lane hit --
      an L0-only result would not exercise the deferred imports at all.

    A deferred import that was deleted rather than moved raises NameError,
    which surfaces as a non-zero exit from `_probe`, not as a quiet pass.

    The subprocess is pinned to `tmp_path` for HOME, dotdir and DB, and
    chdir'd out of the repo: opening a MemoryStore is a *write* (DDL,
    migrations, scope-id generation), so an unpinned run would mutate the
    developer's real store.
    """
    observed = _probe(
        _RETRIEVAL_PROBE
        % {
            "tmp": str(tmp_path),
            "roots": ", ".join(repr(r) for r in _NUMERIC_ROOTS),
        }
    )
    before, after, hits = observed.splitlines()

    assert before == "before:", (
        "importing aelfrice.retrieval pulled the numeric stack in before "
        f"retrieve() was called ({before}). The reading below is then "
        "unattributable, and this test would pass vacuously."
    )
    assert after == "after:numpy,scipy,snowballstemmer", (
        f"retrieve() no longer loads the numeric stack ({after}). A deferred "
        "import was dropped rather than moved, or the L1 lane stopped "
        "running."
    )
    assert hits != "hits:0", (
        "retrieve() returned nothing, so the lanes did not run and the "
        "module-set reading above says nothing about the retrieval path."
    )


# ---------------------------------------------------------------------------
# The driven gate-skipped fire (#1407 review)
# ---------------------------------------------------------------------------
#
# Neither test above can see a regression in the *fire*. The first probes
# `import aelfrice.hook` and stops there; the third drives `retrieve()`, where
# the numeric stack is supposed to load. An eager import added inside
# `user_prompt_submit` above the prompt-shape gate is invisible to both, and
# that is precisely where #1407 first put one: `from aelfrice.bm25 import
# reset_sidecar_outcome`, unconditional, ~60 lines above the gate. Every
# gate-skipped fire -- the majority, and the whole population #1351 exists for
# -- paid numpy + scipy + snowballstemmer to record that it did no index work.

_GATE_SKIP_PROBE = '''
import os
import sys

# Same env pinning as the retrieval probe: the lane resolvers are env-first,
# and opening a MemoryStore is a write, so an unpinned run would both change
# which modules load and mutate the developer's real store.
for _k in [k for k in os.environ if k.startswith(("AELFRICE_", "AELF_"))]:
    del os.environ[_k]

_tmp = %(tmp)r
os.environ["HOME"] = _tmp
os.environ["AELFRICE_DOTDIR"] = os.path.join(_tmp, ".aelfrice")
os.environ["AELFRICE_DB"] = os.path.join(_tmp, "memory.db")
os.environ["AELF_NO_UPDATE_CHECK"] = "1"
os.chdir(_tmp)

import io
import json
import pathlib

import aelfrice.hook as hook

_TARGETS = {%(roots)s}


def _roots():
    return ",".join(sorted({m.split(".")[0] for m in sys.modules} & _TARGETS))


print("imports_ok:%%d" %% int(hook._IMPORTS_OK))
print("before:" + _roots())
rc = hook.user_prompt_submit(
    stdin=io.StringIO(
        json.dumps({"prompt": "ok", "session_id": "s1", "cwd": _tmp})
    ),
    stdout=io.StringIO(),
)
print("after:" + _roots())
print("rc:%%d" %% rc)

rows = []
for _p in sorted(pathlib.Path(_tmp).rglob("*.jsonl")):
    for _line in _p.read_text(encoding="utf-8").splitlines():
        if not _line.strip():
            continue
        _rec = json.loads(_line)
        if _rec.get("hook") == "user_prompt_submit":
            rows.append(_rec)
print("rows:%%d" %% len(rows))
print("gate:" + (str(rows[-1].get("prompt_shape_gate_skip") or "") if rows else ""))
'''


@pytest.mark.timeout(90)
def test_a_gate_skipped_fire_does_not_import_the_numeric_stack(tmp_path) -> None:
    """The gate. A driven fire, in a subprocess, on the skipped path.

    `"ok"` is under `_MIN_PROMPT_LEN`, so `_should_skip_bm25` returns
    `trivial:short` and no retrieval runs. The fire must still write its audit
    row, and must finish with none of numpy / scipy / snowballstemmer in
    `sys.modules`.

    Four readings, each of which has to hold or the assertion below proves
    nothing:

    * **`imports_ok` is 1.** With the sentinel false, `user_prompt_submit`
      returns at its second statement and never reaches any of the code under
      test. Every case in `test_hook_import_resilience.py` patches exactly that
      sentinel, which is why none of them can see this defect either.
    * **`before` is empty.** Importing the hook pulls nothing, so the `after`
      reading is attributable to the fire and to nothing else.
    * **`gate` is the skip reason.** A gate that stopped firing would leave
      `after` empty for the wrong reason -- or, if the prompt stopped being
      trivial, would load the numeric stack legitimately and turn this into a
      false alarm. Either way the assertion below would no longer be about the
      skipped path.
    * **one audit row.** The fire ran to completion. A fire that died early
      also imports nothing.

    Falsifiable by restoring the eager `from aelfrice.bm25 import
    reset_sidecar_outcome` above the shape gate in `user_prompt_submit`.
    """
    observed = _probe(
        _GATE_SKIP_PROBE
        % {
            "tmp": str(tmp_path),
            "roots": ", ".join(repr(r) for r in _NUMERIC_ROOTS),
        }
    )
    imports_ok, before, after, rc, rows, gate = observed.splitlines()

    assert imports_ok == "imports_ok:1", (
        "the hook's runtime deps did not import, so the fire returned at the "
        "_IMPORTS_OK guard and never reached the code under test"
    )
    assert before == "before:", (
        f"importing aelfrice.hook pulled the numeric stack in ({before}); the "
        "reading after the fire is then unattributable"
    )
    assert rc == "rc:0", rc
    assert rows == "rows:1", (
        f"expected exactly one user_prompt_submit audit row, got {rows}; the "
        "fire did not run to completion and imports nothing for that reason"
    )
    assert gate == "gate:trivial:short", (
        f"the fixture prompt was not gate-skipped ({gate}); this test only "
        "says anything about the skipped path"
    )
    assert after == "after:", (
        f"a gate-skipped UserPromptSubmit fire imported {after.split(':', 1)[1]}"
        ". Every hook fire is a fresh process and most of them are "
        "gate-skipped, so this is paid by the majority of fires to do no work "
        "at all. Find the eager edge with:\n"
        "  python -X importtime -c 'import aelfrice.hook'\n"
        "and check for an above-the-gate import inside user_prompt_submit."
    )
