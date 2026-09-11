#!/usr/bin/env python3
"""Re-derive every published figure for the Bash fire cap (#1522).

Usage:

    uv run python scripts/bash_fire_cap_latency.py all
    uv run python scripts/bash_fire_cap_latency.py order --pairs 60
    uv run python scripts/bash_fire_cap_latency.py cap --pairs 40
    uv run python scripts/bash_fire_cap_latency.py bytes
    uv run python scripts/bash_fire_cap_latency.py interleave
    uv run python scripts/bash_fire_cap_latency.py all --dry-run

Exists because #1469 rules that a published number needs a command that
re-derives it. Every figure in `CHANGELOG/unreleased/1522-bash-fire-cap.md`
and in the `hook_search_tool` / `session_ring` docstrings comes from one
of these four probes.

The probes:

  order      What the cap check costs a Bash call that is not a search.
             Copies `src/` twice into a scratch directory and swaps the
             two statements in `_do_search`'s Bash branch in one copy, so
             both arms are real source trees differing only in that
             order. One cold `python -m aelfrice.hook_search_tool` per
             sample, arms strictly interleaved, median of the paired
             deltas.

  cap        What a cap-suppressed fire saves against a firing one. Same
             command string in both arms; the only difference is whether
             the session has already reached `BASH_FIRE_CAP_PER_TURN`.

  bytes      The JSON cost of one session's entry in the ring's Bash map,
             measured as the file-size delta of adding sessions one at a
             time through the shipped writers.

  interleave The A,A,A,B,A,A,A count for the shipped per-session map
             against a record-scoped counter. The record-scoped arm is
             the shipped tree with two edits, which together are what
             holding the counters on the ring record would do: the Bash
             writes go through `_normalize_for_session` instead of
             touching the `bash` key alone, and its fresh-ring branch
             returns an empty Bash map.

Latency figures are wall-clock on the machine that runs this, so they
move with load and hardware. The paired deltas and the emitted/suppressed
counts are the parts that hold; treat the absolute medians as this run's.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TIMEOUT_S = 300.0

_SHIPPED_ORDER = """\
        bash_extracted = _extract_bash_query(payload)
        if bash_extracted is None:
            return
        if _bash_fire_cap_reached(session_id):
            return
"""
_CAP_FIRST_ORDER = """\
        if _bash_fire_cap_reached(session_id):
            return
        bash_extracted = _extract_bash_query(payload)
        if bash_extracted is None:
            return
"""
_KEEPS_BASH_MAP = (
    '            "promotion_dedup": [],\n'
    '            "bash": bash,\n'
    "        }\n"
)
_DROPS_BASH_MAP = (
    '            "promotion_dedup": [],\n'
    '            "bash": {},\n'
    "        }\n"
)

_BASH_SCOPED_WRITE = (
    "            data = _read_ring_unlocked(ring_path)\n"
    '            bash = _normalize_bash_state(data.get("bash"))\n'
)
_RECORD_SCOPED_WRITE = (
    "            data = _read_ring_unlocked(ring_path)\n"
    "            data = _normalize_for_session(\n"
    "                data, session_id, _resolve_ring_max()\n"
    "            )\n"
    '            bash = data["bash"]\n'
)


def _patch(path: Path, old: str, new: str) -> None:
    """Replace the single occurrence of `old` in `path`, or fail loudly."""
    text = path.read_text(encoding="utf-8")
    found = text.count(old)
    if found != 1:
        raise SystemExit(
            f"{path}: expected 1 occurrence of the anchor, found {found}. "
            "The source moved; update this script before trusting it."
        )
    path.write_text(text.replace(old, new), encoding="utf-8")


def _build_tree(root: Path, name: str) -> Path:
    dst = root / name
    shutil.copytree(REPO_ROOT / "src", dst)
    for cached in dst.rglob("__pycache__"):
        shutil.rmtree(cached)
    if name == "capfirst":
        _patch(
            dst / "aelfrice" / "hook_search_tool.py",
            _SHIPPED_ORDER,
            _CAP_FIRST_ORDER,
        )
    elif name == "recordscoped":
        _patch(
            dst / "aelfrice" / "session_ring.py",
            _KEEPS_BASH_MAP,
            _DROPS_BASH_MAP,
        )
        _patch(
            dst / "aelfrice" / "session_ring.py",
            _BASH_SCOPED_WRITE,
            _RECORD_SCOPED_WRITE,
        )
    return dst


def _new_store(work: Path) -> Path:
    """Create an empty but real store under `work` and return its path."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from aelfrice.store import MemoryStore  # noqa: PLC0415

    db = work / "aelfrice" / "memory.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    MemoryStore(str(db)).close()
    return db


def _hook_env(tree: Path, db: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree)
    env["AELFRICE_DB"] = str(db)
    env["AELF_NO_UPDATE_CHECK"] = "1"
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    return env


def _payload(command: str, session: str, cwd: Path) -> str:
    return json.dumps({
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": command},
        "cwd": str(cwd),
        "session_id": session,
    })


def _fire(
    env: dict[str, str], command: str, session: str, cwd: Path,
) -> tuple[float, bool]:
    """Run one hook process; return its wall-clock ms and whether it spoke."""
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "aelfrice.hook_search_tool"],
        input=_payload(command, session, cwd),
        capture_output=True,
        text=True,
        env=env,
        cwd=str(cwd),
        timeout=TIMEOUT_S,
        check=False,
    )
    elapsed = (time.perf_counter() - t0) * 1000.0
    if proc.returncode != 0:
        raise SystemExit(f"hook exited {proc.returncode}: {proc.stderr}")
    return elapsed, bool(proc.stdout.strip())


def _report(name_a: str, a: list[float], name_b: str, b: list[float]) -> None:
    deltas = [x - y for x, y in zip(a, b, strict=True)]
    pairs = len(deltas)
    print(f"  pairs = {pairs}")
    print(f"  {name_a:<26} median = {statistics.median(a):7.1f} ms "
          f"(min {min(a):.1f}, max {max(a):.1f})")
    print(f"  {name_b:<26} median = {statistics.median(b):7.1f} ms "
          f"(min {min(b):.1f}, max {max(b):.1f})")
    print(f"  paired delta median       = {statistics.median(deltas):7.1f} ms")
    print(f"  delta positive in {sum(1 for d in deltas if d > 0)}/{pairs} pairs")


def probe_order(work: Path, pairs: int) -> None:
    """(a) What the cap check costs a Bash call that is not a search."""
    trees = {n: _build_tree(work, n) for n in ("shipped", "capfirst")}
    db = _new_store(work)
    envs = {n: _hook_env(t, db) for n, t in trees.items()}
    session = "order-" + uuid.uuid4().hex[:8]
    cmd = "cat notes.txt"
    for arm, env in envs.items():
        probe = subprocess.run(
            [sys.executable, "-c",
             "import aelfrice.hook_search_tool as m; print(m.__file__)"],
            capture_output=True, text=True, env=env, cwd=str(work),
            timeout=TIMEOUT_S, check=True,
        )
        print(f"  {arm} module: {probe.stdout.strip()}")
    for _ in range(3):  # warm the bytecode caches; neither arm pays compile
        for env in envs.values():
            _fire(env, cmd, session, work)
    shipped: list[float] = []
    capfirst: list[float] = []
    for i in range(pairs):
        order = ("shipped", "capfirst") if i % 2 == 0 else ("capfirst", "shipped")
        got = {}
        for arm in order:
            elapsed, spoke = _fire(envs[arm], cmd, session, work)
            if spoke:
                raise SystemExit(f"{arm}: a non-search Bash call emitted output")
            got[arm] = elapsed
        shipped.append(got["shipped"])
        capfirst.append(got["capfirst"])
    _report("cap first", capfirst, "extract first (shipped)", shipped)


def probe_cap(work: Path, pairs: int) -> None:
    """(b) What a cap-suppressed fire saves against a firing one."""
    tree = _build_tree(work, "shipped")
    db = _new_store(work)
    env = _hook_env(tree, db)
    os.environ["AELFRICE_DB"] = str(db)
    from aelfrice.hook_search_tool import (  # noqa: PLC0415
        BASH_FIRE_CAP_PER_TURN,
    )
    from aelfrice.session_ring import (  # noqa: PLC0415
        read_bash_fire_state,
        record_bash_fire,
        stamp_bash_turn,
    )

    capped = "cap-" + uuid.uuid4().hex[:8]
    stamp_bash_turn(capped)

    def hold_at_cap() -> None:
        """Re-assert the capped session's entry; eviction can drop it."""
        while True:
            state = read_bash_fire_state(capped)
            if state.get("fires", 0) >= BASH_FIRE_CAP_PER_TURN:
                return
            if not state:
                stamp_bash_turn(capped)
            record_bash_fire(capped)

    hold_at_cap()
    for _ in range(3):  # warm the bytecode caches
        _fire(env, "rg warmtoken src/", "warm-" + uuid.uuid4().hex[:8], work)
        hold_at_cap()
        _fire(env, "rg warmtoken src/", capped, work)

    firing: list[float] = []
    suppressed: list[float] = []
    emitted = silent = 0
    for i in range(pairs):
        token = "probetok" + uuid.uuid4().hex[:8]
        fresh = "fresh-" + uuid.uuid4().hex[:8]
        stamp_bash_turn(fresh)
        hold_at_cap()
        if i % 2 == 0:
            f_ms, f_spoke = _fire(env, f"rg {token} src/", fresh, work)
            hold_at_cap()
            s_ms, s_spoke = _fire(env, f"rg {token} src/", capped, work)
        else:
            s_ms, s_spoke = _fire(env, f"rg {token} src/", capped, work)
            f_ms, f_spoke = _fire(env, f"rg {token} src/", fresh, work)
        emitted += int(f_spoke)
        silent += int(not s_spoke)
        firing.append(f_ms)
        suppressed.append(s_ms)
    print(f"  firing arm emitted a block in {emitted}/{pairs}")
    print(f"  suppressed arm stayed silent in {silent}/{pairs}")
    _report("firing", firing, "suppressed", suppressed)
    saving = statistics.median(
        [f - s for f, s in zip(firing, suppressed, strict=True)]
    )
    over_cap = 10 - BASH_FIRE_CAP_PER_TURN
    print(f"  a ten-rg turn suppresses {over_cap} fires "
          f"= {over_cap * saving / 1000.0:.1f} s saved")


def probe_bytes(work: Path) -> None:
    """(e) The JSON cost of one session's entry in the ring's Bash map."""
    db = _new_store(work)
    os.environ["AELFRICE_DB"] = str(db)
    from aelfrice import session_ring as ring  # noqa: PLC0415

    path = ring._session_ring_path()  # noqa: SLF001
    if path is None:
        raise SystemExit("no ring path for this store")
    sizes: list[int] = []
    for _ in range(ring.BASH_STATE_MAX_SESSIONS):
        session = str(uuid.uuid4())
        ring.stamp_bash_turn(session)
        ring.record_bash_fire(session)
        ring.record_bash_fire(session)
        sizes.append(len(path.read_bytes()))
    deltas = [sizes[i] - sizes[i - 1] for i in range(1, len(sizes))]
    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"  file bytes by session count: {sizes}")
    print(f"  marginal bytes per entry:    {deltas}")
    print(f"  median marginal entry       = {statistics.median(deltas):.0f} B")
    print(f"  full {len(data['bash'])}-entry bash map        = "
          f"{len(json.dumps(data['bash']))} B")


def probe_interleave(work: Path) -> None:
    """The A,A,A,B,A,A,A count, per-session map against record-scoped."""
    sequence = [
        ("A", "alphazero"), ("A", "alphaone"), ("A", "alphatwo"),
        ("B", "bravozero"),
        ("A", "alphathree"), ("A", "alphafour"), ("A", "alphafive"),
    ]
    for arm in ("recordscoped", "shipped"):
        tree = _build_tree(work, arm)
        arm_work = work / f"run-{arm}"
        arm_work.mkdir()
        env = _hook_env(tree, _new_store(arm_work))
        fired = [
            _fire(env, f"rg {token} src/", f"interleave-{sid}", arm_work)[1]
            for sid, token in sequence
        ]
        per: dict[str, list[bool]] = {"A": [], "B": []}
        for (sid, _token), spoke in zip(sequence, fired, strict=True):
            per[sid].append(spoke)
        label = "".join(sid for sid, _ in sequence)
        print(f"  {arm:<13} {label}: emitted {sum(fired)} of {len(fired)}"
              f"  (A {sum(per['A'])}/{len(per['A'])},"
              f" B {sum(per['B'])}/{len(per['B'])})")


PROBES = ("order", "cap", "bytes", "interleave")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("probe", choices=(*PROBES, "all"))
    parser.add_argument(
        "--pairs", type=int, default=40,
        help="interleaved pairs per latency probe (default 40)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print what would run, touch nothing",
    )
    args = parser.parse_args(argv)
    chosen = PROBES if args.probe == "all" else (args.probe,)
    if args.pairs < 1:
        parser.error("--pairs must be at least 1")
    if args.dry_run:
        for name in chosen:
            detail = f", {args.pairs} pairs" if name in ("order", "cap") else ""
            print(f"would run probe {name!r} against {REPO_ROOT / 'src'}{detail}")
        return 0
    for name in chosen:
        print(f"\n== {name} ==")
        work = Path(tempfile.mkdtemp(prefix=f"bashcap-{name}-"))
        try:
            if name == "order":
                probe_order(work, args.pairs)
            elif name == "cap":
                probe_cap(work, args.pairs)
            elif name == "bytes":
                probe_bytes(work)
            else:
                probe_interleave(work)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
