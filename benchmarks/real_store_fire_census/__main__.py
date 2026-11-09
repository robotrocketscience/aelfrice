"""Run the fire census over copies of aelfrice stores (#1887).

Usage:

    uv run python -m benchmarks.real_store_fire_census COPY [COPY ...]
        [--named COPY] [--legacy COPY] [--global-dir COPY]
        [--out FILE] [--only A012,A175] [--labels-out FILE]
        [--deadline SECONDS] [--dry-run]

Each COPY is a byte copy of a store folder (the folder that holds
`memory.db` and its sidecars) or of the `memory.db` file itself.
Positional copies are repository stores. `--named` takes a copy of a
`git:<name>` store, `--legacy` a copy of a legacy per-project store, and
`--global-dir` a copy of `~/.aelfrice`, which adds the `global` store and,
when it holds `user/memory.db`, the `user` store.

The runner opens each `memory.db` through a sqlite `mode=ro&immutable=1`
URI with `query_only` on, and reads sidecars with plain reads. It never
copies, writes or deletes anything beside a store; `immutable=1` keeps
sqlite from creating or rewriting `-wal` and `-shm`. Because an immutable
read ignores the `-wal` file, a store whose `-wal` still holds frames is
refused: `copy_store` folds the WAL into the copy.

It refuses any store path whose real path, or the real path of its db,
`-wal` or `-shm`, is a live location as `guard.live_reason` defines it: a
`.git` folder, `~/.aelfrice`, the `$AELFRICE_DB` store, or a `[store]`
marker store, compared case-insensitively. It refuses an `--out` or
`--labels-out` file in a store it reads, in a live location, or in this
repository.

Output is JSON with counts and fixed labels only. Stores appear as
opaque labels (`store01`, `named01`, `legacy01`, `global`, `user`).
Store paths are written only to the `--labels-out` file, which must sit
outside this repository.

`--dry-run` lists the probes and the store labels without opening any
store, and writes no labels file. Exit codes: 0 on a completed run, 2 on a usage error or a refused
path.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any

from .core import (
    DEFAULT_PROBE_DEADLINE_S,
    STORE_LABEL_RE,
    Ctx,
    Layout,
    Probe,
    has_unfolded_wal,
    outcome,
    run_one,
)
from . import guard
from .registry import all_probes, subsystem_of

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
MAX_STORES_PER_LAYOUT = 99


class UsageError(Exception):
    """A refused argument. The message names the argument, never its path."""


@dataclasses.dataclass(frozen=True)
class StoreArg:
    layout: Layout
    store_dir: Path
    db_file: Path


def is_live_location(path: Path) -> bool:
    """True when guard.live_reason refuses the path."""
    return guard.live_reason(path) is not None


def _check_db(db_file: Path, what: str) -> None:
    why = guard.store_live_reason(db_file)
    if why:
        raise UsageError(f"{what} resolves to a live store location ({why}); run on a byte copy")
    try:
        # Beside the link and beside its target: sqlite reads the target's.
        unfolded = has_unfolded_wal(db_file) or has_unfolded_wal(guard.real(db_file))
    except OSError:
        raise UsageError(f"{what} has a -wal file that can't be read") from None
    if unfolded:
        raise UsageError(f"{what} has a non-empty -wal file; make the copy with copy_store")


def _store_arg(raw: str, layout: Layout, what: str) -> StoreArg:
    p = Path(raw)
    why = guard.live_reason(p)
    if why:
        raise UsageError(f"{what} points at a live store location ({why}); run on a byte copy")
    if p.is_dir():
        store_dir, db_file = p, p / "memory.db"
    else:
        store_dir, db_file = p.parent, p
    if not db_file.is_file():
        raise UsageError(f"{what} has no memory.db")
    _check_db(db_file, what)
    return StoreArg(layout, guard.real(store_dir), guard.real(db_file))


def _label(layout: Layout, i: int) -> str:
    prefix = {"repo": "store", "named": "named", "legacy": "legacy"}[layout]
    return f"{prefix}{i:02d}"


def build_contexts(stores: list[str], named: list[str], legacy: list[str],
                   global_dir: str | None) -> tuple[list[Ctx], dict[str, str]]:
    """Contexts with opaque labels, plus the label -> path map. Opens nothing."""
    gdir: Path | None = None
    if global_dir is not None:
        g = Path(global_dir)
        why = guard.live_reason(g)
        if why:
            raise UsageError(f"--global-dir points at a live store location ({why}); run on a copy")
        if not g.is_dir():
            raise UsageError("--global-dir is not a folder")
        gdir = guard.real(g)
        for db in (gdir / "memory.db", gdir / "user" / "memory.db"):
            if db.exists() or db.is_symlink():
                _check_db(db, "--global-dir")
    groups: list[tuple[Layout, list[str], str]] = [
        ("repo", stores, "store argument"), ("named", named, "--named argument"),
        ("legacy", legacy, "--legacy argument"),
    ]
    ctxs: list[Ctx] = []
    private: dict[str, str] = {}
    seen_db: set[Path] = set()
    for layout, raws, what in groups:
        args = [_store_arg(r, layout, f"{what} {i}") for i, r in enumerate(raws, 1)]
        args.sort(key=lambda a: str(a.db_file))
        if len(args) > MAX_STORES_PER_LAYOUT:
            raise UsageError(f"more than {MAX_STORES_PER_LAYOUT} stores of one kind")
        for i, a in enumerate(args, 1):
            if a.db_file in seen_db:
                raise UsageError(f"{what} names a store twice")
            seen_db.add(a.db_file)
            label = _label(layout, i)
            private[label] = str(a.db_file)
            ctxs.append(Ctx(label, a.store_dir, gdir, False, a.db_file, layout))
    if gdir is not None:
        ctxs.append(Ctx("global", gdir, gdir, True, gdir / "memory.db", "global"))
        private["global"] = str(gdir / "memory.db")
        user = gdir / "user"
        if (user / "memory.db").is_file():
            ctxs.append(Ctx("user", user, gdir, True, user / "memory.db", "user"))
            private["user"] = str(user / "memory.db")
    if not ctxs:
        raise UsageError("no stores given")
    return ctxs, private


def _targets(p: Probe, ctxs: list[Ctx]) -> list[Ctx | None]:
    if p.kind == "untraceable":
        return [None]
    if p.kind == "global-log":
        return [c for c in ctxs if c.name == "global"] or [None]
    return list(ctxs)


def run_census(probes: list[Probe], ctxs: list[Ctx],
               deadline_s: float = DEFAULT_PROBE_DEADLINE_S) -> dict[str, Any]:
    """Run every probe on every target context. Returns the output document."""
    per_probe: dict[str, Any] = {}
    status_totals: dict[str, int] = {}
    outcomes: dict[str, int] = {}
    for c in ctxs:
        c.open()
    try:
        for p in probes:
            results: dict[str, Any] = {}
            for ctx in _targets(p, ctxs):
                label = "_" if ctx is None else ctx.name
                r = run_one(p, ctx, deadline_s)
                results[label] = r.to_json()
                status_totals[r.status] = status_totals.get(r.status, 0) + 1
            o = outcome(v["status"] for v in results.values())
            outcomes[o] = outcomes.get(o, 0) + 1
            per_probe[p.item_id] = {
                "name": p.name, "kind": p.kind, "subsystem": subsystem_of(p.item_id),
                "outcome": o, "stores": results,
            }
    finally:
        for c in ctxs:
            c.close()
    for c in ctxs:
        if not STORE_LABEL_RE.match(c.name):
            raise ValueError("store label outside STORE_LABEL_RE")
    return {
        "schema": SCHEMA_VERSION,
        "stores": {c.name: c.layout for c in ctxs},
        "summary": {"probes": len(probes), "status_totals": status_totals,
                    "outcomes": outcomes},
        "probes": per_probe,
    }


def _check_output(flag: str, path: Path, roots: list[Path]) -> None:
    """Refuse an output file in a store the run reads, a live location or the repo."""
    if any(guard.inside(path, r) for r in roots):
        raise UsageError(f"{flag} is inside a store the census reads")
    why = guard.live_reason(path)
    if why:
        raise UsageError(f"{flag} is in a live store location ({why})")
    if guard.inside(path, REPO_ROOT):
        raise UsageError(f"{flag} is inside the repository; write it elsewhere")


def _write_json(path: Path, doc: Any, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    if mode is not None:
        os.chmod(tmp, mode)
    os.replace(tmp, path)


def _parse_only(raw: str, probes: list[Probe]) -> list[Probe]:
    want = {x.strip() for x in raw.split(",") if x.strip()}
    if not want:
        return probes
    known = {p.item_id for p in probes}
    unknown = sorted(want - known)
    if unknown:
        raise UsageError("--only names unknown ids: " + ",".join(unknown[:10]))
    return [p for p in probes if p.item_id in want]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m benchmarks.real_store_fire_census",
        description="Count which aelfrice features fired in copies of real stores.")
    ap.add_argument("stores", nargs="*", metavar="COPY",
                    help="copy of a repository store folder or its memory.db")
    ap.add_argument("--named", action="append", default=[], metavar="COPY",
                    help="copy of a git:<name> store (repeatable)")
    ap.add_argument("--legacy", action="append", default=[], metavar="COPY",
                    help="copy of a legacy per-project store (repeatable)")
    ap.add_argument("--global-dir", default=None, metavar="COPY",
                    help="copy of ~/.aelfrice; adds the global and user stores")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the JSON here instead of standard output")
    ap.add_argument("--only", default="", help="comma-separated probe ids")
    ap.add_argument("--labels-out", type=Path, default=None,
                    help="write the label -> path map here (keep it out of the repo)")
    ap.add_argument("--deadline", type=float, default=DEFAULT_PROBE_DEADLINE_S,
                    help="seconds of sqlite work allowed per probe and store")
    ap.add_argument("--dry-run", action="store_true",
                    help="list probes and store labels without opening a store")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        probes = _parse_only(a.only, all_probes())
        ctxs, private = build_contexts(a.stores, a.named, a.legacy, a.global_dir)
        roots = [c.store_dir for c in ctxs]
        for flag, path in (("--out", a.out), ("--labels-out", a.labels_out)):
            if path is not None:
                _check_output(flag, path, roots)
        if a.deadline <= 0:
            raise UsageError("--deadline must be positive")
    except UsageError as exc:
        print(f"real_store_fire_census: {exc}", file=sys.stderr)
        return 2
    if a.dry_run:
        doc: dict[str, Any] = {
            "dry_run": True,
            "probes": [[p.item_id, p.kind, subsystem_of(p.item_id)] for p in probes],
            "stores": {c.name: c.layout for c in ctxs},
        }
    else:
        doc = run_census(probes, ctxs, a.deadline)
    if a.labels_out is not None:
        if a.dry_run:
            print("real_store_fire_census: dry run, so --labels-out is not written",
                  file=sys.stderr)
        else:
            _write_json(a.labels_out, private, 0o600)
    if a.out is None:
        sys.stdout.write(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    else:
        _write_json(a.out, doc)
    if not a.dry_run:
        print("real_store_fire_census: " + json.dumps(doc["summary"], sort_keys=True),
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
