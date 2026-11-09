"""Run the fire census over copies of aelfrice stores (#1887).

Usage:

    uv run python -m benchmarks.real_store_fire_census COPY [COPY ...]
        [--named COPY] [--legacy COPY] [--global-dir COPY]
        [--out FILE] [--only A012,A175] [--labels-out FILE]
        [--deadline SECONDS] [--dry-run]

Each COPY is a folder that `copy_store` made, or a store folder or
`memory.db` below one. Positional copies are repository stores. `--named`
takes a `git:<name>` store, `--legacy` a legacy per-project store, and
`--global-dir` a copy of `~/.aelfrice`, which adds the `global` store and,
when it holds `user/memory.db`, the `user` store.

The runner reads only copies that `copy_store` made. Each store's db must be
recorded, by inode and size, in the `.fire-census-copy` token at or above
its folder (`copy_token`), whatever the path; a live store has no token.
The live-location checks in `guard` run first as a second line. The db is
opened through a descriptor the runner opened with `O_NOFOLLOW`, checked
against the token with `fstat`, with a `mode=ro&immutable=1` URI and
`query_only` on, so the read changes no byte beside the store. Sidecars are
read with plain reads.

`--out` and `--labels-out` are written through a descriptor for their
folder: a temp file created there with `O_EXCL | O_NOFOLLOW`, then renamed
over the final name. The runner refuses a final path that is an existing
symlink, a folder that holds a store or a copy token, a live location, and
this repository.

Output is JSON with counts and fixed labels only. Stores appear as
opaque labels (`store01`, `named01`, `legacy01`, `global`, `user`).
Store paths are written only to the `--labels-out` file.

`--dry-run` checks the copies and lists the probes and the store labels
without opening any db, and writes no labels file. Exit codes: 0 on a
completed run, 2 on a usage error or a refused path.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

from . import copy_token, guard
from .core import (
    DEFAULT_PROBE_DEADLINE_S,
    STORE_LABEL_RE,
    Ctx,
    IdentityMismatchError,
    Layout,
    Probe,
    UnfoldedWalError,
    has_unfolded_wal,
    outcome,
    run_one,
)
from .registry import all_probes, subsystem_of

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
MAX_STORES_PER_LAYOUT = 99
NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


class UsageError(Exception):
    """A refused argument. The message names the argument, never its path."""


@dataclasses.dataclass(frozen=True)
class StoreArg:
    layout: Layout
    store_dir: Path
    db_file: Path
    expect: tuple[int, int]


def is_live_location(path: Path) -> bool:
    """True when guard.live_reason refuses the path."""
    return guard.live_reason(path) is not None


def _check_db(db_file: Path, what: str) -> tuple[int, int]:
    """Guard, WAL and token checks for one db; returns the token's identity."""
    why = guard.store_live_reason(db_file)
    if why:
        raise UsageError(f"{what} is a live store location ({why}); run on a copy")
    try:
        unfolded = has_unfolded_wal(db_file)
    except OSError:
        raise UsageError(f"{what} has a -wal file that can't be read") from None
    if unfolded:
        raise UsageError(f"{what} has a non-empty -wal file; make the copy with copy_store")
    try:
        return copy_token.verify_db(db_file)
    except copy_token.TokenError as exc:
        raise UsageError(f"{what}: {exc}") from None


def _store_arg(raw: str, layout: Layout, what: str) -> StoreArg:
    p = Path(os.path.abspath(raw))
    why = guard.live_reason(p)
    if why:
        raise UsageError(f"{what} is a live store location ({why}); run on a copy")
    if p.is_dir():
        store_dir, db_file = p, p / "memory.db"
    else:
        store_dir, db_file = p.parent, p
    if not os.path.lexists(db_file):
        raise UsageError(f"{what} has no memory.db")
    expect = _check_db(db_file, what)
    return StoreArg(layout, store_dir, db_file, expect)


def _label(layout: Layout, i: int) -> str:
    prefix = {"repo": "store", "named": "named", "legacy": "legacy"}[layout]
    return f"{prefix}{i:02d}"


def build_contexts(stores: list[str], named: list[str], legacy: list[str],
                   global_dir: str | None) -> tuple[list[Ctx], dict[str, str]]:
    """Contexts with opaque labels, plus the label -> path map. Opens nothing."""
    gdir: Path | None = None
    gexpect: dict[str, tuple[int, int]] = {}
    if global_dir is not None:
        g = Path(os.path.abspath(global_dir))
        why = guard.live_reason(g)
        if why:
            raise UsageError(f"--global-dir is a live store location ({why}); run on a copy")
        if not g.is_dir():
            raise UsageError("--global-dir is not a folder")
        try:
            copy_token.verify_folder(g)
        except copy_token.TokenError as exc:
            raise UsageError(f"--global-dir: {exc}") from None
        gdir = g
        for key, db in (("global", g / "memory.db"), ("user", g / "user" / "memory.db")):
            if os.path.lexists(db):
                gexpect[key] = _check_db(db, "--global-dir")
    groups: list[tuple[Layout, list[str], str]] = [
        ("repo", stores, "store argument"), ("named", named, "--named argument"),
        ("legacy", legacy, "--legacy argument"),
    ]
    ctxs: list[Ctx] = []
    private: dict[str, str] = {}
    seen_db: set[tuple[int, int]] = set()
    for layout, raws, what in groups:
        args = [_store_arg(r, layout, f"{what} {i}") for i, r in enumerate(raws, 1)]
        args.sort(key=lambda a: str(a.db_file))
        if len(args) > MAX_STORES_PER_LAYOUT:
            raise UsageError(f"more than {MAX_STORES_PER_LAYOUT} stores of one kind")
        for i, a in enumerate(args, 1):
            if a.expect in seen_db:
                raise UsageError(f"{what} names a store twice")
            seen_db.add(a.expect)
            label = _label(layout, i)
            private[label] = str(a.db_file)
            ctxs.append(Ctx(label, a.store_dir, gdir, False, a.db_file, layout,
                            expect=a.expect))
    if gdir is not None:
        gdb = gdir / "memory.db"
        ctxs.append(Ctx("global", gdir, gdir, True, gdb if "global" in gexpect else None,
                        "global", expect=gexpect.get("global")))
        private["global"] = str(gdb)
        if "user" in gexpect:
            user = gdir / "user"
            ctxs.append(Ctx("user", user, gdir, True, user / "memory.db", "user",
                            expect=gexpect["user"]))
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
    try:
        for c in ctxs:
            try:
                c.open()
            except (OSError, IdentityMismatchError, UnfoldedWalError):
                raise UsageError("a store changed after it was checked; "
                                 "make a fresh copy with copy_store") from None
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


class OutTarget:
    """An output file, checked and pinned to its folder before the run.

    The folder is opened once, with `O_DIRECTORY | O_NOFOLLOW`, and every
    later step names files relative to that descriptor, so swapping the
    folder for a symlink after the check changes nothing. Where `dir_fd`
    isn't supported (Windows), the file is written with `tempfile.mkstemp`
    in the folder and renamed with `os.replace`.
    """

    def __init__(self, flag: str, path: Path, roots: list[Path]) -> None:
        self.flag = flag
        self.path = Path(os.path.abspath(path))
        self.name = self.path.name
        self.dfd: int | None = None
        self._refuse_path(roots)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if os.open in os.supports_dir_fd and hasattr(os, "O_DIRECTORY"):
            self.dfd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY | NOFOLLOW)
            st = os.fstat(self.dfd)
            if guard.ident(self.path.parent) != (st.st_dev, st.st_ino):
                self.close()
                raise UsageError(f"{flag}'s folder changed while it was checked")
        self._refuse_folder()

    def _refuse_path(self, roots: list[Path]) -> None:
        flag = self.flag
        if os.path.lexists(self.path):
            mode = os.lstat(self.path).st_mode
            if stat.S_ISLNK(mode):
                raise UsageError(f"{flag} is an existing symlink")
            if not stat.S_ISREG(mode):
                raise UsageError(f"{flag} exists and is not a regular file")
        if any(guard.inside(self.path, r) for r in roots):
            raise UsageError(f"{flag} is inside a store the census reads")
        why = guard.live_reason(self.path)
        if why:
            raise UsageError(f"{flag} is in a live store location ({why})")
        if guard.inside(self.path, REPO_ROOT):
            raise UsageError(f"{flag} is inside the repository; write it elsewhere")

    def _exists(self, name: str) -> bool:
        try:
            if self.dfd is not None:
                os.stat(name, dir_fd=self.dfd, follow_symlinks=False)
            else:
                os.lstat(self.path.parent / name)
        except FileNotFoundError:
            return False
        return True

    def _refuse_folder(self) -> None:
        for name in ("memory.db", copy_token.TOKEN_FILENAME):
            if self._exists(name):
                self.close()
                raise UsageError(f"{self.flag}'s folder holds a store or a copy token")

    def write(self, doc: Any, mode: int = 0o644) -> None:
        data = (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode("utf-8")
        if self.dfd is None:
            import tempfile
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix="." + self.name + ".")
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.chmod(tmp, mode)
            os.replace(tmp, self.path)
            return
        import secrets
        tmp = f".{self.name}.{secrets.token_hex(8)}.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | NOFOLLOW, mode,
                     dir_fd=self.dfd)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        try:
            final = os.stat(self.name, dir_fd=self.dfd, follow_symlinks=False)
            if stat.S_ISLNK(final.st_mode):
                raise UsageError(f"{self.flag} became a symlink during the run")
        except FileNotFoundError:
            pass
        except UsageError:
            os.unlink(tmp, dir_fd=self.dfd)
            raise
        os.rename(tmp, self.name, src_dir_fd=self.dfd, dst_dir_fd=self.dfd)

    def close(self) -> None:
        if self.dfd is not None:
            os.close(self.dfd)
            self.dfd = None


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
    targets: dict[str, OutTarget] = {}
    try:
        try:
            probes = _parse_only(a.only, all_probes())
            ctxs, private = build_contexts(a.stores, a.named, a.legacy, a.global_dir)
            roots = [c.store_dir for c in ctxs]
            if a.deadline <= 0:
                raise UsageError("--deadline must be positive")
            for flag, path in (("--out", a.out), ("--labels-out", a.labels_out)):
                if path is not None:
                    targets[flag] = OutTarget(flag, path, roots)
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
            try:
                doc = run_census(probes, ctxs, a.deadline)
            except UsageError as exc:
                print(f"real_store_fire_census: {exc}", file=sys.stderr)
                return 2
        try:
            if "--labels-out" in targets:
                if a.dry_run:
                    print("real_store_fire_census: dry run, so --labels-out is not written",
                          file=sys.stderr)
                else:
                    targets["--labels-out"].write(private, 0o600)
            if "--out" in targets:
                targets["--out"].write(doc)
            else:
                sys.stdout.write(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        except UsageError as exc:
            print(f"real_store_fire_census: {exc}", file=sys.stderr)
            return 2
        if not a.dry_run:
            print("real_store_fire_census: " + json.dumps(doc["summary"], sort_keys=True),
                  file=sys.stderr)
        return 0
    finally:
        for t in targets.values():
            t.close()


if __name__ == "__main__":
    sys.exit(main())
