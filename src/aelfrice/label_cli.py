"""Interactive corpus-row labelling CLI (#859, enabler for #819).

`aelf label <module>` walks the operator through stub-row JSONL input
(`{id, query, beliefs}`) and produces a schema-valid output JSONL with
`gold_top_k`, `gold_ordering` (optional), `labeller_note`, `k`, plus the
common envelope (`provenance`, `label`).

The stub input is produced upstream — either by `retrieve_v2(query, k=200)`
over a real store (lab-side / private corpora) or by hand-authoring
synthetic queries (public tree). This module does NOT generate stubs;
it removes the JSONL-typing friction from the labelling-decision loop.

Validation here is structural (indices in pool, non-empty strings, k≥1) —
the full corpus contract is enforced by `tests/test_corpus_schema.py`
once rows land in `tests/corpus/v2_0/<module>/`. Keeping the heavy
validator in `tests/` (not shipped in wheel) and the lightweight
typo-catcher here is intentional.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TextIO


def _read_stub_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"{path}:{lineno}: invalid JSON ({exc.msg})"
                ) from None
            if not isinstance(row, dict):
                raise SystemExit(
                    f"{path}:{lineno}: row must be an object, got {type(row).__name__}"
                )
            for field in ("id", "query", "beliefs"):
                if field not in row:
                    raise SystemExit(
                        f"{path}:{lineno}: stub row missing required field {field!r}"
                    )
            if not isinstance(row["beliefs"], list) or not row["beliefs"]:
                raise SystemExit(
                    f"{path}:{lineno}: 'beliefs' must be a non-empty list"
                )
            rows.append(row)
    return rows


def _existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = row.get("id")
            if isinstance(rid, str) and rid:
                ids.add(rid)
    return ids


def _parse_indices(raw: str, pool_size: int) -> list[int]:
    """Parse '1,3,7' → [0,2,6]. '0' alone returns []. Raises ValueError on bad input."""
    raw = raw.strip()
    if raw == "0":
        return []
    if not raw:
        raise ValueError("empty selection")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("empty selection")
    out: list[int] = []
    seen: set[int] = set()
    for p in parts:
        try:
            idx = int(p)
        except ValueError:
            raise ValueError(f"not an integer: {p!r}") from None
        if idx < 1 or idx > pool_size:
            raise ValueError(f"index {idx} out of range 1..{pool_size}")
        zero_idx = idx - 1
        if zero_idx in seen:
            raise ValueError(f"duplicate index: {idx}")
        seen.add(zero_idx)
        out.append(zero_idx)
    return out


def _prompt(stdin: TextIO, out: TextIO, msg: str) -> str:
    print(msg, end="", file=out, flush=True)
    line = stdin.readline()
    if line == "":
        raise EOFError
    return line.rstrip("\n")


def _show_row(out: TextIO, idx: int, total: int, row: dict[str, Any]) -> None:
    print(f"\n[{idx}/{total}] id={row['id']}", file=out)
    print(f"  query: {row['query']}", file=out)
    print(f"  beliefs ({len(row['beliefs'])}):", file=out)
    for i, b in enumerate(row["beliefs"], 1):
        text = b.get("text", "")
        if len(text) > 120:
            text = text[:117] + "..."
        print(f"    {i:>3}. [{b.get('id', '?')}] {text}", file=out)


def _label_one(
    stub: dict[str, Any],
    *,
    stdin: TextIO,
    out: TextIO,
    default_k: int,
    no_ordering: bool,
) -> dict[str, Any] | None:
    """Return labelled row, or None if operator chose to skip."""
    pool = stub["beliefs"]
    pool_size = len(pool)

    while True:
        try:
            raw = _prompt(stdin, out, "  gold_top_k indices (comma-separated, 0=skip row): ")
        except EOFError:
            return None
        try:
            zero_idx = _parse_indices(raw, pool_size)
        except ValueError as exc:
            print(f"  invalid: {exc}; retry.", file=out)
            continue
        if not zero_idx:
            return None
        gold_top_k = [pool[i]["id"] for i in zero_idx]
        break

    gold_ordering: list[str] | None = None
    if not no_ordering:
        while True:
            try:
                raw = _prompt(
                    stdin, out,
                    "  gold_ordering indices (comma-separated, blank to omit): ",
                )
            except EOFError:
                return None
            if not raw.strip():
                break
            try:
                ord_idx = _parse_indices(raw, pool_size)
            except ValueError as exc:
                print(f"  invalid: {exc}; retry.", file=out)
                continue
            ord_ids = [pool[i]["id"] for i in ord_idx]
            missing = [bid for bid in gold_top_k if bid not in ord_ids]
            if missing:
                print(
                    f"  invalid: gold_ordering must contain every gold_top_k id; "
                    f"missing {missing}; retry.",
                    file=out,
                )
                continue
            gold_ordering = ord_ids
            break

    while True:
        try:
            note = _prompt(stdin, out, "  labeller_note (one line, required): ")
        except EOFError:
            return None
        if note.strip():
            break
        print("  invalid: labeller_note must be non-empty; retry.", file=out)

    k_val = stub.get("k", default_k)
    if not isinstance(k_val, int) or isinstance(k_val, bool) or k_val < 1:
        k_val = default_k

    provenance = stub.get("provenance") or "synthetic-v0.1"

    row: dict[str, Any] = {
        "id": stub["id"],
        "provenance": provenance,
        "labeller_note": note.strip(),
        "label": "graded",
        "query": stub["query"],
        "beliefs": pool,
        "gold_top_k": gold_top_k,
        "k": k_val,
    }
    if gold_ordering is not None:
        row["gold_ordering"] = gold_ordering
    return row


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        fh.write("\n")


def cmd_label(args: argparse.Namespace, out: object) -> int:
    """Entry point for `aelf label <module>`."""
    stream: TextIO = out if hasattr(out, "write") else sys.stdout  # type: ignore[assignment]
    stdin: TextIO = getattr(args, "stdin", None) or sys.stdin

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"error: --input {input_path} does not exist", file=stream)
        return 2

    stubs = _read_stub_rows(input_path)
    if not stubs:
        print(f"error: --input {input_path} contained no rows", file=stream)
        return 2

    skip_ids = _existing_ids(output_path) if args.resume else set()
    pending = [s for s in stubs if s["id"] not in skip_ids]
    total = len(pending)
    print(
        f"labelling module={args.module} input={input_path} "
        f"output={output_path} pending={total} "
        f"({len(stubs) - total} already in output, skipped)",
        file=stream,
    )

    labelled = 0
    skipped = 0
    try:
        for i, stub in enumerate(pending, 1):
            _show_row(stream, i, total, stub)
            row = _label_one(
                stub,
                stdin=stdin,
                out=stream,
                default_k=int(args.k),
                no_ordering=bool(args.no_ordering),
            )
            if row is None:
                skipped += 1
                print(f"  -> skipped ({i}/{total})", file=stream)
                continue
            _append_jsonl(output_path, row)
            labelled += 1
            print(f"  -> wrote ({i}/{total}, {labelled} labelled total)", file=stream)
    except KeyboardInterrupt:
        print(
            f"\ninterrupted: {labelled} labelled, {skipped} skipped, "
            f"{total - labelled - skipped} remaining. Re-run with --resume.",
            file=stream,
        )
        return 130

    print(
        f"done: {labelled} labelled, {skipped} skipped, output={output_path}",
        file=stream,
    )
    return 0


__all__ = ["cmd_label"]
