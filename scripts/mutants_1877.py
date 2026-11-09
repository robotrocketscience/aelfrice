#!/usr/bin/env python3
"""Mutation check for `aelf doctor --retire-relay-rows` (#1877): each mutant must fail a test.

Each mutant replaces one exact snippet of a source file the pass uses. The script
copies the tree into a temporary folder, applies one mutant at a time there, runs
the relay-row, #1780, #1875, and cross-session notice tests against the copy, and
puts the snippet back. Your checkout is never changed.

Usage:
    uv run python scripts/mutants_1877.py --dry-run      # check every anchor, run nothing
    uv run python scripts/mutants_1877.py --list         # print each mutant and retired name
    uv run python scripts/mutants_1877.py                # run every mutant
    uv run python scripts/mutants_1877.py NAME [NAME...] # run the named mutants
    uv run python scripts/mutants_1877.py --shard 0/4    # run every 4th mutant, from the first

A full run takes about 15 minutes and needs one copy of the tree, so check free disk
space first. It prints one line per mutant, `killed NAME by TEST` or `SURVIVED NAME`,
and exits 1 when a mutant survives or its anchor isn't found exactly once.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RR = "src/aelfrice/relay_rows.py"
NF = "src/aelfrice/noise_filter.py"
SP = "src/aelfrice/scheduled_prompts.py"
CLI = "src/aelfrice/cli.py"
ST = "src/aelfrice/store.py"
UL = "src/aelfrice/ulid.py"
TESTS = (
    "tests/test_relay_rows_1877.py", "tests/test_retire_scheduled_prompts_1780.py",
    "tests/test_relay_scaffolding.py", "tests/test_cross_session_notice.py",
)
COPIED = ("src", "tests", "docs", "scripts", "CHANGELOG", ".github")
COPIED_FILES = ("pyproject.toml", "README.md", "uv.lock")
TEST_TIMEOUT_SECONDS = 800

# (name, file, exact snippet, replacement). Each snippet must occur once in its file.
MUTANTS: list[tuple[str, str, str, str]] = [
    ('pin_off', RR,
     '                pinned = turns.pinned(i, n, capped, src, dst)\n',
     '                pinned = True\n'),
    ('cutoff_tz_minus7h', RR,
     'RAW_RECORD_CUTOFF_MS: Final[int] = 1_790_805_153_000',
     'RAW_RECORD_CUTOFF_MS: Final[int] = 1_790_779_953_000'),
    ('cutoff_tz_plus7h', RR,
     'RAW_RECORD_CUTOFF_MS: Final[int] = 1_790_805_153_000',
     'RAW_RECORD_CUTOFF_MS: Final[int] = 1_790_830_353_000'),
    ('cutoff_merge_time', RR,
     'RAW_RECORD_CUTOFF_MS: Final[int] = 1_790_805_153_000',
     'RAW_RECORD_CUTOFF_MS: Final[int] = 1_790_811_412_000'),
    ('cutoff_one_day_early', RR,
     'RAW_RECORD_CUTOFF_MS: Final[int] = 1_790_805_153_000',
     'RAW_RECORD_CUTOFF_MS: Final[int] = 1_790_718_753_000'),
    ('cutoff_release_510', RR,
     'RAW_RECORD_CUTOFF_MS: Final[int] = 1_790_805_153_000',
     'RAW_RECORD_CUTOFF_MS: Final[int] = 1_791_346_666_000'),
    ('cutoff_nov_1', RR,
     'RAW_RECORD_CUTOFF_MS: Final[int] = 1_790_805_153_000',
     'RAW_RECORD_CUTOFF_MS: Final[int] = 1_793_491_200_000'),
    ('cutoff_seconds_unit', RR,
     'RAW_RECORD_CUTOFF_MS: Final[int] = 1_790_805_153_000',
     'RAW_RECORD_CUTOFF_MS: Final[int] = 1_790_805_153'),
    ('cutoff_plus_1ms', RR,
     'RAW_RECORD_CUTOFF_MS: Final[int] = 1_790_805_153_000',
     'RAW_RECORD_CUTOFF_MS: Final[int] = 1_790_805_153_001'),
    ('cutoff_minus_1ms', RR,
     'RAW_RECORD_CUTOFF_MS: Final[int] = 1_790_805_153_000',
     'RAW_RECORD_CUTOFF_MS: Final[int] = 1_790_805_152_999'),
    ('late_check_off', RR,
     '        if any((rows.keys[i][0], rows.texts[i]) in rows.late for i, _, _, _ in takes):\n            continue\n',
     ''),
    ('late_any_session', RR,
     'if any((rows.keys[i][0], rows.texts[i]) in rows.late for i, _, _, _ in takes):',
     'if any(t == rows.texts[i] for i, _, _, _ in takes for _, t in rows.late):'),
    ('late_last_row', RR,
     'if any((rows.keys[i][0], rows.texts[i]) in rows.late for i, _, _, _ in takes):',
     'if any((rows.keys[i][0], rows.texts[i + n - 1]) in rows.late for i, n, _, _ in takes):'),
    ('notice_label_off', RR,
     '                first = j == 0 and notice\n',
     '                first = False\n'),
    ('take_past_cut', RR,
     'runs.append((i, min(n, len(head)), notice,',
     'runs.append((i, n, notice,'),
    ('next_any_path', RR,
     '            sid, _, path = self._keys[t]\n',
     '            sid, _, path = self._keys[t]\n            path = None\n'),
    ('ulid_len_check_off', UL,
     '    if len(value) != _ULID_LEN:\n        return None\n',
     ''),
    ('ulid_bad_char_ok', UL,
     '        if digit < 0:\n            return None\n',
     ''),
    ('ulid_base_off', UL,
     '        ms = ms * 32 + digit',
     '        ms = ms * 31 + digit'),
    ('ulid_case_off', UL,
     '    for ch in value[:10].upper():',
     '    for ch in value[:10]:'),
    ('host_form_loose', NF,
     '    host = _HOST_TAG_RE.fullmatch(tag)',
     '    host = re.fullmatch(r"<(" + _RELAY_TAG_ALT + r")((?:\\s+[^\\s=>]+=(?:\\"[^\\"]*\\"|\'[^\']*\'|[^\\s>]*))+)\\s*>", tag, re.I)'),
    ('crlf_split_lf_only', NF,
     '_LINE_BREAK_RE: Final[re.Pattern[str]] = re.compile(r"\\r\\n|\\n|\\r")',
     '_LINE_BREAK_RE: Final[re.Pattern[str]] = re.compile(r"\\n")'),
    ('veto_opener_first_only', RR,
     '                if relay_row_shape(whole[0]) is not None:',
     '                if relay_row_shape(whole[0]) == RELAY_ROW_OPENER:'),
    ('shape_drop_rest_check', NF,
     '        and not text[opener.end():].strip() and _host_shaped(tag.group(0))',
     '        and _host_shaped(tag.group(0))'),
    ('shape_no_host_attrs', NF,
     '        and not text[opener.end():].strip() and _host_shaped(tag.group(0))',
     '        and not text[opener.end():].strip()'),
    ('shape_notice_to_line_end', NF,
     '    if _NOTICE_TEMPLATE_LINE_RE.match(text):\n        return RELAY_ROW_NOTICE',
     '    if _CROSS_SESSION_NOTICE_RE.match(text):\n        return RELAY_ROW_NOTICE'),
    ('cut_tags_unchecked', NF,
     '        if tag is None or not _host_shaped(tag.group(0)):\n            return False',
     '        pass'),
    ('tag_none_ok', NF,
     'if tag is None or not _host_shaped(tag.group(0)):',
     'if tag is not None and not _host_shaped(tag.group(0)):'),
    ('cut_notices_unchecked', NF,
     '    return not any(\n        _CROSS_SESSION_NOTICE_RE.match(line)',
     '    return not any(\n        False and _CROSS_SESSION_NOTICE_RE.match(line)'),
    ('anchor_host_check_off', RR,
     'relay = rest != anchor and relay_cut_is_host_shaped(cut)',
     'relay = rest != anchor'),
    ('veto_first_row_only', RR,
     '                        vetoed.update(range(i, i + n))',
     '                        vetoed.add(i)'),
    ('veto_off', RR,
     '                        vetoed.update(range(i, i + n))',
     '                        pass'),
    ('veto_not_applied_to_blocks', RR,
     '    return {i: shape for i, shape in found.items() if i not in vetoed}, vetoed',
     '    return found, vetoed'),
    ('notice_rows_without_anchor', RR,
     '            if relay_row_shape(text) == RELAY_ROW_OPENER:\n                shape_at[i] = RELAY_ROW_OPENER',
     '            if relay_row_shape(text) is not None:\n                shape_at[i] = str(relay_row_shape(text))'),
    ('key_without_path', RR,
     '        key = (sid, ts, path)\n',
     '        key = (sid, ts, None)\n'),
    ('no_holed_fallback', RR,
     'for keep_spans in (True, False):',
     'for keep_spans in (True,):'),
    ('capped_need_whole', RR,
     'need = len(whole) - 1 if capped else len(whole)',
     'need = len(whole)'),
    ('uncapped_need_minus_one', RR,
     'need = len(whole) - 1 if capped else len(whole)',
     'need = len(whole) - 1'),
    ('cap_strict_gt', RR,
     'capped = len(anchor) >= ANCHOR_TEXT_MAX_LEN',
     'capped = len(anchor) > ANCHOR_TEXT_MAX_LEN'),
    ('no_head_prefix_check', RR,
     'if not head or whole[: len(head)] != head:',
     'if not head:'),
    ('any_row_relay', RR,
     'every_row_relay.get(bid, True) and shape is not None',
     'every_row_relay.get(bid, True) or shape is not None'),
    ('split_evidence_bypass', RR,
     '    report.candidates = [bid for bid, _ in candidates]',
     '    report.candidates = sorted(relay_only)'),
    ('foreign_not_skipped', RR,
     '        store, (b for b in relay_only if b not in foreign),',
     '        store, (b for b in relay_only),'),
    ('restore_foreign_ok', RR,
     '            and not store.find_foreign_owner(bid)\n',
     '\n'),
    ('agent_inferred_negated', RR,
     'belief.origin == ORIGIN_AGENT_INFERRED',
     'belief.origin != ORIGIN_AGENT_INFERRED'),
    ('no_soft_delete', RR,
     '            store.soft_delete_belief(bid, ts)\n',
     ''),
    ('no_audit_row', RR,
     '            store.insert_feedback_event(bid, 0.0, SOURCE_RELAY_RETIRE, ts)\n',
     ''),
    ('restore_any_retired', RR,
     'belief is not None and belief.valid_to == ts',
     'belief is not None and belief.valid_to is not None'),
    ('restore_no_user_restored', RR,
     '                store.insert_feedback_event(bid, 1.0, SOURCE_USER_RESTORED, ts)\n',
     '                pass\n'),
    ('dry_run_applies', RR,
     '    report = RelayRowsReport(dry_run=dry_run)\n    if dry_run:\n        _select(store, report)\n        return report',
     '    report = RelayRowsReport(dry_run=dry_run)\n    if False:\n        return report'),
    ('split_no_lock_check', SP,
     '        if belief.lock_level == LOCK_USER:\n            locked.append(bid)\n            continue\n',
     ''),
    ('cli_swap_restore', CLI,
     'return _cmd_doctor_relay_rows(args, out, restore=True)',
     'return _cmd_doctor_relay_rows(args, out, restore=False)'),
    ('cli_not_exclusive', CLI,
     '            ("--restore-relay-rows", "restore_relay_rows"),\n',
     ''),
    ('store_sentences_any_kind', ST,
     '"WHERE source_kind = ? ORDER BY id",\n            (INGEST_SOURCE_TRANSCRIPT,),',
     '"WHERE source_kind != ? ORDER BY id",\n            ("none",),'),
    ('store_latest_is_first', ST,
     'SELECT belief_id, MAX(created_at) FROM feedback_history',
     'SELECT belief_id, MIN(created_at) FROM feedback_history'),
    ('raw_cutoff_off', RR,
     '        raw = ms is not None and ms < RAW_RECORD_CUTOFF_MS\n',
     '        raw = True\n'),
    ('cutoff_le', RR,
     'ms is not None and ms < RAW_RECORD_CUTOFF_MS',
     'ms is not None and ms <= RAW_RECORD_CUTOFF_MS'),
    ('cutoff_none_is_raw', RR,
     'ms is not None and ms < RAW_RECORD_CUTOFF_MS',
     'ms is None or ms < RAW_RECORD_CUTOFF_MS'),
    ('pin_dst_off', RR,
     '        return dst in self._head[t] and nxt is not None and src in self._head[nxt]',
     '        return nxt is not None and src in self._head[nxt]'),
    ('pin_src_off', RR,
     '        return dst in self._head[t] and nxt is not None and src in self._head[nxt]',
     '        return dst in self._head[t] and nxt is not None'),
    ('pin_head_any_row', RR,
     '                self._head[-1] = derives[i]',
     '                self._head[-1] = self._head[-1] | derives[i]'),
    ('pin_uncapped_fill_off', RR,
     '        if self._keys[t][0] is None or (not capped and i + n != self._end[t]):',
     '        if self._keys[t][0] is None:'),
    ('pin_sid_none_ok', RR,
     '        if self._keys[t][0] is None or (not capped and i + n != self._end[t]):',
     '        if not capped and i + n != self._end[t]:'),
    ('next_ignores_session', RR,
     '            sid, _, path = self._keys[t]\n',
     '            sid, _, path = self._keys[t]\n            sid = None\n'),
    ('host_case_insensitive', NF,
     '    r"<(" + _RELAY_TAG_ALT + r\')((?: [a-z0-9_-]+="[^"]+")+)>\',\n)',
     '    r"<(" + _RELAY_TAG_ALT + r\')((?: [a-z0-9_-]+="[^"]+")+)>\',\n    re.IGNORECASE,\n)'),
    ('host_value_check_off', NF,
     '    if not all(_host_value(value) for _, value in attributes):\n        return False\n',
     ''),
    ('host_value_breaks_ok', NF,
     '    return value.splitlines() == [value] and any(',
     '    return any('),
    ('host_value_visible_off', NF,
     '        c.isprintable() and not c.isspace() for c in value',
     '        True for c in value'),
    ('host_value_space_ok', NF,
     '        c.isprintable() and not c.isspace() for c in value',
     '        c.isprintable() for c in value'),
    ('host_required_attrs_off', NF,
     '    return HOST_RELAY_ATTRIBUTES[host.group(1)] <= {name for name, _ in attributes}',
     '    return True'),
    ('veto_not_applied_to_openers', RR,
     '        if rows.opens[i] and i not in vetoed:',
     '        if rows.opens[i]:'),
    ('shape_any_position', RR,
     '        if rows.opens[i] and i not in vetoed:',
     '        if i not in vetoed:'),
    ('opener_at_split_turn', RR,
     '        if rows.opens[i] and i not in vetoed:',
     '        if rows.starts[i] and i not in vetoed:'),
    ('anchor_at_split_turn', RR,
     '        if rows.opens[i]:\n            starts_by_text',
     '        if rows.starts[i]:\n            starts_by_text'),
    ('anchor_start_any_position', RR,
     '        if rows.opens[i]:\n            starts_by_text',
     '        if True:\n            starts_by_text'),
    ('opens_ignores_repeat', RR,
     '        opens_turn = first_text.setdefault(key, held[0][1]) == held[0][1]\n',
     '        opens_turn = True\n'),
    ('unread_stretches_read', RR,
     '        if not all_raw:\n            continue\n',
     ''),
    ('late_rows_counted_raw', RR,
     '        stretches[-1] = (k, held, all_raw and raw)',
     '        stretches[-1] = (k, held, all_raw)'),
    ('ambiguity_off', RR,
     '        if len(labels) > 1 or len(bodies) > 1:\n            report.anchors_ambiguous += 1\n            continue\n',
     ''),
    ('labels_check_off', RR,
     '        if len(labels) > 1 or len(bodies) > 1:',
     '        if len(bodies) > 1:'),
    ('bodies_check_off', RR,
     '        if len(labels) > 1 or len(bodies) > 1:',
     '        if len(labels) > 1:'),
    ('takes_all_runs', RR,
     '        takes = [r for r in runs if r[3]]',
     '        takes = runs'),
    ('copies_not_taken', RR,
     '        takes = [r for r in runs if rows.keys[r[0]][:2] in records]\n',
     ''),
    ('veto_same_session_as_first', RR,
     'for i, n in _runs(rows, starts_by_text, whole, capped):\n                        vetoed.update(range(i, i + n))',
     'for i, n in [r for r in _runs(rows, starts_by_text, whole, capped) if rows.keys[r[0]][0] == rows.keys[_runs(rows, starts_by_text, whole, capped)[0][0]][0]]:\n                        vetoed.update(range(i, i + n))'),
]

# Mutants dropped from the set, with the reason. A test can't tell them from the code.
RETIRED: dict[str, str] = {
    "no_seen_dedupe": "equivalent: a second sentence split equal to the first repeats the "
                      "same veto and take calls, which change nothing the second time",
    "key_not_checked_in_run": "equivalent: a run that leaves its stretch fits an edge only when "
                              "the anchor is capped, and then the stretch's last row, inside the "
                              "run, has to derive the edge's earlier end, so the record that "
                              "wrote the edge would end on a sentence it held before the cap",
}


def _check_anchors(root: Path) -> list[str]:
    """Return a line for each mutant whose snippet isn't in its file exactly once."""
    bad: list[str] = []
    for name, path, old, _new in MUTANTS:
        count = (root / path).read_text(encoding="utf-8").count(old)
        if count != 1:
            bad.append(f"NOTFOUND {name} {count}")
    return bad


def _copy_tree(into: Path) -> None:
    for name in COPIED:
        if (ROOT / name).exists():
            shutil.copytree(ROOT / name, into / name, ignore=shutil.ignore_patterns("__pycache__"))
    for name in COPIED_FILES:
        if (ROOT / name).exists():
            shutil.copy(ROOT / name, into / name)


def _run_one(root: Path, name: str, path: str, old: str, new: str) -> bool:
    """Apply one mutant in `root`, run the tests, restore the file. True when killed."""
    target = root / path
    original = target.read_text(encoding="utf-8")
    target.write_text(original.replace(old, new), encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONDONTWRITEBYTECODE="1")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", *TESTS, "-x", "-q", "--no-header",
             "-p", "no:cacheprovider", "--tb=no", "--timeout=240"],
            cwd=root, env=env, capture_output=True, text=True, timeout=TEST_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        print(f"killed   {name} by timeout", flush=True)
        return True
    finally:
        target.write_text(original, encoding="utf-8")
    if result.returncode == 0:
        print(f"SURVIVED {name}", flush=True)
        return False
    failed = [line for line in result.stdout.splitlines() if line.startswith(("FAILED", "ERROR"))]
    first = failed[0].split("::")[-1].split(" - ")[0][:70] if failed else f"rc={result.returncode}"
    print(f"killed   {name} by {first}", flush=True)
    return True


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Mutation check for aelf doctor --retire-relay-rows (#1877).")
    parser.add_argument("names", nargs="*", help="run only these mutants")
    parser.add_argument("--dry-run", action="store_true", help="check every anchor and run nothing")
    parser.add_argument("--list", action="store_true", help="print each mutant and retired name")
    parser.add_argument("--shard", default="0/1", help="I/N: run mutants whose index mod N is I")
    args = parser.parse_args(argv)
    names = {m[0] for m in MUTANTS}
    if len(names) != len(MUTANTS):
        print("duplicate mutant names", file=sys.stderr)
        return 1
    unknown = sorted(set(args.names) - names)
    if unknown:
        print("unknown mutants: " + " ".join(unknown), file=sys.stderr)
        return 1
    if args.list:
        for name, path, _old, _new in MUTANTS:
            print(f"{name}\t{path}")
        for name, why in sorted(RETIRED.items()):
            print(f"{name}\tretired: {why}")
        return 0
    bad = _check_anchors(ROOT)
    for line in bad:
        print(line)
    if args.dry_run:
        print(f"checked {len(MUTANTS)} anchors, {len(bad)} not found exactly once")
        return 1 if bad else 0
    shard, shards = (int(part) for part in args.shard.split("/"))
    survivors = 0
    killed = 0
    with tempfile.TemporaryDirectory(prefix="mutants-1877-") as folder:
        root = Path(folder)
        _copy_tree(root)
        for number, (name, path, old, new) in enumerate(MUTANTS):
            if (args.names and name not in args.names) or number % shards != shard:
                continue
            if (root / path).read_text(encoding="utf-8").count(old) != 1:
                continue
            if _run_one(root, name, path, old, new):
                killed += 1
            else:
                survivors += 1
    print(f"killed {killed}, survived {survivors}, anchors not found {len(bad)}")
    return 1 if survivors or bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
