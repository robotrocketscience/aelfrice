#!/usr/bin/env bash
# Scan files against a pattern file, one extended regex per line.
#
# Both pattern jobs in .github/workflows/staging-gate.yml call this, with
# the pattern file decoded from the SCAN_RULES_B64 secret. It exists so the
# loop can be tested: the inline loops it replaced read grep's exit status
# 2 (a pattern that does not compile) as "no hit", so one malformed rule
# switched itself off and the job still passed.
#
# Usage: scripts/scan_patterns.sh [--dry-run] --mode tree|history PATTERNS FILE...
#   --mode tree     report each hit with file and line (grep -HnI: binary
#                   files are skipped), error text "pattern hit".
#   --mode history  report only that a hit exists, never the matched text;
#                   binary-looking input is still searched.
#   --dry-run       check the pattern file and every pattern, print the
#                   counts, and scan nothing. Still exits 2 on a bad file.
#
# Blank lines and lines starting with '#' are skipped. A pattern is named
# in messages by its line number N in PATTERNS, never by its text: the
# patterns are secret, so nothing here prints one.
#
# Exit status: 0 no hit, 1 at least one hit, 2 the scan could not be
# trusted (usage error, missing or empty pattern file, no patterns, a
# pattern that does not compile, or grep failing to read an input).
set -euo pipefail

usage() {
    sed -n '/^# Usage:/,/^# Exit status/p' "$0" | sed '$d; s/^# \{0,1\}//' >&2
    exit 2
}

dry_run=0
mode=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        --mode) [ $# -ge 2 ] || usage; mode="$2"; shift 2 ;;
        --) shift; break ;;
        -*) usage ;;
        *) break ;;
    esac
done

case "$mode" in
    tree) grep_flags="-EHnI" ;;
    history) grep_flags="-El" ;;
    *) usage ;;
esac

[ $# -ge 2 ] || usage
patterns="$1"
shift

if [ ! -s "$patterns" ]; then
    echo "::error::pattern file is missing or empty"
    exit 2
fi

hit=0
broken=0
count=0
lineno=0
# `|| [ -n "$p" ]` keeps a last line that has no trailing newline, which
# a plain `read` loop drops. Reading on fd 3 keeps grep off the pattern
# file's stdin.
while IFS= read -r p <&3 || [ -n "$p" ]; do
    lineno=$((lineno + 1))
    [ -z "$p" ] && continue
    case "$p" in '#'*) continue ;; esac
    count=$((count + 1))

    # Compile check against empty input: 1 means the pattern compiled and
    # matched nothing, 2 means it did not compile. Kept apart from the
    # scan so a bad pattern and an unreadable input get different errors.
    rc=0
    grep -E -e "$p" -- /dev/null >/dev/null 2>&1 || rc=$?
    if [ "$rc" -ge 2 ]; then
        echo "::error::pattern $lineno did not compile"
        broken=1
        continue
    fi
    [ "$dry_run" -eq 1 ] && continue

    # grep's stderr is dropped: some implementations quote part of the
    # pattern in their warnings.
    rc=0
    matches=$(grep "$grep_flags" -e "$p" -- "$@" 2>/dev/null) || rc=$?
    if [ "$rc" -ge 2 ]; then
        echo "::error::pattern $lineno: grep could not read the input (exit $rc)"
        broken=1
    elif [ "$rc" -eq 0 ]; then
        hit=1
        if [ "$mode" = "tree" ]; then
            echo "::error::pattern hit"
            echo "$matches"
        else
            echo "::error::pattern hit in history"
        fi
    fi
done 3< "$patterns"

if [ "$count" -eq 0 ]; then
    echo "::error::pattern file holds no patterns"
    exit 2
fi
if [ "$dry_run" -eq 1 ]; then
    echo "dry run: $count patterns, $# input files, nothing scanned"
fi
if [ "$broken" -ne 0 ]; then
    exit 2
fi
exit "$hit"
