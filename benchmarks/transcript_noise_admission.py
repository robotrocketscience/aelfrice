"""#1371 §1 — what the transcript noise filter admits, before and after.

`is_transcript_noise` decides what never becomes a belief. #1159 §1 reported
that it discards the product's own policy statements; this harness measures
that against a real corpus rather than against constructed examples, and it
reports the two arms separately because they turned out to behave very
differently.

## Population

This repo's archived transcripts (`<git-common-dir>/aelfrice/transcripts/`),
`role == "user"` turns only, split with the same `extraction.extract_sentences`
the ingest path uses. Read-only: it opens no store at all, so it cannot
perturb what it measures.

## The two arms

* **Ack** — a measured defect. The pattern allowed 40 characters of arbitrary
  content after the keyword, so `"No telemetry, no network calls, no
  accounts."` matched `No` + a short tail and was discarded.
* **Shell** — hardening only. The prefixes are matched by bare `startswith`,
  so `"pytest is the only test runner we support."` is discarded in principle;
  on this corpus the fix rescues nothing. Read that as a limit of the rule
  rather than as a clean population: at least two still-discarded rows are
  prose about pytest, not pasted commands, and they stay discarded only
  because they carry no terminal full stop. Reported separately rather than
  folded into the ack number, because a rescue count of 0 is the honest
  figure for this arm.

Usage:

    uv run python benchmarks/transcript_noise_admission.py [--json out.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any

from aelfrice.extraction import extract_sentences
from aelfrice.noise_filter import is_transcript_noise, is_transcript_scaffolding

# The pre-#1371 predicate, kept verbatim so the "before" column is a real
# measurement rather than a remembered one. Copied deliberately: importing
# the shipped one would make both columns identical and the harness useless.
_OLD_SHELL_PREFIXES = ("cd /", "git ", "gh ", "uv run", "pytest", "python ")
_OLD_XML_PREFIXES = (
    "<worktree", "<output-file", "<task-", "<summary>Background",
    "<summary>Monitor", "<tool-use-id", "<usage", "<event", "<total_tokens",
    "<system-reminder", "<command-name", "<command-message", "<command-args",
    "<local-command-stdout", "<local-command-caveat",
)
_OLD_BOXDRAW = frozenset("┌┐└┘├┤┬┴┼─│╭╮╰╯╞╡═")
_OLD_PROGRESS = re.compile(r"^[A-Z][a-z]+ing\.$")
_OLD_ACK = re.compile(r"^(Yes|No|Standing by|Ready|Nothing|Polling)( .{0,40})?\.?$")


def _old_is_transcript_noise(s: str) -> bool:
    if not s or not s.strip():
        return False
    if any(s.startswith(p) for p in _OLD_SHELL_PREFIXES):
        return True
    if s.startswith("⏺"):
        return True
    if any(s.startswith(p) for p in _OLD_XML_PREFIXES):
        return True
    st = s.lstrip()
    if st.startswith("</"):
        return True
    if st and st[0] in _OLD_BOXDRAW:
        return True
    if _OLD_PROGRESS.match(s) is not None:
        return True
    return _OLD_ACK.match(s) is not None


def _transcript_dir() -> str:
    """Resolve the transcript directory the logger actually writes to.

    Via `transcript_logger.transcripts_dir()` rather than a hard-coded
    path, so this runs against whatever checkout it is invoked from and
    honours the git-common-dir resolution the logger uses in a worktree.
    """
    from aelfrice.transcript_logger import transcripts_dir

    return str(transcripts_dir())


def _load(tdir: str) -> tuple[list[str], list[str]]:
    prompts: list[str] = []
    for path in sorted(glob.glob(f"{tdir}/archive/*.jsonl")) + [f"{tdir}/turns.jsonl"]:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("role") == "user" and (row.get("text") or "").strip():
                    prompts.append(row["text"])
    sentences = [
        s.strip() for p in prompts for s in extract_sentences(p) if s.strip()
    ]
    return prompts, sentences


def measure(tdir: str) -> dict[str, Any]:
    prompts, sentences = _load(tdir)
    old = [s for s in sentences if _old_is_transcript_noise(s)]
    new = [s for s in sentences if is_transcript_noise(s)]
    rescued = [s for s in old if not is_transcript_noise(s)]
    newly = [s for s in new if not _old_is_transcript_noise(s)]

    shell_hits = [s for s in sentences if any(s.startswith(p) for p in _OLD_SHELL_PREFIXES)]
    shell_rescued = [s for s in shell_hits if not is_transcript_noise(s)]
    ack_hits = [s for s in sentences if _OLD_ACK.match(s)]
    ack_rescued = [s for s in ack_hits if not is_transcript_noise(s)]

    def logger_drops_new(p: str) -> bool:
        if is_transcript_scaffolding(p):
            return True
        if not is_transcript_noise(p):
            return False
        parts = [s for s in extract_sentences(p) if s.strip()]
        return all(is_transcript_noise(s) for s in parts)

    # Row counts and distinct counts are different claims and the row count
    # is the misleading one: three sentences account for most of the rescue
    # at four occurrences each, so "20 rescued" reads as 20 content items and
    # is 10. Both are reported; neither is derivable from the other.
    return {
        "sentences": len(sentences),
        "prompts": len(prompts),
        "sentence_discards_before": len(old),
        "sentence_discards_after": len(new),
        "admission_rate_before": round(1 - len(old) / max(1, len(sentences)), 6),
        "admission_rate_after": round(1 - len(new) / max(1, len(sentences)), 6),
        "rescued_total": len(rescued),
        "rescued_distinct": len(set(rescued)),
        "newly_discarded": len(newly),
        "newly_discarded_distinct": len(set(newly)),
        "newly_discarded_examples": sorted({s[:120] for s in newly})[:10],
        "ack_arm": {
            "matched_old_pattern": len(ack_hits),
            "matched_old_pattern_distinct": len(set(ack_hits)),
            "rescued": len(ack_rescued),
            "rescued_distinct": len(set(ack_rescued)),
            "examples": sorted({s[:120] for s in ack_rescued})[:15],
        },
        "shell_arm": {
            "prefixed_sentences": len(shell_hits),
            "prefixed_sentences_distinct": len(set(shell_hits)),
            "rescued": len(shell_rescued),
            "rescued_distinct": len(set(shell_rescued)),
            "examples": sorted({s[:120] for s in shell_rescued})[:10],
        },
        "whole_prompt_drops_before": sum(
            1 for p in prompts if _old_is_transcript_noise(p)
        ),
        "whole_prompt_drops_after": sum(1 for p in prompts if logger_drops_new(p)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--transcripts", default=_transcript_dir())
    ap.add_argument("--json", dest="json_out")
    args = ap.parse_args(argv)
    if not os.path.isdir(args.transcripts):
        print(f"no transcript dir at {args.transcripts}", file=sys.stderr)
        return 1
    report = measure(args.transcripts)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
