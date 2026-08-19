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

**The corpus grows every time a session archives, so a headline count with no
corpus identity beside it is unreproducible** — the failure #1398 was filed
over. `corpus_identity` is therefore printed on every run, not offered as a
flag, and it carries two things because neither pins the input on its own:
the checkout (`sha`, `dirty`), which fixes the *code*, and a digest over the
transcript files themselves, which fixes the *data*. The transcripts are
untracked, so the sha says nothing about them; `turns.jsonl` is the live file
the running session is still appending to, so two runs minutes apart can
legitimately differ. Quote `corpus_sha256` with any figure taken from here.

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
import hashlib
import json
import os
import re
import subprocess
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


def _corpus_paths(tdir: str) -> list[str]:
    """The files `_load` reads, in read order.

    Shared with `_corpus_identity` on purpose: a digest computed over a
    different file list than the one measured would certify the wrong
    corpus, which is worse than printing no digest at all.
    """
    paths = sorted(glob.glob(f"{tdir}/archive/*.jsonl")) + [f"{tdir}/turns.jsonl"]
    return [p for p in paths if os.path.exists(p)]


def _git(root: str, *args: str) -> str:
    """`git -C root ...`, or the empty string if git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "-C", root, *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _corpus_identity(tdir: str, root: str = ".") -> dict[str, Any]:
    """Everything needed to reproduce this run's inputs (#1398).

    `sha`/`dirty` identify the *code* — the shape of `is_transcript_noise`
    and of the pre-fix replica above. They do not identify the *data*:
    the transcript archive is untracked and `turns.jsonl` is appended to
    while this runs, so the digest below is the half that matters for the
    corpus counts. Both are printed because a figure needs both to be
    reproducible, and neither implies the other.
    """
    digest = hashlib.sha256()
    files: list[dict[str, Any]] = []
    for path in _corpus_paths(tdir):
        with open(path, "rb") as fh:
            data = fh.read()
        digest.update(os.path.basename(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        files.append({"name": os.path.basename(path), "bytes": len(data)})
    dirty = _git(root, "status", "--porcelain")
    return {
        "transcripts": os.path.abspath(tdir),
        "corpus_files": len(files),
        "corpus_bytes": sum(f["bytes"] for f in files),
        "corpus_sha256": digest.hexdigest(),
        "sha": _git(root, "rev-parse", "HEAD") or "unknown",
        "dirty": bool(dirty),
        "dirty_paths": len(dirty.splitlines()) if dirty else 0,
    }


def _load(tdir: str) -> tuple[list[str], list[str]]:
    prompts: list[str] = []
    for path in _corpus_paths(tdir):
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


# --------------------------------------------------------------------------
# #1490 — what KIND of sentence the filter rescues
# --------------------------------------------------------------------------
#
# `rescued_distinct` says how many sentences the relaxation admits. It does
# not say whether admitting them is good, and the three kinds behave very
# differently once stored: a durable policy statement earns its slot, an
# ephemeral session-status line ("No failures yet.") is true of one run and
# of nothing after it, yet both enter at the same undeflated
# `user_transcript` prior of 0.75 and compete for the same injection slots.
#
# First match wins, so reordering changes the split without changing the
# total — the same property `scan_admission_funnel.NOISE_BUCKETS` documents.
#
# These are REPORTING heuristics. Nothing here reaches the ingest write path,
# and nothing here decides what is stored; a wrong bucket produces a wrong
# number in a manually-run report, never a dropped belief. The shape detector
# that WOULD have touched ingest was rejected twice (2026-08-11, 2026-08-12)
# as "the same widening #1371 §1 just narrowed".
#
# `unclassified` is the point of the exercise, not a leftover. CI can never
# watch this number — the corpus is an untracked local archive — so a growing
# `unclassified` count is the only signal that the closed set of three has
# stopped describing what the filter actually admits.
RESCUE_BUCKETS = (
    "ephemeral_status",
    "operator_directive",
    "durable_policy",
    "unclassified",
)

# "true when written, false later": an explicit temporal hedge, or a report of
# what has happened so far.
_EPHEMERAL_MARKERS = (
    " yet",
    "so far",
    "to date",
    "as of now",
    "currently",
    "at this point",
)
_EPHEMERAL_REPORT = re.compile(
    r"\b(attempted|tried|observed|encountered|reproduced|detected|"
    r"triggered|fired|reported|seen)\b",
    re.IGNORECASE,
)

# A statement about the product's standing behaviour or configuration.
_POLICY_TOKENS = re.compile(
    r"\b(aelf|aelfrice|telemetry|network calls?|accounts?|by default|"
    r"default|config|configuration|setup|install)\b",
    re.IGNORECASE,
)

# A terse instruction or prohibition addressed to whoever is working.
_DIRECTIVE = re.compile(
    r"^(no|do not|don't|never|always|avoid|use|prefer)\b", re.IGNORECASE
)


def _rescue_bucket(sentence: str) -> str:
    """Which `RESCUE_BUCKETS` member `sentence` falls in. Never returns None.

    Total by construction: the final branch is `unclassified`, so a sentence
    matching no rule is counted rather than dropped. A silent drop here would
    make the buckets sum to less than `rescued_distinct` and hide exactly the
    novel shape this split exists to surface.
    """
    lowered = sentence.lower()
    if any(m in lowered for m in _EPHEMERAL_MARKERS):
        return "ephemeral_status"
    if _EPHEMERAL_REPORT.search(sentence):
        return "ephemeral_status"
    if _POLICY_TOKENS.search(sentence):
        return "durable_policy"
    if _DIRECTIVE.match(sentence.strip()):
        return "operator_directive"
    return "unclassified"


def _bucket_counts(sentences: list[str]) -> dict[str, int]:
    """Every declared bucket present, so a zero column reads as measured."""
    counts = {b: 0 for b in RESCUE_BUCKETS}
    for s in sentences:
        counts[_rescue_bucket(s)] += 1
    return counts


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
        "corpus_identity": _corpus_identity(tdir),
        "sentences": len(sentences),
        "prompts": len(prompts),
        "sentence_discards_before": len(old),
        "sentence_discards_after": len(new),
        "admission_rate_before": round(1 - len(old) / max(1, len(sentences)), 6),
        "admission_rate_after": round(1 - len(new) / max(1, len(sentences)), 6),
        "rescued_total": len(rescued),
        "rescued_distinct": len(set(rescued)),
        # #1490: the buckets partition each set exactly — see
        # test_transcript_noise_buckets_1490.
        "rescue_buckets_distinct": _bucket_counts(sorted(set(rescued))),
        "rescue_buckets_rows": _bucket_counts(rescued),
        "rescue_bucket_examples": {
            b: sorted({s[:120] for s in set(rescued) if _rescue_bucket(s) == b})[:5]
            for b in RESCUE_BUCKETS
        },
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
