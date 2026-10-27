"""The 0 to 3 relevance scale of the #1804 calibration, and its score reader.

#1804 has graders score how relevant a note from a project's memory is
to a prompt, from 0 to 3 or null, with the instructions below. The
calibration harness, `grader_calibration_1804`, and the real-store A/B
harness, `real_store_ab_1795` (#1809), hand the graders these
instructions and read their score files here.

A score file is JSONL, one `{"id": str, "score": 0|1|2|3|null}` object
per line. Each grader's file is its own and scores exactly the key's
rows; `require_distinct_files` and `require_scores` check both.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, cast

SCORES: Final[tuple[int, ...]] = (0, 1, 2, 3)

SCALE: Final[tuple[tuple[str, str], ...]] = (
    ("3", ("Needed: without the note, the answer would be wrong or would miss a rule,"
           " a preference, or an earlier decision about this exact task.")),
    ("2", ("Would help: reading the note would change or directly inform what the"
           " assistant does for this prompt.")),
    ("1", ("Same topic only: the note is about the same project, file, tool, or topic,"
           " but wouldn't change the answer.")),
    ("0", "Unrelated, or it only shares words with the prompt."),
    ("null", "You can't tell what the prompt asks or what the note means."),
)
"""The pre-registration's scale table, one `(score, meaning)` row per score."""

INSTRUCTIONS: Final[str] = (
    "Each row of packet.jsonl pairs a prompt a user typed to a coding assistant\n"
    "with one note from that project's memory. Score each row from 0 to 3, or\n"
    "null:\n\n"
    + "".join(f"  {score:<5} {meaning}\n" for score, meaning in SCALE)
    + "\nWrite one JSON object per line to your own labels file:\n"
    '  {"id": "<row id>", "score": 2}\n'
    "The score is 0, 1, 2, 3, or null. Score every row. Rows come in no\n"
    "particular order. Don't discuss the rows with the other graders.\n"
)

Scores = Mapping[str, int | None]


class InputError(ValueError):
    """Bad input: a malformed file, a missing field, or a bad argument."""


def parse_scores(path: Path, data: bytes) -> dict[str, int | None]:
    """{id: 0 to 3 or None} from `data`, JSONL of {"id", "score"} read from `path`.

    InputError when malformed. A row with a `correct` field is refused,
    with or without a score, and the error says so: the field holds a
    binary label of the rubric the scale replaced, and a row that holds
    both leaves unclear which label the grader meant.
    """
    scores: dict[str, int | None] = {}
    for n, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        obj: object = json.loads(line)
        if not isinstance(obj, dict):
            raise InputError(f"{path}:{n}: not a JSON object")
        row = cast(dict[str, object], obj)
        if "correct" in row:
            raise InputError(
                f"{path}:{n}: a binary \"correct\" label; score each row 0, 1, 2, 3,"
                " or null with the 0 to 3 instructions")
        rid, score = row.get("id"), row.get("score")
        # `True == 1` in Python, so a boolean is refused by type, not by value.
        valid = score is None or (type(score) is int and score in SCORES)
        if not isinstance(rid, str) or "score" not in row or not valid:
            raise InputError(f"{path}:{n}: need an id and a score of 0, 1, 2, 3, or null")
        if rid in scores:
            raise InputError(f"{path}:{n}: repeats row {rid}")
        scores[rid] = cast(int | None, score)
    return scores


def binarize(scores: Scores, cut: int) -> dict[str, bool | None]:
    """{id: score at or above `cut`}, a null score staying None."""
    return {i: None if s is None else s >= cut for i, s in scores.items()}


def require_distinct_files(flag: str, paths: Sequence[Path]) -> None:
    """InputError when two of `paths` are one file, by any spelling or link.

    One grader's file given for two graders would agree with itself on
    every row.
    """
    for n, p in enumerate(paths):
        for q in paths[n + 1:]:
            if p.samefile(q):
                raise InputError(f"{flag} {p} and {q} are one file; each grader needs its own")


def require_scores(ids: Sequence[str], scores: Mapping[str, Scores]) -> None:
    """InputError unless each grader scored exactly the key's ids."""
    wanted = set(ids)
    for grader, s in scores.items():
        missing = sorted(wanted - s.keys())
        if missing:
            raise InputError(f"grader {grader} has no score for {len(missing)} key rows,"
                             f" {missing[0]} among them")
        extra = sorted(s.keys() - wanted)
        if extra:
            raise InputError(f"grader {grader} scored {len(extra)} rows the key lacks,"
                             f" {extra[0]} among them")
