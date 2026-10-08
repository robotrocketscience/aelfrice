"""The LLM onboard classifier's request text, as a leaf module (#1658).

`aelfrice.llm_classifier` sends this system prompt and builds each user
message with `build_user_message`, and re-exports both. They live here
so `classification_core.rule_set_hash()` can hash them on the ingest
path without importing `llm_classifier`, whose import costs about 17 ms
in a hook process that would otherwise never load it.

Imports here must stay leaf: the standard library only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Final

SYSTEM_PROMPT: Final[str] = """\
You are classifying short text candidates extracted from a software
project's documentation, git history, and Python docstrings. Each
candidate becomes a unit of memory in a Bayesian belief store.

For each candidate, return a JSON object with three fields:
  belief_type: one of "factual", "correction", "preference",
               "requirement"
  origin:      one of "document_recent", "agent_inferred"
  persist:     true if the candidate should become a stored belief,
               false if it should be dropped (questions, headings,
               table-of-contents lines, navigational text,
               meta-commentary, anything ephemeral).

Definitions:
  factual      A statement of fact, decision, or analysis. Default.
  correction   A statement that overrides or corrects a previous
               claim ("not X but Y", "actually Z", "the earlier
               version was wrong because ...").
  preference   A stated preference, taste, or convention ("we prefer
               composition", "always use uv", "I want explicit
               types").
  requirement  A hard rule, constraint, must-do, or invariant ("CI
               must be green", "no global state", "Python 3.12+").

  document_recent   The candidate reads as committed prose from the
                    project's own documentation or commit history --
                    something a human wrote down deliberately.
                    Default for paragraphs from .md/.rst files and
                    for git commit subjects.
  agent_inferred    The candidate reads as machine-extracted or
                    incidental -- a docstring fragment, a templated
                    line, anything where the underlying assertion
                    was not necessarily reviewed by a human.

Return one JSON object per candidate, in input order, as a JSON
array. No prose before or after the array. No markdown fences.
"""


@dataclass
class CandidateInput:
    """One candidate to send to the classifier.

    Mirrors `scanner.SentenceCandidate` but with a stable `index`
    that lets the response array match results back to inputs.
    """

    index: int
    text: str
    source: str


def build_user_message(candidates: list[CandidateInput]) -> str:
    """Return the user-message body for one classifier request.

    The body is a JSON array; the model returns a JSON array of the
    same length. Spec § 5 locks the response to JSON only.
    """
    payload = [
        {"index": c.index, "source": c.source, "text": c.text}
        for c in candidates
    ]
    return json.dumps(payload)
