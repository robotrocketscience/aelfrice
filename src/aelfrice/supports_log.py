"""Log reworded restatements as SUPPORTS evidence without applying them (#1650).

After transcript ingest commits a turn the user typed, this pass checks
each sentence the turn inserted against the beliefs that sentence might
restate in other words, and records each match in
``belief_support_evidence`` with ``applied`` set to 0. It writes no
``SUPPORTS`` edge, moves no posterior, and calls no ``apply_feedback``.
A later change applies the logged rows, to phantoms only, after the
matcher passes its precision gate (``docs/design/feature-supports-writer.md``).

A supporter counts only when it is provably the user's: its
``ingest_log`` row is a ``transcript`` ingest whose ``raw_meta`` records
``"role": "user"`` and whose first derived belief is the supporter. A
supporter with no session is skipped, because nothing shows a target
came from another session. A target is any active belief whose origin
is not in ``USER_ORIGINS``, that holds every candidate term of the user
sentence (``reworded_restatement.candidate_terms``), that was not
created in the supporter's session (restating yourself in one
conversation is emphasis, not evidence), and that was created before the
supporter. Ingest stamps the supporter with the turn's own timestamp, so
a backfill of an old transcript can't log a sentence the user typed
before the target existed. At most
``REWORDED_CANDIDATE_TOP_K`` targets are matched per sentence.

Candidates come from the store's full-text index, which holds each
belief as stored. The matcher reads text normalized, so a target whose
words are in fullwidth letters or split by a format character is never a
candidate, although the matcher alone would accept it. The log
undercounts on such text; it does not log a wrong match.

The pass fails open. A read-only store is skipped, and any exception is
reported on stderr and swallowed, so it never fails or rolls back the
ingest that called it.

The pass is off by default. It runs when:

  1. the ``AELFRICE_LOG_REWORDED_SUPPORT`` env var is truthy, or
  2. the env var is unset or unrecognized, and ``[ingest]
     log_reworded_support`` in ``.aelfrice.toml`` is ``true``.
"""
from __future__ import annotations

import os
import sys
import tomllib
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Final, cast

from aelfrice.config_discovery import discover_config
from aelfrice.models import (
    INGEST_SOURCE_TRANSCRIPT,
    ORIGIN_USER_CORRECTED,
    ORIGIN_USER_STATED,
    ORIGIN_USER_TRANSCRIPT,
    ORIGIN_USER_VALIDATED,
    SUPPORT_EVIDENCE_KIND_REWORDED,
)
from aelfrice.store import MemoryStore

if TYPE_CHECKING:
    from aelfrice.reworded_restatement import RewordedMatch

ENV_LOG_REWORDED_SUPPORT: Final[str] = "AELFRICE_LOG_REWORDED_SUPPORT"
SECTION: Final[str] = "ingest"
LOG_REWORDED_SUPPORT_KEY: Final[str] = "log_reworded_support"

# --- Pinned constants (see detector_thresholds.py) -----------------------

# The most candidate targets matched against one user sentence. The
# candidates already hold every term of the sentence, so the cut binds
# only on a store with many near-copies of one belief, and it bounds the
# matcher's cost per sentence however large the store grows.
REWORDED_CANDIDATE_TOP_K: Final[int] = 20

# Origins that mark a belief as the user's own words. A belief with one
# of these is never a target: support is evidence for a belief the user
# did not write.
USER_ORIGINS: Final[frozenset[str]] = frozenset({
    ORIGIN_USER_STATED,
    ORIGIN_USER_CORRECTED,
    ORIGIN_USER_VALIDATED,
    ORIGIN_USER_TRANSCRIPT,
})

_USER_ROLE: Final[str] = "user"
_ENV_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
_ENV_FALSY: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})


def _env_override(env: Mapping[str, str]) -> bool | None:
    raw = env.get(ENV_LOG_REWORDED_SUPPORT)
    if raw is None:
        return None
    norm = raw.strip().lower()
    if norm in _ENV_TRUTHY:
        return True
    if norm in _ENV_FALSY:
        return False
    return None


def _read_toml(start: Path | None) -> bool | None:
    """`[ingest] log_reworded_support` from the nearest `.aelfrice.toml`.

    None on a missing file, section, or key, on an unreadable or
    unparsable file, and on a non-bool value. Never raises: ingest runs
    inside hooks.
    """
    serr: IO[str] = sys.stderr
    try:
        candidate = discover_config(start)
    except (OSError, ValueError) as exc:  # ValueError: a NUL in the path
        print(f"aelfrice supports_log: cannot look for .aelfrice.toml: {exc}",
              file=serr)
        return None
    if candidate is None:
        return None
    try:
        parsed: dict[str, Any] = tomllib.loads(
            candidate.read_bytes().decode("utf-8", errors="replace"),
        )
    except (OSError, tomllib.TOMLDecodeError, RecursionError) as exc:
        print(
            f"aelfrice supports_log: cannot read {LOG_REWORDED_SUPPORT_KEY} "
            f"in {candidate}: {exc}",
            file=serr,
        )
        return None
    section: Any = parsed.get(SECTION, {})
    if not isinstance(section, dict):
        return None
    val: Any = cast("dict[str, Any]", section).get(LOG_REWORDED_SUPPORT_KEY)
    return val if isinstance(val, bool) else None


def is_reworded_support_log_enabled(
    *,
    start: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Whether ingest logs reworded restatements. Default False.

    The environment variable wins over the TOML key. `env` defaults to
    `os.environ`; pass a mapping to keep a caller off the ambient
    environment.
    """
    override = _env_override(env if env is not None else os.environ)
    if override is not None:
        return override
    toml_value = _read_toml(start)
    if toml_value is not None:
        return toml_value
    return False


def _user_typed(entry: Mapping[str, object] | None, belief_id: str) -> bool:
    """True when `entry` is a transcript row the user typed that made `belief_id`."""
    if entry is None or entry.get("source_kind") != INGEST_SOURCE_TRANSCRIPT:
        return False
    meta = entry.get("raw_meta")
    if not isinstance(meta, dict):
        return False
    if cast("dict[str, object]", meta).get("role") != _USER_ROLE:
        return False
    ids = entry.get("derived_belief_ids")
    if not isinstance(ids, list) or not ids:
        return False
    return cast("list[object]", ids)[0] == belief_id


@dataclass(frozen=True)
class SupportCandidate:
    """One user sentence and one belief it may restate, with the matcher's answer.

    `match` is None when `match_reworded` rejects the pair.
    """

    supporter_id: str
    supporter_text: str
    ingest_log_id: str
    session_id: str
    target_id: str
    target_text: str
    match: RewordedMatch | None


def iter_reworded_candidates(
    store: MemoryStore, supporters: Iterable[tuple[str, str]],
) -> Iterator[SupportCandidate]:
    """Each candidate target of each eligible supporter, matched or not.

    Applies every rule in the module docstring, so a candidate yielded
    with a match is exactly a row `log_reworded_support` records. Reads
    only, so it runs on a read-only store.
    """
    # Imported here, not at module scope: the matcher pulls in the BM25
    # stack (numpy, scipy), about 130 ms, and ingest imports this module
    # to read the flag on every run, including with the flag off.
    from aelfrice.reworded_restatement import (  # noqa: PLC0415
        candidate_terms,
        match_reworded,
    )

    for belief_id, log_id in supporters:
        entry = store.get_ingest_log_entry(log_id)
        if not _user_typed(entry, belief_id):
            continue
        session = cast("Mapping[str, object]", entry).get("session_id")
        if not isinstance(session, str) or not session:
            continue
        supporter = store.get_belief(belief_id)
        if supporter is None:
            continue
        terms = candidate_terms(supporter.content)
        if not terms:
            continue
        candidates = store.support_candidates(
            terms, limit=REWORDED_CANDIDATE_TOP_K,
            supporter_id=belief_id, supporter_session_id=session,
            supporter_created_at=supporter.created_at,
            exclude_origins=USER_ORIGINS,
        )
        for target_id, content in candidates:
            yield SupportCandidate(
                supporter_id=belief_id, supporter_text=supporter.content,
                ingest_log_id=log_id, session_id=session,
                target_id=target_id, target_text=content,
                match=match_reworded(supporter.content, content),
            )


def _log(store: MemoryStore, supporters: Iterable[tuple[str, str]]) -> int:
    recorded = 0
    with store.transaction():
        for cand in iter_reworded_candidates(store, supporters):
            if cand.match is None:
                continue
            if store.record_support_evidence(
                cand.supporter_id, cand.target_id,
                kind=SUPPORT_EVIDENCE_KIND_REWORDED,
                target_sentence_index=cand.match.target_sentence_index,
                supporter_session_id=cand.session_id,
                supporter_ingest_log_id=cand.ingest_log_id,
                matched_span=cand.match.target_sentence,
                applied=False,
            ):
                recorded += 1
    return recorded


def log_reworded_support(
    store: MemoryStore, supporters: Iterable[tuple[str, str]],
) -> int:
    """Log each reworded restatement by `supporters`; return the rows added.

    `supporters` holds `(belief_id, ingest_log_id)` pairs: the beliefs a
    turn inserted and the log rows that produced them. Each row records
    the matched target sentence's index and text, with `applied` 0. A
    pair already logged is not counted again.

    Fails open: returns 0 on a read-only store, and on any exception
    after printing one line to stderr. The rows of a pass that raised
    are rolled back with it unless the caller holds an open transaction.
    """
    if store.read_only:
        return 0
    try:
        return _log(store, supporters)
    except Exception as exc:  # noqa: BLE001 - must never fail ingest
        print(
            f"aelfrice supports_log: skipped logging reworded support: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 0


__all__ = [
    "ENV_LOG_REWORDED_SUPPORT",
    "LOG_REWORDED_SUPPORT_KEY",
    "REWORDED_CANDIDATE_TOP_K",
    "SECTION",
    "USER_ORIGINS",
    "SupportCandidate",
    "is_reworded_support_log_enabled",
    "iter_reworded_candidates",
    "log_reworded_support",
]
