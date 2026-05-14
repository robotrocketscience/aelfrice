"""Adaptive expansion-gate for retrieval (#741).

Cheap deterministic prompt-shape gate that decides whether to run the
expensive expansion layers (BFS multi-hop, HRR structural) for a given
query. Broad natural-language prompts that lack structural markers are
unlikely to benefit from graph expansion and pay the highest latency
cost on those lanes — the gate short-circuits expansion on them while
keeping L0 / L1 / L2.5-entity always on.

Honors v3.0 PHILOSOPHY (#605): deterministic narrow surface, stdlib
only, no embeddings, no model calls.

Resolver precedence (first decisive wins):
  1. ``AELFRICE_FORCE_EXPANSION=1`` env var → always run expansion
     (escape hatch for testing and bench).
  2. ``AELFRICE_NO_EXPANSION_GATE=1`` env var → disable the gate
     entirely; behaves as pre-gate (all expansion lanes run).
  3. ``[retrieval] expansion_gate_enabled`` in ``.aelfrice.toml``.
     Default ``True``.
  4. Run heuristics; return decision.

Heuristics (all deterministic, stdlib-only):
  - Length: prompt token count > ``BROAD_PROMPT_TOKEN_THRESHOLD``
    (default 80) → broad signal.
  - Structural-marker absence: no ``#NNN`` issue refs, no file paths
    (``src/...``, ``tests/...``, ``docs/...``, ``benchmarks/...``),
    no snake_case / camelCase identifiers, no edge-type names
    (``SUPPORTS``, ``CONTRADICTS``, ...) → broad signal.
  - Question-form prefix: starts with ``what``, ``why``, ``how``,
    ``which``, ``who``, ``tell me``, ``explain`` → broad signal.

If ANY signal fires "broad", expansion is skipped. Conservative by
design: prefer skipping when in doubt; users with a narrow query that
the heuristic misclassifies can opt back into the full stack via
``AELFRICE_FORCE_EXPANSION=1`` or ``aelf reason`` (which never gates).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from aelfrice.store import MemoryStore

from aelfrice.models import EDGE_TYPES

# --- Config keys ----------------------------------------------------------

CONFIG_FILENAME: Final[str] = ".aelfrice.toml"
RETRIEVAL_SECTION: Final[str] = "retrieval"
EXPANSION_GATE_FLAG: Final[str] = "expansion_gate_enabled"

ENV_FORCE_EXPANSION: Final[str] = "AELFRICE_FORCE_EXPANSION"
ENV_NO_EXPANSION_GATE: Final[str] = "AELFRICE_NO_EXPANSION_GATE"

_ENV_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
_ENV_FALSY: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})

# --- Heuristic thresholds -------------------------------------------------

BROAD_PROMPT_TOKEN_THRESHOLD: Final[int] = 80

_QUESTION_FORM_PREFIXES: Final[tuple[str, ...]] = (
    "what",
    "why",
    "how",
    "which",
    "who",
    "tell me",
    "explain",
)

# Patterns for structural-marker detection. Any match counts as "narrow"
# evidence (the prompt references specific code / belief structure).
_ISSUE_REF_RE: Final[re.Pattern[str]] = re.compile(r"#\d+\b")
_FILE_PATH_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:src|tests|docs|benchmarks|scripts)/[A-Za-z0-9_./\-]+",
)
# snake_case identifiers (at least one underscore between word chars).
_SNAKE_CASE_RE: Final[re.Pattern[str]] = re.compile(
    r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b",
)
# camelCase identifiers (lowercase prefix followed by an internal capital).
_CAMEL_CASE_RE: Final[re.Pattern[str]] = re.compile(
    r"\b[a-z][a-z0-9]*[A-Z][A-Za-z0-9]+\b",
)
# Edge-type names live in models.EDGE_TYPES. Compile once at import time.
_EDGE_TYPE_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:" + "|".join(sorted(EDGE_TYPES, key=len, reverse=True)) + r")\b",
)


@dataclass(frozen=True)
class ExpansionDecision:
    """Per-call decision from :func:`should_run_expansion`.

    Attributes
    ----------
    run_bfs:
        When ``False`` the BFS multi-hop lane is skipped in
        :func:`aelfrice.retrieval.retrieve`. L0 / L2.5-entity / L1
        always stay on.
    run_hrr_structural:
        When ``False`` the HRR structural-query lane is skipped.
        v3.0 default keeps this on regardless of gate verdict; the
        field is reserved for forward-compat once the bench shows
        broad-prompt HRR-structural cost is load-bearing.
    reason:
        Short human-readable tag used by telemetry / ``aelf doctor``
        to explain *why* the gate fired. Always populated.
    """

    run_bfs: bool
    run_hrr_structural: bool
    reason: str


def _env_force_expansion() -> bool | None:
    """``True`` when AELFRICE_FORCE_EXPANSION is truthy, ``False`` when
    explicitly falsy, ``None`` when unset / unrecognised.
    """
    raw = os.environ.get(ENV_FORCE_EXPANSION)
    if raw is None:
        return None
    norm = raw.strip().lower()
    if norm in _ENV_TRUTHY:
        return True
    if norm in _ENV_FALSY:
        return False
    return None


def _env_no_expansion_gate() -> bool | None:
    """``True`` when AELFRICE_NO_EXPANSION_GATE is truthy, ``False``
    when explicitly falsy, ``None`` when unset / unrecognised.
    """
    raw = os.environ.get(ENV_NO_EXPANSION_GATE)
    if raw is None:
        return None
    norm = raw.strip().lower()
    if norm in _ENV_TRUTHY:
        return True
    if norm in _ENV_FALSY:
        return False
    return None


def _read_toml_flag(start: Path | None = None) -> bool | None:
    """Read ``[retrieval] expansion_gate_enabled`` from
    ``.aelfrice.toml``. Returns ``None`` when the file / section /
    key is absent or the value is not a bool. Fail-soft: malformed
    TOML returns ``None``.
    """
    try:
        import tomllib
    except ImportError:  # Python <3.11 fallback; aelfrice ships >=3.11
        return None
    cur = (start or Path.cwd()).resolve()
    seen: set[Path] = set()
    while True:
        candidate = cur / CONFIG_FILENAME
        if candidate.is_file():
            try:
                with candidate.open("rb") as fh:
                    data = tomllib.load(fh)
            except (OSError, ValueError, tomllib.TOMLDecodeError):
                return None
            section = data.get(RETRIEVAL_SECTION)
            if isinstance(section, dict):
                value = section.get(EXPANSION_GATE_FLAG)
                if isinstance(value, bool):
                    return value
            return None
        if cur in seen or cur.parent == cur:
            return None
        seen.add(cur)
        cur = cur.parent


def _has_structural_markers(text: str) -> bool:
    """Return ``True`` if ``text`` contains at least one structural
    marker — an issue ref, file path, identifier, or edge-type name.
    """
    if _ISSUE_REF_RE.search(text):
        return True
    if _FILE_PATH_RE.search(text):
        return True
    if _SNAKE_CASE_RE.search(text):
        return True
    if _CAMEL_CASE_RE.search(text):
        return True
    if _EDGE_TYPE_RE.search(text):
        return True
    return False


def _starts_with_question_form(text: str) -> bool:
    """Return ``True`` if ``text`` begins with a recognised
    question-form prefix (case-insensitive).
    """
    lowered = text.lstrip().lower()
    for prefix in _QUESTION_FORM_PREFIXES:
        if lowered.startswith(prefix):
            # Require a word boundary so ``whatever`` doesn't match
            # ``what``. End-of-string or any non-alnum char qualifies.
            tail_pos = len(prefix)
            if tail_pos >= len(lowered):
                return True
            tail = lowered[tail_pos]
            if not tail.isalnum() and tail != "_":
                return True
    return False


def should_run_expansion(
    query: str,
    *,
    start: Path | None = None,
    store: MemoryStore | None = None,
    now_ts: int | None = None,
) -> ExpansionDecision:
    """Decide whether to run BFS / HRR-structural expansion for ``query``.

    Resolver precedence: env-force > env-disable-gate > TOML > heuristics.

    Parameters
    ----------
    query:
        The user prompt or retrieval query. Whitespace-only / empty
        queries short-circuit to ``run_bfs=True`` (no gating; the
        upstream retrieve() returns L0-only on empty input anyway).
    start:
        Optional directory to start the ``.aelfrice.toml`` walk from.
        Defaults to ``Path.cwd()``.
    store:
        Optional :class:`~aelfrice.store.MemoryStore` for the #760
        meta-belief token-threshold resolver. When ``None``, the
        heuristic falls back to the static
        :data:`BROAD_PROMPT_TOKEN_THRESHOLD` (80) regardless of the
        meta-belief env flag.
    now_ts:
        UTC epoch seconds for the meta-belief resolver's decay
        calculation. When ``None`` and ``store`` is provided,
        defaults to ``int(time.time())``.

    Returns
    -------
    ExpansionDecision
        Populated decision with ``reason`` set for telemetry.
    """
    env_force = _env_force_expansion()
    if env_force is True:
        return ExpansionDecision(
            run_bfs=True,
            run_hrr_structural=True,
            reason="env-force-expansion",
        )

    env_no_gate = _env_no_expansion_gate()
    if env_no_gate is True:
        return ExpansionDecision(
            run_bfs=True,
            run_hrr_structural=True,
            reason="env-no-gate",
        )

    toml_value = _read_toml_flag(start)
    if toml_value is False:
        return ExpansionDecision(
            run_bfs=True,
            run_hrr_structural=True,
            reason="toml-disabled",
        )

    text = query.strip()
    if not text:
        return ExpansionDecision(
            run_bfs=True,
            run_hrr_structural=True,
            reason="empty-query",
        )

    # Gate evaluation. Any "broad" signal trips it.
    # #760: resolve the token threshold through the meta-belief when the
    # env flag is on and a store was supplied; fall back to the static
    # BROAD_PROMPT_TOKEN_THRESHOLD (80) otherwise so behaviour is
    # byte-identical to pre-#760 on cold-start or flag-off installs.
    if store is not None:
        import time as _time  # noqa: PLC0415
        from aelfrice.retrieval import (  # noqa: PLC0415
            resolve_expansion_gate_token_threshold_with_meta,
        )
        _effective_ts = now_ts if now_ts is not None else int(_time.time())
        token_threshold = resolve_expansion_gate_token_threshold_with_meta(
            store, now_ts=_effective_ts,
        )
    else:
        token_threshold = BROAD_PROMPT_TOKEN_THRESHOLD
    tokens = text.split()
    long_prompt = len(tokens) > token_threshold
    has_markers = _has_structural_markers(text)
    question_form = _starts_with_question_form(text)

    reasons: list[str] = []
    if long_prompt:
        reasons.append(f"long({len(tokens)}>{token_threshold})")
    if not has_markers:
        reasons.append("no-markers")
    if question_form:
        reasons.append("question-form")

    if reasons:
        # HRR-structural stays on for v1 — only BFS is gated. The
        # field is present so a future flag can toggle it without
        # an API break.
        return ExpansionDecision(
            run_bfs=False,
            run_hrr_structural=True,
            reason="broad:" + ",".join(reasons),
        )

    return ExpansionDecision(
        run_bfs=True,
        run_hrr_structural=True,
        reason="narrow",
    )
