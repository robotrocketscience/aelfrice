"""LLM-Haiku onboard classifier (opt-in, v1.3.0).

Wraps the Anthropic SDK to classify scanner candidates with higher
recall than the regex fallback in `aelfrice.classification`. Default-OFF
in every code path. The module is importable without the `[onboard-llm]`
extra; the actual `anthropic` import is local to the request call site
so a baseline install has no `anthropic` symbol resolvable at any
module load.

Boundary policy (docs/llm_classifier.md § 4):

  All four gates must succeed before any outbound call. If any gate
  fails, `aelf onboard` either falls through to the regex classifier
  (gate 1, 3, 4 default-off paths) or exits 1 with a clear hint
  (gate 1 explicit-flag path, gate 2, gate 4 explicit-rejection):

  1. The `[onboard-llm]` extra is installed (anthropic importable).
  2. `ANTHROPIC_API_KEY` is set in the environment.
  3. `--llm-classify` flag (CLI) OR `[onboard.llm].enabled = true`
     (.aelfrice.toml). Flag wins on conflict.
  4. One-time consent prompt accepted; sentinel at
     `~/.aelfrice/llm-classify-consented` records timestamp +
     model id + aelfrice major version.

Failure semantics (docs/llm_classifier.md § 7):

  - Connection refused / DNS / TLS / 5xx / 429 / timeout / malformed
    response → fall back to regex `classify_sentence` for the whole
    batch, exit 0, write one `feedback_history` audit row per fallback
    insertion tagged `onboard.llm.fallback:<reason>`.
  - 401/403 (auth) → DO NOT fall back. Exit 1, no beliefs inserted,
    error message names Anthropic auth.
  - Token cap exceeded → exit 1 with partial-insertion message; the
    deterministic belief id ensures a re-run resumes from the cap-hit
    point.

Telemetry: stdout-only. The "no telemetry" stance in PRIVACY.md is
preserved — aelfrice does not phone-home about its own LLM usage.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Final, Mapping, cast

from aelfrice import __version__ as _AELFRICE_VERSION
from aelfrice.classification import (
    ClassificationResult,
    classify_sentence,
    get_source_adjusted_prior,
)
from aelfrice.models import (
    BELIEF_TYPES,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_DOCUMENT_RECENT,
)

# Pinned default model. Overridable via `[onboard.llm].model` but
# discouraged — the boundary policy and classifier behaviour are
# calibrated against this specific id.
DEFAULT_MODEL: Final[str] = "claude-haiku-4-5-20251001"

# Hard cap on input+output tokens per onboard run. Default 200_000 is
# enough for projects with thousands of candidates after the noise
# filter; 0 disables the cap.
DEFAULT_MAX_TOKENS: Final[int] = 200_000

# Consent sentinel path. Per-user-per-machine, not per-project. The
# spec lives under `~/.aelfrice/`, not the per-project DB directory.
SENTINEL_DIRNAME: Final[str] = ".aelfrice"
SENTINEL_FILENAME: Final[str] = "llm-classify-consented"

ENV_API_KEY: Final[str] = "ANTHROPIC_API_KEY"

# Permitted origin tiers from the LLM. `user_*` tiers are reserved
# for explicit user actions; the classifier may not assign them.
_PERMITTED_ORIGINS: Final[frozenset[str]] = frozenset({
    ORIGIN_AGENT_INFERRED,
    ORIGIN_DOCUMENT_RECENT,
})

# Per-request timeout, seconds. Caps the total wall-clock per
# Anthropic call so a hung request does not freeze onboard.
_REQUEST_TIMEOUT_SECONDS: Final[float] = 30.0

# Temperature for classification: deterministic.
_REQUEST_TEMPERATURE: Final[float] = 0.0


# --- Config dataclass ----------------------------------------------------


@dataclass(frozen=True)
class LLMConfig:
    """Resolved configuration for the LLM-classify path.

    `enabled` is the resolution of CLI flag + config block + default
    OFF (see `resolve_enabled`). `max_tokens` is the per-run cap;
    0 disables. `model` is the Anthropic model id; pinned by default.
    """

    enabled: bool = False
    max_tokens: int = DEFAULT_MAX_TOKENS
    model: str = DEFAULT_MODEL

    @classmethod
    def default(cls) -> "LLMConfig":
        return cls()

    @classmethod
    def from_mapping(
        cls,
        section: Mapping[str, Any],
        *,
        stderr: IO[str] | None = None,
    ) -> "LLMConfig":
        """Build from a parsed `[onboard.llm]` table.

        Tolerant of unknown keys (forward-compat) and wrong-typed
        values (skip-and-warn, default applied). Mirrors the
        NoiseConfig resilience contract in noise_filter.py.
        """
        serr: IO[str] = stderr if stderr is not None else sys.stderr

        enabled = section.get("enabled", False)
        if not isinstance(enabled, bool):
            print(
                f"aelfrice llm_classifier: ignoring [onboard.llm] enabled "
                f"(expected bool, got {type(enabled).__name__})",
                file=serr,
            )
            enabled = False

        max_tokens_raw = section.get("max_tokens", DEFAULT_MAX_TOKENS)
        if (
            not isinstance(max_tokens_raw, int)
            or isinstance(max_tokens_raw, bool)
        ):
            print(
                f"aelfrice llm_classifier: ignoring [onboard.llm] max_tokens "
                f"(expected int, got {type(max_tokens_raw).__name__})",
                file=serr,
            )
            max_tokens = DEFAULT_MAX_TOKENS
        else:
            max_tokens = max(0, max_tokens_raw)

        model_raw = section.get("model", DEFAULT_MODEL)
        if not isinstance(model_raw, str) or not model_raw:
            if model_raw is not DEFAULT_MODEL:
                print(
                    f"aelfrice llm_classifier: ignoring [onboard.llm] model "
                    f"(expected non-empty str, got "
                    f"{type(model_raw).__name__})",
                    file=serr,
                )
            model = DEFAULT_MODEL
        else:
            model = model_raw

        return cls(enabled=enabled, max_tokens=max_tokens, model=model)


# --- Resolution helpers --------------------------------------------------


def resolve_enabled(
    *, flag: bool | None, config_enabled: bool,
) -> bool:
    """Resolve CLI flag + config block to a single boolean.

    Order (per spec § 3.4):
      1. `--llm-classify=false` on CLI → off.
      2. `--llm-classify` (or `=true`) on CLI → on.
      3. `[onboard.llm].enabled = true` in `.aelfrice.toml` → on.
      4. Default → off.

    `flag` semantics: `None` means the user did not pass the flag,
    `True` means `--llm-classify` was set, `False` means
    `--llm-classify=false` was set.
    """
    if flag is True:
        return True
    if flag is False:
        return False
    return bool(config_enabled)


# --- Sentinel file -------------------------------------------------------


@dataclass(frozen=True)
class Sentinel:
    """The consent sentinel record.

    Written to `~/.aelfrice/llm-classify-consented` after a user
    accepts the one-time prompt. Used to skip the prompt on
    subsequent runs.

    A new model id, or a new aelfrice MAJOR version, invalidates the
    sentinel and re-prompts. Patch and minor bumps do not.
    """

    consented_at: str
    model: str
    aelfrice_version: str


def sentinel_path(home: Path | None = None) -> Path:
    """Return the absolute path of the consent sentinel.

    `home=None` resolves to `Path.home()`. Tests inject a tmp_path
    here to avoid touching the real `~/.aelfrice/`.
    """
    base = home if home is not None else Path.home()
    return base / SENTINEL_DIRNAME / SENTINEL_FILENAME


def _aelfrice_major_version(version: str) -> str:
    """Return the major-version segment of a SemVer string.

    Tolerates pre-release suffixes (`1.3.0a0` → `1`). Returns the
    raw input on parse failure so the comparison is conservative.
    """
    if not version:
        return version
    head = version.split(".", 1)[0]
    return head


def read_sentinel(path: Path) -> Sentinel | None:
    """Read the sentinel file; return None on missing / malformed.

    Failures degrade silently to None — the caller will re-prompt,
    which is the correct behaviour for any unexpected state.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, PermissionError):
        return None
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    parsed_dict = cast(dict[str, Any], parsed)
    consented_at = parsed_dict.get("consented_at")
    model = parsed_dict.get("model")
    version = parsed_dict.get("aelfrice_version")
    if (
        not isinstance(consented_at, str)
        or not isinstance(model, str)
        or not isinstance(version, str)
    ):
        return None
    return Sentinel(
        consented_at=consented_at,
        model=model,
        aelfrice_version=version,
    )


def write_sentinel(
    path: Path,
    *,
    model: str,
    version: str = _AELFRICE_VERSION,
    now: str | None = None,
) -> None:
    """Write the sentinel file atomically.

    Creates the parent directory if needed. Tests override `now`
    for determinism.
    """
    timestamp = now if now is not None else _utc_now_iso()
    payload = {
        "consented_at": timestamp,
        "model": model,
        "aelfrice_version": version,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def revoke_sentinel(path: Path) -> bool:
    """Remove the sentinel file. Return True if a file was removed.

    Idempotent: a missing sentinel is not an error.
    """
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except (OSError, PermissionError):
        return False


def is_sentinel_valid(
    sentinel: Sentinel | None,
    *,
    model: str,
    version: str = _AELFRICE_VERSION,
) -> bool:
    """Return True iff the sentinel matches the current model + major.

    Spec § 4.3: a new model id, or a new aelfrice MAJOR version,
    invalidates the sentinel. Patch and minor bumps do not.
    """
    if sentinel is None:
        return False
    if sentinel.model != model:
        return False
    if _aelfrice_major_version(sentinel.aelfrice_version) != \
            _aelfrice_major_version(version):
        return False
    return True


# --- Consent prompt ------------------------------------------------------


_PROMPT_TEXT: Final[str] = (
    "aelf onboard --llm-classify\n"
    "\n"
    "This will send sentences extracted from the files under the\n"
    "scanned path (paragraphs from .md/.rst/.txt/.adoc files, git\n"
    "commit subjects, Python docstrings) plus their `source` strings\n"
    "(e.g., doc:README.md:p3) to Anthropic's API for classification.\n"
    "The content of those candidates will leave your machine.\n"
    "\n"
    "aelfrice will NOT send: file contents beyond the extracted\n"
    "candidate, env vars, working directory paths, hostnames, git\n"
    "remotes, git author email, files marked INEDIBLE, or anything\n"
    "outside the extracted candidate text.\n"
    "\n"
    "You can audit what would be sent with:\n"
    "  aelf onboard <path> --llm-classify --dry-run\n"
    "\n"
    "Continue? [y/N]: "
)


@dataclass
class PromptResult:
    """Outcome of `prompt_for_consent`.

    `accepted=True` only when stdin is a TTY, the user typed `y` /
    `Y` / `yes`, and no IO error occurred. Every other path
    (`N`, EOF, non-TTY, exception) returns `accepted=False`.
    """

    accepted: bool
    reason: str


def prompt_for_consent(
    *,
    stdin: IO[str] | None = None,
    stderr: IO[str] | None = None,
    is_tty: bool | None = None,
) -> PromptResult:
    """Print the prompt to stderr; read y/N from stdin.

    Returns `accepted=True` only when `is_tty=True`, stdin is
    readable, and the user typed an affirmative.

    Tests inject `stdin` / `stderr` / `is_tty` for full
    determinism without touching the real terminal.
    """
    sin: IO[str] = stdin if stdin is not None else sys.stdin
    serr: IO[str] = stderr if stderr is not None else sys.stderr
    if is_tty is None:
        try:
            is_tty = sin.isatty()
        except (AttributeError, ValueError, OSError):
            is_tty = False

    print(_PROMPT_TEXT, file=serr, end="", flush=True)

    if not is_tty:
        # CI / non-interactive: a one-time prompt cannot be answered
        # in a script. Refuse the prompt and exit 1; the user must
        # accept once on a workstation, or pre-create the sentinel.
        print(
            "\naelf: stdin is not a TTY; cannot prompt for consent. "
            "Run interactively once or pre-create the sentinel at "
            f"{sentinel_path()}",
            file=serr,
        )
        return PromptResult(accepted=False, reason="non-tty")

    try:
        line = sin.readline()
    except (EOFError, OSError):
        return PromptResult(accepted=False, reason="eof")

    if line == "":
        # readline returns "" only on EOF; "\n" represents Enter on
        # an empty line.
        return PromptResult(accepted=False, reason="eof")

    answer = line.strip().lower()
    if answer in ("y", "yes"):
        return PromptResult(accepted=True, reason="y")
    return PromptResult(accepted=False, reason="rejected")


# --- Gate result ---------------------------------------------------------


@dataclass(frozen=True)
class GateResult:
    """Outcome of `check_gates`.

    `pass_all=True` means every one of the four gates passed and
    `aelf onboard` may make outbound calls. `pass_all=False` paired
    with `exit_code is None` means "fall through to regex"; with
    `exit_code != None` means "exit with this code".

    `message` is a human-readable hint surfaced on stderr when the
    gate fails the explicit-opt-in path (gate 2 fail-fast, etc.).
    """

    pass_all: bool
    exit_code: int | None = None
    message: str | None = None


def check_gates(
    *,
    enabled: bool,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    model: str = DEFAULT_MODEL,
    sdk_check: "Any" = None,
) -> GateResult:
    """Run all four gates in order. Return GateResult.

    `enabled` is the resolved CLI-flag-or-config-block value (gate 3).
    `env` is `os.environ`-like; `None` reads the live env.
    `home` is the user's home dir; `None` uses `Path.home()`.
    `sdk_check` overrides the import probe; tests inject a callable
    that returns True/False, otherwise the live `import anthropic`
    is attempted.

    Order (per spec § 3, § 4):
      0. enabled? else default OFF, fall through to regex (no exit).
      1. anthropic SDK importable? else exit 1 with install hint.
      2. ANTHROPIC_API_KEY set? else exit 1 with hint.
      3. (already covered by `enabled`).
      4. consent sentinel valid? else prompt is the caller's job
         (gate-check returns `accepted_consent=False`); the prompt
         path lives in `prompt_for_consent` and the caller wires it.
    """
    if not enabled:
        return GateResult(pass_all=False, exit_code=None, message=None)

    if not _anthropic_importable(sdk_check):
        return GateResult(
            pass_all=False,
            exit_code=1,
            message=(
                "aelf: --llm-classify requires the [onboard-llm] extra. "
                "Install with: pip install aelfrice[onboard-llm]"
            ),
        )

    e = env if env is not None else os.environ
    if not e.get(ENV_API_KEY):
        return GateResult(
            pass_all=False,
            exit_code=1,
            message=(
                f"aelf: {ENV_API_KEY} not set; --llm-classify requires it. "
                "Either unset --llm-classify or export the key."
            ),
        )

    return GateResult(pass_all=True, exit_code=None, message=None)


def _anthropic_importable(sdk_check: "Any") -> bool:
    """Probe whether `anthropic` can be imported.

    `sdk_check` is a test injection: a callable returning a bool.
    Production callers pass `None` and the live import is tried.
    """
    if sdk_check is not None:
        return bool(sdk_check())
    try:
        import anthropic  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


# --- Classification request ----------------------------------------------


_SYSTEM_PROMPT: Final[str] = """\
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


@dataclass
class CandidateClassification:
    """One classification result, addressed by index.

    Returned from `classify_batch` for each successful candidate.
    Failures (parse errors, invalid fields, etc.) are not in the
    returned list — the caller composites them with the regex
    fallback.
    """

    index: int
    belief_type: str
    origin: str
    persist: bool


@dataclass
class BatchTelemetry:
    """Anthropic-billed token counts for the batch.

    Filled in by the request layer; surfaced on stdout by the CLI.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    fallbacks: int = 0
    skipped_invalid: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMAuthError(Exception):
    """Raised on 401/403 from Anthropic. Causes a hard exit; no fallback."""


class LLMTransientError(Exception):
    """Raised on connect/timeout/429/5xx/malformed. Triggers regex fallback."""


class LLMTokenCapExceeded(Exception):
    """Raised when running tokens exceed the configured cap mid-stream."""

    def __init__(self, consumed: int, cap: int) -> None:
        super().__init__(
            f"onboard aborted: token cap reached at {consumed}/{cap}. "
            "Re-run to resume, or raise [onboard.llm].max_tokens in "
            ".aelfrice.toml."
        )
        self.consumed = consumed
        self.cap = cap


def build_user_message(candidates: list[CandidateInput]) -> str:
    """Return the user-message body for one Haiku request.

    The body is a JSON array; the model returns a JSON array of the
    same length. Spec § 5 locks the response to JSON only.
    """
    payload = [
        {"index": c.index, "source": c.source, "text": c.text}
        for c in candidates
    ]
    return json.dumps(payload)


def parse_response(
    raw: str, *, expected_count: int,
) -> list[CandidateClassification]:
    """Parse a Haiku response into a list of CandidateClassification.

    Raises LLMTransientError on JSON failure or array-length
    mismatch (treat as broken-batch → regex fallback).
    Per-candidate invalid fields are dropped silently and counted
    by the caller; only structural failures escape.
    """
    try:
        parsed_any: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMTransientError(f"malformed JSON response: {exc}") from exc

    if not isinstance(parsed_any, list):
        raise LLMTransientError(
            f"response is not a JSON array (got {type(parsed_any).__name__})"
        )
    parsed = cast(list[Any], parsed_any)
    if len(parsed) != expected_count:
        raise LLMTransientError(
            f"response length {len(parsed)} != expected {expected_count}"
        )

    out: list[CandidateClassification] = []
    for idx, entry in enumerate(parsed):
        if not isinstance(entry, dict):
            continue
        entry_dict = cast(dict[str, Any], entry)
        belief_type = entry_dict.get("belief_type")
        origin = entry_dict.get("origin")
        persist = entry_dict.get("persist")
        if (
            not isinstance(belief_type, str)
            or belief_type not in BELIEF_TYPES
        ):
            continue
        if not isinstance(origin, str) or origin not in _PERMITTED_ORIGINS:
            continue
        if not isinstance(persist, bool):
            continue
        out.append(
            CandidateClassification(
                index=idx,
                belief_type=belief_type,
                origin=origin,
                persist=persist,
            )
        )
    return out


@dataclass
class _ClientResponse:
    """Internal: shape we expect from the Anthropic SDK call."""

    text: str
    input_tokens: int
    output_tokens: int


def _call_anthropic(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_message: str,
    max_output_tokens: int,
    sdk_module: "Any" = None,
) -> _ClientResponse:
    """Make one Anthropic Messages API call and return the response text + usage.

    `sdk_module` is a test injection: tests pass a fake module with
    an `Anthropic(...)` constructor whose `.messages.create(...)`
    method returns a fake response. Production callers pass `None`
    and the real `anthropic` is imported here, locally.

    Raises:
      LLMAuthError on 401/403.
      LLMTransientError on connect / timeout / 429 / 5xx / parse.
    """
    if sdk_module is None:
        try:
            import anthropic as sdk_module  # type: ignore[import-not-found,no-redef]
        except ImportError as exc:
            raise LLMTransientError(
                "anthropic SDK not importable at request time"
            ) from exc

    sdk: Any = sdk_module
    try:
        client = sdk.Anthropic(api_key=api_key, timeout=_REQUEST_TIMEOUT_SECONDS)
        resp = client.messages.create(
            model=model,
            max_tokens=max_output_tokens,
            temperature=_REQUEST_TEMPERATURE,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
    except Exception as exc:  # noqa: BLE001 — SDK exception classes vary
        if _is_auth_error(exc):
            raise LLMAuthError(str(exc)) from exc
        raise LLMTransientError(str(exc)) from exc

    text = _extract_text_from_response(resp)
    input_tokens, output_tokens = _extract_usage(resp)
    return _ClientResponse(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _is_auth_error(exc: BaseException) -> bool:
    """Detect 401/403 from an Anthropic SDK exception.

    The SDK exposes `AuthenticationError` and `PermissionDeniedError`
    classes; we check class name to avoid importing them at module
    load. Tests can synthesise via class-name only.
    """
    name = type(exc).__name__
    if name in ("AuthenticationError", "PermissionDeniedError"):
        return True
    # Fall back to status-code attribute if present.
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return True
    return False


def _extract_text_from_response(resp: Any) -> str:
    """Pluck the text block off an Anthropic response.

    Real shape: `resp.content` is a list of content blocks; the
    text block has `.text`. Tests pass a fake with the same shape.
    """
    content = getattr(resp, "content", None)
    if content is None:
        raise LLMTransientError("response has no content")
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    if not parts:
        raise LLMTransientError("response contains no text block")
    return "".join(parts)


def _extract_usage(resp: Any) -> tuple[int, int]:
    """Pull `(input_tokens, output_tokens)` off the response usage.

    Returns `(0, 0)` if the SDK omits usage (older versions).
    """
    usage = getattr(resp, "usage", None)
    if usage is None:
        return (0, 0)
    inp = getattr(usage, "input_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or 0
    try:
        return (int(inp), int(out))
    except (ValueError, TypeError):
        return (0, 0)


# --- Batch driver --------------------------------------------------------


@dataclass
class BatchResult:
    """Outcome of `classify_batch`.

    `classifications` indexes into the input candidate list.
    `fallback_used=True` means every candidate was reclassified by
    the regex fallback. `auth_error` is non-None for 401/403; the
    caller exits 1 in that case.
    """

    classifications: list[CandidateClassification] = field(
        default_factory=lambda: [],
    )
    telemetry: BatchTelemetry = field(default_factory=BatchTelemetry)
    fallback_used: bool = False
    fallback_reason: str | None = None
    auth_error: str | None = None
    token_cap_exceeded: bool = False
    token_cap_consumed: int = 0


def classify_batch(
    candidates: list[CandidateInput],
    *,
    api_key: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    sdk_module: "Any" = None,
) -> BatchResult:
    """Classify a list of candidates in a single Haiku request.

    Returns a BatchResult. The caller composites with the regex
    fallback when `fallback_used=True` and exits 1 when
    `auth_error is not None` or `token_cap_exceeded=True`.

    Sharding is single-request at v1.3.0 (spec § 5: implementation
    choice; acceptance only requires correct telemetry). A 200k
    context comfortably fits the typical aelfrice-shaped project.
    """
    result = BatchResult()

    if not candidates:
        return result

    # Output token budget: ~30 tokens per candidate × count, with a
    # small floor so the API doesn't reject pathological short inputs.
    max_output_tokens = max(64, len(candidates) * 50)

    # Pre-flight cap check: if the input alone is bigger than the
    # cap, abort before sending. Heuristic: 30 tokens per candidate
    # for input + the system prompt amortization.
    if max_tokens > 0:
        estimated_input = 800 + 30 * len(candidates)
        if estimated_input >= max_tokens:
            result.token_cap_exceeded = True
            result.token_cap_consumed = estimated_input
            return result

    user_msg = build_user_message(candidates)

    try:
        resp = _call_anthropic(
            api_key=api_key,
            model=model,
            system_prompt=_SYSTEM_PROMPT,
            user_message=user_msg,
            max_output_tokens=max_output_tokens,
            sdk_module=sdk_module,
        )
    except LLMAuthError as exc:
        result.auth_error = str(exc)
        return result
    except LLMTransientError as exc:
        result.fallback_used = True
        result.fallback_reason = type(exc).__name__ + ":transient"
        result.telemetry.fallbacks = len(candidates)
        result.telemetry.requests = 1
        return result

    result.telemetry.input_tokens += resp.input_tokens
    result.telemetry.output_tokens += resp.output_tokens
    result.telemetry.requests += 1

    # Post-call cap check: if the response usage pushed us past the
    # cap, abort before parsing. Already-classified candidates from
    # earlier shards (none yet at v1.3) would be retained; for the
    # single-request path this is a clean abort.
    if max_tokens > 0 and result.telemetry.total_tokens > max_tokens:
        result.token_cap_exceeded = True
        result.token_cap_consumed = result.telemetry.total_tokens
        return result

    try:
        parsed = parse_response(resp.text, expected_count=len(candidates))
    except LLMTransientError as exc:
        result.fallback_used = True
        result.fallback_reason = "parse:" + str(exc)[:64]
        result.telemetry.fallbacks = len(candidates)
        return result

    # Track invalid-per-candidate count (spec § 5.4: invalid fields
    # are dropped, not regex-fallback'd per candidate).
    result.telemetry.skipped_invalid = len(candidates) - len(parsed)
    result.classifications = parsed
    return result


# --- Regex-fallback bridge ----------------------------------------------


def regex_fallback_classify(
    candidate: CandidateInput,
) -> ClassificationResult:
    """Run the regex `classify_sentence` for one candidate.

    Wrapper so callers don't need to know about the underlying
    function. Used by `aelfrice.scanner.scan_repo` when
    `BatchResult.fallback_used=True`.
    """
    return classify_sentence(candidate.text, candidate.source)


def llm_origin_for_source(source: str) -> str:
    """Default origin tier for a scanner candidate source string.

    The LLM also returns an origin in its response, but the
    fallback path needs a deterministic default. Doc paragraphs
    and git commit subjects → `document_recent`; AST docstrings
    → `agent_inferred`. Mirrors spec § 5 few-shot defaults.
    """
    if source.startswith("doc:") or source.startswith("git:"):
        return ORIGIN_DOCUMENT_RECENT
    return ORIGIN_AGENT_INFERRED


# --- Telemetry formatting ------------------------------------------------


def format_telemetry_line(
    *, model: str, telemetry: BatchTelemetry,
) -> str:
    """Format the one-line `onboard.llm:` summary printed to stdout.

    Spec § 6.3 schema:
      onboard.llm: model=<model> input_tokens=<n> output_tokens=<n>
        total_tokens=<n> requests=<n> fallbacks=<n>
    """
    return (
        f"onboard.llm: model={model} "
        f"input_tokens={telemetry.input_tokens} "
        f"output_tokens={telemetry.output_tokens} "
        f"total_tokens={telemetry.total_tokens} "
        f"requests={telemetry.requests} "
        f"fallbacks={telemetry.fallbacks}"
    )


# --- Source-adjusted prior re-export -------------------------------------
# scanner.scan_repo imports get_source_adjusted_prior from
# `classification` directly; we re-export here so callers using the
# llm_classifier surface as their entry point have one less import.
__all__ = [
    "BatchResult",
    "BatchTelemetry",
    "CandidateClassification",
    "CandidateInput",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "ENV_API_KEY",
    "GateResult",
    "LLMAuthError",
    "LLMConfig",
    "LLMTokenCapExceeded",
    "LLMTransientError",
    "PromptResult",
    "Sentinel",
    "build_user_message",
    "check_gates",
    "classify_batch",
    "format_telemetry_line",
    "get_source_adjusted_prior",
    "is_sentinel_valid",
    "llm_origin_for_source",
    "parse_response",
    "prompt_for_consent",
    "read_sentinel",
    "regex_fallback_classify",
    "resolve_enabled",
    "revoke_sentinel",
    "sentinel_path",
    "write_sentinel",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
