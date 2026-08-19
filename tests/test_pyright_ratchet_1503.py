"""No document may assert that pyright strict passes (#1503).

`RELEASING.md` step 6 ran `uv run pyright src/` with the comment `# strict`,
implying it passes. It emits 987 errors over 76 files, and **no workflow ran
pyright at all**, so the tick was self-reported and unenforced. Three other
documents repeated the claim. The cost is on the record: `CHANGELOG/v4.md`
notes a `NameError` that reached `main` behind it.

Two properties are pinned here, and neither of them is the error count — that
lives in `pyright_baseline.json` and moves with every fix.

  1. **The baseline exists and is non-empty.** A check whose baseline is
     missing or `{}` passes everything; that is the self-reported tick again,
     wearing a CI badge.
  2. **No document re-asserts the unqualified claim.** The words come back
     easily — the four sites were written by four different changes — and a
     document that says "strict must pass" next to a ratchet that permits 987
     errors is worse than no document.

Running pyright itself is CI's job (`.github/workflows/pyright-ratchet.yml`).
It takes minutes on this tree, so it is deliberately not run from the unit
suite.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "pyright_baseline.json"
SCRIPT = REPO / "scripts" / "check_pyright_baseline.py"
WORKFLOW = REPO / ".github" / "workflows" / "pyright-ratchet.yml"

CLAIM_SITES = (
    "docs/concepts/RELEASING.md",
    "CONTRIBUTING.md",
    ".github/pull_request_template.md",
    "docs/concepts/ARCHITECTURE.md",
)

# "strict passes / must pass / is clean", with no qualifier alongside.
_UNQUALIFIED = re.compile(
    r"pyright[^.\n]{0,40}(--)?strict[^.\n]{0,40}\b(passes|must pass|is clean|"
    r"clean)\b",
    re.IGNORECASE,
)


def test_the_ratchet_is_wired() -> None:
    assert SCRIPT.exists(), "the baseline checker is gone"
    assert WORKFLOW.exists(), (
        "no pyright-ratchet workflow — an unenforced checker is what #1503 "
        "was filed about"
    )
    wf = WORKFLOW.read_text(encoding="utf-8")
    assert "check_pyright_baseline.py" in wf, (
        "the workflow no longer runs the baseline checker"
    )
    # A *setting*, not the substring: the workflow's own comment explains why
    # it has no `continue-on-error`, and a naive `in` check fires on that
    # prose. Matched on a non-comment line with a truthy value.
    soft = [
        ln for ln in wf.splitlines()
        if re.match(r"^\s*continue-on-error:\s*(true|yes|on)\s*$", ln, re.I)
    ]
    assert not soft, (
        f"the ratchet job cannot fail ({soft}); a gate that cannot fail is "
        "the self-reported tick with a CI badge on it"
    )


def test_the_baseline_is_present_and_not_vacuous() -> None:
    assert BASELINE.exists(), f"{BASELINE} is missing; the ratchet has no floor"
    data = json.loads(BASELINE.read_text(encoding="utf-8"))
    files = data.get("files", {})
    assert files, "the baseline is empty, so every file would pass unchecked"
    assert data.get("total") == sum(files.values()), (
        "the baseline's recorded total disagrees with its per-file counts"
    )
    assert all(isinstance(v, int) and v >= 0 for v in files.values())


def test_no_document_claims_strict_passes() -> None:
    """The unqualified claim must not come back.

    A qualified sentence is fine and expected — CONTRIBUTING states plainly
    that strict does *not* pass and that the ratchet permits the existing
    count. What must not reappear is the bare assertion.
    """
    offenders: list[str] = []
    for rel in CLAIM_SITES:
        path = REPO / rel
        if not path.exists():
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "does not pass" in line or "no file may" in line:
                continue  # the corrected, qualified form
            if _UNQUALIFIED.search(line):
                offenders.append(f"{rel}:{i}: {line.strip()[:110]}")
    assert not offenders, (
        "document(s) assert pyright strict passes, which is false:\n  "
        + "\n  ".join(offenders)
        + "\n\nState the enforced baseline instead."
    )
