"""Unit + replay tests for `scripts/issue_creation_audit.py`."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "issue_creation_audit.py"
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "issue_creation_audit"


def _load_module():
    spec = importlib.util.spec_from_file_location("issue_creation_audit", SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


audit = _load_module()


def test_extract_module_path():
    body = "Adds `src/aelfrice/foo_bar.py` and wires it."
    out = audit.extract(body)
    assert ("module_path", "src/aelfrice/foo_bar.py") in out


def test_extract_class_name():
    body = "## Acceptance\n- `class FooBar` exposing `rewrite()`."
    out = audit.extract(body)
    assert ("class_name", "FooBar") in out


def test_extract_file_path_test_and_doc():
    body = "Tests at tests/test_foo.py and spec at docs/foo.md."
    kinds = {kv for kv in audit.extract(body)}
    assert ("file_path", "tests/test_foo.py") in kinds
    assert ("file_path", "docs/foo.md") in kinds


def test_extract_api_symbol():
    body = "Exposes `Foo.bar(` for callers."
    out = audit.extract(body)
    assert ("api_symbol", "Foo.bar") in out


def test_extract_dedup():
    body = "src/aelfrice/x.py and again src/aelfrice/x.py and `class Foo` and `class Foo`."
    out = audit.extract(body)
    assert sum(1 for k, _ in out if k == "module_path") == 1
    assert sum(1 for k, _ in out if k == "class_name") == 1


def test_extract_strips_refs_section():
    body = (
        "## Acceptance\n\n"
        "- New module `src/aelfrice/foo.py`.\n\n"
        "## Refs\n\n"
        "- Parent: see `src/aelfrice/parent_module.py`\n"
    )
    out = audit.extract(body)
    paths = {v for k, v in out if k == "module_path"}
    assert "src/aelfrice/foo.py" in paths
    assert "src/aelfrice/parent_module.py" not in paths


def test_extract_strips_out_of_scope_section():
    body = (
        "## Acceptance\n\n- `class Foo`.\n\n"
        "## Out of scope\n\n- Existing `class Bar` is not touched here.\n"
    )
    out = {v for k, v in audit.extract(body) if k == "class_name"}
    assert "Foo" in out
    assert "Bar" not in out


def test_extract_ignores_lowercase_class_name():
    body = "uses `class foo`"
    out = audit.extract(body)
    assert not any(k == "class_name" for k, _ in out)


def test_hit_hash_stable_across_order():
    a = audit.hit_hash(["a", "b", "c"])
    b = audit.hit_hash(["c", "a", "b"])
    assert a == b


def test_render_includes_marker_and_hash():
    out = audit.render(["x"])
    assert audit.COMMENT_MARKER_PREFIX in out
    assert "hits:" in out
    assert "audit-ack" in out


def test_render_bullets_match_hits():
    out = audit.render(["alpha", "beta"])
    assert "  - alpha" in out
    assert "  - beta" in out


def test_main_emits_nothing_on_clean_body(tmp_path: Path):
    body = tmp_path / "body.md"
    body.write_text("Plain prose with no module paths or class names.\n")
    result = subprocess.run(
        ["python3", str(SCRIPT_PATH), "--body-file", str(body), "--main-ref", "HEAD"],
        capture_output=True, text=True, cwd=REPO_ROOT, check=False,
    )
    assert result.returncode == 0
    assert result.stdout == ""


def _have_main_with_vocab_bridge() -> bool:
    """Skip the replay test when the local repo has no origin/main containing #433's ship."""
    for ref in ("origin/main", "github/main", "main"):
        r = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", ref, "--", "src/aelfrice/vocab_bridge.py"],
            capture_output=True, text=True, cwd=REPO_ROOT, check=False,
        )
        if r.returncode == 0 and r.stdout.strip() == "src/aelfrice/vocab_bridge.py":
            return True
    return False


def _resolve_main_ref() -> str:
    for ref in ("origin/main", "github/main", "main"):
        r = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", ref],
            capture_output=True, text=True, cwd=REPO_ROOT, check=False,
        )
        if r.returncode == 0:
            return ref
    return "HEAD"


@pytest.mark.skipif(not _have_main_with_vocab_bridge(),
                    reason="origin/main does not contain #433 ship; replay test depends on shipped surface")
def test_replay_issue_521_surfaces_known_artifacts():
    """Regression: the audit script must surface the known #433 artifacts when re-run against #521's body."""
    fixture = FIXTURES / "issue_521_body.md"
    assert fixture.is_file(), f"missing fixture {fixture}"
    main_ref = _resolve_main_ref()
    result = subprocess.run(
        ["python3", str(SCRIPT_PATH), "--body-file", str(fixture), "--main-ref", main_ref],
        capture_output=True, text=True, cwd=REPO_ROOT, check=False,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "src/aelfrice/vocab_bridge.py" in out
    assert "VocabBridge" in out
    assert "tests/test_vocab_bridge.py" in out
    assert "docs/feature-hrr-vocab-bridge.md" in out


def test_no_false_positive_on_unrelated_module(tmp_path: Path):
    body = tmp_path / "body.md"
    body.write_text("Adds `src/aelfrice/this_module_will_not_ever_exist_zzz_123.py`.\n")
    main_ref = _resolve_main_ref()
    result = subprocess.run(
        ["python3", str(SCRIPT_PATH), "--body-file", str(body), "--main-ref", main_ref],
        capture_output=True, text=True, cwd=REPO_ROOT, check=False,
    )
    assert result.returncode == 0
    assert result.stdout == ""
