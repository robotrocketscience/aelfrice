"""Unit tests for scripts/check-commit-msg.py prefix validation.

Each test is deterministic and completes well under 1 s.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Load the script as a module without installing it as a package.
# ---------------------------------------------------------------------------
_SCRIPT = Path(__file__).parent.parent / "scripts" / "check-commit-msg.py"


def _load() -> object:
    spec = importlib.util.spec_from_file_location("check_commit_msg", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_mod = _load()
validate_subject = _mod.validate_subject  # type: ignore[attr-defined]
main = _mod.main  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# validate_subject — direct unit tests
# ---------------------------------------------------------------------------

class TestValidSubject:
    def test_bare_prefix(self) -> None:
        assert validate_subject("feat: add thing") is True

    def test_fix_prefix(self) -> None:
        assert validate_subject("fix: correct off-by-one") is True

    def test_ci_prefix(self) -> None:
        assert validate_subject("ci: add staging gate job") is True

    def test_docs_prefix(self) -> None:
        assert validate_subject("docs: update CONTRIBUTING") is True

    def test_scoped_prefix(self) -> None:
        assert validate_subject("feat(cli): add --verbose flag") is True

    def test_scoped_breaking_prefix(self) -> None:
        assert validate_subject("feat(api)!: remove deprecated endpoint") is True

    def test_breaking_no_scope(self) -> None:
        assert validate_subject("refactor!: rename module") is True

    def test_all_allowed_types(self) -> None:
        allowed = (
            "feat", "fix", "perf", "refactor", "test", "docs", "build",
            "ci", "style", "revert", "exp", "chore", "release", "gate", "audit",
        )
        for t in allowed:
            assert validate_subject(f"{t}: some description") is True, t

    def test_merge_auto_subject_exempt(self) -> None:
        assert validate_subject("Merge pull request #42 from foo/bar") is True

    def test_revert_auto_subject_exempt(self) -> None:
        assert validate_subject("Revert \"feat: add thing\"") is True

    def test_leading_whitespace_stripped(self) -> None:
        assert validate_subject("  feat: stripped  ") is True


class TestInvalidSubject:
    def test_missing_colon(self) -> None:
        assert validate_subject("feat add thing") is False

    def test_unknown_prefix(self) -> None:
        assert validate_subject("wip: work in progress") is False

    def test_no_space_after_colon(self) -> None:
        assert validate_subject("feat:no space") is False

    def test_empty_subject(self) -> None:
        assert validate_subject("") is False

    def test_whitespace_only_subject(self) -> None:
        assert validate_subject("   ") is False

    def test_capitalised_type_rejected(self) -> None:
        # Type tokens must be lowercase.
        assert validate_subject("Feat: add thing") is False

    def test_bare_word(self) -> None:
        assert validate_subject("just a commit") is False

    def test_type_without_description(self) -> None:
        # "feat: " has a trailing space; strip() reduces it to "feat:" which
        # no longer matches the ": " separator — reasonably rejected.
        assert validate_subject("feat: ") is False

    def test_type_colon_no_space(self) -> None:
        assert validate_subject("feat:nodesc") is False


# ---------------------------------------------------------------------------
# main() — CLI interface (reads from file or --subject flag)
# ---------------------------------------------------------------------------

class TestMainCLI:
    def test_subject_flag_valid(self) -> None:
        assert main(["--subject", "ci: enforce prefix in CI"]) == 0

    def test_subject_flag_invalid(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["--subject", "wip: something"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "valid conventional-commit prefix" in captured.err

    def test_file_valid(self, tmp_path: Path) -> None:
        f = tmp_path / "COMMIT_EDITMSG"
        f.write_text("feat(scope): do something\n\nBody text here.\n")
        assert main([str(f)]) == 0

    def test_file_invalid(self, tmp_path: Path) -> None:
        f = tmp_path / "COMMIT_EDITMSG"
        f.write_text("bad commit message\n")
        assert main([str(f)]) == 1

    def test_file_merge_exempt(self, tmp_path: Path) -> None:
        f = tmp_path / "COMMIT_EDITMSG"
        f.write_text("Merge branch 'main' into my-feature\n")
        assert main([str(f)]) == 0

    def test_file_revert_exempt(self, tmp_path: Path) -> None:
        f = tmp_path / "COMMIT_EDITMSG"
        f.write_text('Revert "feat: add thing"\n')
        assert main([str(f)]) == 0

    def test_no_args_returns_usage_error(self) -> None:
        assert main([]) == 2

    def test_subject_flag_missing_value(self) -> None:
        assert main(["--subject"]) == 2

    def test_nonexistent_file(self) -> None:
        assert main(["/nonexistent/COMMIT_EDITMSG"]) == 2
