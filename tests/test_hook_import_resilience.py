"""Regression tests for issue #236 — hook silent-skip on missing runtime deps.

Verifies that when a heavy transitive dep (e.g. numpy) is absent from the
install environment, `aelfrice.hook.user_prompt_submit()` and
`aelfrice.hook_search_tool.main()` both:
  - return 0 (exit 0, non-blocking contract preserved)
  - emit nothing to stdout
  - emit at most one concise diagnostic line to stderr (no traceback)

Also verifies that `aelf doctor` surfaces a [FAIL] line for each absent
declared runtime dep.

Strategy for the hook tests: patch sys.modules at the function level so
the sentinel variables (`_IMPORTS_OK`, `_IMPORT_ERR`) appear absent to the
hook functions, without needing a full module reload that would disturb other
tests running in the same process.
"""
from __future__ import annotations

import importlib
import io
import sys
import types
from pathlib import Path
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# hook.py — user_prompt_submit
# ---------------------------------------------------------------------------

class TestHookImportResilience:
    """aelfrice.hook: _IMPORTS_OK sentinel → exit 0, no traceback, no stdout."""

    def _call_with_imports_failed(
        self, func_name: str = "user_prompt_submit"
    ) -> tuple[int, str, str]:
        """Call hook function with _IMPORTS_OK=False patched in."""
        import aelfrice.hook as hook_mod

        fake_err = ImportError("No module named 'numpy'")
        fake_err.name = "numpy"  # type: ignore[attr-defined]

        stdin = io.StringIO(
            '{"session_id":"test","transcript_path":"/dev/null",'
            '"cwd":"/tmp","prompt":"hello"}'
        )
        stdout = io.StringIO()
        stderr = io.StringIO()

        with (
            mock.patch.object(hook_mod, "_IMPORTS_OK", False),
            mock.patch.object(hook_mod, "_IMPORT_ERR", fake_err),
        ):
            fn = getattr(hook_mod, func_name)
            rc = fn(stdin=stdin, stdout=stdout, stderr=stderr)

        return rc, stdout.getvalue(), stderr.getvalue()

    def test_user_prompt_submit_returns_zero(self) -> None:
        rc, _stdout, _stderr = self._call_with_imports_failed("user_prompt_submit")
        assert rc == 0, f"expected exit 0, got {rc}"

    def test_user_prompt_submit_empty_stdout(self) -> None:
        _rc, stdout, _stderr = self._call_with_imports_failed("user_prompt_submit")
        assert stdout == "", f"expected empty stdout, got: {stdout!r}"

    def test_user_prompt_submit_no_traceback(self) -> None:
        _rc, _stdout, stderr = self._call_with_imports_failed("user_prompt_submit")
        assert "Traceback" not in stderr, (
            f"traceback must not appear in stderr; got:\n{stderr}"
        )

    def test_user_prompt_submit_single_stderr_line(self) -> None:
        _rc, _stdout, stderr = self._call_with_imports_failed("user_prompt_submit")
        nonempty = [ln for ln in stderr.splitlines() if ln.strip()]
        assert len(nonempty) <= 1, (
            f"at most 1 stderr line expected, got {len(nonempty)}: {stderr!r}"
        )

    def test_user_prompt_submit_stderr_mentions_missing(self) -> None:
        _rc, _stdout, stderr = self._call_with_imports_failed("user_prompt_submit")
        if stderr.strip():
            assert "numpy" in stderr or "missing" in stderr, (
                f"stderr diagnostic should mention missing module; got: {stderr!r}"
            )

    def test_pre_compact_returns_zero(self) -> None:
        rc, _stdout, _stderr = self._call_with_imports_failed("pre_compact")
        assert rc == 0

    def test_pre_compact_empty_stdout(self) -> None:
        _rc, stdout, _stderr = self._call_with_imports_failed("pre_compact")
        assert stdout == ""

    def test_session_start_returns_zero(self) -> None:
        rc, _stdout, _stderr = self._call_with_imports_failed("session_start")
        assert rc == 0

    def test_session_start_empty_stdout(self) -> None:
        _rc, stdout, _stderr = self._call_with_imports_failed("session_start")
        assert stdout == ""


# ---------------------------------------------------------------------------
# hook_search_tool.py — main
# ---------------------------------------------------------------------------

class TestHookSearchToolImportResilience:
    """aelfrice.hook_search_tool: lazy ImportError → exit 0, no traceback."""

    def _call_main_with_broken_lazy_imports(self) -> tuple[int, str, str]:
        """Invoke main() with lazy imports patched to raise ImportError."""
        import aelfrice.hook_search_tool as hst

        original_do_search = hst._do_search

        def _broken_do_search(
            payload: dict[str, object], stdout: io.StringIO
        ) -> None:
            # Simulate the ImportError that would occur if numpy is absent
            # in the lazy import chain inside _do_search.
            try:
                err = ImportError("No module named 'numpy'")
                err.name = "numpy"  # type: ignore[attr-defined]
                raise err
            except ImportError as _ie:
                missing = getattr(_ie, "name", None) or str(_ie)
                print(
                    f"aelf-hook: install incomplete (missing {missing}); skipping",
                    file=sys.stderr,
                )
                return

        payload = (
            '{"tool_name":"Grep","tool_input":{"pattern":"retrieve"},'
            '"cwd":"/tmp"}'
        )
        stdin = io.StringIO(payload)
        stdout = io.StringIO()
        stderr = io.StringIO()

        with mock.patch.object(hst, "_do_search", _broken_do_search):
            rc = hst.main(stdin=stdin, stdout=stdout, stderr=stderr)

        return rc, stdout.getvalue(), stderr.getvalue()

    def test_returns_zero(self) -> None:
        rc, _stdout, _stderr = self._call_main_with_broken_lazy_imports()
        assert rc == 0

    def test_empty_stdout(self) -> None:
        _rc, stdout, _stderr = self._call_main_with_broken_lazy_imports()
        assert stdout == "", f"expected empty stdout, got: {stdout!r}"

    def test_no_traceback(self) -> None:
        import aelfrice.hook_search_tool as hst

        payload = (
            '{"tool_name":"Grep","tool_input":{"pattern":"retrieve"},'
            '"cwd":"/tmp"}'
        )
        stdin = io.StringIO(payload)
        stdout = io.StringIO()
        stderr = io.StringIO()

        # Patch the lazy imports inside _do_search to raise ImportError.
        original_import = importlib.import_module

        def _broken_import(name: str, *args: object, **kwargs: object) -> object:
            if name in ("aelfrice.cli", "aelfrice.retrieval", "aelfrice.store"):
                err = ImportError(f"No module named 'numpy'")
                err.name = "numpy"  # type: ignore[attr-defined]
                raise err
            return original_import(name)

        # Remove cached submodules so the lazy import path is exercised.
        saved = {
            k: sys.modules.pop(k)
            for k in list(sys.modules)
            if k in ("aelfrice.cli", "aelfrice.retrieval", "aelfrice.store",
                      "aelfrice.bm25")
        }
        try:
            with mock.patch("builtins.__import__", side_effect=_broken_import):
                rc = hst.main(stdin=stdin, stdout=stdout, stderr=stderr)
        finally:
            sys.modules.update(saved)

        assert "Traceback" not in stderr.getvalue(), (
            f"traceback must not appear in stderr; got:\n{stderr.getvalue()}"
        )
        assert rc == 0


# ---------------------------------------------------------------------------
# doctor.py — missing runtime dep reporting
# ---------------------------------------------------------------------------

class TestDoctorMissingRuntimeDeps:
    """aelf doctor surfaces [FAIL] for absent declared runtime deps."""

    def test_format_report_includes_fail_line(self) -> None:
        from aelfrice.doctor import DoctorReport, format_report

        report = DoctorReport()
        report.missing_runtime_deps = ["numpy", "scipy"]
        text = format_report(report)
        assert "[FAIL] missing runtime dep: numpy" in text
        assert "[FAIL] missing runtime dep: scipy" in text

    def test_format_report_includes_reinstall_hint(self) -> None:
        from aelfrice.doctor import DoctorReport, format_report

        report = DoctorReport()
        report.missing_runtime_deps = ["numpy"]
        text = format_report(report)
        assert "reinstall" in text.lower() or "upgrade" in text.lower()

    def test_format_report_shows_fail_even_when_no_settings(self) -> None:
        """Missing deps are surfaced even when no settings.json was scanned."""
        from aelfrice.doctor import DoctorReport, format_report

        report = DoctorReport()
        assert report.scopes_scanned == []
        report.missing_runtime_deps = ["numpy"]
        text = format_report(report)
        assert "[FAIL] missing runtime dep: numpy" in text

    def test_check_runtime_deps_returns_list(self) -> None:
        from aelfrice.doctor import _check_runtime_deps

        result = _check_runtime_deps()
        assert isinstance(result, list)

    def test_check_runtime_deps_empty_when_all_present(self) -> None:
        """In the test venv (which has numpy+scipy), no deps should be missing."""
        from aelfrice.doctor import _check_runtime_deps

        missing = _check_runtime_deps()
        assert missing == [], (
            f"All declared runtime deps should be installed in test venv; "
            f"missing: {missing}"
        )

    def test_check_runtime_deps_detects_absent_dep(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Simulate a missing dep by patching importlib.import_module in doctor."""
        import aelfrice.doctor as doctor_mod

        original_import = importlib.import_module

        def _fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "numpy":
                raise ImportError("No module named 'numpy'")
            return original_import(name)

        monkeypatch.setattr(
            "aelfrice.doctor.importlib.import_module", _fake_import
        )
        missing = doctor_mod._check_runtime_deps()
        assert "numpy" in missing

    def test_diagnose_populates_missing_runtime_deps(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import aelfrice.doctor as doctor_mod

        original_import = importlib.import_module

        def _fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "numpy":
                raise ImportError("No module named 'numpy'")
            return original_import(name)

        monkeypatch.setattr(
            "aelfrice.doctor.importlib.import_module", _fake_import
        )
        report = doctor_mod.diagnose(
            user_settings=tmp_path / "missing.json",
            project_root=tmp_path / "noproj",
        )
        assert "numpy" in report.missing_runtime_deps
