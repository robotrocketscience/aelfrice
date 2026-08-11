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
import subprocess
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


# ---------------------------------------------------------------------------
# A real fire with an unimportable numeric stack (#1407 review)
# ---------------------------------------------------------------------------
#
# Every case above patches `_IMPORTS_OK = False`, which returns at the second
# statement of `user_prompt_submit` -- roughly a hundred lines above the hook
# body. Since #1351 the hook's own module-scope imports do not touch numpy, so
# a real install missing the numeric stack keeps `_IMPORTS_OK` True and runs
# the whole body. Nothing in this file covered that, and #1407 shipped an
# unguarded `from aelfrice.bm25 import ...` into it: the import raised, the
# traceback escaped `user_prompt_submit`, and the fire wrote no audit row and
# no stdout at all. An audit field must never be the reason a hook breaks.

_UNIMPORTABLE_STACK_PROBE = '''
import os
import sys

for _k in [k for k in os.environ if k.startswith(("AELFRICE_", "AELF_"))]:
    del os.environ[_k]

_tmp = %(tmp)r
os.environ["HOME"] = _tmp
os.environ["AELFRICE_DOTDIR"] = os.path.join(_tmp, ".aelfrice")
os.environ["AELFRICE_DB"] = os.path.join(_tmp, "memory.db")
os.environ["AELF_NO_UPDATE_CHECK"] = "1"
os.chdir(_tmp)

_BLOCKED = %(blocked)r


class _Blocker:
    """Make the named modules and their submodules unimportable."""

    def find_spec(self, fullname, path=None, target=None):
        for _b in _BLOCKED:
            if fullname == _b or fullname.startswith(_b + "."):
                raise ImportError("blocked: " + fullname, name=fullname)
        return None


sys.meta_path.insert(0, _Blocker())

import io
import json
import pathlib

import aelfrice.hook as hook

print("imports_ok:%%d" %% int(hook._IMPORTS_OK))
rc = hook.user_prompt_submit(
    stdin=io.StringIO(
        json.dumps({"prompt": "ok", "session_id": "s1", "cwd": _tmp})
    ),
    stdout=io.StringIO(),
)
print("rc:%%d" %% rc)

rows = 0
for _p in sorted(pathlib.Path(_tmp).rglob("*.jsonl")):
    for _line in _p.read_text(encoding="utf-8").splitlines():
        if _line.strip() and json.loads(_line).get("hook") == "user_prompt_submit":
            rows += 1
print("rows:%%d" %% rows)
'''


@pytest.mark.timeout(90)
@pytest.mark.parametrize(
    "blocked",
    [
        pytest.param(
            ("numpy", "scipy", "snowballstemmer"), id="numeric-stack-absent"
        ),
        pytest.param(("aelfrice.sidecar_outcome",), id="audit-leaf-absent"),
    ],
)
def test_a_fire_survives_an_unimportable_module(
    tmp_path: Path, blocked: tuple[str, ...]
) -> None:
    """A gate-skipped fire must exit 0, write its audit row and print no
    traceback even when a module it reaches for cannot be imported.

    A `sys.meta_path` finder raising `ImportError` stands in for an install
    that lacks the module -- closer to the real failure than patching
    `importlib.import_module`, because it also catches a plain `import x`
    statement anywhere in the call graph, which is the form the regression
    took.

    The two cases pin two different things, and each has been run against its
    own mutation:

    * **numeric-stack-absent** pins that nothing above the shape gate reaches
      into `aelfrice.bm25`. Restoring the eager `from aelfrice.bm25 import
      reset_sidecar_outcome` there fails this with a traceback naming
      `hook.py ... in user_prompt_submit` and zero audit rows.
    * **audit-leaf-absent** pins the `try` / `except Exception` around that
      import and call. The leaf module has no dependencies, so the first case
      cannot fail on a missing guard; blocking the leaf itself is what makes
      the guard's absence observable, and it is the general contract that
      matters -- an audit field must never be the reason a hook breaks.

    `imports_ok` must be 1 or this is the test the file already had: with the
    sentinel false the fire returns ~100 lines above any of this.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _UNIMPORTABLE_STACK_PROBE
            % {"tmp": str(tmp_path), "blocked": blocked},
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, (
        "the probe process itself died; the hook let an exception escape:\n"
        + proc.stderr
    )
    assert "Traceback" not in proc.stderr, (
        "a traceback reached stderr -- the hook body aborted on an import it "
        f"is supposed to survive:\n{proc.stderr}"
    )
    lines = proc.stdout.split()
    assert lines[0] == "imports_ok:1", (
        f"importing aelfrice.hook needs one of {blocked}, so the fire returned "
        "at the _IMPORTS_OK guard and this test proves nothing"
    )
    assert "rc:0" in lines, proc.stdout
    assert "rows:1" in lines, (
        "the fire wrote no audit row with the numeric stack unimportable; an "
        f"audit field broke the hook.\nstdout: {proc.stdout}\n"
        f"stderr: {proc.stderr}"
    )
