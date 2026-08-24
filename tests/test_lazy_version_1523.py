"""#1523: `aelfrice.__version__` resolves lazily.

The package used to call `importlib.metadata.version()` at import time, so
every hook process paid a metadata read in a fresh interpreter. These tests
pin both halves of the fix: the version still resolves, and importing the
package no longer drags `importlib.metadata` in.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_version_resolves() -> None:
    import aelfrice

    assert isinstance(aelfrice.__version__, str)
    assert aelfrice.__version__


def test_version_is_cached_after_first_access() -> None:
    import aelfrice

    first = aelfrice.__version__
    # The lazy hook writes the resolved value into module globals, so the
    # second read must not go through __getattr__ again.
    assert "__version__" in vars(aelfrice)
    assert aelfrice.__version__ is first


def test_unknown_attribute_still_raises() -> None:
    import aelfrice

    try:
        aelfrice.does_not_exist  # noqa: B018
    except AttributeError as exc:
        assert "does_not_exist" in str(exc)
    else:  # pragma: no cover - the raise above is the contract
        raise AssertionError("expected AttributeError for unknown attribute")


@pytest.mark.timeout(30)
def test_import_does_not_pull_importlib_metadata() -> None:
    """The perf contract: a bare `import aelfrice` stays off importlib.metadata.

    Asserted in a subprocess because the parent test session has almost
    certainly imported it already via pytest's own plugin machinery.
    """
    code = (
        "import sys; import aelfrice; "
        "assert 'importlib.metadata' not in sys.modules, "
        "'importlib.metadata imported by bare `import aelfrice`'; "
        "print('clean')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


@pytest.mark.timeout(30)
def test_version_access_still_works_in_fresh_interpreter() -> None:
    code = "import aelfrice; v = aelfrice.__version__; assert isinstance(v, str) and v; print(v)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()
