"""Adapter-side exit-code tests for #479.

Each canonical adapter is expected to `sys.exit(2)` when its data
source is missing or filters reduce the input set to zero rows. We
exercise this by monkey-patching the adapter's loader to either raise
FileNotFoundError or return an empty list, then invoking main() with
minimal arg vectors and asserting on the SystemExit code.

Per-adapter, we cover both shapes the issue calls out:
- data-dir absent → FileNotFoundError → exit(2)
- filters match no rows → empty list → exit(2)

The structmemeval adapter has its own discover_cases path that already
returns [] when the task directory is missing; we cover that with a
pure-empty-list fixture.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest


# Optional benchmark deps that some adapters import at module load.
# The dev/test environment installs only the core extras; CI uses the
# same posture. We stub these at sys.modules so adapter import works
# even without the `benchmarks` extra installed. The real loaders get
# replaced by the per-test patches before main() is invoked.
_OPTIONAL_DEP_STUBS = ("nltk", "nltk.stem", "datasets", "tiktoken")


def _stub_optional_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    import types
    for name in _OPTIONAL_DEP_STUBS:
        if name in sys.modules:
            continue
        stub = types.ModuleType(name)
        # nltk.stem.PorterStemmer surface used by locomo / mab adapters.
        if name == "nltk.stem":
            class _PorterStemmer:
                def stem(self, w: str) -> str:
                    return w
            stub.PorterStemmer = _PorterStemmer  # type: ignore[attr-defined]
        # datasets.load_dataset surface used by HF-backed adapters; the
        # patched loader gets called instead, so a no-op is fine.
        if name == "datasets":
            stub.load_dataset = lambda *a, **kw: []  # type: ignore[attr-defined]
        # tiktoken.encoding_for_model surface used by mab adapter.
        if name == "tiktoken":
            class _Enc:
                def encode(self, s: str) -> list[int]:
                    return [0] * len(s)
            stub.encoding_for_model = lambda *a, **kw: _Enc()  # type: ignore[attr-defined]
            stub.get_encoding = lambda *a, **kw: _Enc()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, name, stub)


def _run_adapter_main(monkeypatch: pytest.MonkeyPatch, module_name: str, argv: list[str], **patches: Any) -> int:
    """Import-fresh the module, patch its loader symbol(s), then call
    main() and return the SystemExit code (0 when main returns).
    """
    _stub_optional_deps(monkeypatch)
    import importlib
    mod = importlib.import_module(module_name)
    for name, fn in patches.items():
        monkeypatch.setattr(mod, name, fn)
    monkeypatch.setattr(sys, "argv", argv)
    try:
        mod.main()
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 0
    return 0


def test_structmemeval_exits_2_when_no_cases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rc = _run_adapter_main(
        monkeypatch,
        "benchmarks.structmemeval_adapter",
        ["structmemeval_adapter", "--task", "location", "--bench", "big",
         "--data", str(tmp_path / "missing")],
        discover_cases=lambda *a, **kw: [],
    )
    assert rc == 2


def test_locomo_exits_2_when_data_file_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _raise(*a: Any, **kw: Any) -> None:
        raise FileNotFoundError(str(tmp_path / "absent.json"))
    rc = _run_adapter_main(
        monkeypatch,
        "benchmarks.locomo_adapter",
        ["locomo_adapter", "--data", str(tmp_path / "absent.json")],
        load_locomo=_raise,
    )
    assert rc == 2


def test_locomo_exits_2_when_load_returns_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rc = _run_adapter_main(
        monkeypatch,
        "benchmarks.locomo_adapter",
        ["locomo_adapter", "--data", str(tmp_path / "empty.json")],
        load_locomo=lambda *a, **kw: [],
    )
    assert rc == 2


def test_longmemeval_exits_2_when_file_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _raise(*a: Any, **kw: Any) -> None:
        raise FileNotFoundError(str(tmp_path / "absent.json"))
    rc = _run_adapter_main(
        monkeypatch,
        "benchmarks.longmemeval_adapter",
        ["longmemeval_adapter", "--data", str(tmp_path / "absent.json")],
        load_from_file=_raise,
    )
    assert rc == 2


def test_longmemeval_exits_2_when_raw_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rc = _run_adapter_main(
        monkeypatch,
        "benchmarks.longmemeval_adapter",
        ["longmemeval_adapter", "--data", str(tmp_path / "empty.json")],
        load_from_file=lambda *a, **kw: [],
    )
    assert rc == 2


def test_mab_exits_2_when_load_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*a: Any, **kw: Any) -> None:
        raise FileNotFoundError("HuggingFace cache absent")
    rc = _run_adapter_main(
        monkeypatch,
        "benchmarks.mab_adapter",
        ["mab_adapter", "--split", "Conflict_Resolution"],
        load_mab_split=_raise,
    )
    assert rc == 2


def test_mab_exits_2_when_no_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    rc = _run_adapter_main(
        monkeypatch,
        "benchmarks.mab_adapter",
        ["mab_adapter", "--split", "Conflict_Resolution"],
        load_mab_split=lambda *a, **kw: [],
    )
    assert rc == 2


def test_mab_entity_index_exits_2_when_load_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(*a: Any, **kw: Any) -> None:
        raise FileNotFoundError("HuggingFace cache absent")
    rc = _run_adapter_main(
        monkeypatch,
        "benchmarks.mab_entity_index_adapter",
        ["mab_entity_index_adapter", "--split", "Conflict_Resolution"],
        load_mab_split=_raise,
    )
    assert rc == 2


def test_mab_entity_index_exits_2_when_no_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rc = _run_adapter_main(
        monkeypatch,
        "benchmarks.mab_entity_index_adapter",
        ["mab_entity_index_adapter", "--split", "Conflict_Resolution"],
        load_mab_split=lambda *a, **kw: [],
    )
    assert rc == 2


def test_amabench_exits_2_when_no_episodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rc = _run_adapter_main(
        monkeypatch,
        "benchmarks.amabench_adapter",
        ["amabench_adapter", "--max-episodes", "5"],
        load_amabench=lambda *a, **kw: [],
    )
    assert rc == 2
