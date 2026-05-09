"""Tests for the @mcp.tool-decorated wrapper layer in mcp_server.serve().

The pure-handler tests (tests/test_mcp_server.py) cover the business
logic. These tests cover the wiring layer that the host LLM actually
sees: tool registration, schema generation from Pydantic Field hints,
annotation propagation, error path through fastmcp.

Two strategies:

1. **Static AST guards** — work without fastmcp installed. Verify each
   `aelf_*` wrapper inside serve() delegates to its matching `tool_*`
   pure handler and manages the store lifetime via try/finally.

2. **fastmcp shim** — when fastmcp is unavailable in dev, install a
   minimal fake into sys.modules so `serve()` runs and we can capture
   registered tools. This catches drift in the registration call shape
   that the AST-only tests would miss.
"""
from __future__ import annotations

import ast
import importlib
import sys
import types
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Static AST guards
# ---------------------------------------------------------------------------


def _serve_function() -> ast.FunctionDef:
    import aelfrice.mcp_server as mod

    src = open(mod.__file__, "r", encoding="utf-8").read()
    tree = ast.parse(src)
    serve_fn = next(
        (n for n in tree.body
         if isinstance(n, ast.FunctionDef) and n.name == "serve"),
        None,
    )
    assert serve_fn is not None, "serve() not found"
    return serve_fn


_EXPECTED_TOOLS: dict[str, str] = {
    "aelf_onboard": "tool_onboard",
    "aelf_search": "tool_search",
    "aelf_lock": "tool_lock",
    "aelf_locked": "tool_locked",
    "aelf_demote": "tool_demote",
    "aelf_validate": "tool_validate",
    "aelf_unlock": "tool_unlock",
    "aelf_promote": "tool_promote",
    "aelf_feedback": "tool_feedback",
    "aelf_confirm": "tool_confirm",
    "aelf_stats": "tool_stats",
    "aelf_health": "tool_health",
}


def _wrapper_funcs(serve_fn: ast.FunctionDef) -> dict[str, ast.FunctionDef]:
    """Return mapping of aelf_* wrapper name -> ast.FunctionDef."""
    out: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(serve_fn):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("aelf_"):
            out[node.name] = node
    return out


def test_all_expected_wrappers_present() -> None:
    serve_fn = _serve_function()
    wrappers = _wrapper_funcs(serve_fn)
    missing = set(_EXPECTED_TOOLS) - set(wrappers)
    assert not missing, f"missing wrapper functions in serve(): {missing}"


def test_each_wrapper_calls_its_matching_pure_handler() -> None:
    """aelf_X must internally call tool_X with the store. Catches typos
    where a wrapper accidentally points at the wrong handler."""
    serve_fn = _serve_function()
    wrappers = _wrapper_funcs(serve_fn)

    bad: list[tuple[str, str]] = []
    for wrapper_name, expected_handler in _EXPECTED_TOOLS.items():
        fn = wrappers[wrapper_name]
        called = {
            node.func.id
            for node in ast.walk(fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
        }
        if expected_handler not in called:
            bad.append((wrapper_name, expected_handler))
    assert not bad, (
        f"wrappers not delegating to expected handlers: {bad}"
    )


def test_each_wrapper_opens_and_closes_store() -> None:
    """Wrapper bodies must open a store via _open_default_store() and
    close it in a finally block. Catches resource-leak refactors."""
    serve_fn = _serve_function()
    wrappers = _wrapper_funcs(serve_fn)

    leaks: list[str] = []
    for name, fn in wrappers.items():
        opens = any(
            isinstance(c, ast.Call)
            and isinstance(c.func, ast.Name)
            and c.func.id == "_open_default_store"
            for c in ast.walk(fn)
        )
        closes_in_finally = False
        for try_node in (n for n in ast.walk(fn) if isinstance(n, ast.Try)):
            for stmt in try_node.finalbody:
                for c in ast.walk(stmt):
                    if (
                        isinstance(c, ast.Call)
                        and isinstance(c.func, ast.Attribute)
                        and c.func.attr == "close"
                    ):
                        closes_in_finally = True
        if not (opens and closes_in_finally):
            leaks.append(
                f"{name} (opens={opens}, closes_in_finally={closes_in_finally})"
            )
    assert not leaks, (
        f"wrappers missing store lifetime hygiene: {leaks}"
    )


# ---------------------------------------------------------------------------
# fastmcp shim — capture tool registrations dynamically
# ---------------------------------------------------------------------------


class _CapturedTool:
    __slots__ = ("fn", "annotations")

    def __init__(self, fn: Any, annotations: dict[str, Any] | None) -> None:
        self.fn = fn
        self.annotations = annotations or {}


class _FakeFastMCP:
    """Minimal stand-in for fastmcp.FastMCP. Records every decorated
    function and the kwargs passed to @mcp.tool(...)."""

    def __init__(self, **server_kwargs: Any) -> None:
        self.server_kwargs = server_kwargs
        self.tools: dict[str, _CapturedTool] = {}

    def tool(self, **decorator_kwargs: Any):
        annotations = decorator_kwargs.get("annotations")

        def decorator(fn: Any) -> Any:
            self.tools[fn.__name__] = _CapturedTool(fn, annotations)
            return fn

        return decorator

    # serve() also calls mcp.run() at the end. Make it a no-op so the
    # test exits cleanly instead of blocking on stdio.
    def run(self) -> None:
        return None


@pytest.fixture
def fastmcp_shim(
    monkeypatch: pytest.MonkeyPatch,
) -> _FakeFastMCP:
    """Install a fake fastmcp module + return the shim instance the
    serve() under test will receive (after we call it)."""
    captured: dict[str, _FakeFastMCP] = {}

    fake_module = types.ModuleType("fastmcp")

    class _FakeFactory:
        def __call__(self, **kwargs: Any) -> _FakeFastMCP:
            shim = _FakeFastMCP(**kwargs)
            captured["instance"] = shim
            return shim

    fake_module.FastMCP = _FakeFactory()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastmcp", fake_module)

    # serve() also imports `pydantic.Field` lazily. pydantic is a
    # transitive dep of fastmcp; provide a minimal stand-in if it's
    # absent in dev. Real Field is a callable returning a sentinel; for
    # AST/registration purposes any callable works.
    if "pydantic" not in sys.modules:
        fake_pydantic = types.ModuleType("pydantic")
        fake_pydantic.Field = lambda *args, **kwargs: None  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "pydantic", fake_pydantic)

    # Force a fresh import so serve() re-binds against the fake fastmcp.
    if "aelfrice.mcp_server" in sys.modules:
        importlib.reload(sys.modules["aelfrice.mcp_server"])

    from aelfrice.mcp_server import serve

    serve()
    shim = captured.get("instance")
    assert shim is not None, "_FakeFastMCP was never instantiated"
    return shim


def test_shim_registers_all_twelve_tools(fastmcp_shim: _FakeFastMCP) -> None:
    expected = set(_EXPECTED_TOOLS)
    got = set(fastmcp_shim.tools)
    missing = expected - got
    extra = got - expected
    assert not missing, f"tools never registered: {missing}"
    assert not extra, f"unexpected tools registered: {extra}"


def test_shim_every_tool_has_annotations_dict(fastmcp_shim: _FakeFastMCP) -> None:
    bad = [
        name for name, captured in fastmcp_shim.tools.items()
        if not captured.annotations
    ]
    assert not bad, f"tools registered without annotations: {bad}"


def test_shim_annotation_keys_are_complete(fastmcp_shim: _FakeFastMCP) -> None:
    required = {"readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"}
    incomplete: list[tuple[str, set[str]]] = []
    for name, captured in fastmcp_shim.tools.items():
        missing = required - set(captured.annotations)
        if missing:
            incomplete.append((name, missing))
    assert not incomplete, (
        f"tools missing required annotation keys: {incomplete}"
    )


def test_shim_read_only_set_matches_expected(fastmcp_shim: _FakeFastMCP) -> None:
    """Catch annotation drift: the read-only set is canonical."""
    expected_read_only = {"aelf_search", "aelf_locked", "aelf_stats", "aelf_health"}
    got_read_only = {
        name for name, captured in fastmcp_shim.tools.items()
        if captured.annotations.get("readOnlyHint") is True
    }
    assert got_read_only == expected_read_only, (
        f"readOnlyHint set drifted: expected {expected_read_only}, "
        f"got {got_read_only}"
    )


def test_shim_destructive_set_matches_expected(fastmcp_shim: _FakeFastMCP) -> None:
    """Only aelf_demote is annotated destructiveHint=True."""
    expected_destructive = {"aelf_demote"}
    got_destructive = {
        name for name, captured in fastmcp_shim.tools.items()
        if captured.annotations.get("destructiveHint") is True
    }
    assert got_destructive == expected_destructive


def test_shim_server_received_instructions_kwarg(
    fastmcp_shim: _FakeFastMCP,
) -> None:
    instructions = fastmcp_shim.server_kwargs.get("instructions")
    assert instructions, "FastMCP(instructions=) not passed to constructor"
    assert isinstance(instructions, str)
    assert len(instructions.strip()) > 100


def test_shim_server_name_is_aelfrice(fastmcp_shim: _FakeFastMCP) -> None:
    assert fastmcp_shim.server_kwargs.get("name") == "aelfrice"
