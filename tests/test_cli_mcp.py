"""`aelf mcp` CLI subcommand — start the FastMCP stdio server.

The MCP server module ships in every install, but the FastMCP runtime
ships under the `[mcp]` extra. These tests verify that the subcommand
is wired and that invoking it without the extra produces an actionable
error rather than an opaque ImportError or silent no-op.
"""
from __future__ import annotations

import argparse
import io


def test_mcp_subcommand_registered() -> None:
    from aelfrice.cli import _known_cli_subcommands

    assert "mcp" in _known_cli_subcommands()


def test_cmd_mcp_returns_one_with_actionable_error_when_fastmcp_missing(
    capsys: object,
) -> None:
    """fastmcp is not in dev deps; _cmd_mcp must return 1 + stderr hint."""
    from aelfrice.cli import _cmd_mcp

    ns = argparse.Namespace()
    out = io.StringIO()
    rc = _cmd_mcp(ns, out)
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert rc == 1
    assert "aelfrice[mcp]" in captured.err
    assert "fastmcp" in captured.err.lower()


def test_mcp_server_module_has_main_guard() -> None:
    """`python -m aelfrice.mcp_server` is the documented fallback entry.

    Verify the module file contains the `__main__` guard so tooling that
    introspects the module can confirm the entry point exists. This is
    a structural check, not a runtime check (running the guard would
    block on stdio).
    """
    import aelfrice.mcp_server as mod

    src = open(mod.__file__, "r", encoding="utf-8").read()
    assert 'if __name__ == "__main__"' in src
    assert "serve()" in src.split('if __name__ == "__main__"')[1]


def test_mcp_subcommand_help_string_mentions_mcp_extra() -> None:
    """`aelf --help` should advertise that mcp needs the [mcp] extra."""
    from aelfrice.cli import build_parser

    parser = build_parser()
    buf = io.StringIO()
    parser.print_help(file=buf)
    text = buf.getvalue()
    assert "mcp" in text
    assert "aelfrice[mcp]" in text


def test_mcp_subcommand_help_short_circuits_invocation() -> None:
    """`aelf mcp --help` must not actually start the server."""
    from aelfrice.cli import build_parser

    parser = build_parser()
    try:
        parser.parse_args(["mcp", "--help"])
    except SystemExit as exc:
        # argparse's --help raises SystemExit(0); the test passes if we
        # got here without serve() ever running.
        assert exc.code == 0
        return
    assert False, "argparse --help should have raised SystemExit"


def test_python_dash_m_mcp_server_module_resolves() -> None:
    """`python -m aelfrice.mcp_server` resolves to the right module.

    We don't actually run it (would block on stdio); we just confirm the
    module is importable as a script target via runpy's name resolution.
    """
    import importlib.util

    spec = importlib.util.find_spec("aelfrice.mcp_server")
    assert spec is not None
    assert spec.name == "aelfrice.mcp_server"
    assert spec.origin and spec.origin.endswith("mcp_server.py")


# --- tool description coverage (FastMCP reads decorator-fn docstring) --


def test_server_passes_instructions_to_fastmcp() -> None:
    """The FastMCP constructor call inside serve() must pass an
    `instructions=` argument so hosts get a server-level overview of the
    tool surface. Static AST guard.
    """
    import ast
    import aelfrice.mcp_server as mod

    src = open(mod.__file__, "r", encoding="utf-8").read()
    tree = ast.parse(src)

    serve_fn = next(
        (n for n in tree.body
         if isinstance(n, ast.FunctionDef) and n.name == "serve"),
        None,
    )
    assert serve_fn is not None

    # Be lenient on which symbol invokes the constructor — search for any
    # call passing `name=` with a string arg, which is fastmcp's signature.
    constructor_calls = [
        node for node in ast.walk(serve_fn)
        if isinstance(node, ast.Call)
        and any(kw.arg == "name" for kw in node.keywords)
    ]
    assert constructor_calls, "no FastMCP-style constructor call found in serve()"

    have_instructions = [
        c for c in constructor_calls
        if any(kw.arg == "instructions" for kw in c.keywords)
    ]
    assert have_instructions, (
        "FastMCP(...) constructor missing instructions= kwarg — host LLMs "
        "will receive no server-level overview at registration time"
    )

    # Sanity: the module must define a non-empty _SERVER_INSTRUCTIONS.
    assert hasattr(mod, "_SERVER_INSTRUCTIONS")
    assert len(mod._SERVER_INSTRUCTIONS.strip()) > 100, (
        "_SERVER_INSTRUCTIONS too short to be useful (< 100 chars stripped)"
    )


def test_every_decorated_aelf_tool_has_annotations() -> None:
    """Every @mcp.tool() must pass an `annotations={...}` dict with the
    four MCP behavioral hints. Hosts use these to gate dangerous tools
    (e.g. require approval for destructiveHint=True). An unannotated
    tool defaults to destructiveHint=True and openWorldHint=True per
    spec — the worst-of-both-worlds default.

    Static AST guard: parse mcp_server.py, find every @mcp.tool() call
    inside serve(), assert the call kwargs include 'annotations' and
    that the annotations dict has all four required hint keys.
    """
    import ast
    import aelfrice.mcp_server as mod

    required_keys = {
        "readOnlyHint",
        "destructiveHint",
        "idempotentHint",
        "openWorldHint",
    }

    src = open(mod.__file__, "r", encoding="utf-8").read()
    tree = ast.parse(src)

    serve_fn = next(
        (n for n in tree.body
         if isinstance(n, ast.FunctionDef) and n.name == "serve"),
        None,
    )
    assert serve_fn is not None

    missing: list[str] = []
    incomplete: list[tuple[str, set[str]]] = []
    seen = 0
    for node in ast.walk(serve_fn):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            if (isinstance(dec, ast.Call)
                    and isinstance(dec.func, ast.Attribute)
                    and dec.func.attr == "tool"):
                seen += 1
                ann_kw = next(
                    (kw for kw in dec.keywords if kw.arg == "annotations"),
                    None,
                )
                if ann_kw is None:
                    missing.append(node.name)
                    break
                if not isinstance(ann_kw.value, ast.Dict):
                    missing.append(node.name)
                    break
                hint_keys = {
                    k.value for k in ann_kw.value.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)
                }
                if not required_keys.issubset(hint_keys):
                    incomplete.append((node.name, required_keys - hint_keys))
                break

    assert seen >= 12, f"expected >=12 @mcp.tool decorators, found {seen}"
    assert not missing, f"@mcp.tool decorators missing annotations=: {missing}"
    assert not incomplete, (
        f"@mcp.tool annotations missing required hint keys: {incomplete}"
    )


def test_every_decorated_aelf_tool_has_a_docstring() -> None:
    """FastMCP exposes the @mcp.tool-decorated function's docstring as the
    tool's `description`. An empty docstring means the host LLM gets no
    guidance on when/how to call the tool — discoverability collapses.

    This is a static guard: parse mcp_server.py with `ast`, find every
    `@mcp.tool()` decorator inside `serve()`, and assert the decorated
    function has a non-empty docstring.
    """
    import ast
    import aelfrice.mcp_server as mod

    src = open(mod.__file__, "r", encoding="utf-8").read()
    tree = ast.parse(src)

    serve_fn = next(
        (node for node in tree.body
         if isinstance(node, ast.FunctionDef) and node.name == "serve"),
        None,
    )
    assert serve_fn is not None, "serve() function not found in mcp_server.py"

    decorated_fns: list[ast.FunctionDef] = []
    for node in ast.walk(serve_fn):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                # Match `@mcp.tool()` — Call whose func is Attribute(attr='tool')
                if (isinstance(dec, ast.Call)
                        and isinstance(dec.func, ast.Attribute)
                        and dec.func.attr == "tool"):
                    decorated_fns.append(node)
                    break

    assert len(decorated_fns) >= 12, (
        f"expected at least 12 @mcp.tool functions, found {len(decorated_fns)}"
    )

    missing = [
        fn.name for fn in decorated_fns
        if not (ast.get_docstring(fn) or "").strip()
    ]
    assert not missing, (
        f"@mcp.tool functions missing docstrings (host LLM sees no "
        f"description): {missing}"
    )


# --- stdout discipline check (regression guard) -------------------------


def test_mcp_handlers_never_print_to_stdout() -> None:
    """stdio MCP servers must never write to stdout (it carries JSON-RPC).

    Static check: scan mcp_server.py source for `print(`. Any hit must be
    prefixed with `# allow-print` to be acceptable. This catches future
    regressions where someone adds a debug print without realizing the
    constraint.
    """
    import aelfrice.mcp_server as mod

    src = open(mod.__file__, "r", encoding="utf-8").read()
    for lineno, line in enumerate(src.splitlines(), 1):
        if "print(" in line and "# allow-print" not in line:
            # `print(` may appear in strings or comments — only fail on
            # actual statements. Crude heuristic: skip lines that are
            # inside a docstring or comment.
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"'):
                continue
            assert False, (
                f"mcp_server.py:{lineno} contains a print() call: {line!r}. "
                "stdio MCP servers must not write to stdout. Use stderr or "
                "fastmcp's logging API instead."
            )
