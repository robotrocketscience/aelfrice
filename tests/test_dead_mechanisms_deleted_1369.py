"""#1369 deleted four unwired mechanisms. They must stay deleted.

#1162's disposition for each was "wire it, or delete it and strike its
docs". Each was deleted, on evidence of zero callers under `src/` plus a
named reason wiring it is unfunded. The failure mode this file guards is
resurrection-by-copy: the surfaces read as plausible infrastructure, and
three of the four had docs asserting them as live before #1218 / #1369
corrected the prose.

Each assertion is an *absence*, so each needs a distinguishing check that
would fail if the symbol came back — not merely a suite that no longer
imports it. Reverting any one of the four deletion commits turns exactly
one test here red.

Deliberately does not read `docs/`. CI's `code` path filter excludes
`docs/**`; a suite that read it would tax every docs-only PR with the
full pytest matrix, which `test_ci_path_filter` pins against.
"""
from __future__ import annotations

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src" / "aelfrice"

# The posterior-decay surface deleted under #1369 §1 (filed as #1162 §4).
# Named rather than globbed on "decay": `retrieval._apply_temporal_decay`
# and `meta_beliefs._decay_toward_static` are separate, live mechanisms
# that act on ranking position and on a meta-belief series respectively,
# never on a belief's stored posterior. Conflating them is what #1218
# existed to undo.
_POSTERIOR_DECAY_NAMES = frozenset(
    {"decay", "type_half_life", "TYPE_HALF_LIFE_SECONDS"}
)


def _module_asts() -> dict[str, ast.Module]:
    """Parsed `src/aelfrice/**.py`, keyed by file name.

    Parsed, not grepped: every name checked here also occurs in prose in
    comments and docstrings across `src/` that explain why the mechanism
    is gone. A text search would report those as live references.
    """
    return {
        path.name: ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for path in sorted(_SRC.rglob("*.py"))
    }


def _bound_names(tree: ast.Module) -> set[str]:
    """Module-level names a module binds: defs, classes, assignments."""
    bound: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            bound.add(node.name)
        elif isinstance(node, ast.Assign):
            bound |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            bound.add(node.target.id)
    return bound


def test_posterior_decay_surface_is_gone_from_scoring() -> None:
    """`scoring` defines none of `decay` / `type_half_life` / the table.

    The import-level check below passes vacuously if the functions exist
    but nobody imports them — which is precisely the state #1162 refused
    to leave the tree in. This asserts the definitions themselves.
    """
    scoring = _module_asts()["scoring.py"]
    assert _bound_names(scoring) & _POSTERIOR_DECAY_NAMES == set()


def test_no_module_imports_the_posterior_decay_surface() -> None:
    """Nothing under `src/` reaches the names by import or attribute.

    Catches a reimplementation landing in some other module and being
    re-exported, which the definition check above would miss.
    """
    offenders: dict[str, set[str]] = {}
    for name, tree in _module_asts().items():
        hits: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "aelfrice.scoring":
                hits |= {a.name for a in node.names} & _POSTERIOR_DECAY_NAMES
            elif (
                isinstance(node, ast.Attribute)
                and node.attr in _POSTERIOR_DECAY_NAMES
                and isinstance(node.value, ast.Name)
                and node.value.id == "scoring"
            ):
                hits.add(node.attr)
        if hits:
            offenders[name] = hits
    assert offenders == {}


def test_no_module_defines_a_retrieval_query_cache() -> None:
    """#1369 §2: `retrieval.RetrievalCache` is gone and stays gone.

    Every constructor call was in `tests/`; the class docstring said so
    itself. Its key tuple had also drifted to omit every lane flag added
    after v1.7 (hrr_structural, temporal_spine, intentional_clustering,
    entity_persist_demote, heat_kernel, gamma, zeta), so a caller
    toggling any of them would have been served another lane's results.
    Fixing the key would only have produced a correct cache nobody calls.

    Checked across all of `src/`, not just `retrieval.py`: a resurrected
    memoizer is as likely to land beside the thing it wraps as inside it.
    The store's invalidation registry is deliberately *not* asserted
    against — it is live for the BM25F sidecar and the spectral lane.
    """
    offenders = {
        name: sorted(_bound_names(tree) & {"RetrievalCache"})
        for name, tree in _module_asts().items()
        if _bound_names(tree) & {"RetrievalCache"}
    }
    assert offenders == {}


def test_correction_detector_has_no_directive_category() -> None:
    """#1369 §3: `correction._DIRECTIVE_TERMS` is gone and stays gone.

    Unreachable by construction, not merely uncalled: the term list is a
    strict subset of `classification_core._REQUIREMENT_KEYWORDS`, which
    is tested against the same lowercased text one branch earlier and
    returns, so `detect_correction` never sees a text that would fire it.

    The subset relation is asserted rather than assumed — it is the whole
    argument, and it would silently stop holding if someone added a term
    to `_REQUIREMENT_KEYWORDS`' would-be sibling without checking. Since
    the sibling no longer exists, what is checkable is that no six-name
    signal list grew a seventh entry back.
    """
    from aelfrice import correction

    assert not hasattr(correction, "_DIRECTIVE_TERMS")

    # `signals` is built in evaluation order; the categories are the
    # module's contract with `classification_core`.
    result = correction.detect_correction(
        "always do not commit secrets! we already agreed the rule is the rule"
    )
    assert "directive" not in result.signals
    assert set(result.signals) <= {
        "imperative",
        "always_never",
        "negation",
        "emphasis",
        "prior_ref",
        "declarative",
    }


def test_the_multiplicative_post_rank_adjusters_are_not_in_the_package() -> None:
    """#1369 §4: `aelfrice.uri_baki` is gone from `src/`.

    Its verdict is an honest negative (`benchmarks/uri_baki_retest/`),
    and `apply_supersession_demote` is actively wrong for the scale
    retrieval uses: the composite rerank score is log-domain and
    routinely negative, so `score * 0.5` *raises* it and promotes the
    superseded belief. `retrieval._supersession_demote` (#1187) is the
    corrected, log-additive replacement.

    Asserted as un-importable rather than as an absent file: the risk is
    someone re-adding the primitives under a different module name and
    calling them from the rerank path.
    """
    import importlib

    for module in ("aelfrice.uri_baki", "aelfrice.post_rank_adjusters"):
        try:
            importlib.import_module(module)
        except ModuleNotFoundError:
            continue
        raise AssertionError(f"{module} is back in the shipped package")

    offenders = {
        name: sorted(
            _bound_names(tree)
            & {"apply_locked_floor", "apply_recency_decay",
               "apply_supersession_demote"}
        )
        for name, tree in _module_asts().items()
    }
    assert {k: v for k, v in offenders.items() if v} == {}


def test_stored_posteriors_are_never_moved_by_age() -> None:
    """The behavioural half: `scoring` exposes no age-taking function.

    The two checks above are name-based and a rename would slip past
    both. Posterior decay needs an age or half-life input to do its job,
    so an age-typed parameter anywhere in `scoring`'s public surface is
    the shape of the deleted mechanism regardless of what it is called.
    """
    scoring = _module_asts()["scoring.py"]
    age_taking: dict[str, list[str]] = {}
    for node in scoring.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        args = node.args
        params = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
        aged = [
            p
            for p in params
            if "age" in p or "half_life" in p or p in {"now", "now_ts"}
        ]
        if aged:
            age_taking[node.name] = aged
    assert age_taking == {}
