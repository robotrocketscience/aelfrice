"""aelfrice: Bayesian memory designed for feedback-driven learning."""

_LAZY_NAMES = ("__version__", "PackageNotFoundError")

# Declarations only — no value is bound, so attribute access still falls
# through to `__getattr__` below. They exist so static analysis keeps the
# types the eager block used to provide: without them `__version__` widens
# to the `__getattr__` return type and every downstream consumer of it
# degrades (4 new pyright errors across cli.py and llm_classifier.py).
# Deliberately not behind `if TYPE_CHECKING:` — that would import `typing`
# at module scope and spend part of the startup this change exists to save.
__version__: str
PackageNotFoundError: type[Exception]


def __dir__() -> list[str]:
    """Keep the lazy names discoverable (PEP 562).

    A module `__getattr__` does not participate in default directory
    enumeration, so without this `dir(aelfrice)` would omit `__version__`
    until something had already read it — an introspection surface that
    changed depending on access order.
    """
    return sorted(set(globals()) | set(_LAZY_NAMES))


def __getattr__(name: str) -> object:
    """Resolve the metadata names on first access (PEP 562).

    `importlib.metadata` reads distribution metadata off disk. Resolving the
    version at import time charged that cost to every hook process, in a
    fresh interpreter each fire, for an attribute almost none of them read.
    Resolved values are cached into module globals, so only the first access
    pays and subsequent lookups bypass this hook entirely.

    Two threads racing the first access both resolve and both write the same
    value; the write is idempotent, so the race costs a duplicated read and
    nothing else. It is deliberately unguarded — importing `threading` here
    to serialise it would reintroduce the import cost this defers.
    """
    if name == "__version__":
        try:
            from importlib.metadata import version as _meta_version

            resolved: object = _meta_version("aelfrice")
        except Exception:
            resolved = "0.0.0"
        globals()["__version__"] = resolved
        return resolved
    if name == "PackageNotFoundError":
        # Re-exported for compatibility: the eager block bound this into the
        # package namespace, so `from aelfrice import PackageNotFoundError`
        # worked even though no in-repo caller uses it. Kept lazy so the
        # compatibility costs nothing until someone actually asks for it.
        from importlib.metadata import PackageNotFoundError as _not_found

        globals()["PackageNotFoundError"] = _not_found
        return _not_found
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
