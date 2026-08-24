"""aelfrice: Bayesian memory designed for feedback-driven learning."""


def __getattr__(name: str) -> str:
    """Resolve `__version__` on first access (PEP 562).

    `importlib.metadata` reads distribution metadata off disk. Resolving the
    version at import time charged that cost to every hook process, in a
    fresh interpreter each fire, for an attribute almost none of them read.
    The resolved value is cached into module globals, so only the first
    access pays and subsequent lookups bypass this hook entirely.
    """
    if name == "__version__":
        try:
            from importlib.metadata import version as _meta_version

            resolved = _meta_version("aelfrice")
        except Exception:
            resolved = "0.0.0"
        globals()["__version__"] = resolved
        return resolved
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
