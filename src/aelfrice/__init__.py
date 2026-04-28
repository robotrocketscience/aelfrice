"""aelfrice: Bayesian memory designed for feedback-driven learning."""
try:
    from importlib.metadata import version as _meta_version, PackageNotFoundError
    __version__ = _meta_version("aelfrice")
except Exception:
    __version__ = "0.0.0"
