"""Shared noun-phrase regex pattern.

Extracted from `triple_extractor` so that downstream consumers
(`entity_extractor`, etc.) can use the same NP shape without
introducing a module-import cycle. Leaf module — no aelfrice
imports.
"""
from __future__ import annotations

from typing import Final

_DET: Final[str] = (
    r"(?:the|a|an|our|their|its|this|that|these|those|my|your|his|her)"
)
_TOKEN: Final[str] = r"[A-Za-z][\w-]*"
_NP: Final[str] = (
    rf"(?:(?:{_DET})\s+)?{_TOKEN}(?:\s+{_TOKEN}){{0,4}}"
)

NOUN_PHRASE_PATTERN: Final[str] = _NP
