"""`empty_parameter_set_mark = "fail_at_collect"` stays set in pyproject.toml.

pytest's default for a `parametrize` over an empty list is to collect one
test and skip it. A skip is a non-failing outcome nothing gates on, so a
helper that builds its case list from a fixture file or a filter and comes
back empty turns every test it feeds green without running an assertion
(#1786). `fail_at_collect` reports the empty set as a collection error in
the file that produced it.

Two pins: the TOML, so a revert of the line fails here, and the live
session config, so an override from another config file fails here too.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

EXPECTED = "fail_at_collect"


def test_pyproject_sets_empty_parameter_set_mark() -> None:
    with _PYPROJECT.open("rb") as fh:
        config = tomllib.load(fh)
    options = config["tool"]["pytest"]["ini_options"]
    assert options.get("empty_parameter_set_mark") == EXPECTED, (
        "an empty parametrize list collects as a skip unless pyproject.toml "
        f'sets empty_parameter_set_mark = "{EXPECTED}"'
    )


def test_session_honors_empty_parameter_set_mark(
    pytestconfig: pytest.Config,
) -> None:
    assert pytestconfig.getini("empty_parameter_set_mark") == EXPECTED, (
        "the running session did not read empty_parameter_set_mark from "
        "pyproject.toml; another config file is taking precedence"
    )
