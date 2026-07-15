"""Unit tests for the pure `aelfrice.category` module (#1126).

Covers trigger (de)serialization, deterministic matching, name/lock
validation, and the default-off config gate. No store, no disk.
"""
from __future__ import annotations

import pytest

from aelfrice import category as cat


# --- CategoryTrigger round-trip ----------------------------------------


def test_trigger_json_roundtrip() -> None:
    t = cat.CategoryTrigger(
        keywords=("commit and push", "rebase"),
        tool_globs=("git push*",),
        file_globs=("tests/**",),
    )
    back = cat.CategoryTrigger.from_json(t.to_json())
    assert back == t


def test_trigger_json_is_stable_sorted_keys() -> None:
    t = cat.CategoryTrigger(keywords=("a",), tool_globs=("b",))
    # Byte-stable: sorted keys, compact separators.
    assert t.to_json() == '{"file_globs":[],"keywords":["a"],"tool_globs":["b"]}'


@pytest.mark.parametrize("raw", ["", None, "not json", "[1,2,3]", "42", "{}"])
def test_trigger_from_json_tolerant(raw: str | None) -> None:
    # Malformed / wrong-typed input degrades to an empty trigger, never raises.
    assert cat.CategoryTrigger.from_json(raw) == cat.CategoryTrigger()


def test_trigger_from_json_drops_bad_entries() -> None:
    t = cat.CategoryTrigger.from_json(
        '{"keywords": ["ok", "", 5, null, "  "], "tool_globs": "notalist"}'
    )
    assert t.keywords == ("ok",)
    assert t.tool_globs == ()


def test_trigger_is_empty() -> None:
    assert cat.CategoryTrigger().is_empty()
    assert not cat.CategoryTrigger(keywords=("x",)).is_empty()


# --- name / lock validation --------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [("git-workflow", "git-workflow"), ("  Repo_Rules ", "repo_rules"), ("a1", "a1")],
)
def test_normalize_name_ok(raw: str, expected: str) -> None:
    assert cat.normalize_name(raw) == expected


@pytest.mark.parametrize("bad", ["", "  ", "-leading", "has space", "sym$bol", "_us"])
def test_normalize_name_rejects(bad: str) -> None:
    with pytest.raises(ValueError):
        cat.normalize_name(bad)


def test_normalize_default_lock() -> None:
    assert cat.normalize_default_lock(None) == cat.DEFAULT_LOCK_NONE
    assert cat.normalize_default_lock("LOCKED") == "locked"
    with pytest.raises(ValueError):
        cat.normalize_default_lock("bogus")


# --- keyword matching ---------------------------------------------------


def test_keyword_hit_word_boundary() -> None:
    trig = cat.CategoryTrigger(keywords=("push",))
    assert cat.keyword_hit("please push the branch", trig)
    # word-boundary: 'pushover' must NOT match 'push'
    assert not cat.keyword_hit("this is a pushover", trig)


def test_keyword_hit_case_insensitive() -> None:
    trig = cat.CategoryTrigger(keywords=("commit",))
    assert cat.keyword_hit("COMMIT this now", trig)


def test_keyword_hit_phrase_whitespace_tolerant() -> None:
    trig = cat.CategoryTrigger(keywords=("commit and push",))
    assert cat.keyword_hit("please commit and push now", trig)
    assert cat.keyword_hit("commit  and   push", trig)  # collapsed whitespace
    assert not cat.keyword_hit("commit or push", trig)


def test_keyword_hit_empty_inputs() -> None:
    assert not cat.keyword_hit("", cat.CategoryTrigger(keywords=("x",)))
    assert not cat.keyword_hit("something", cat.CategoryTrigger())


# --- glob matching (command / file lanes) ------------------------------


def test_command_hit() -> None:
    trig = cat.CategoryTrigger(tool_globs=("git push*",))
    assert cat.command_hit("git push origin main", trig)
    assert not cat.command_hit("git status", trig)
    assert not cat.command_hit("git push origin main", cat.CategoryTrigger())


def test_paths_hit() -> None:
    trig = cat.CategoryTrigger(file_globs=("test_*.py",))
    assert cat.paths_hit(["test_foo.py", "src/x.py"], trig)
    assert not cat.paths_hit(["src/x.py"], trig)


# --- match_prompt (the v1 lane) ----------------------------------------


def _cats() -> list[cat.Category]:
    return [
        cat.Category(name="repo-rules", always_on=True),
        cat.Category(
            name="git-workflow",
            trigger=cat.CategoryTrigger(keywords=("push", "commit")),
        ),
        cat.Category(
            name="prose",
            trigger=cat.CategoryTrigger(keywords=("write",)),
        ),
    ]


def test_match_prompt_always_on_always_fires() -> None:
    fired = cat.match_prompt("nothing relevant here", _cats())
    assert [c.name for c in fired] == ["repo-rules"]


def test_match_prompt_keyword_fires() -> None:
    fired = cat.match_prompt("please push the branch", _cats())
    # deterministic name-ASC order: git-workflow, repo-rules
    assert [c.name for c in fired] == ["git-workflow", "repo-rules"]


def test_match_prompt_multiple_and_deterministic() -> None:
    fired = cat.match_prompt("write docs then commit", _cats())
    assert [c.name for c in fired] == ["git-workflow", "prose", "repo-rules"]
    # Re-running yields identical order (byte-determinism).
    again = cat.match_prompt("write docs then commit", _cats())
    assert [c.name for c in fired] == [c.name for c in again]


def test_match_prompt_dedup_by_name() -> None:
    dupes = [
        cat.Category(name="x", always_on=True),
        cat.Category(name="x", trigger=cat.CategoryTrigger(keywords=("push",))),
    ]
    fired = cat.match_prompt("push", dupes)
    assert [c.name for c in fired] == ["x"]


# --- config gate --------------------------------------------------------


def test_is_enabled_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(cat.ENV_ENABLED, raising=False)
    assert cat.is_enabled(None) is False
    assert cat.is_enabled({}) is False
    assert cat.is_enabled({"belief_categories": {"enabled": False}}) is False


def test_is_enabled_via_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(cat.ENV_ENABLED, raising=False)
    assert cat.is_enabled({"belief_categories": {"enabled": True}}) is True


def test_is_enabled_non_bool_config_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(cat.ENV_ENABLED, raising=False)
    # truthy-but-not-bool must NOT enable (mirrors sentiment gate).
    assert cat.is_enabled({"belief_categories": {"enabled": "yes"}}) is False


def test_is_enabled_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(cat.ENV_ENABLED, "1")
    assert cat.is_enabled(None) is True
    monkeypatch.setenv(cat.ENV_ENABLED, "off")
    assert cat.is_enabled({"belief_categories": {"enabled": True}}) is False


# --- seed set -----------------------------------------------------------


def test_seed_categories_wellformed() -> None:
    names = [c.name for c in cat.SEED_CATEGORIES]
    assert names == [
        "repo-rules",
        "git-workflow",
        "secrets-and-safety",
        "prose-and-docs",
        "testing",
    ]
    for c in cat.SEED_CATEGORIES:
        assert cat.normalize_name(c.name) == c.name
        assert c.default_lock in cat.DEFAULT_LOCK_VALUES
        # each seed either fires always-on or carries a keyword lane
        assert c.always_on or c.trigger.keywords
