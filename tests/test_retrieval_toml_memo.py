"""#1135 TOML parse memo: cached per (mtime_ns, size), rewrite honoured."""
from __future__ import annotations

from pathlib import Path

import aelfrice.retrieval as retrieval


def test_toml_flag_rewrite_is_honoured(tmp_path: Path) -> None:
    cfg = tmp_path / ".aelfrice.toml"
    cfg.write_text("[retrieval]\nuse_bfs = true\n")
    assert retrieval._read_toml_flag_for("use_bfs", start=tmp_path) is True
    cfg.write_text("[retrieval]\nuse_bfs = false\n")
    assert retrieval._read_toml_flag_for("use_bfs", start=tmp_path) is False
    cfg.unlink()
    assert retrieval._read_toml_flag_for("use_bfs", start=tmp_path) is None


def test_toml_parse_is_memoized(tmp_path: Path, monkeypatch) -> None:
    cfg = tmp_path / ".aelfrice.toml"
    cfg.write_text("[retrieval]\nposterior_weight = 0.4\n")
    parses: list[int] = []
    import tomllib

    real_loads = tomllib.loads

    def counting_loads(*args: object, **kwargs: object) -> object:
        parses.append(1)
        return real_loads(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(retrieval.tomllib, "loads", counting_loads)
    for _ in range(10):
        assert retrieval._read_toml_float_for(
            "posterior_weight", start=tmp_path,
        ) == 0.4
    assert len(parses) <= 1, f"parsed {len(parses)}x for an unchanged file"


def _count_config_walks(monkeypatch) -> list[int]:
    """Count real `.aelfrice.toml` discovery walks, not resolver calls.

    A wall-clock assertion would re-flake on the next slow machine; the
    defect in #1289 is a count, so this counts.
    """
    walks: list[int] = []
    real_is_file = Path.is_file

    def counting_is_file(self: Path) -> bool:
        if self.name == retrieval.CONFIG_FILENAME:
            walks.append(1)
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", counting_is_file)
    return walks


def test_one_retrieval_walks_for_the_config_once(monkeypatch) -> None:
    """#1289: the walk was per flag, so it scaled with the number of flags.

    ~22 `[retrieval]` resolvers each re-walked from cwd to the first
    `.aelfrice.toml`. The parse was memoized; the discovery was not. The
    cost was O(flags x path depth) per `retrieve()` and grew with every lane
    flag added — 22 walks and 173 `posix.stat` calls on a 5-belief store.

    Counted against a one-walk baseline measured in the same process rather
    than against a fixed number, so the assertion does not encode this
    checkout's path depth.
    """
    probes = _count_config_walks(monkeypatch)
    retrieval._read_toml_flag_for("use_bfs")
    one_walk = len(probes)
    assert one_walk > 0, "baseline measured no filesystem probes"

    probes.clear()
    with retrieval.config_discovery_scope():
        for key in ("use_bfs", "use_hrr_expand", "use_entity_expand"):
            retrieval._read_toml_flag_for(key)
        retrieval._read_toml_float_for("posterior_weight")
    assert len(probes) == one_walk, (
        f"{len(probes)} probes for 4 resolvers; one walk costs {one_walk}"
    )


def test_adding_a_flag_does_not_add_a_walk(monkeypatch) -> None:
    """The property that stops this regrowing (#1289 acceptance).

    Holding the probe count at exactly one walk for 4 resolvers and again
    for 12 is what makes the cost independent of the flag count, rather
    than merely smaller than it was.
    """
    probes = _count_config_walks(monkeypatch)
    retrieval._read_toml_flag_for("use_bfs")
    one_walk = len(probes)

    probes.clear()
    with retrieval.config_discovery_scope():
        for _ in range(12):
            retrieval._read_toml_flag_for("use_bfs")
    assert len(probes) == one_walk, (
        f"{len(probes)} probes for 12 resolvers; one walk costs {one_walk}"
    )


def test_a_config_created_between_retrievals_is_still_seen(
    tmp_path: Path,
) -> None:
    """The staleness case a process-lifetime cache would have broken.

    The memo is scoped to one retrieval precisely so that a `.aelfrice.toml`
    written after the first call is honoured by the second. A cache keyed on
    the process would return the stale "no config found" until restart.
    """
    with retrieval.config_discovery_scope():
        assert retrieval._read_toml_flag_for("use_bfs", start=tmp_path) is None

    (tmp_path / ".aelfrice.toml").write_text("[retrieval]\nuse_bfs = true\n")

    with retrieval.config_discovery_scope():
        assert retrieval._read_toml_flag_for("use_bfs", start=tmp_path) is True


def test_a_config_deleted_between_retrievals_is_still_seen(
    tmp_path: Path,
) -> None:
    """The mirror case: a removed config must stop applying."""
    cfg = tmp_path / ".aelfrice.toml"
    cfg.write_text("[retrieval]\nuse_bfs = true\n")
    with retrieval.config_discovery_scope():
        assert retrieval._read_toml_flag_for("use_bfs", start=tmp_path) is True

    cfg.unlink()

    with retrieval.config_discovery_scope():
        assert retrieval._read_toml_flag_for("use_bfs", start=tmp_path) is None


def test_the_memo_does_not_leak_across_scopes(tmp_path: Path) -> None:
    """Distinguishing arm: the memo must be off outside a scope.

    Without this, the two tests above would pass for the wrong reason — a
    memo that never caches anything also never goes stale. Inside one scope
    a mid-scope write is deliberately *not* observed, which is what makes
    the cross-scope freshness above a real property.
    """
    with retrieval.config_discovery_scope():
        assert retrieval._read_toml_flag_for("use_bfs", start=tmp_path) is None
        (tmp_path / ".aelfrice.toml").write_text(
            "[retrieval]\nuse_bfs = true\n",
        )
        assert retrieval._read_toml_flag_for("use_bfs", start=tmp_path) is None

    assert retrieval._read_toml_flag_for("use_bfs", start=tmp_path) is True
