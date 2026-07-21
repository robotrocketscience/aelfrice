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
