"""The transcript-admission harness identifies the corpus it measured (#1398).

This harness's population is a live archive: every session that ends appends
to it, and `turns.jsonl` is being written while the harness reads it. A
headline count from it is therefore unreproducible unless the run says which
bytes it saw, which is the failure #1398 was filed over.

Each test below fails on a different mutation of `_corpus_identity`:

* a digest that ignores file *content* (hashes only names, or nothing) —
  `test_the_digest_tracks_content`;
* a digest computed over a different file list than the one measured, the
  subtle version, where the number is certified against bytes nobody read —
  `test_the_digest_covers_exactly_the_measured_files`;
* an identity block that is computed but never reaches the report —
  `test_measure_publishes_the_identity`.
"""
from __future__ import annotations

import json
from pathlib import Path

from benchmarks import transcript_noise_admission as harness


def _corpus(root: Path, *, archive: list[str], live: list[str]) -> Path:
    """Write a transcript dir in the shape `_load` expects."""
    tdir = root / "transcripts"
    (tdir / "archive").mkdir(parents=True)

    def _rows(texts: list[str]) -> str:
        return "".join(
            json.dumps({"role": "user", "text": t}) + "\n" for t in texts
        )

    (tdir / "archive" / "a.jsonl").write_text(_rows(archive), encoding="utf-8")
    (tdir / "turns.jsonl").write_text(_rows(live), encoding="utf-8")
    return tdir


def test_the_digest_tracks_content(tmp_path: Path) -> None:
    """Same bytes, same digest; one more archived turn, different digest.

    The stability half matters as much as the sensitivity half: a digest
    seeded with the clock or with `id()` would pass "it changed" and still
    be useless for reproducing a figure.
    """
    one = _corpus(
        tmp_path / "one",
        archive=["No telemetry, no network calls, no accounts."],
        live=["Rebase onto main and re-run the gate."],
    )
    same = _corpus(
        tmp_path / "same",
        archive=["No telemetry, no network calls, no accounts."],
        live=["Rebase onto main and re-run the gate."],
    )
    grown = _corpus(
        tmp_path / "grown",
        archive=["No telemetry, no network calls, no accounts."],
        live=["Rebase onto main and re-run the gate.", "No failures yet."],
    )

    a = harness._corpus_identity(str(one))
    b = harness._corpus_identity(str(same))
    c = harness._corpus_identity(str(grown))

    assert a["corpus_sha256"] == b["corpus_sha256"]
    assert a["corpus_sha256"] != c["corpus_sha256"]
    assert a["corpus_files"] == c["corpus_files"] == 2
    assert c["corpus_bytes"] > a["corpus_bytes"]


def test_the_digest_covers_exactly_the_measured_files(tmp_path: Path) -> None:
    """The digest's file list is the list `_load` reads — not a wider glob.

    Two mutations this catches and a content-only assertion does not: a
    `_corpus_paths` that drops `turns.jsonl` (the live file, so every
    in-flight run would certify stale bytes), and one that sweeps in
    files `_load` never opens (so the digest moves for reasons the
    counts never see). Asserted by making each side move on its own.
    """
    tdir = _corpus(
        tmp_path,
        archive=["No telemetry, no network calls, no accounts."],
        live=["Rebase onto main and re-run the gate."],
    )
    paths = harness._corpus_paths(str(tdir))
    assert [Path(p).name for p in paths] == ["a.jsonl", "turns.jsonl"]
    assert harness._corpus_identity(str(tdir))["corpus_files"] == len(paths)

    before = harness._corpus_identity(str(tdir))["corpus_sha256"]
    prompts_before, _ = harness._load(str(tdir))

    # A file the harness does not read must not move the digest.
    (tdir / "notes.txt").write_text("scratch", encoding="utf-8")
    assert harness._corpus_identity(str(tdir))["corpus_sha256"] == before

    # The live file, which it does read, must.
    with (tdir / "turns.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"role": "user", "text": "No failures yet."}) + "\n")
    after = harness._corpus_identity(str(tdir))
    prompts_after, _ = harness._load(str(tdir))
    assert after["corpus_sha256"] != before
    assert len(prompts_after) == len(prompts_before) + 1


def test_measure_publishes_the_identity(tmp_path: Path) -> None:
    """Computing the identity and not reporting it is the same as not having it."""
    tdir = _corpus(
        tmp_path,
        archive=["No telemetry, no network calls, no accounts."],
        live=["Rebase onto main and re-run the gate."],
    )
    report = harness.measure(str(tdir))
    identity = report["corpus_identity"]
    assert identity["corpus_sha256"] == harness._corpus_identity(str(tdir))[
        "corpus_sha256"
    ]
    assert identity["corpus_files"] == 2
    assert set(identity) >= {"sha", "dirty", "dirty_paths", "transcripts"}
    # The report must be JSON-serialisable: `main` dumps it, and a Path or a
    # set in the identity block would only fail at the point of publication.
    json.dumps(report, sort_keys=True)
