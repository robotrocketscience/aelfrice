"""User locks get a share of a hook fire, not the whole of it (#1944).

User-scope locks are injected in every repository. Before this change
they spent the same payload as the repository's locks and the retrieval
hits, so a user store of 70 locks filled every prompt's block with locks.
These tests pin that user locks past their share don't push out a
retrieval hit or a repository lock, that they still fill room nothing
else wants, that a cut one is named, that a share of 1 is the old
behavior, that retrieval charges user locks only up to the share, and
that `aelf doctor` warns when the user store holds more than the share
shows.

Every path is a temp path. The conftest pins `AELFRICE_DB` and
`AELFRICE_USER_DB` at a sandbox; each test repoints both.
"""

from __future__ import annotations

import io
import json
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice import user_locks as ul
from aelfrice.doctor import (
    DoctorReport,
    _format_user_lock_share_section,  # pyright: ignore[reportPrivateUsage]
    diagnose_user_lock_share,
)
from aelfrice.hook import user_prompt_submit
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.retrieval import retrieve
from aelfrice.store import MemoryStore

N_USER = 40
USER_PAD = "keep the user rule in mind for every change you make. " * 6


def _lock(bid: str, content: str, *, locked_at: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=9.0,
        beta=0.5,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_USER,
        locked_at=locked_at,
        created_at="2026-10-01T00:00:00+00:00",
        last_retrieved_at=None,
    )


def _plain(bid: str, content: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-10-01T00:00:00+00:00",
        last_retrieved_at=None,
    )


def _seed(path: Path, beliefs: list[Belief]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    s = MemoryStore(str(path))
    try:
        for b in beliefs:
            s.insert_belief(b)
    finally:
        s.close()


def _user_locks(n: int = N_USER) -> list[Belief]:
    # Newest first is the store's order, so u00 renders first.
    return [
        _lock(f"u{i:02d}", f"User rule {i}: {USER_PAD}",
              locked_at=f"2026-10-{28 - (i % 27):02d}T{i % 24:02d}:00:00+00:00")
        for i in range(n)
    ]


def _repo_beliefs() -> list[Belief]:
    out: list[Belief] = [
        _lock("r1", "This repository builds with make, not cmake.",
              locked_at="2026-10-01T00:00:00+00:00"),
    ]
    for i in range(20):
        out.append(_plain(
            f"p{i:02d}",
            f"widget calibration note {i}: the widget calibration step "
            f"runs before the widget assembly step on line {i}.",
        ))
    return out


@pytest.fixture()
def stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """(repo_db, user_db), both temp paths, neither created yet."""
    repo = tmp_path / "repo" / "memory.db"
    user = tmp_path / "home" / ".aelfrice" / "user" / "memory.db"
    repo.parent.mkdir(parents=True)
    monkeypatch.setenv("AELFRICE_DB", str(repo))
    monkeypatch.setenv("AELFRICE_USER_DB", str(user))
    monkeypatch.delenv("AELFRICE_HOOK_AUDIT", raising=False)
    monkeypatch.delenv(ul.ENV_USER_LOCK_SHARE, raising=False)
    monkeypatch.delenv("AELFRICE_HOOK_BLOCK_CEILING", raising=False)
    monkeypatch.delenv("AELFRICE_RETRIEVAL_TOKEN_BUDGET", raising=False)
    monkeypatch.chdir(tmp_path)
    return repo, user


@pytest.fixture(autouse=True)
def _reset_user_lock_cache() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    ul._WARNED.clear()  # pyright: ignore[reportPrivateUsage]
    ul._CACHE.clear()  # pyright: ignore[reportPrivateUsage]
    yield
    ul._WARNED.clear()  # pyright: ignore[reportPrivateUsage]
    ul._CACHE.clear()  # pyright: ignore[reportPrivateUsage]


def _prompt(tmp_path: Path, prompt: str, session: str = "s1944") -> str:
    out, err = io.StringIO(), io.StringIO()
    payload = json.dumps({
        "prompt": prompt, "session_id": session, "cwd": str(tmp_path),
    })
    assert user_prompt_submit(
        stdin=io.StringIO(payload), stdout=out, stderr=err,
    ) == 0
    assert "Traceback" not in err.getvalue()
    return out.getvalue()


def _rendered(text: str, prefix: str) -> list[str]:
    return sorted(set(re.findall(rf'<belief id="({prefix}\d+)"', text)))


def _named(text: str) -> int:
    m = re.search(r"aelfrice: (\d+) user lock\(s\) did not fit", text)
    return int(m.group(1)) if m else 0


# --- the share -------------------------------------------------------------


def test_fit_is_first_fit_in_render_order() -> None:
    kept, past = ul.fit_user_lock_share(
        [("a", 40), ("b", 70), ("c", 30), ("d", 31)], 100,
    )
    assert kept == ["a", "c"]
    assert past == ["b", "d"]


def test_fit_keeps_a_lock_that_lands_exactly_on_the_share() -> None:
    kept, past = ul.fit_user_lock_share([("a", 60), ("b", 40)], 100)
    assert (kept, past) == (["a", "b"], [])


def test_share_resolves_env_then_toml_then_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ul.ENV_USER_LOCK_SHARE, raising=False)
    monkeypatch.chdir(tmp_path)
    assert ul.resolve_user_lock_share(start=tmp_path) == ul.DEFAULT_USER_LOCK_SHARE
    (tmp_path / ".aelfrice.toml").write_text("[retrieval]\nuser_lock_share = 0.5\n")
    assert ul.resolve_user_lock_share(start=tmp_path) == 0.5
    monkeypatch.setenv(ul.ENV_USER_LOCK_SHARE, "0.2")
    assert ul.resolve_user_lock_share(start=tmp_path) == 0.2


@pytest.mark.parametrize("raw", ["1.5", "-0.1", "lots"])
def test_a_bad_env_share_warns_and_falls_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".aelfrice.toml").write_text("[retrieval]\nuser_lock_share = 0.5\n")
    monkeypatch.setenv(ul.ENV_USER_LOCK_SHARE, raw)
    err = io.StringIO()
    assert ul.resolve_user_lock_share(start=tmp_path, stderr=err) == 0.5
    assert ul.ENV_USER_LOCK_SHARE in err.getvalue()


def test_a_bad_toml_share_warns_and_uses_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ul.ENV_USER_LOCK_SHARE, raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".aelfrice.toml").write_text("[retrieval]\nuser_lock_share = 2\n")
    err = io.StringIO()
    assert ul.resolve_user_lock_share(start=tmp_path, stderr=err) == (
        ul.DEFAULT_USER_LOCK_SHARE)
    assert "user_lock_share" in err.getvalue()


def test_the_share_bounds_are_both_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    for raw, want in (("0", 0.0), ("1", 1.0)):
        monkeypatch.setenv(ul.ENV_USER_LOCK_SHARE, raw)
        assert ul.resolve_user_lock_share(start=tmp_path) == want


# --- the prompt hook ---------------------------------------------------------


@pytest.mark.timeout(120)
def test_user_locks_past_the_share_leave_the_hits_their_room(
    stores: tuple[Path, Path], tmp_path: Path,
) -> None:
    """The issue's failure: a large user store took the hits' room."""
    repo, user = stores
    _seed(repo, _repo_beliefs())
    _seed(user, _user_locks())
    text = _prompt(tmp_path, "widget calibration step order")
    hits = _rendered(text, "p")
    shown = _rendered(text, "u")
    assert len(hits) >= 5, hits
    assert 'id="r1"' in text
    assert 0 < len(shown) < N_USER
    # Every user lock is shown or named.
    assert len(shown) + _named(text) == N_USER


@pytest.mark.timeout(120)
def test_a_share_of_one_is_the_old_behavior(
    stores: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, user = stores
    _seed(repo, _repo_beliefs())
    _seed(user, _user_locks())
    monkeypatch.setenv(ul.ENV_USER_LOCK_SHARE, "1")
    text = _prompt(tmp_path, "widget calibration step order")
    # The user locks take the room, as before #1944, and push out hits.
    assert len(_rendered(text, "u")) > 10
    shared = _rendered(text, "p")
    monkeypatch.setenv(ul.ENV_USER_LOCK_SHARE, "0.35")
    ul._CACHE.clear()  # pyright: ignore[reportPrivateUsage]
    text2 = _prompt(tmp_path, "widget calibration step order", session="s2")
    assert len(_rendered(text2, "p")) > len(shared)


@pytest.mark.timeout(120)
def test_user_locks_past_the_share_fill_room_nothing_else_wants(
    stores: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no hit competing, the share changes nothing."""
    repo, user = stores
    _seed(repo, [_lock("r1", "This repository builds with make, not cmake.",
                       locked_at="2026-10-01T00:00:00+00:00")])
    _seed(user, _user_locks())
    monkeypatch.setenv(ul.ENV_USER_LOCK_SHARE, "1")
    full = _rendered(_prompt(tmp_path, "zebra quokka"), "u")
    monkeypatch.setenv(ul.ENV_USER_LOCK_SHARE, "0")
    ul._CACHE.clear()  # pyright: ignore[reportPrivateUsage]
    none_reserved = _rendered(_prompt(tmp_path, "zebra quokka", session="s2"), "u")
    assert len(full) > 10
    assert none_reserved == full


@pytest.mark.timeout(120)
def test_repository_locks_keep_their_room(
    stores: tuple[Path, Path], tmp_path: Path,
) -> None:
    repo, user = stores
    repo_locks = [
        _lock(f"r{i:02d}", f"Repository rule {i}: run the widget build first.",
              locked_at=f"2026-09-{i + 1:02d}T00:00:00+00:00")
        for i in range(8)
    ]
    _seed(repo, repo_locks + _repo_beliefs()[1:])
    _seed(user, _user_locks())
    text = _prompt(tmp_path, "widget calibration step order")
    assert len(_rendered(text, "r")) == 8
    assert len(_rendered(text, "p")) >= 5
    assert _rendered(text, "u")


@pytest.mark.timeout(120)
def test_user_locks_that_fit_the_room_still_yield_to_hits_past_the_share(
    stores: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The locks alone fit the room; with the hits they don't."""
    repo, user = stores
    long_hits = [
        _plain(f"p{i:02d}", f"widget calibration note {i}: "
               + "the widget calibration step runs before assembly. " * 7)
        for i in range(20)
    ]
    _seed(repo, long_hits)
    _seed(user, _user_locks(12))
    monkeypatch.setenv(ul.ENV_USER_LOCK_SHARE, "1")
    old = _prompt(tmp_path, "widget calibration step order")
    assert len(_rendered(old, "u")) == 12
    monkeypatch.delenv(ul.ENV_USER_LOCK_SHARE)
    ul._CACHE.clear()  # pyright: ignore[reportPrivateUsage]
    text = _prompt(tmp_path, "widget calibration step order", session="s2")
    assert len(_rendered(text, "p")) > len(_rendered(old, "p"))
    assert len(_rendered(text, "u")) + _named(text) == 12


@pytest.mark.timeout(120)
def test_the_gate_skip_branch_keeps_the_core_lane_its_room(
    stores: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A short first prompt skips retrieval; its `<core>` lane is the hit."""
    repo, user = stores
    core: list[Belief] = []
    for i in range(12):
        b = _plain(f"c{i:02d}", f"core fact {i}: the build uses make and runs tests.")
        b.alpha, b.beta = 9.0, 1.0
        core.append(b)
    _seed(repo, core)
    _seed(user, _user_locks())
    monkeypatch.setenv(ul.ENV_USER_LOCK_SHARE, "1")
    old = _rendered(_prompt(tmp_path, "ok"), "c")
    monkeypatch.delenv(ul.ENV_USER_LOCK_SHARE)
    ul._CACHE.clear()  # pyright: ignore[reportPrivateUsage]
    text = _prompt(tmp_path, "ok", session="s2")
    assert "<core>" in text
    assert len(_rendered(text, "c")) > len(old)
    assert len(_rendered(text, "u")) + _named(text) == N_USER


@pytest.mark.timeout(120)
def test_user_locks_past_the_share_never_shed_the_recent_work_lane(
    stores: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aelfrice import hook

    repo, user = stores
    _seed(repo, [_lock("r1", "This repository builds with make, not cmake.",
                       locked_at="2026-10-01T00:00:00+00:00")])
    _seed(user, _user_locks())
    recent = ("<recent-work>\n<branch>main</branch>\n<commits>\n"
              + "".join(f'<commit sha="{i:07d}">fix the widget build {i}</commit>\n'
                        for i in range(30))
              + "</commits>\n</recent-work>")
    def fixed_recent_work(cwd: Path | None = None, commit_limit: int = 0) -> str:
        return recent

    monkeypatch.setattr(hook, "_build_recent_work_subblock", fixed_recent_work)
    monkeypatch.setenv(ul.ENV_USER_LOCK_SHARE, "1")
    assert "<recent-work>" not in _prompt(tmp_path, "ok")
    monkeypatch.delenv(ul.ENV_USER_LOCK_SHARE)
    ul._CACHE.clear()  # pyright: ignore[reportPrivateUsage]
    text = _prompt(tmp_path, "ok", session="s2")
    assert "<recent-work>" in text
    assert len(_rendered(text, "u")) + _named(text) == N_USER


def _prompt_from(
    payload_cwd: Path, prompt: str, session: str, err: io.StringIO,
) -> str:
    out = io.StringIO()
    payload = json.dumps({
        "prompt": prompt, "session_id": session, "cwd": str(payload_cwd),
    })
    assert user_prompt_submit(
        stdin=io.StringIO(payload), stdout=out, stderr=err,
    ) == 0
    return out.getvalue()


@pytest.mark.timeout(120)
def test_the_share_resolves_from_the_payload_cwd_not_the_process_cwd(
    stores: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Project config follows the payload's cwd (#909/#887), as the rest does."""
    repo, user = stores
    _seed(repo, _repo_beliefs())
    _seed(user, _user_locks())
    project = tmp_path / "project"
    elsewhere = tmp_path / "elsewhere"
    project.mkdir()
    elsewhere.mkdir()
    (project / ".aelfrice.toml").write_text("[retrieval]\nuser_lock_share = 1\n")
    (elsewhere / ".aelfrice.toml").write_text("[retrieval]\nuser_lock_share = 0\n")
    monkeypatch.chdir(elsewhere)
    text = _prompt_from(project, "widget calibration step order", "cwd1",
                        io.StringIO())
    # Share 1: the user locks take the room, as before #1944.
    assert len(_rendered(text, "u")) > 10
    assert len(_rendered(text, "p")) <= 2
    monkeypatch.chdir(project)
    ul._CACHE.clear()  # pyright: ignore[reportPrivateUsage]
    text = _prompt_from(elsewhere, "widget calibration step order", "cwd2",
                        io.StringIO())
    # Share 0: user locks take only room the hits leave.
    assert len(_rendered(text, "p")) >= 5


@pytest.mark.timeout(120)
def test_a_bad_share_warns_once_per_prompt_on_the_hook_stderr(
    stores: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, user = stores
    _seed(repo, _repo_beliefs())
    _seed(user, _user_locks())
    monkeypatch.setenv(ul.ENV_USER_LOCK_SHARE, "lots")
    err = io.StringIO()
    text = _prompt_from(tmp_path, "widget calibration step order", "bad1", err)
    assert err.getvalue().count(ul.ENV_USER_LOCK_SHARE) == 1
    assert ul.ENV_USER_LOCK_SHARE not in capsys.readouterr().err
    # It fell through to the default share.
    assert len(_rendered(text, "p")) >= 5
    # The pin ends with the fire: a later read resolves the env again.
    assert ul.resolve_user_lock_share(stderr=io.StringIO()) == (
        ul.DEFAULT_USER_LOCK_SHARE)
    monkeypatch.setenv(ul.ENV_USER_LOCK_SHARE, "0.5")
    assert ul.resolve_user_lock_share() == 0.5


@pytest.mark.timeout(120)
def test_a_user_lock_sharing_a_repo_id_is_not_capped_as_a_user_lock(
    stores: tuple[Path, Path],
) -> None:
    from aelfrice.hook import (
        _user_scope_lock_ids,  # pyright: ignore[reportPrivateUsage]
    )

    repo, user = stores
    same = "This repository builds with make, not cmake."
    _seed(repo, [_lock("r1", same, locked_at="2026-10-01T00:00:00+00:00")])
    _seed(user, [_lock("r1", same, locked_at="2026-10-01T00:00:00+00:00"),
                 _lock("u1", "Something else.", locked_at="2026-10-01T00:00:00+00:00")])
    s = MemoryStore(str(repo))
    try:
        assert _user_scope_lock_ids(s) == frozenset({"u1"})
    finally:
        s.close()


# --- retrieval ---------------------------------------------------------------


def _hit_count(repo: Path, budget: int) -> int:
    s = MemoryStore(str(repo))
    try:
        ids = [b.id for b in retrieve(s, "widget calibration", token_budget=budget)]
    finally:
        s.close()
    return len([i for i in ids if i.startswith("p")])


@pytest.mark.timeout(120)
def test_retrieval_charges_user_locks_only_up_to_the_share(
    stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, user = stores
    _seed(repo, _repo_beliefs())
    _seed(user, _user_locks())
    monkeypatch.setenv(ul.ENV_USER_LOCK_SHARE, "1")
    floor_only = _hit_count(repo, 1500)
    monkeypatch.setenv(ul.ENV_USER_LOCK_SHARE, "0.1")
    assert _hit_count(repo, 1500) > floor_only


@pytest.mark.timeout(120)
def test_the_structural_lane_charges_user_locks_up_to_the_share(
    stores: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aelfrice.hrr_index import HRRStructIndexCache
    from aelfrice.models import EDGE_CONTRADICTS, Edge
    from aelfrice.retrieval import retrieve_v2

    repo, _user = stores
    targets = [_plain(f"b{i:02d}", f"structural note {i} " * 8) for i in range(30)]
    _seed(repo, [_plain("hub", "hub")] + targets)
    s = MemoryStore(str(repo))
    try:
        for t in targets:
            s.insert_edge(Edge(src=t.id, dst="hub", type=EDGE_CONTRADICTS,
                               weight=1.0))
    finally:
        s.close()
    users = _user_locks(6)
    for u in users:
        setattr(u, ul.USER_SCOPE_ATTR, ul.USER_SCOPE_VALUE)

    def tail(share: str) -> int:
        monkeypatch.setenv(ul.ENV_USER_LOCK_SHARE, share)
        s = MemoryStore(str(repo))
        try:
            result = retrieve_v2(
                s, "CONTRADICTS:hub", budget=600,
                use_hrr_structural=True,
                hrr_struct_index_cache=HRRStructIndexCache(store=s, dim=512, seed=42),
                user_locks=users,
            )
        finally:
            s.close()
        assert result.locked_ids == [u.id for u in users]
        return len(result.beliefs) - len(users)

    floor_only = tail("1")
    assert tail("0.1") > floor_only


def test_l0_charge_is_the_plain_sum_without_user_locks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ul.ENV_USER_LOCK_SHARE, "0")
    locks = [_lock(f"r{i}", "x" * 40, locked_at="2026-10-01T00:00:00+00:00")
             for i in range(3)]
    assert ul.l0_charge(locks, lambda b: 100, 150) == 300


def test_l0_charge_caps_only_the_user_part(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ul.ENV_USER_LOCK_SHARE, "0.5")
    repo = [_lock("r1", "x", locked_at="2026-10-01T00:00:00+00:00")]
    users = [_lock(f"u{i}", "y", locked_at="2026-10-01T00:00:00+00:00")
             for i in range(4)]
    for u in users:
        setattr(u, ul.USER_SCOPE_ATTR, ul.USER_SCOPE_VALUE)
    # Repository 100 in full; users 400 capped at 0.5 * 300 = 150.
    assert ul.l0_charge(repo + users, lambda b: 100, 300) == 250
    # Under the cap, users are charged what they cost.
    assert ul.l0_charge(repo + users[:1], lambda b: 100, 300) == 200


# --- doctor ------------------------------------------------------------------


def _share_lines(report: DoctorReport) -> list[str]:
    lines: list[str] = []
    _format_user_lock_share_section(report, lines)
    return lines


@pytest.mark.timeout(60)
def test_doctor_warns_when_the_user_store_outgrows_its_share(
    stores: tuple[Path, Path], tmp_path: Path,
) -> None:
    repo, user = stores
    _seed(repo, _repo_beliefs())
    _seed(user, _user_locks())
    st = diagnose_user_lock_share(str(repo), str(user), project_root=tmp_path)
    assert st.known and st.total == N_USER
    assert 0 < st.fit < N_USER
    assert st.share_chars == int(9500 * ul.DEFAULT_USER_LOCK_SHARE)
    lines = _share_lines(DoctorReport(user_lock_share=st))
    text = "\n".join(lines)
    assert f"holds {N_USER} locks" in text
    assert f"about {st.fit} fit" in text
    assert "aelf unlock --user" in text


@pytest.mark.timeout(60)
def test_doctor_counts_the_locks_the_hook_shows_within_the_share(
    stores: tuple[Path, Path], tmp_path: Path,
) -> None:
    """Doctor's estimate is the hook's own count on a fire with no other block."""
    repo, user = stores
    _seed(repo, _repo_beliefs())
    _seed(user, _user_locks())
    st = diagnose_user_lock_share(str(repo), str(user), project_root=tmp_path)
    cost = ul.user_lock_render_chars(_user_locks()[0])
    assert st.fit == st.share_chars // cost


@pytest.mark.timeout(60)
def test_doctor_does_not_count_an_expired_user_lock(
    stores: tuple[Path, Path], tmp_path: Path,
) -> None:
    repo, user = stores
    _seed(repo, _repo_beliefs())
    locks = _user_locks(3)
    locks[0].lock_expires_at = "2026-10-02T00:00:00+00:00"
    _seed(user, locks)
    st = diagnose_user_lock_share(
        str(repo), str(user), project_root=tmp_path,
        now="2026-11-01T00:00:00+00:00",
    )
    assert st.total == 2


@pytest.mark.timeout(60)
def test_doctor_reads_the_share_from_the_project_root(
    stores: tuple[Path, Path], tmp_path: Path,
) -> None:
    repo, user = stores
    _seed(repo, _repo_beliefs())
    _seed(user, _user_locks(12))
    project = tmp_path / "project"
    project.mkdir()
    st = diagnose_user_lock_share(str(repo), str(user), project_root=project)
    assert st.fit < 12
    (project / ".aelfrice.toml").write_text("[retrieval]\nuser_lock_share = 1\n")
    st = diagnose_user_lock_share(str(repo), str(user), project_root=project)
    assert st.share == 1.0
    assert st.share_chars == 9500
    assert st.fit == st.total == 12


@pytest.mark.timeout(60)
def test_doctor_leaves_out_user_locks_the_hooks_drop_as_duplicates(
    stores: tuple[Path, Path], tmp_path: Path,
) -> None:
    repo, user = stores
    rule = "This repository builds with make, not cmake."
    _seed(repo, _repo_beliefs())
    _seed(user, _user_locks(3) + [
        _lock("r1", rule, locked_at="2026-10-01T00:00:00+00:00"),
        _lock("dup1", rule.replace(".", "!"), locked_at="2026-10-01T00:00:00+00:00"),
    ])
    st = diagnose_user_lock_share(str(repo), str(user), project_root=tmp_path)
    assert st.total == 3


@pytest.mark.timeout(60)
def test_doctor_is_silent_when_every_user_lock_fits(
    stores: tuple[Path, Path], tmp_path: Path,
) -> None:
    repo, user = stores
    _seed(repo, _repo_beliefs())
    _seed(user, _user_locks(2))
    st = diagnose_user_lock_share(str(repo), str(user), project_root=tmp_path)
    assert st.known and st.total == 2 and st.fit == 2
    assert _share_lines(DoctorReport(user_lock_share=st)) == []


@pytest.mark.timeout(60)
def test_doctor_never_creates_a_missing_user_store(
    stores: tuple[Path, Path], tmp_path: Path,
) -> None:
    repo, user = stores
    _seed(repo, _repo_beliefs())
    st = diagnose_user_lock_share(str(repo), str(user), project_root=tmp_path)
    assert st.known and not st.user_store_present and st.total == 0
    assert not user.exists()
    assert _share_lines(DoctorReport(user_lock_share=st)) == []


@pytest.mark.timeout(60)
def test_doctor_skips_a_user_store_that_is_the_repo_store(
    stores: tuple[Path, Path], tmp_path: Path,
) -> None:
    repo, _user = stores
    _seed(repo, _repo_beliefs() + _user_locks())
    st = diagnose_user_lock_share(str(repo), str(repo), project_root=tmp_path)
    assert st.known and st.total == 0
