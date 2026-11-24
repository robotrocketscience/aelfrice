#!/usr/bin/env python3
"""Probe which Antigravity CLI (agy) hook events fire and which outputs reach the model.

Usage:
    agy_hook_probe.py --hook --state DIR             # hook mode: agy runs this; reads stdin JSON
    agy_hook_probe.py --install WS --state DIR [--dry-run] [--decoys]
    agy_hook_probe.py --uninstall WS [--dry-run]
    agy_hook_probe.py --scenario NAME --state DIR [--dry-run]
    agy_hook_probe.py --report --state DIR [--transcripts DIR]
    agy_hook_probe.py --self-test

--install writes WS/.agents/hooks.json wiring the six documented agy events
(SessionStart, PreToolUse, PostToolUse, PreInvocation, PostInvocation, Stop)
to this script. --decoys also wires Claude-only event keys (UserPromptSubmit,
PreCompact, PostCompact, SessionEnd, Notification) under a second hook name,
to confirm agy never fires them.

--scenario writes DIR/scenario.json, which tells hook mode what to print per
event: an injectSteps step type carrying a unique AELFPROBE-<EVENT>-<hex>
marker, a PreToolUse stdout variant, a Stop "continue", and an exit code.
Hook mode appends one record per fire to DIR/fires.jsonl (payload keys,
truncated payload, hook-process env var NAMES only, emitted markers).
--report lists events fired and which emitted markers appear in the agy
transcripts and in DIR/out-*.txt (captured `agy -p` stdout), i.e. which
output channels reach the model.

Stdlib only. Records environment variable names, never values. Exits
non-zero on any failure outside hook mode.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, cast

EVENTS: Final[tuple[str, ...]] = (
    "SessionStart", "PreToolUse", "PostToolUse",
    "PreInvocation", "PostInvocation", "Stop",
)
TOOL_EVENTS: Final[frozenset[str]] = frozenset({"PreToolUse", "PostToolUse"})
DECOY_EVENTS: Final[tuple[str, ...]] = (
    "UserPromptSubmit", "PreCompact", "PostCompact", "SessionEnd", "Notification",
)
HOOK_NAME: Final[str] = "aelf-probe"
DECOY_NAME: Final[str] = "aelf-probe-decoy"
PAYLOAD_CAP: Final[int] = 4096
MAX_LOG_LINES: Final[int] = 100_000
SCRIPT: Final[Path] = Path(__file__).resolve()
TRANSCRIPT_ROOT: Final[Path] = Path.home() / ".gemini" / "antigravity-cli" / "brain"

# Scenario: event -> rule. Rule keys: inject (systemMessage|userMessage|
# ephemeralMessage|None), pretool (allow|empty_object|no_output|deny|
# auto_approve), stop_continue (bool), exit (int).
SCENARIOS: Final[dict[str, dict[str, dict[str, object]]]] = {
    "inject-system": {
        "SessionStart": {"inject": "systemMessage"},
        "PreInvocation": {"inject": "systemMessage"},
        "PostInvocation": {"inject": "systemMessage"},
        "PreToolUse": {"pretool": "allow"},
    },
    "inject-user": {
        "SessionStart": {"inject": "userMessage"},
        "PreInvocation": {"inject": "userMessage"},
        "PreToolUse": {"pretool": "allow"},
    },
    "inject-ephemeral": {
        "SessionStart": {"inject": "ephemeralMessage"},
        "PreInvocation": {"inject": "ephemeralMessage"},
        "PreToolUse": {"pretool": "allow"},
    },
    "pretool-empty-object": {"PreToolUse": {"pretool": "empty_object"}},
    "pretool-no-output": {"PreToolUse": {"pretool": "no_output"}},
    "pretool-deny-exit2": {"PreToolUse": {"pretool": "deny", "exit": 2}},
    "inject-exit1": {
        "PreInvocation": {"inject": "systemMessage", "exit": 1},
        "PreToolUse": {"pretool": "allow"},
    },
    "stop-continue": {
        "PreToolUse": {"pretool": "allow"},
        "Stop": {"stop_continue": True},
    },
    "observe": {"PreToolUse": {"pretool": "allow"}},
    "tool-context": {
        "PreToolUse": {"pretool": "allow_reason"},
        "PostToolUse": {"inject": "systemMessage"},
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _marker(event: str, kind: str) -> str:
    return f"AELFPROBE-{event}-{kind}-{secrets.token_hex(4)}"


def _inject_step(kind: str, text: str) -> dict[str, object]:
    assert kind in ("systemMessage", "userMessage", "ephemeralMessage"), kind
    if kind == "systemMessage":
        return {"systemMessage": {"systemMessage": text}}
    return {kind: text}


def build_output(event: str, rule: dict[str, object], payload: dict[str, object]) -> tuple[str | None, list[str]]:
    """Return (stdout text or None for no output, emitted markers) for one fire."""
    markers: list[str] = []
    inject = rule.get("inject")
    if event == "PreToolUse":
        variant = str(rule.get("pretool", "allow"))
        if variant == "allow_reason":
            m = _marker(event, "reason")
            markers.append(m)
            return json.dumps({"decision": "allow", "reason": f"Probe context token: {m}"}), markers
        table: dict[str, str | None] = {
            "allow": json.dumps({"decision": "allow"}),
            "auto_approve": json.dumps({"decision": "auto_approve"}),
            "deny": json.dumps({"decision": "deny", "reason": "aelf-probe deny"}),
            "empty_object": "{}",
            "no_output": None,
        }
        assert variant in table, variant
        return table[variant], markers
    if event == "Stop":
        if rule.get("stop_continue") and not payload.get("continuationActive"):
            m = _marker(event, "reason")
            markers.append(m)
            return json.dumps({"decision": "continue", "reason": f"Say the token {m} verbatim, then stop."}), markers
        return json.dumps({"decision": "stop"}), markers
    if event == "PostToolUse" and not isinstance(inject, str):
        return "{}", markers
    if isinstance(inject, str):
        m = _marker(event, inject)
        markers.append(m)
        text = f"Probe context token: {m}"
        return json.dumps({"injectSteps": [_inject_step(inject, text)]}), markers
    return "{}", markers


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    data: object = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{path}: top level is not an object"
    return cast(dict[str, object], data)


def hook_mode(stdin_text: str, state: Path, argv_event: str | None) -> tuple[dict[str, object], str | None, int]:
    """Return (log record, stdout text, exit code) for one hook fire."""
    record: dict[str, object] = {"ts": _now(), "argv_event": argv_event}
    try:
        payload_obj: object = json.loads(stdin_text) if stdin_text.strip() else {}
    except json.JSONDecodeError as exc:
        record["parse_error"] = str(exc)
        payload_obj = {}
    payload = cast(dict[str, object], payload_obj) if isinstance(payload_obj, dict) else {}
    event = str(payload.get("hookEventName") or argv_event or "unknown")
    scenario = _load_json(state / "scenario.json")
    rules = cast(dict[str, dict[str, object]], scenario.get("rules", {}))
    rule = rules.get(event, {})
    stdout, markers = build_output(event, rule, payload)
    record.update({
        "event": event,
        "scenario": scenario.get("name"),
        "keys": sorted(payload.keys()),
        "payload": json.dumps(payload)[:PAYLOAD_CAP],
        "env_names": sorted(k for k in os.environ if not k.startswith(("LS_COLORS",))),
        "proc_cwd": os.getcwd(),
        "stdout": stdout,
        "emitted": markers,
    })
    return record, stdout, int(cast(int, rule.get("exit", 0)))


def run_hook(state: Path, argv_event: str | None) -> int:
    try:
        record, stdout, code = hook_mode(sys.stdin.read(), state, argv_event)
        state.mkdir(parents=True, exist_ok=True)
        with (state / "fires.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception as exc:  # noqa: BLE001 - a probe bug must not wedge agy
        sys.stderr.write(f"aelf-probe: {exc}\n")
        sys.stdout.write(json.dumps({"decision": "allow"}) if argv_event == "PreToolUse" else "{}")
        return 0
    if stdout is not None:
        sys.stdout.write(stdout)
    return code


def _command(state: Path, event: str) -> str:
    return f"python3 -I {SCRIPT} --hook --event {event} --state {state}"


def planned_hooks(existing: dict[str, object], state: Path, decoys: bool) -> dict[str, object]:
    out = {k: v for k, v in existing.items() if k not in (HOOK_NAME, DECOY_NAME)}
    spec: dict[str, object] = {}
    for event in EVENTS:
        handler = {"type": "command", "command": _command(state, event), "timeout": 20}
        spec[event] = [{"matcher": "*", "hooks": [handler]}] if event in TOOL_EVENTS else [handler]
    out[HOOK_NAME] = spec
    if decoys:
        out[DECOY_NAME] = {e: [{"type": "command", "command": _command(state, e), "timeout": 20}] for e in DECOY_EVENTS}
    return out


def _hooks_file(ws: Path) -> Path:
    return ws / ".agents" / "hooks.json"


def install(ws: Path, state: Path, dry_run: bool, decoys: bool) -> int:
    path = _hooks_file(ws)
    planned = planned_hooks(_load_json(path), state.resolve(), decoys)
    text = json.dumps(planned, indent=2) + "\n"
    if dry_run:
        print(f"[dry-run] would write {path}:\n{text}")
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path}")
    return 0


def uninstall(ws: Path, dry_run: bool) -> int:
    path = _hooks_file(ws)
    if not path.exists():
        print(f"nothing to remove: {path} absent")
        return 0
    kept = {k: v for k, v in _load_json(path).items() if k not in (HOOK_NAME, DECOY_NAME)}
    if dry_run:
        print(f"[dry-run] would leave {sorted(kept)} in {path}")
        return 0
    if kept:
        path.write_text(json.dumps(kept, indent=2) + "\n", encoding="utf-8")
    else:
        path.unlink()
    print(f"removed probe hooks from {path}")
    return 0


def set_scenario(name: str, state: Path, dry_run: bool) -> int:
    if name not in SCENARIOS:
        print(f"unknown scenario {name}; known: {sorted(SCENARIOS)}", file=sys.stderr)
        return 1
    body = {"name": name, "rules": SCENARIOS[name]}
    if dry_run:
        print(f"[dry-run] would write {state / 'scenario.json'}: {json.dumps(body)}")
        return 0
    state.mkdir(parents=True, exist_ok=True)
    (state / "scenario.json").write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    print(f"scenario {name}")
    return 0


def _read_fires(state: Path) -> list[dict[str, object]]:
    path = state / "fires.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) <= MAX_LOG_LINES, "fires.jsonl over cap"
    return [cast(dict[str, object], json.loads(line)) for line in lines if line.strip()]


def _haystacks(state: Path, transcripts: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(state.glob("out-*.txt")):
        out[str(p)] = p.read_text(encoding="utf-8", errors="replace")
    if transcripts.exists():
        for p in sorted(transcripts.rglob("*.jsonl"))[:2000]:
            out[str(p)] = p.read_text(encoding="utf-8", errors="replace")
    return out


def report(state: Path, transcripts: Path) -> int:
    fires = _read_fires(state)
    if not fires:
        print(f"no fires recorded in {state / 'fires.jsonl'}")
        return 1
    hay = _haystacks(state, transcripts)
    by_event: dict[str, int] = {}
    keys: dict[str, set[str]] = {}
    for rec in fires:
        ev = str(rec.get("event"))
        by_event[ev] = by_event.get(ev, 0) + 1
        keys.setdefault(ev, set()).update(cast(list[str], rec.get("keys", [])))
    print("events fired:")
    for ev, n in sorted(by_event.items()):
        print(f"  {ev}: {n}  keys={sorted(keys[ev])}")
    print("markers:")
    for rec in fires:
        for m in cast(list[str], rec.get("emitted", [])):
            where = [Path(k).name for k, v in hay.items() if m in v]
            print(f"  [{rec.get('scenario')}] {m}: {'found in ' + ', '.join(where) if where else 'NOT FOUND'}")
    return 0


def _run_hook_subprocess(payload: dict[str, object], state: Path, event: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--hook", "--event", event, "--state", str(state)],
        input=json.dumps(payload), capture_output=True, text=True, timeout=30, check=False,
    )


def self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        state, ws = root / "state", root / "ws"
        assert set_scenario("inject-system", state, False) == 0
        r = _run_hook_subprocess({"hookEventName": "PreInvocation", "invocationNum": 0}, state, "PreInvocation")
        out = json.loads(r.stdout)
        step = out["injectSteps"][0]["systemMessage"]["systemMessage"]
        assert "AELFPROBE-PreInvocation-systemMessage-" in step, r.stdout
        r = _run_hook_subprocess({"hookEventName": "PreToolUse"}, state, "PreToolUse")
        assert json.loads(r.stdout) == {"decision": "allow"}, r.stdout
        assert set_scenario("pretool-no-output", state, False) == 0
        r = _run_hook_subprocess({"hookEventName": "PreToolUse"}, state, "PreToolUse")
        assert r.stdout == "", r.stdout
        assert set_scenario("pretool-deny-exit2", state, False) == 0
        r = _run_hook_subprocess({"hookEventName": "PreToolUse"}, state, "PreToolUse")
        assert r.returncode == 2 and json.loads(r.stdout)["decision"] == "deny", (r.returncode, r.stdout)
        assert set_scenario("stop-continue", state, False) == 0
        r = _run_hook_subprocess({"hookEventName": "Stop", "continuationActive": False}, state, "Stop")
        assert json.loads(r.stdout)["decision"] == "continue", r.stdout
        r = _run_hook_subprocess({"hookEventName": "Stop", "continuationActive": True}, state, "Stop")
        assert json.loads(r.stdout)["decision"] != "continue", r.stdout
        r = _run_hook_subprocess({}, state, "PreToolUse")
        assert r.returncode == 0, r.stderr
        fires = _read_fires(state)
        assert len(fires) == 7, len(fires)
        assert all("env_names" in f for f in fires)
        hooks_file = _hooks_file(ws)
        hooks_file.parent.mkdir(parents=True)
        hooks_file.write_text(json.dumps({"other": {"Stop": []}}), encoding="utf-8")
        assert install(ws, state, False, True) == 0
        data = _load_json(hooks_file)
        assert set(data) == {"other", HOOK_NAME, DECOY_NAME}, sorted(data)
        assert uninstall(ws, False) == 0
        assert set(_load_json(hooks_file)) == {"other"}
    print("self-test passed")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--hook", action="store_true")
    mode.add_argument("--install", type=Path, metavar="WS")
    mode.add_argument("--uninstall", type=Path, metavar="WS")
    mode.add_argument("--scenario", choices=sorted(SCENARIOS))
    mode.add_argument("--report", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    ap.add_argument("--state", type=Path)
    ap.add_argument("--event")
    ap.add_argument("--transcripts", type=Path, default=TRANSCRIPT_ROOT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--decoys", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.uninstall:
        return uninstall(args.uninstall, args.dry_run)
    if args.state is None:
        ap.error("--state is required for this mode")
    if args.hook:
        return run_hook(args.state, args.event)
    if args.install:
        return install(args.install, args.state, args.dry_run, args.decoys)
    if args.scenario:
        return set_scenario(args.scenario, args.state, args.dry_run)
    return report(args.state, args.transcripts)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
