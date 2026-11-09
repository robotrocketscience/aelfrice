"""Every census probe, one module per subsystem (#1887).

Each `probes_*` module exports `PROBES: list[Probe]`. The ids together are
A001 to A351, one probe per feature in the inventory of `main` at
ad14ef545, and no id appears twice.
"""
from __future__ import annotations

from types import ModuleType

from . import (
    probes_beliefs_graph_feedback,
    probes_cli_setup_hosts_export,
    probes_hook_injection,
    probes_ingest_extraction,
    probes_locks_core_federation,
    probes_retrieval,
    probes_store_maintenance_telemetry,
    probes_wonder_reason,
)
from .core import Probe

SUBSYSTEM_MODULES: dict[str, ModuleType] = {
    "beliefs-graph-feedback": probes_beliefs_graph_feedback,
    "cli-setup-hosts-export": probes_cli_setup_hosts_export,
    "hook-injection": probes_hook_injection,
    "ingest-extraction": probes_ingest_extraction,
    "locks-core-federation": probes_locks_core_federation,
    "retrieval": probes_retrieval,
    "store-maintenance-telemetry": probes_store_maintenance_telemetry,
    "wonder-reason": probes_wonder_reason,
}

FIRST_ID = 1
LAST_ID = 351

# Traceable probes that no test makes report `fired` or `ambiguous` (a trace
# found), and why. Every other traceable probe has a test in which it reports
# one of the two. (A028's writer was removed in 5.1.0, so its fired test
# plants the row with SQL.)
NO_FIRED_TEST: dict[str, str] = {
    "A082": "fires only on audit rows from before 5.1.0 or a block over a bound "
            "the hook enforces",
    "A120": "the recap line prints only; reports untraceable with a replay in detail",
    "A143": "store-marker routing leaves no trace of the rule that chose the store",
    "A161": "the filter records no drop; untraceable once scan rows exist",
    "A211": "the suffix goes to stderr only; untraceable when a belief qualifies",
    "A236": "no production path writes the struct index",
    "A240": "the lane persists nothing; untraceable once its edges exist",
    "A254": "the rerank persists nothing; untraceable once its edges exist",
    "A280": "doctor only reads the hooks' telemetry rows",
}

# Traceable probes that no test sees report `silent`, and why. Both lists are
# checked against the statuses the probe tests produce, in both directions.
NO_SILENT_RESULT: dict[str, str] = {
    "A107": "a search-hook fire whose hits were already in the ring leaves no id",
    "A120": "the recap line prints only",
    "A143": "store-marker routing leaves no trace",
    "A171": "the speaker gate writes nothing; fired is inferred from dropped lines",
    "A236": "no production path writes the struct index",
    "A280": "doctor only reads the hooks' telemetry rows",
    "A292": "the migration guard deletes its marker on success",
}


def all_probes() -> list[Probe]:
    """Every probe sorted by id. Raises on a duplicate id or a non-Probe entry."""
    seen: set[str] = set()
    out: list[Probe] = []
    for mod in SUBSYSTEM_MODULES.values():
        entries: list[object] = list(getattr(mod, "PROBES"))
        for p in entries:
            if not isinstance(p, Probe):
                raise TypeError(f"{mod.__name__}: PROBES entry is not a Probe")
            if p.item_id in seen:
                raise ValueError(f"duplicate probe for {p.item_id}")
            seen.add(p.item_id)
            out.append(p)
    return sorted(out, key=lambda p: p.item_id)


def subsystem_of(item_id: str) -> str:
    """The subsystem whose module holds `item_id`."""
    for name, mod in SUBSYSTEM_MODULES.items():
        entries: list[Probe] = list(getattr(mod, "PROBES"))
        if any(p.item_id == item_id for p in entries):
            return name
    raise KeyError(item_id)
