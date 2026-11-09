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
