"""pytest plugin: log every fire-census probe status the tests observe (#1887).

Loaded with `-p tests.fire_census_status_log` by the test that checks
`registry.NO_FIRED_TEST` and `registry.NO_SILENT_RESULT` against what the
probe tests actually see. Each `core.run_one` call appends one JSON line,
`[item_id, status]`, to the file `$FIRE_CENSUS_STATUS_LOG` names.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from benchmarks.real_store_fire_census import core

_LOG = Path(os.environ["FIRE_CENSUS_STATUS_LOG"])
_run_one = core.run_one


def _logged(p: core.Probe, ctx: core.Ctx | None,
            deadline_s: float = core.DEFAULT_PROBE_DEADLINE_S) -> core.Result:
    r = _run_one(p, ctx, deadline_s)
    with _LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps([p.item_id, r.status]) + "\n")
    return r


core.run_one = _logged
