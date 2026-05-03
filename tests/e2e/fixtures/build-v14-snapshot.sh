#!/usr/bin/env bash
# Regenerate tests/e2e/fixtures/v14-snapshot.db from a clean install of
# aelfrice 1.4.0. Used by scenario #5 of the e2e suite (#334).
#
# Idempotent. Run from the repo root or anywhere — paths resolve to the
# location of this script.
#
# The snapshot is checked in as a binary fixture so the e2e suite does
# not need network access to PyPI at test time. Re-run this script when
# the v1.4 install path changes (rare) or to verify the fixture is
# reproducible.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="${here}/v14-snapshot.db"
build="$(mktemp -d)"
trap 'rm -rf "${build}"' EXIT

uv venv --python 3.12 "${build}/.venv" >/dev/null
uv pip install --python "${build}/.venv/bin/python" --quiet aelfrice==1.4.0

db="${build}/v14.sqlite3"
aelf="${build}/.venv/bin/aelf"

AELFRICE_DB="${db}" "${aelf}" lock \
  "Quokkas calibrate the knob carefully on Tuesdays."
AELFRICE_DB="${db}" "${aelf}" lock \
  "The aardvark counter resets at midnight."
AELFRICE_DB="${db}" "${aelf}" lock \
  "Wibble pickling requires the canonical protocol header bytes."

cp "${db}" "${out}"
echo "wrote ${out} ($(wc -c <"${out}") bytes)"
