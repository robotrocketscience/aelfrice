#!/usr/bin/env bash
# Run the v2.0 bench-gate harness against a mounted lab corpus (#319).
#
# The corpus content lives in the private lab repo (see #307); this
# script just points the public harness at it. Default expects the
# standard two-repo layout:
#
#   ~/projects/aelfrice         <- public, this repo
#   ~/projects/aelfrice-lab     <- private, holds the corpus
#
# Override via AELFRICE_CORPUS_ROOT.
set -euo pipefail

: "${AELFRICE_CORPUS_ROOT:=$HOME/projects/aelfrice-lab/tests/corpus/v2_0}"
export AELFRICE_CORPUS_ROOT

if [[ ! -d "$AELFRICE_CORPUS_ROOT" ]]; then
    echo "error: AELFRICE_CORPUS_ROOT does not exist: $AELFRICE_CORPUS_ROOT" >&2
    echo "       (mount the lab corpus or set the env var)" >&2
    exit 1
fi

echo "bench-gate corpus root: $AELFRICE_CORPUS_ROOT"
exec uv run pytest tests/bench_gate/ -v -m bench_gated "$@"
