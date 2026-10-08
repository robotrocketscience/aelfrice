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
# Override via AELFRICE_CORPUS_ROOT. A release run must read the lab
# corpus's `main` (docs/concepts/RELEASING.md step 7).
#
# Usage: scripts/run_bench_gate.sh [--dry-run] [pytest args...]
#   --dry-run  print the corpus report and the pytest command, then exit 0.
set -euo pipefail

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=1
    shift
fi

: "${AELFRICE_CORPUS_ROOT:=$HOME/projects/aelfrice-lab/tests/corpus/v2_0}"
export AELFRICE_CORPUS_ROOT

if [[ ! -d "$AELFRICE_CORPUS_ROOT" ]]; then
    echo "error: AELFRICE_CORPUS_ROOT does not exist: $AELFRICE_CORPUS_ROOT" >&2
    echo "       (mount the lab corpus or set the env var)" >&2
    exit 1
fi

echo "bench-gate corpus root: $AELFRICE_CORPUS_ROOT"
# #1735: the default root follows whatever branch the lab checkout has
# checked out, so say which revision this run reads. Git's location
# variables are cleared so an inherited GIT_DIR (a git hook, say) can't
# report another repository.
corpus_git() {
    env -u GIT_DIR -u GIT_WORK_TREE -u GIT_COMMON_DIR -u GIT_INDEX_FILE \
        git -C "$AELFRICE_CORPUS_ROOT" "$@"
}
warn_release() {
    echo "warning: a release run must read the corpus at a clean main (RELEASING.md step 7)" >&2
}
if top=$(corpus_git rev-parse --show-toplevel 2>/dev/null); then
    branch=$(corpus_git symbolic-ref --short -q HEAD || echo "(detached HEAD)")
    commit=$(corpus_git rev-parse --short=12 HEAD 2>/dev/null || echo "(no commits)")
    # --ignored: a corpus file no commit tracks is read all the same.
    if [[ -n "$(corpus_git status --porcelain --ignored -- . 2>/dev/null)" ]]; then
        changes="yes"
    else
        changes="no"
    fi
    echo "bench-gate corpus checkout: $top"
    echo "bench-gate corpus branch: $branch"
    echo "bench-gate corpus commit: $commit"
    echo "bench-gate corpus uncommitted changes: $changes"
    if [[ "$branch" != "main" || "$changes" == "yes" || "$commit" == "(no commits)" ]]; then
        warn_release
    fi
else
    echo "bench-gate corpus checkout: none (the root is not inside a git checkout)"
    warn_release
fi

cmd=(uv run pytest tests/bench_gate/ -v -m bench_gated "$@")
if [[ "$dry_run" == 1 ]]; then
    printf "dry run, would run:"
    printf " %q" "${cmd[@]}"
    printf "\n"
    exit 0
fi
exec "${cmd[@]}"
