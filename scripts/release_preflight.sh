#!/usr/bin/env bash
# Refuse to publish a release tag that main does not contain, or whose
# pyproject.toml version does not match the tag (audit CI-03, CI-04, F10).
#
# publish.yml fires on any pushed tag shaped like vX.Y.Z, on any commit.
# This script is its first step after checkout, ahead of the tests, the
# build and the upload. It checks:
#
#   1. the tag names a final release: vMAJOR.MINOR.PATCH;
#   2. the tagged commit is an ancestor of (or equal to) the main ref;
#   3. the [project] version in pyproject.toml at the tagged commit equals
#      the tag without its leading "v".
#
# The main ref must already exist locally. publish.yml fetches it right
# before running this script. pyproject.toml is read from the tagged
# commit, not from the working tree, so a dirty checkout cannot change
# the answer.
#
# Usage: scripts/release_preflight.sh [--dry-run] [--main-ref REF] [TAG]
#   TAG         the release tag; defaults to $GITHUB_REF_NAME.
#   --main-ref  the ref the tagged commit must be on; default origin/main.
#   --dry-run   run every check and print its result, then exit 0 even when
#               a check fails. A usage error still exits 2.
#
# Exit status: 0 when every check passes (or under --dry-run), 1 when a
# check fails, 2 on a usage error or a ref that does not resolve.
set -euo pipefail

usage() {
    echo "usage: scripts/release_preflight.sh [--dry-run] [--main-ref REF] [TAG]" >&2
    exit 2
}

dry_run=0
main_ref="origin/main"
tag=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        --main-ref)
            [[ $# -ge 2 && -n "$2" ]] || usage
            main_ref="$2"; shift 2 ;;
        -h|--help) usage ;;
        -*) echo "error: unknown option: $1" >&2; usage ;;
        *)
            [[ -z "$tag" ]] || { echo "error: more than one tag given" >&2; usage; }
            tag="$1"; shift ;;
    esac
done
tag="${tag:-${GITHUB_REF_NAME:-}}"
if [[ -z "$tag" ]]; then
    echo "error: no tag given and GITHUB_REF_NAME is unset" >&2
    usage
fi

fail=0
report() {
    # report PASS|FAIL message
    if [[ "$1" == "PASS" ]]; then
        echo "preflight PASS: $2"
    else
        fail=1
        if [[ "$dry_run" == 1 ]]; then
            echo "preflight FAIL (dry run, not fatal): $2"
        else
            echo "::error::release preflight: $2"
            echo "preflight FAIL: $2" >&2
        fi
    fi
}

if [[ "$tag" =~ ^v([0-9]+\.[0-9]+\.[0-9]+)$ ]]; then
    tag_version="${BASH_REMATCH[1]}"
    report PASS "tag $tag names a final release"
else
    tag_version=""
    report FAIL "tag $tag is not vMAJOR.MINOR.PATCH"
fi

if ! tag_sha=$(git rev-parse --verify --quiet "refs/tags/${tag}^{commit}"); then
    echo "error: tag $tag does not resolve to a commit in this repository" >&2
    exit 2
fi
if ! main_sha=$(git rev-parse --verify --quiet "${main_ref}^{commit}"); then
    echo "error: main ref $main_ref does not resolve; fetch it first" >&2
    exit 2
fi
echo "preflight tag commit: $tag_sha"
echo "preflight $main_ref commit: $main_sha"

if git merge-base --is-ancestor "$tag_sha" "$main_sha"; then
    report PASS "tag commit is on $main_ref"
else
    report FAIL "tag commit $tag_sha is not an ancestor of $main_ref ($main_sha)"
fi

# The first `version = "..."` inside the [project] table. A version under
# another table (a tool's own version key) is not the package version.
if ! pyproject=$(git show "${tag_sha}:pyproject.toml" 2>/dev/null); then
    pyproject=""
fi
project_version=$(printf '%s\n' "$pyproject" | awk '
    /^[[:space:]]*\[/ {
        in_project = ($0 ~ /^[[:space:]]*\[project\][[:space:]]*(#.*)?$/)
        next
    }
    in_project && /^[[:space:]]*version[[:space:]]*=/ {
        line = $0
        sub(/^[^=]*=[[:space:]]*/, "", line)
        if (match(line, /^"[^"]*"/)) {
            print substr(line, 2, RLENGTH - 2)
            exit
        }
    }
')
if [[ -z "$project_version" ]]; then
    report FAIL "no [project] version found in pyproject.toml at $tag_sha"
elif [[ "$project_version" == "$tag_version" ]]; then
    report PASS "pyproject.toml version $project_version matches tag $tag"
else
    report FAIL "pyproject.toml version $project_version does not match tag $tag"
fi

if [[ "$dry_run" == 1 ]]; then
    if [[ "$fail" == 1 ]]; then
        echo "dry run: a real run would exit 1"
    else
        echo "dry run: a real run would exit 0"
    fi
    exit 0
fi
exit "$fail"
