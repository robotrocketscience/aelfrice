#!/usr/bin/env bash
#
# sync-repo-settings.sh — apply the canonical GitHub repo settings for aelfrice.
#
# Usage:
#   scripts/sync-repo-settings.sh           # dry-run: print the diff
#   scripts/sync-repo-settings.sh --apply   # actually write the settings
#
# Idempotent. Re-running with --apply is a no-op when state already matches.
#
# Surface: this script does NOT touch branch protection, secrets, codeowners,
# webhooks, deploy keys, or any irreversible permission. It only manages the
# About blurb, topics, and the four "feature toggles" surfaced under
# Settings → General → Features (Wiki, Issues, Projects, Discussions).
#
# Desired state is the source of truth — adjust the variables below and re-run.

set -euo pipefail

REPO="${REPO:-robotrocketscience/aelfrice}"

# ---- Desired state ---------------------------------------------------------
DESIRED_DESCRIPTION="Bayesian memory that learns from feedback for LLM agents"
DESIRED_HOMEPAGE="https://pypi.org/project/aelfrice/"
DESIRED_WIKI="false"          # in-repo docs/ is the single source of truth
DESIRED_ISSUES="true"         # primary feedback channel
DESIRED_PROJECTS="false"      # not used; ROADMAP.md is the planning surface
DESIRED_DISCUSSIONS="true"    # Q&A forum for users
# Topics shown on the repo landing page. Order matters for display.
DESIRED_TOPICS=(
  agent-memory ai-agents anthropic bayesian bayesian-inference
  claude-code fts5 llm mcp mcp-server memory python retrieval
  semantic-memory sqlite
)
# ----------------------------------------------------------------------------

APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1

bold() { printf '\033[1m%s\033[0m\n' "$1"; }
diff_line() {
  local label="$1" current="$2" desired="$3"
  if [ "$current" = "$desired" ]; then
    printf '  %-22s %s\n' "$label" "$desired"
  else
    printf '  %-22s %s -> %s\n' "$label" "$current" "$desired"
  fi
}

bold "Reading current state for $REPO..."
current=$(gh repo view "$REPO" --json description,homepageUrl,hasWikiEnabled,hasIssuesEnabled,hasProjectsEnabled,hasDiscussionsEnabled,repositoryTopics)

cur_desc=$(jq -r '.description // ""'             <<<"$current")
cur_home=$(jq -r '.homepageUrl // ""'             <<<"$current")
cur_wiki=$(jq -r '.hasWikiEnabled'                <<<"$current")
cur_iss=$(jq -r  '.hasIssuesEnabled'              <<<"$current")
cur_proj=$(jq -r '.hasProjectsEnabled'            <<<"$current")
cur_disc=$(jq -r '.hasDiscussionsEnabled'         <<<"$current")
cur_topics=$(jq -r '.repositoryTopics | map(.name) | sort | join(",")' <<<"$current")
desired_topics=$(printf '%s\n' "${DESIRED_TOPICS[@]}" | sort | paste -sd, -)

bold "\nDesired vs current:"
diff_line "description"  "$cur_desc"  "$DESIRED_DESCRIPTION"
diff_line "homepage"     "$cur_home"  "$DESIRED_HOMEPAGE"
diff_line "wiki"         "$cur_wiki"  "$DESIRED_WIKI"
diff_line "issues"       "$cur_iss"   "$DESIRED_ISSUES"
diff_line "projects"     "$cur_proj"  "$DESIRED_PROJECTS"
diff_line "discussions"  "$cur_disc"  "$DESIRED_DISCUSSIONS"
diff_line "topics"       "$cur_topics" "$desired_topics"

if [ "$APPLY" -ne 1 ]; then
  bold "\nDry-run only. Re-run with --apply to write."
  exit 0
fi

bold "\nApplying..."

# About blurb + homepage + features in one call.
gh repo edit "$REPO" \
  --description "$DESIRED_DESCRIPTION" \
  --homepage "$DESIRED_HOMEPAGE" \
  --enable-wiki="$DESIRED_WIKI" \
  --enable-issues="$DESIRED_ISSUES" \
  --enable-projects="$DESIRED_PROJECTS" \
  --enable-discussions="$DESIRED_DISCUSSIONS" >/dev/null

# Topics: gh repo edit --add-topic doesn't remove unlisted ones, so reset.
# Read the live list, remove what's not in desired, add what's missing.
live_topics_csv=$(gh repo view "$REPO" --json repositoryTopics --jq '.repositoryTopics | map(.name) | join(",")')
to_remove=$(comm -23 <(printf '%s\n' "$live_topics_csv" | tr ',' '\n' | sort -u) <(printf '%s\n' "${DESIRED_TOPICS[@]}" | sort -u))
to_add=$(comm -13 <(printf '%s\n' "$live_topics_csv" | tr ',' '\n' | sort -u) <(printf '%s\n' "${DESIRED_TOPICS[@]}" | sort -u))

if [ -n "$to_remove" ]; then
  while IFS= read -r t; do
    [ -z "$t" ] && continue
    gh repo edit "$REPO" --remove-topic "$t" >/dev/null
  done <<<"$to_remove"
fi
if [ -n "$to_add" ]; then
  while IFS= read -r t; do
    [ -z "$t" ] && continue
    gh repo edit "$REPO" --add-topic "$t" >/dev/null
  done <<<"$to_add"
fi

bold "Done."
echo ""
echo "Things this script does NOT touch (do them in the GitHub web UI):"
echo "  - Social preview image (Settings -> General -> Social preview)."
echo "    Suggested source: a 1280x640 crop of an image in docs/assets/."
echo "  - Pinned issues (repo home -> Customize your pins)."
echo "  - Branch protection rules (deferred per CLAUDE.md until public flip)."
