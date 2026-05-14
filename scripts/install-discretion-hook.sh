#!/usr/bin/env bash
# install-discretion-hook.sh — write the canonical pre-push discretion hook to
# .git/hooks/pre-push in the aelfrice working tree.
#
# Usage:
#   scripts/install-discretion-hook.sh [--force] [--help]
#
#   --force   Overwrite an existing hook even if it differs from the canonical
#             content. Without --force the script prints a diff and exits 1.
#
# The hook blocks banned paths, banned vocabulary, and CLAUDE.md-derived phrases
# from being pushed — both in diff content and in commit messages.
#
# This installer is idempotent: if .git/hooks/pre-push already matches the
# canonical content byte-for-byte, it exits 0 with "already up to date".
#
# Must be run from inside the aelfrice git working tree.

set -euo pipefail

FORCE=0

for arg in "$@"; do
    case "$arg" in
        --help|-h)
            sed -n '2,/^$/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        --force)
            FORCE=1
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            echo "Usage: $0 [--force] [--help]" >&2
            exit 1
            ;;
    esac
done

# Resolve the repo root — works from any directory inside the working tree.
if ! REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null); then
    echo "ERROR: not inside a git working tree." >&2
    exit 1
fi

HOOK_PATH="$REPO_ROOT/.git/hooks/pre-push"

# Canonical hook content lives here as a heredoc. This is the single source of
# truth. If you update the hook logic, update it here and re-run the installer.
CANONICAL=$(cat <<'HOOK_EOF'
#!/bin/bash
# Pre-push guard for ~/projects/aelfrice (public repo).
# Blocks any push whose diff or commit messages contain:
#   1. Files under tests/corpus/v2_0/**/*.jsonl  (corpus content lives in lab repo only)
#   2. Files under .claude/, .planning/, ~/.claude/handoffs/  (session/handoff content)
#   3. Banned vocabulary in diff content (codenames, model identifiers, pipeline vocab)
#   4. Banned vocabulary in commit messages
#   5. Banned phrases from ~/.claude/CLAUDE.md in commit messages
#
# To override for an emergency, set ALLOW_DISCRETION_OVERRIDE=1 in the env. The override
# logs to ~/.aelfrice/discretion-override.log; investigate every entry afterward.
#
# DO NOT REMOVE OR WEAKEN THIS FILE WITHOUT EXPLICIT USER REQUEST.

set -euo pipefail

remote="${1:-}"
url="${2:-}"

z40=0000000000000000000000000000000000000000

# Banned paths — extended regex
BANNED_PATHS='^(tests/corpus/v2_0/.+\.jsonl|\.claude/|\.planning/|.*/handoffs/.*\.md)$'

# Banned vocabulary in diff content — case-insensitive extended regex.
# Word boundaries are intentional to avoid blocking on substrings like "claude.md".
BANNED_VOCAB='\b(Setr|Kulili|Gylf|Sonnet|Opus|Haiku|Anthropic|Claude Code|parallel session|rook.tier|queen.tier|delegate to (Sonnet|Haiku)|Agent tool|Generated with \[Claude Code\])\b'

# Banned phrases from ~/.claude/CLAUDE.md — multi-word fragments that indicate
# fixture/corpus content was seeded from the user's private global instructions.
# Added 2026-05-11 after PR #673 fixture leak (paraphrase of two-repo workflow
# section slipped past BANNED_VOCAB). This is a deliberately narrow allowlist
# of high-signal phrases; the durable fix is a similarity-check redesign
# (filed as a separate aelfrice issue).
BANNED_PHRASES='(branch off main|staging gate|gitleaks|pii (scan|pattern)|commit.history audit|DO_NOT_PUSH_PERSONAL_DATA|aelfrice-lab|PII_PATTERNS_SECRET|two-repo workflow|publish-to-quarantine|publish-to-github)'

# Read each ref being pushed.
fail=0
while read -r local_ref local_sha remote_ref remote_sha; do
    [ -z "${local_ref:-}" ] && continue

    # Branch deletion — local_sha is all zeros. Allow.
    if [ "$local_sha" = "$z40" ]; then
        continue
    fi

    # Diff against merge-base with main — covers new-branch, fast-forward,
    # and force-push (rebase) cases without false-positives from main commits
    # that came in under the old base on a rebased branch.
    base=$(git merge-base main "$local_sha" 2>/dev/null || echo "")
    if [ -z "$base" ]; then
        # No common ancestor with main — diff against empty tree (full content scan).
        range="$(git hash-object -t tree /dev/null)..$local_sha"
        # For commit-message checks, walk the whole reachable history.
        msg_range="$local_sha"
    else
        range="$base..$local_sha"
        msg_range="$base..$local_sha"
    fi

    # Check 1: banned paths.
    bad_paths=$(git diff --name-only "$range" | grep -E "$BANNED_PATHS" || true)
    if [ -n "$bad_paths" ]; then
        echo "" >&2
        echo "PRE-PUSH BLOCKED: forbidden paths in $local_ref" >&2
        echo "$bad_paths" | sed 's/^/  /' >&2
        echo "" >&2
        fail=1
    fi

    # Check 2: banned vocabulary in diff content.
    bad_vocab=$(git diff "$range" -- . ':(exclude)CLAUDE.md' ':(exclude).gitleaks*.toml' \
        ':(exclude)scripts/install-discretion-hook.sh' \
        | grep -E '^\+' \
        | grep -niE "$BANNED_VOCAB" \
        || true)
    if [ -n "$bad_vocab" ]; then
        echo "" >&2
        echo "PRE-PUSH BLOCKED: banned vocabulary in diff for $local_ref" >&2
        echo "$bad_vocab" | head -20 | sed 's/^/  /' >&2
        echo "" >&2
        fail=1
    fi

    # Check 3: banned phrases lifted from ~/.claude/CLAUDE.md.
    bad_phrases=$(git diff "$range" -- . ':(exclude)CLAUDE.md' ':(exclude).gitleaks*.toml' \
        ':(exclude).git/hooks/pre-push' ':(exclude)scripts/install-discretion-hook.sh' \
        | grep -E '^\+' \
        | grep -niE "$BANNED_PHRASES" \
        || true)
    if [ -n "$bad_phrases" ]; then
        echo "" >&2
        echo "PRE-PUSH BLOCKED: CLAUDE.md-derived phrase in diff for $local_ref" >&2
        echo "$bad_phrases" | head -20 | sed 's/^/  /' >&2
        echo "" >&2
        fail=1
    fi

    # Check 4: banned vocabulary in commit messages.
    # Path-excludes do not apply to commit messages — a banned codename in a
    # commit subject is always a leak regardless of which files were changed.
    bad_msg_vocab=$(git log --format=%B "$msg_range" \
        | grep -niE "$BANNED_VOCAB" \
        || true)
    if [ -n "$bad_msg_vocab" ]; then
        echo "" >&2
        echo "PRE-PUSH BLOCKED: banned vocabulary in commit message(s) for $local_ref" >&2
        echo "$bad_msg_vocab" | head -10 | sed 's/^/  /' >&2
        echo "" >&2
        fail=1
    fi

    # Check 5: banned phrases in commit messages.
    bad_msg_phrases=$(git log --format=%B "$msg_range" \
        | grep -niE "$BANNED_PHRASES" \
        || true)
    if [ -n "$bad_msg_phrases" ]; then
        echo "" >&2
        echo "PRE-PUSH BLOCKED: CLAUDE.md-derived phrase in commit message(s) for $local_ref" >&2
        echo "$bad_msg_phrases" | head -10 | sed 's/^/  /' >&2
        echo "" >&2
        fail=1
    fi
done

if [ "$fail" -ne 0 ]; then
    if [ "${ALLOW_DISCRETION_OVERRIDE:-0}" = "1" ]; then
        mkdir -p "$HOME/.aelfrice"
        echo "$(date -u +%FT%TZ) override remote=$remote url=$url cwd=$(pwd) refs=$(git rev-parse HEAD)" \
            >> "$HOME/.aelfrice/discretion-override.log"
        echo "PRE-PUSH OVERRIDE accepted; logged to ~/.aelfrice/discretion-override.log" >&2
        exit 0
    fi
    echo "Push blocked. Investigate the listed paths/lines above." >&2
    echo "If the content is genuinely safe and must ship, set ALLOW_DISCRETION_OVERRIDE=1 and retry." >&2
    exit 1
fi

exit 0
HOOK_EOF
)

# Idempotency check.
if [ -f "$HOOK_PATH" ]; then
    existing=$(cat "$HOOK_PATH")
    if [ "$existing" = "$CANONICAL" ]; then
        echo "scripts/install-discretion-hook.sh: .git/hooks/pre-push already up to date."
        exit 0
    fi

    if [ "$FORCE" -eq 0 ]; then
        echo "ERROR: .git/hooks/pre-push exists and differs from the canonical content." >&2
        echo "Diff (existing vs canonical):" >&2
        diff <(echo "$existing") <(echo "$CANONICAL") >&2 || true
        echo "" >&2
        echo "Re-run with --force to overwrite, or inspect the diff above first." >&2
        exit 1
    fi

    echo "scripts/install-discretion-hook.sh: overwriting existing hook (--force)."
fi

# Write the canonical hook.
printf '%s\n' "$CANONICAL" > "$HOOK_PATH"
chmod +x "$HOOK_PATH"
echo "scripts/install-discretion-hook.sh: installed .git/hooks/pre-push"
