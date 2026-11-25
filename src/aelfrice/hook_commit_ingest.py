"""The `aelf-commit-ingest` hook. It runs, exits 0, and stores nothing.

`aelf setup` installs this command as each repository's git `post-commit`
hook (#1698). The block that setup writes resolves the commit, skips a
commit that a rebase replays, and starts `aelf-commit-ingest --git-hook
<hash>` in the background. An older setup ran the same command as a
Claude Code `PostToolUse:Bash` hook, which reads a JSON payload on stdin.

The command used to store the relation phrases in each commit message as
beliefs and edges. It no longer opens a store or writes a file, on either
entry point. The hook stays installed and keeps its command line, so a
later design can give it a job without a reinstall.

Both entry points return 0, so a commit never fails because of this hook.
"""
from __future__ import annotations

import sys
from typing import IO, Final

from aelfrice.stream_encoding import ensure_utf8_streams, read_payload_text

GIT_HOOK_FLAG: Final[str] = "--git-hook"
"""The argument the git `post-commit` block passes before the commit hash."""


def main(
    *,
    stdin: IO[str] | None = None,
    stderr: IO[str] | None = None,
    argv: list[str] | None = None,
) -> int:
    """Hook entry point. Returns 0 and writes nothing.

    With `--git-hook`, git's `post-commit` block started it. Otherwise a
    host started it with a `PostToolUse` payload on stdin, which is read
    and dropped, so the host's write to the pipe completes.
    """
    args = sys.argv[1:] if argv is None else argv
    if GIT_HOOK_FLAG in args:
        return 0
    sin = stdin if stdin is not None else sys.stdin
    serr = stderr if stderr is not None else sys.stderr
    ensure_utf8_streams((serr,))
    try:
        read_payload_text(sin, serr)
    except (OSError, ValueError):
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
