### Fixed

- **The context rebuilder now reads aelfrice's own turn log from a linked worktree ([#1706](https://github.com/robotrocketscience/aelfrice/issues/1706)).**
  - **The cause:** to find `turns.jsonl`, the rebuilder walked up from the hook payload's directory to the first `.git` entry. In a linked worktree, `.git` is a file, so it built a path that could never exist and fell back to the host's own transcript. The transcript logger writes the log under the git common dir.
  - **The fix:** the reader now uses the logger's own resolver: `AELFRICE_TRANSCRIPTS_DIR`, else the git common dir, else `~/.aelfrice/transcripts/`. It resolves from the process's working directory, like every other in-turn reader ([#1630](https://github.com/robotrocketscience/aelfrice/issues/1630)), rather than from the payload's.
  - **Affected readers:** the compaction rebuild, the cadence rebuild, the conversation-aware query, and `aelf rebuild`.
  - **Two side effects:**
    - A hook payload without a `cwd` now reads the turn log too.
    - The conversation-aware query, which is on by default, now makes one `git rev-parse` call per prompt to find the log.
  - **Tests:** the test suite now pins `AELFRICE_TRANSCRIPTS_DIR` to its sandbox, so no test reads a contributor's real turn log.
