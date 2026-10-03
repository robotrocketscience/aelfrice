### Added

- **Aelfrice flags an ignored `~/.aelfrice.toml` ([#1652](https://github.com/robotrocketscience/aelfrice/issues/1652)).** Since #1582, aelfrice never reads a per-user config, so settings kept there stopped applying with no message. When `~/.aelfrice.toml` exists and the project has no `.aelfrice.toml`:
  - **SessionStart hook:** adds one line to the agent's context naming the file, saying it's ignored, and giving the fix (copy it to the project's root). It goes on stdout, like the failed-lock notice, because a hook's stderr reaches only the host's debug log when the hook succeeds. It resolves the project from the session's working directory and doesn't repeat after a compaction.
  - **`aelf doctor`:** reports the same as a warning, which doesn't change its exit code.

  Both stay quiet when the project has its own config. `docs/user/CONFIG.md` now also explains that a linked work tree reads only the `.aelfrice.toml` at its own root.
