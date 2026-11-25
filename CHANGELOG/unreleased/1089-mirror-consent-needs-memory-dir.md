### Documentation

- **`aelf setup` records claude-memory mirror consent only where a claude-memory directory exists ([#1089](https://github.com/robotrocketscience/aelfrice/issues/1089)).** `docs/user/CONFIG.md` said that the first `aelf setup` for a project writes the mirror's consent sentinel. Setup skips the reconcile, and so the sentinel, when the project has no claude-memory directory. On such a project the mirror stays off until a later `aelf setup` or `aelf reconcile-claude-memory` records consent. The page now says so.
