### Added

- **User locks reach your prompts in every repository ([#1681](https://github.com/robotrocketscience/aelfrice/issues/1681)).** A lock written with `aelf lock --user` now joins the L0 lock set next to the repository's locks. The SessionStart baseline, the UserPromptSubmit memory block and its `<locked>` section, the PreCompact rebuild, the search-tool and agent-context hooks, and `aelf search` all inject it.
  - **Shared limits:** user locks count against the same token budget, relevance floor, and payload ceiling as repository locks. They don't get a budget of their own.
  - **Duplicates:** a user lock that near-duplicates a repository lock is dropped, and the repository copy is injected. The test is the one the `aelf lock` near-duplicate warning uses: Jaccard of at least 0.8 and Levenshtein ratio of at least 0.85.
  - **Order:** repository locks come first, then user locks, each in its store's order (newest lock first, then by id). Two runs over the same stores inject the same order.
  - **Read-only:** the hooks open the user store read-only and never create it. A missing user store changes nothing. A user store that can't be opened or read prints one warning to stderr, and the hooks inject the repository locks only. A user lock past its `--for` or `--until` window isn't injected, even before a write sweeps it.
  - **Benchmarks:** the retrieval benchmark, `aelf eval`, and the gap analysis in `aelf wonder` read the repository locks only, so their results don't depend on the user locks of the machine they run on.
