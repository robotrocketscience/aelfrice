### Performance

- **The prompt hook asks git for the repository's location once instead of five times ([#1734](https://github.com/robotrocketscience/aelfrice/issues/1734)).**
  - **Before:** every lookup of the store, the repository identity, and the turn log ran its own `git rev-parse --git-common-dir`. One prompt ran it 5 times, at about 44 ms each on a loaded machine.
  - **Now:** the answer is cached for the life of the process, keyed on the working directory and the environment variables git reads to find a repository (`GIT_DIR`, `GIT_COMMON_DIR`, `GIT_WORK_TREE`, `GIT_CEILING_DIRECTORIES`, and `GIT_DISCOVERY_ACROSS_FILESYSTEM`). A "not in a repository" answer is cached too, so the fallback stays the same on every call.
  - **Measured:** over 15 interleaved runs of the prompt hook in a fresh repository, the median wall time fell from 458 ms to 256 ms.
  - **One limit:** a process that runs `git init` in a directory it has already looked up keeps the old answer until it calls `db_paths.clear_git_common_dir_cache()`. No aelfrice command does this; the test suite clears the cache before each test.
