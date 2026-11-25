### Removed

- **The commit hook stores nothing ([#2010](https://github.com/robotrocketscience/aelfrice/issues/2010)).** After each commit, the git `post-commit` hook that `aelf setup` installs still starts `aelf-commit-ingest`, and the command now exits 0 without opening a store or writing a file. It no longer stores the words around a relation phrase such as "X supersedes Y" as beliefs, adds an edge between them, or writes `ingest_log`, `sessions`, or corroboration rows. The `PostToolUse:Bash` form, which older settings and the Codex host run, reads its payload and stores nothing too. `ingest_triples` and the commit hook's own pattern bank are removed from `triple_extractor`. `extract_triples` keeps the bank the context rebuilder queries with, so the query it records for a prompt doesn't change. Beliefs and edges that an earlier release stored stay in your store, and retrieval still reads them. The detector-thresholds manifest moves to version 9, and `benchmarks/triple_precision_1376.py` goes with the bank it measured.

  To upgrade, you don't need to do anything. A `post-commit` block from an earlier release runs the same command, so it stores nothing either. Its comment says that it stores phrases until you run `aelf setup` in that repository again, which rewrites the comment. The command line stays the same while setup resolves `aelf-commit-ingest` to the same path; if that path changed, setup writes the new one.

  - On by default: the hook is installed by default, as before, and it writes nothing.
  - Reached from the hook path: yes. The `post-commit` block and the `PostToolUse:Bash` entry both run the changed entry point.
  - Measured on: nothing measured.
