### CI

- **The weekly mutation job runs as time-boxed shards, so it finishes and reports ([#1747](https://github.com/robotrocketscience/aelfrice/issues/1747)).**
  - **The cause:** every weekly run hit the job's 240-minute limit while mutmut was still generating mutants, so no mutant ever ran. `cli.py` alone produced 17,338 mutants in a 650 MB file that mutmut couldn't finish parsing. 6,137 of those came from `build_parser`, which is argparse declarations and help text.
  - **Shards:** `scripts/mutation_shards.py plan` counts each function's mutants with mutmut's own generator and assigns them to 8 shards of about 7,660 mutants each. A file with more than 4,000 mutants is split by function across shards. Each shard is its own matrix job, scoped with `only_mutate` and with `# pragma: no mutate block` on the functions of a split file that belong to other shards.
  - **Excluded:** `cli.py`'s `build_parser` is mutated in no shard. If it's renamed, the plan stops, and a test in the ordinary suite fails first.
  - **Time box:** each shard's `mutmut run` gets SIGINT after 330 minutes. mutmut keeps every result it saved, so a shard that runs out of time reports what it checked and warns. It fails only when it lists no mutant or checks none.
  - **Report:** a last job merges the shards' reports into one summary, and fails when a planned shard left no report or no shard checked a mutant.
  - **Measured locally:** the plan takes 38 s for 61,318 mutants. After the pragmas, mutmut's own mutant count matches the plan for every file in every shard. Generating the largest piece of `cli.py` takes about 4 minutes, against more than 23 CPU-minutes for the whole file before.
