### Documentation

- **The search hook's docs no longer say the model sees memory before it chooses a search ([#1646](https://github.com/robotrocketscience/aelfrice/issues/1646)).** The hook runs before the search tool, but the host shows its results next to the tool's own output ([hooks reference](https://code.claude.com/docs/en/hooks)). By then the model has already chosen the tool and its query. The results shape the model's next step, not the search that fired the hook, and they don't let it skip that search. The module docstring, the search-tool comment, the `aelf setup --search-tool` help text, the design note and its diagram, and a test docstring now say so. The hook's own closing line, which the model reads, said "you may skip the tool call". It now says there's no need to search further if memory answers the question, and otherwise to use the tool's result to fill the gaps. Nothing else changed.

  The new closing line is 21 bytes longer than the old one, so each search-hook block grows by 21 bytes. The search-tool block measures 3,326 bytes before the budget and 2,630 after; the search-tool-bash block measures 1,931 before and 1,699 after. The 5.0.0 notes keep the counts measured at that release, and the markers that bound those counts to `benchmarks/injection_budget_bytes.py` move here, re-derived.
  <!-- derived: benchmarks/injection_budget_bytes.py#search_tool_bytes_before = 3326 -->
  <!-- derived: benchmarks/injection_budget_bytes.py#search_tool_bytes_after = 2630 -->
  <!-- derived: benchmarks/injection_budget_bytes.py#search_tool_bash_bytes_before = 1931 -->
  <!-- derived: benchmarks/injection_budget_bytes.py#search_tool_bash_bytes_after = 1699 -->
