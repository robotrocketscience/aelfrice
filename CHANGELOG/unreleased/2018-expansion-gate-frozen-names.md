### Fixed

- **The BFS expansion gate matches a fixed list of edge-type names ([#2018](https://github.com/robotrocketscience/aelfrice/issues/2018)).** The gate keeps breadth-first search (BFS) for a prompt that names an edge type such as `SUPPORTS`, because the name counts as a structural marker. It built that pattern from `models.EDGE_TYPES` at import time, so adding, renaming, or removing an edge type changed which prompts skip BFS without any change to the gate. `EDGE_TYPES` is a frozenset, so the pattern text also changed with `PYTHONHASHSEED`, although matching didn't. The gate now builds the pattern from a literal tuple of today's 10 names in a fixed order, longest first, with names of equal length in alphabetical order. The detector-thresholds manifest pins the pattern as its first entry that gates a read instead of an edge write, and it moves to version 10. The pattern matches the same text as before, so retrieval output doesn't change for any prompt.

  To upgrade, you don't need to do anything.

  - On by default: yes. The gate runs on every retrieval, and its verdict matters only on a call that turns BFS on, which is off by default.
  - Reached from the hook path: yes. The hook's `retrieve()` reaches `retrieve_with_tiers`, which calls the gate.
  - Measured on: nothing measured. A test compares the new pattern with the old one on 99 probe strings and finds the same matches and spans.
