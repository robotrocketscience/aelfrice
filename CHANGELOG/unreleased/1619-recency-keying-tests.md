### Internal

- **The git-recency lookup is tested with more than one file ([#1619](https://github.com/robotrocketscience/aelfrice/issues/1619)).** Every earlier recency test used one file and a one-entry map, so a lookup that gave every file the first entry's date passed the whole suite. New two-file fixtures, with distinct dates and the other file listed first, cover `extract_ast`, `extract_filesystem`, and the onboard handshake through `accept_classifications`. That mutant now fails two tests at each lookup site. No production code changed. The shipped lookups were already correct.
