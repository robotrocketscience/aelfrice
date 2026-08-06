"""uri_baki retest harness — issue #153.

Offline-only research benchmark. Does not land as part of `aelf bench`.

Self-contained since #1369: the post-rank adjusters it scores live in
`adjusters.py` beside it, not in `src/aelfrice/`. They never had a
production importer and the retest verdict in `RESULTS.md` is an honest
negative, so the package no longer ships them.
"""
