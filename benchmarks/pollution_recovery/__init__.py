"""Pollution-recovery benchmark (#1011 doc-chunk signal/noise).

Measures whether user-stated facts survive retrieval ranking when the
store is flooded with keyword-overlapping document chunks — the
"doc-chunk drowning" problem from #1011. Deterministic: synthetic
fixture, BM25 retrieval, no LLM, no network.

The harness is origin-agnostic about *how* a fix works; it just scores
`retrieve()` output, so it can compare the BM25 baseline against any
default-off rerank lane (e.g. the #1013 origin-tier rerank).
"""
