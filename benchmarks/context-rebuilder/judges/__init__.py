"""Judge stages for the context-rebuilder eval harness.

Each judge module writes a per-run request file and reads a per-run
response file. The host CLI (or operator skill / MCP host) does the
model dispatch in its own context — aelfrice and the benchmarks code
never import a provider SDK directly. Mirrors `/aelf:onboard`'s
polymorphic LLM-classify pattern.
"""
