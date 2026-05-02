#!/usr/bin/env bash
# Install the working tree as an isolated uv-tool. Single dumb line per spec.
# Used by the e2e workflow's `uv-tool` matrix leg (#334).
set -euo pipefail
uv tool install --force .
