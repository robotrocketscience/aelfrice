#!/usr/bin/env bash
# Install the working tree via pipx. Single dumb line per spec.
# Used by the e2e workflow's `pipx` matrix leg (#334).
set -euo pipefail
pipx install --force .
