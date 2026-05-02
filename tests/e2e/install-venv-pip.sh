#!/usr/bin/env bash
# Install the working tree into a fresh venv via pip. Single dumb line per
# spec; the venv's bin/ is exported on PATH by the workflow step.
# Used by the e2e workflow's `venv-pip` matrix leg (#334).
set -euo pipefail
python -m venv .e2e-venv
.e2e-venv/bin/pip install --quiet --upgrade pip
.e2e-venv/bin/pip install --quiet .
