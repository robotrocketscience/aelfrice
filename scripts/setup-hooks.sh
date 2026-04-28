#!/bin/sh
# Wire .githooks/ as the git hooks directory for this repo.
# Run once after cloning:  sh scripts/setup-hooks.sh
set -e
git config core.hooksPath .githooks
echo "Hooks installed. core.hooksPath = .githooks"
