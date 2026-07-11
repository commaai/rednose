#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd "$DIR"

# *** build ***
scons -j8

# *** lint + test ***
ruff check .
mypy --ignore-missing-imports .
pytest

# *** all done ***
GREEN='\033[0;32m'
NC='\033[0m'
printf "\n${GREEN}All good!${NC} Finished lint and test in ${SECONDS}s\n"
