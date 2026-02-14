#!/bin/bash

# Script to check that all .ipynb files in notebooks/ have corresponding .py source files in notebooks/src/
# This pre-commit hook ensures that notebooks are properly tracked as jupytext Python scripts

set -euo pipefail

# Change to the repository root directory
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

# Color definitions for output (only if stdout is a TTY)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BOLD=''
    NC=''
fi

# Find all .ipynb files in the notebooks directory (excluding subdirectories)
declare -a orphaned_notebooks=()

# Find all .ipynb files in notebooks/ (not in subdirectories)
while IFS= read -r -d '' ipynb_file; do
    # Get the basename without extension
    basename=$(basename "$ipynb_file" .ipynb)

    # Check if corresponding .py file exists in notebooks/src/
    py_source="notebooks/src/${basename}.py"

    if [ ! -f "$py_source" ]; then
        orphaned_notebooks+=("$ipynb_file")
    fi
done < <(find notebooks -maxdepth 1 -type f -name "*.ipynb" -print0)

# Report results
if [ "${#orphaned_notebooks[@]}" -gt 0 ]; then
    echo -e "${RED}Error:${NC} Found ${BOLD}${#orphaned_notebooks[@]}${NC} notebook(s) without corresponding source files in ${BOLD}notebooks/src/${NC}:" >&2
    for nb in "${orphaned_notebooks[@]}"; do
        basename=$(basename "$nb" .ipynb)
        echo -e "  - ${BOLD}$nb${NC} is missing ${BOLD}notebooks/src/${basename}.py${NC}" >&2
    done
    echo -e "" >&2
    echo -e "${YELLOW}All notebooks in notebooks/ must have corresponding jupytext Python source files in notebooks/src/${NC}" >&2
    echo -e "${YELLOW}Please create the source file or remove the orphaned notebook.${NC}" >&2
    exit 1
else
    echo -e "${GREEN}All notebooks have corresponding source files${NC}"
    exit 0
fi
