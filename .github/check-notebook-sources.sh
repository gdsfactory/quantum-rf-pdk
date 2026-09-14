#!/bin/bash

# Script to check that all .ipynb files in notebooks/ have corresponding source files in notebooks/src/
# This pre-commit hook ensures that notebooks are properly tracked as jupytext source scripts.
# Source files use `.py` for Python-kernel notebooks and `.m` for MATLAB-kernel notebooks.
#
# It also checks that no notebook contains a leaked jupytext YAML header cell. jupytext only
# recognises the header when its `---` fence is on its own line; if a formatter reflows the
# source's comment block the fence merges with the next line, and the whole header silently
# becomes a regular cell that then renders in the built documentation.

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

    # Check if a corresponding source file exists in notebooks/src/ for any
    # supported jupytext extension (.py for Python, .m for MATLAB).
    if [ ! -f "notebooks/src/${basename}.py" ] && [ ! -f "notebooks/src/${basename}.m" ]; then
        orphaned_notebooks+=("$ipynb_file")
    fi
done < <(find notebooks -maxdepth 1 -type f -name "*.ipynb" -print0)

# Report results
if [ "${#orphaned_notebooks[@]}" -gt 0 ]; then
    echo -e "${RED}Error:${NC} Found ${BOLD}${#orphaned_notebooks[@]}${NC} notebook(s) without corresponding source files in ${BOLD}notebooks/src/${NC}:" >&2
    for nb in "${orphaned_notebooks[@]}"; do
        basename=$(basename "$nb" .ipynb)
        echo -e "  - ${BOLD}$nb${NC} is missing ${BOLD}notebooks/src/${basename}.py${NC} (or ${BOLD}.m${NC})" >&2
    done
    echo -e "" >&2
    echo -e "${YELLOW}All notebooks in notebooks/ must have a corresponding jupytext source file in notebooks/src/ (.py for Python, .m for MATLAB).${NC}" >&2
    echo -e "${YELLOW}Please create the source file or remove the orphaned notebook.${NC}" >&2
    exit 1
fi

echo -e "${GREEN}All notebooks have corresponding source files${NC}"

# Check that no notebook carries the jupytext YAML header as a visible cell.
# Prefer uv (as the notebook conversion hook does); fall back to the system
# interpreter since the checker is standard-library only.
if command -v uv >/dev/null 2>&1; then
    header_check=(uv run --script .github/check_jupytext_header.py)
else
    header_check=(python3 .github/check_jupytext_header.py)
fi

declare -a leaked_notebooks=()
while IFS= read -r -d '' ipynb_file; do
    if "${header_check[@]}" "$ipynb_file"; then
        continue
    fi
    leaked_notebooks+=("$ipynb_file")
done < <(find notebooks -maxdepth 1 -type f -name "*.ipynb" -print0)

if [ "${#leaked_notebooks[@]}" -gt 0 ]; then
    echo -e "${RED}Error:${NC} Found ${BOLD}${#leaked_notebooks[@]}${NC} notebook(s) containing the jupytext YAML header as a cell:" >&2
    for nb in "${leaked_notebooks[@]}"; do
        echo -e "  - ${BOLD}$nb${NC}" >&2
    done
    echo -e "" >&2
    echo -e "${YELLOW}The header must stay a comment block in notebooks/src/ with its '---' fence on its own line.${NC}" >&2
    echo -e "${YELLOW}Check that no formatter reflowed it, then regenerate the notebook.${NC}" >&2
    exit 1
fi

echo -e "${GREEN}No notebooks leak their jupytext header${NC}"
exit 0
