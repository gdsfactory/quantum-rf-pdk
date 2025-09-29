#!/bin/bash

# Script to convert jupytext scripts from notebooks/src/ to ipynb format in notebooks/
# This script is called by the pre-commit hook and by the Makefile

set -euo pipefail

# Change to the repository root directory
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

# Color definitions for output (only if stdout is a TTY)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    CYAN=''
    BOLD=''
    NC=''
fi

if [ "$#" -gt 0 ]; then
    echo -e "${CYAN}Converting jupytext notebooks for provided files to notebooks/${NC}"
    declare -a changed_files=()
    for py_file in "$@"; do
        if [ -f "$py_file" ]; then
            basename=$(basename "$py_file" .py)
            ipynb_path="notebooks/${basename}.ipynb"

            # Handle missing commands (exit 127) explicitly
            set +e
            show_changes_output=$(uvx jupytext --update --to ipynb "$py_file" --output "$ipynb_path" --show-changes 2>&1)
            uvx_status=$?
            if [ $uvx_status -ne 0 ]; then
                show_changes_output=$(jupytext --update --to ipynb "$py_file" --output "$ipynb_path" --show-changes 2>&1)
                jupytext_status=$?
                if [ $uvx_status -eq 127 ] && [ $jupytext_status -eq 127 ]; then
                    echo -e "${RED}Error:${NC} You need to have ${BOLD}'uv'${NC} or ${BOLD}'jupytext'${NC} in your PATH" >&2
                    exit 127
                fi
            fi
            set -e
            echo -e "$show_changes_output"

            # Record file if jupytext reported changes (stdout does not include 'Unchanged')
            if ! printf '%s' "$show_changes_output" | grep -q 'Unchanged'; then
                changed_files+=("$ipynb_path")

                echo -e "${BLUE}Converting${NC} ${BOLD}$py_file${NC} ${BLUE}to${NC} ${BOLD}${ipynb_path}${NC}"
                uvx jupytext --update --to ipynb "$py_file" --output "$ipynb_path" || jupytext --update --to ipynb "$py_file" --output "$ipynb_path"

            else
                # Ensure the notebook is tracked and has no unstaged changes
                if ! git ls-files --error-unmatch -- "$ipynb_path" > /dev/null 2>&1; then
                    echo -e "${RED}Error:${NC} ${BOLD}$ipynb_path${NC} is not tracked/staged in git."
                    echo -e "${YELLOW}Please add it to the index before committing:${NC} git add \"$ipynb_path\""
                    exit 1
                fi
                if git diff --name-only -- "$ipynb_path" | grep -q .; then
                    echo -e "${RED}Error:${NC} ${BOLD}$ipynb_path${NC} has unstaged changes."
                    echo -e "${YELLOW}Please stage or discard changes before committing.${NC}"
                    exit 1
                fi
            fi
        fi
    done
    if [ "${#changed_files[@]}" -gt 0 ]; then
        echo -e "${RED}Error:${NC} The following notebooks were modified by jupytext conversion:"
        for f in "${changed_files[@]}"; do
            echo -e "  - ${BOLD}$f${NC}"
        done
        echo -e "${YELLOW}Please add and commit these changes, then retry.${NC}"
        exit 1
    else
        echo -e "${GREEN}Conversion completed successfully${NC}"
    fi
else
    echo -e "${YELLOW}No Python files provided to convert${NC}"
fi
