#!/bin/bash

# Script to convert jupytext scripts from notebooks/src/ to ipynb format in notebooks/
# This script is called by the pre-commit hook and by the Makefile

set -euo pipefail

# Change to the repository root directory
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

# Create the notebooks output directory if it doesn't exist
mkdir -p notebooks

# Convert all .py files from notebooks/src/ to .ipynb files in notebooks/
if [ -d "notebooks/src" ] && [ "$(ls -A notebooks/src/*.py 2>/dev/null)" ]; then
    echo "Converting jupytext notebooks from notebooks/src/ to notebooks/"
    for py_file in notebooks/src/*.py; do
        if [ -f "$py_file" ]; then
            basename=$(basename "$py_file" .py)
            echo "  Converting $py_file to notebooks/${basename}.ipynb"
            uvx jupytext --to ipynb "$py_file" --output "notebooks/${basename}.ipynb"
        fi
    done
    echo "Conversion completed successfully"
else
    echo "No Python files found in notebooks/src/ directory"
fi
