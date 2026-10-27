#!/bin/bash

# Pre-commit hook to lint Verilog-A models with OpenVAF-Reloaded (openvaf-r).
# Each file is compiled with --dry-run, so the full front end (preprocessor,
# name resolution, type checking, lowering) runs without writing an .osdi.
# Lint warnings are promoted to errors so that they cannot accumulate.
#
# The compiler is taken from the OPENVAF environment variable, as in
# qpdk.simulation.vacask, and otherwise from PATH. Release binaries are at
# https://github.com/OpenVAF/OpenVAF-Reloaded/releases, and the VACASK
# packages for Linux and Windows bundle one in their bin/ directory.

set -euo pipefail

openvaf="${OPENVAF:-openvaf-r}"
if ! command -v "$openvaf" >/dev/null 2>&1; then
    echo "Error: '$openvaf' not found. Install OpenVAF-Reloaded and put openvaf-r on PATH," >&2
    echo "or point the OPENVAF environment variable at the binary." >&2
    echo "Releases: https://github.com/OpenVAF/OpenVAF-Reloaded/releases" >&2
    exit 1
fi

status=0
for file in "$@"; do
    if ! "$openvaf" --dry-run --deny warnings "$file"; then
        status=1
    fi
done
exit "$status"
