#!/usr/bin/env python3
"""Generate Makefile help documentation."""

import subprocess
from pathlib import Path

from qpdk.config import PATH

# Output file path
output_file = PATH.docs / "makefile_help.txt"

# Run make help and capture output
result = subprocess.run(
    ["make", "help"],
    cwd=PATH.repo,
    capture_output=True,
    text=True,
    check=True,
)

# Filter out make debug messages and empty lines at the end
lines = result.stdout.split("\n")
filtered_lines = [
    line for line in lines
    if not line.startswith("make[") and not line.startswith("make: ")
]

# Remove trailing empty lines
while filtered_lines and filtered_lines[-1] == "":
    filtered_lines.pop()

filtered_output = "\n".join(filtered_lines)

# Write output to file
output_file.write_text(filtered_output)

print(f"Makefile help written to {output_file}")
