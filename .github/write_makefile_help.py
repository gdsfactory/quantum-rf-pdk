#!/usr/bin/env python3
"""Generate Makefile help documentation."""

import re
import shutil
import subprocess

from qpdk.config import PATH

# Output file path
output_file = PATH.docs / "makefile_help.txt"

# Find the make executable
make_path = shutil.which("make")
if not make_path:
    msg = "make executable not found"
    raise FileNotFoundError(msg)

# Run make help and capture output
result = subprocess.run(
    [make_path, "help"],
    cwd=PATH.repo,
    capture_output=True,
    text=True,
    check=True,
)

# ANSI escape sequence pattern
ansi_escape = re.compile(r"\x1b\[[0-9;]*m")

# Pattern for make directory messages
make_dir_pattern = re.compile(r"make\[\d+\]:.*directory.*")

# Get the output
output = result.stdout

# Remove ANSI color codes
output = ansi_escape.sub("", output)

# Split into lines and filter
lines = output.split("\n")
filtered_lines = []

for line in lines:
    # Skip make directory messages
    if make_dir_pattern.search(line):
        continue
    filtered_lines.append(line)

# Join lines back
filtered_output = "\n".join(filtered_lines)

# Remove any trailing make directory message that might be concatenated
filtered_output = make_dir_pattern.sub("", filtered_output)

# Remove trailing whitespace and empty lines
filtered_output = filtered_output.rstrip()

# Write output to file
output_file.write_text(filtered_output)

print(f"Makefile help written to {output_file}")
