#!/usr/bin/env python3
"""Generate Justfile help documentation."""

import re
import shutil
import subprocess

from qpdk.config import PATH

# Output file path
output_file = PATH.docs / "justfile_help.txt"

# Find the just executable
just_path = shutil.which("just")
if not just_path:
    msg = "just executable not found"
    raise FileNotFoundError(msg)

# Run just --list and capture output
result = subprocess.run(  # noqa: S603
    [just_path, "--list"],
    cwd=PATH.repo,
    capture_output=True,
    text=True,
    check=True,
)

# ANSI escape sequence pattern
ansi_escape = re.compile(r"\x1b\[[0-9;]*m")

# Get the output
output = result.stdout

# Remove ANSI color codes
output = ansi_escape.sub("", output)

# Remove trailing whitespace and empty lines
output = output.rstrip()

# Write output to file
output_file.write_text(output)

print(f"Justfile help written to {output_file}")
