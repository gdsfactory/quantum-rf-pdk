#!/bin/bash

# Pre-commit hook to check that the version ranges documented in AGENTS.md stay
# in sync with pyproject.toml. Checks the `requires-python` range and the
# gdsfactory dependency constraint, and fails when AGENTS.md drifts from
# pyproject.toml (e.g. after a dependency bump without a docs update).

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

# A missing AGENTS.md is a repo defect, not a legitimate skip: the hook only
# runs when AGENTS.md or pyproject.toml changed, so the file should exist.
if [ ! -f "AGENTS.md" ]; then
    echo -e "${RED}Error:${NC} AGENTS.md not found — nothing to check against pyproject.toml." >&2
    exit 1
fi

# Parse pyproject.toml with tomllib instead of pattern-matching its text
# formatting (quoting style, extras such as gdsfactory[cad], whitespace).
# Prints "<requires-python>\t<gdsfactory-spec>" (tab-separated, since a spec
# may itself contain spaces); a field is empty only when the corresponding key
# is genuinely absent from pyproject.toml. A parse failure
# exits nonzero, which aborts this script under 'set -e' — the hook must fail
# loudly rather than silently skip a check.
pyproject_values=$(python3 - <<'PYEOF'
import sys

try:
    import tomllib
except ImportError:
    print("Error: python3 lacks tomllib (Python 3.11+ required to parse pyproject.toml)", file=sys.stderr)
    sys.exit(2)

try:
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
except (OSError, tomllib.TOMLDecodeError) as exc:
    print(f"Error: could not parse pyproject.toml: {exc}", file=sys.stderr)
    sys.exit(2)

requires_python = data.get("project", {}).get("requires-python") or ""
gdsfactory_spec = ""
for dep in data.get("project", {}).get("dependencies", []):
    dep = dep.split(";", 1)[0].strip()  # drop environment markers
    if dep.lower().startswith("gdsfactory"):
        rest = dep[len("gdsfactory"):].lstrip()
        if rest.startswith("["):  # drop extras such as gdsfactory[cad]
            rest = rest[rest.index("]", 1) + 1 :].lstrip()
        gdsfactory_spec = rest
        break

print(requires_python + "\t" + gdsfactory_spec)
PYEOF
)
requires_python="${pyproject_values%%$'\t'*}"
gdsfactory_spec="${pyproject_values#*$'\t'}"

# Extract the ranges documented in the Technology and Tools section of
# AGENTS.md. AGENTS.md escapes '<' as '\<' in markdown, so unescape before
# comparing. The ranges must equal the pyproject.toml ones: containment is not
# enough, because a documented range that merely contains the pyproject spec
# (e.g. an upper bound that no longer exists) is drift.
agents_section=$(awk '/^## Technology and Tools/{flag=1; next} /^## /{flag=0} flag' AGENTS.md)
python_doc=$(grep -F '**Python versions**' <<<"$agents_section" | sed -n 's/.*(currently \([^)]*\)).*/\1/p' | head -n 1 | sed 's/\\\([<>]\)/\1/g' || true)
gdsfactory_doc=$(grep -F 'gdsfactory (' <<<"$agents_section" | sed -n 's/.*gdsfactory (\([^)]*\)).*/\1/p' | head -n 1 | sed 's/\\\([<>]\)/\1/g' || true)

drift=0
if [ -n "$requires_python" ]; then
    if [ "$python_doc" != "$requires_python" ]; then
        echo -e "${RED}Error:${NC} AGENTS.md documents '${python_doc:-nothing}' for Python versions but pyproject.toml specifies ${BOLD}$requires_python${NC}" >&2
        drift=1
    fi
else
    echo -e "${YELLOW}Warning:${NC} no requires-python key in pyproject.toml — skipping that check." >&2
fi
if [ -n "$gdsfactory_spec" ]; then
    if [ "$gdsfactory_doc" != "$gdsfactory_spec" ]; then
        echo -e "${RED}Error:${NC} AGENTS.md documents '${gdsfactory_doc:-nothing}' for gdsfactory but pyproject.toml specifies ${BOLD}$gdsfactory_spec${NC}" >&2
        drift=1
    fi
else
    echo -e "${YELLOW}Warning:${NC} no gdsfactory dependency in pyproject.toml — skipping that check." >&2
fi

if [ "$drift" -eq 1 ]; then
    echo -e "" >&2
    echo -e "${YELLOW}Update the version ranges in AGENTS.md (Technology and Tools section) to match pyproject.toml.${NC}" >&2
    exit 1
else
    echo -e "${GREEN}AGENTS.md version ranges match pyproject.toml${NC}"
    exit 0
fi
