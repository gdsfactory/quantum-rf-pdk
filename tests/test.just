PYTEST_COMMAND := "uv run --extra models --extra graphics --extra hfss --extra qutip --group dev pytest -n auto"

# Check if Git LFS is available and pull LFS files
[private]
[group('test')]
check-lfs:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! command -v git-lfs >/dev/null 2>&1; then
        echo ""
        echo "Error: Git LFS is not installed!"
        echo ""
        echo "This repository uses Git LFS to store test data files."
        echo "Please install Git LFS before running tests:"
        echo ""
        echo "  Ubuntu/Debian:  sudo apt-get install git-lfs"
        echo "  RHEL/Rocky:     sudo dnf install git-lfs"
        echo "  macOS:          brew install git-lfs"
        echo "  Windows:        Download from https://git-lfs.github.com/"
        echo ""
        echo "After installing, run: git lfs install && git lfs pull"
        echo ""
        exit 1
    fi
    echo "Git LFS is available. Pulling LFS files…"
    git lfs pull

# Run the full test suite in parallel using pytest
[group('test')]
test *args: check-lfs
    {{ PYTEST_COMMAND }} {{ args }}

# Run optical port position tests (tests/test_pdk.py::test_optical_port_positions)
[group('test')]
test-ports:
    {{ PYTEST_COMMAND }} -s tests/test_pdk.py::test_optical_port_positions

# Run GDS regressions tests (tests/test_pdk.py)
[group('test')]
test-gds:
    {{ PYTEST_COMMAND }} -s tests/test_pdk.py

# Run GDS regressions tests (tests/test_pdk.py) and regenerate
[group('test')]
test-gds-force:
    {{ PYTEST_COMMAND }} -s tests/test_pdk.py --force-regen

# Run GDS regressions tests (tests/test_pdk.py) and stop at first failure
[group('test')]
test-gds-fail-fast:
    {{ PYTEST_COMMAND }} -s tests/test_pdk.py -x

# Run model regression tests (tests/test_models_regression.py)
[group('test')]
test-models:
    {{ PYTEST_COMMAND }} -s tests/test_models_regression.py

# Regenerate model regression reference files
[group('test')]
test-models-force:
    {{ PYTEST_COMMAND }} -s tests/test_models_regression.py --force-regen

# Run HFSS simulation tests (requires HFSS to be installed)
[group('test')]
test-hfss *args: check-lfs
    uv run --all-extras --group dev pytest -m hfss {{ args }}
