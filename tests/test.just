# Helper to get extras based on python version
[private]
get-extras:
    #!/usr/bin/env bash
    EXTRAS="--extra models --extra graphics --extra qutip"
    # Only include hfss if python version is not 3.14 (due to vtk dependency)
    # Check UV_PYTHON env var first, then actual python version
    VERSION="${UV_PYTHON:-$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')}"
    if [[ "$VERSION" != "3.14"* ]]; then
        EXTRAS="$EXTRAS --extra hfss"
    fi
    echo "$EXTRAS"

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
    #!/usr/bin/env bash
    EXTRAS=$(just get-extras)
    uv run $EXTRAS --group dev pytest -n auto {{ args }}

# Run optical port position tests (tests/test_pdk.py::test_optical_port_positions)
[group('test')]
test-ports:
    #!/usr/bin/env bash
    EXTRAS=$(just get-extras)
    uv run $EXTRAS --group dev pytest -s tests/test_pdk.py::test_optical_port_positions

# Run GDS regressions tests (tests/test_pdk.py)
[group('test')]
test-gds:
    #!/usr/bin/env bash
    EXTRAS=$(just get-extras)
    uv run $EXTRAS --group dev pytest -s tests/test_pdk.py

# Run GDS regressions tests (tests/test_pdk.py) and regenerate
[group('test')]
test-gds-force:
    #!/usr/bin/env bash
    EXTRAS=$(just get-extras)
    uv run $EXTRAS --group dev pytest -s tests/test_pdk.py --force-regen

# Run GDS regressions tests (tests/test_pdk.py) and stop at first failure
[group('test')]
test-gds-fail-fast:
    #!/usr/bin/env bash
    EXTRAS=$(just get-extras)
    uv run $EXTRAS --group dev pytest -s tests/test_pdk.py -x

# Run HFSS simulation tests (requires HFSS to be installed)
[group('test')]
test-hfss *args: check-lfs
    #!/usr/bin/env bash
    VERSION="${UV_PYTHON:-$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')}"
    if [[ "$VERSION" == "3.14"* ]]; then
        echo "Skipping HFSS tests on Python 3.14 (vtk incompatible)"
        exit 0
    fi
    uv run --all-extras --group dev pytest -m hfss {{ args }}
