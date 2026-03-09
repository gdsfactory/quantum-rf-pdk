# List available commands
default:
    @just --list

# Install the package and all development dependencies
install:
    uv sync --all-extras

# Install KLayout technology files for the PDK
install-tech:
    uv run --dev qpdk/install_tech.py

# Remove samples folder
rm-samples:
    rm -rf qpdk/samples

# Clean up all build, test, coverage and Python artifacts
clean:
    rm -rf dist build *.egg-info docs/_build docs/notebooks

###########
# Testing #
###########

PYTEST_COMMAND := "uv run --all-extras --group dev pytest -n auto"

# Check if Git LFS is available and pull LFS files
check-lfs:
    @echo "Checking for Git LFS…"
    @if ! command -v git-lfs >/dev/null 2>&1; then \
    echo ""; \
    echo "Error: Git LFS is not installed!"; \
    echo ""; \
    echo "This repository uses Git LFS to store test data files."; \
    echo "Please install Git LFS before running tests:"; \
    echo ""; \
    echo "  Ubuntu/Debian:  sudo apt-get install git-lfs"; \
    echo "  RHEL/Rocky:     sudo dnf install git-lfs"; \
    echo "  macOS:          brew install git-lfs"; \
    echo "  Windows:        Download from https://git-lfs.github.com/"; \
    echo ""; \
    echo "After installing, run: git lfs install && git lfs pull"; \
    echo ""; \
    exit 1; \
    fi
    @echo "Git LFS is available. Pulling LFS files…"
    @git lfs pull

# Run the full test suite in parallel using pytest
test *args: check-lfs
    {{PYTEST_COMMAND}} {{args}}

# Run optical port position tests (tests/test_pdk.py::test_optical_port_positions)
test-ports:
	{{PYTEST_COMMAND}} -s tests/test_pdk.py::test_optical_port_positions

# Run GDS regressions tests (tests/test_pdk.py)
test-gds:
    {{PYTEST_COMMAND}} -s tests/test_pdk.py

# Run GDS regressions tests (tests/test_pdk.py) and regenerate
test-gds-force:
    {{PYTEST_COMMAND}} -s tests/test_pdk.py --force-regen

# Run GDS regressions tests (tests/test_pdk.py) and stop at first failure
test-gds-fail-fast:
    {{PYTEST_COMMAND}} -s tests/test_pdk.py -x

# Run HFSS simulation tests (requires HFSS to be installed)
test-hfss *args: check-lfs
    uv run --all-extras --group dev pytest -m hfss {{args}}

# Update pre-commit hooks to the latest revisions
update-pre:
    #!/usr/bin/env bash
    set -euo pipefail
    # Calculate number of jobs: (nproc / 2) rounded up
    NPROC=$(nproc)
    JOBS=$(($NPROC / 2 + $NPROC % 2))
    uvx prek autoupdate -j "$JOBS"

# Run all pre-commit hooks on all files
run-pre:
    uvx prek run --all-files

# Build the Python package (install build tool and create dist)
build:
    rm -rf dist
    uv build

#################
# Documentation #
#################

# Write cell outputs into documentation notebooks (used when building docs)
write-cells:
    uv run --group docs .github/write_cells.py

# Write model outputs into documentation notebooks (used when building docs)
write-models:
    uv run --extra models --group docs .github/write_models.py

# Write Justfile help output to documentation
write-justfile-help:
    uv run --group docs .github/write_justfile_help.py

# Convert jupytext scripts from notebooks/src to ipynb format in notebooks
convert-notebooks:
    ./.github/convert-notebooks.sh notebooks/src/*.py

# Copy all sample scripts to use as notebooks docs
copy-sample-notebooks:
    mkdir -p docs/notebooks
    cp notebooks/src/*.py docs/notebooks/

# Temporarily setup IPython configuration for documentation build
setup-ipython-config-temporary-before:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p ~/.ipython/profile_default
    if [ -f ~/.ipython/profile_default/ipython_config.py ]; then
        mv ~/.ipython/profile_default/ipython_config.py ~/.ipython/profile_default/ipython_config.py.bak
    fi
    cp docs/ipython_config.py ~/.ipython/profile_default/ipython_config.py

    mkdir -p ~/.config/matplotlib/stylelib/
    if [ -f ~/.config/matplotlib/stylelib/qpdk.mplstyle ]; then
        mv ~/.config/matplotlib/stylelib/qpdk.mplstyle ~/.config/matplotlib/stylelib/qpdk.mplstyle.bak
    fi
    cp docs/qpdk.mplstyle ~/.config/matplotlib/stylelib/qpdk.mplstyle

# Restore original IPython configuration
setup-ipython-config-temporary-after:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -f ~/.ipython/profile_default/ipython_config.py.bak ]; then
        mv ~/.ipython/profile_default/ipython_config.py.bak ~/.ipython/profile_default/ipython_config.py
    else
        rm -f ~/.ipython/profile_default/ipython_config.py
    fi

    if [ -f ~/.config/matplotlib/stylelib/qpdk.mplstyle.bak ]; then
        mv ~/.config/matplotlib/stylelib/qpdk.mplstyle.bak ~/.config/matplotlib/stylelib/qpdk.mplstyle
    else
        rm -f ~/.config/matplotlib/stylelib/qpdk.mplstyle
    fi

# Shared prerequisites for building documentation (runs in parallel)
[parallel]
docs-prerequisites: write-cells write-models write-justfile-help copy-sample-notebooks

# Build the HTML documentation
docs: docs-prerequisites setup-ipython-config-temporary-before
    #!/usr/bin/env bash
    set -euo pipefail
    trap 'just setup-ipython-config-temporary-after' EXIT INT TERM
    uv run --all-extras --group docs jb build docs

# Setup LaTeX for PDF documentation
docs-latex: docs-prerequisites setup-ipython-config-temporary-before
    #!/usr/bin/env bash
    set -euo pipefail
    trap 'just setup-ipython-config-temporary-after' EXIT INT TERM
    uv run --all-extras --group docs jb build docs --builder latex

# Build PDF documentation (requires a TeXLive installation)
docs-pdf: docs-latex
    #!/usr/bin/env bash
    set -euo pipefail
    cd "docs/_build/latex"
    XINDYOPTS="-M sphinx.xdy" latexmk -pdfxe -xelatex -interaction=nonstopmode -f -file-line-error || {
        if [ -f qpdk.pdf ]; then
            echo "PDF generated despite warnings"
            exit 0
        else
            exit 1
        fi
    }
