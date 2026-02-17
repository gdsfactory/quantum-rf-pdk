# List available commands
default:
    @just --list

# Install the package and all development dependencies
install:
    uv sync --all-extras --all-groups

# Remove samples folder
rm-samples:
    rm -rf qpdk/samples

# Clean up all build, test, coverage and Python artifacts
clean:
    rm -rf dist build *.egg-info docs/_build docs/notebooks

###########
# Testing #
###########

PYTEST_COMMAND := "uv run --group dev pytest"

# Check if Git LFS is available and pull LFS files
check-lfs:
    @echo "Checking for Git LFS..."
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
    @echo "Git LFS is available. Pulling LFS files..."
    @git lfs pull

# Run the full test suite in parallel using pytest
test: check-lfs
    {{PYTEST_COMMAND}} -n auto

# Run GDS regressions tests (tests/test_pdk.py)
test-gds:
    {{PYTEST_COMMAND}} -s tests/test_pdk.py

# Run GDS regressions tests (tests/test_pdk.py) and regenerate
test-gds-force:
    {{PYTEST_COMMAND}} -s tests/test_pdk.py --force-regen

# Run GDS regressions tests (tests/test_pdk.py) and stop at first failure
test-gds-fail-fast:
    {{PYTEST_COMMAND}} -s tests/test_pdk.py -x

# Update pre-commit hooks to the latest revisions
update-pre:
    uvx prek autoupdate -j `expr $(nproc) / 2 + $(expr $(nproc) % 2)`

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
    uv run --group docs .github/write_models.py

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

# Setup IPython configuration for documentation build
setup-ipython-config:
    mkdir -p ~/.ipython/profile_default
    cp docs/ipython_config.py ~/.ipython/profile_default/ipython_config.py
    mkdir -p ~/.config/matplotlib/stylelib/
    cp docs/qpdk.mplstyle ~/.config/matplotlib/stylelib/qpdk.mplstyle

# Build the HTML documentation
docs: write-cells write-models write-justfile-help copy-sample-notebooks
    uv run --group docs jb build docs

# Setup LaTeX for PDF documentation
docs-latex: write-cells write-models write-justfile-help copy-sample-notebooks
    uv run --group docs jb build docs --builder latex

# Build PDF documentation (requires a TeXLive installation)
docs-pdf: docs-latex
    cd "docs/_build/latex" && XINDYOPTS="-M sphinx.xdy" latexmk -pdfxe -xelatex -interaction=nonstopmode -f -file-line-error || (test -f qpdk.pdf && echo "PDF generated despite warnings" && exit 0)
