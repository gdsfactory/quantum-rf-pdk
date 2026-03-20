set dotenv-load := true

pdk := env('pdk', 'qpdk')
cpus := num_cpus()

import 'tests/test.just'
import 'docs/docs.just'

# List available commands
default:
    @just --list

# Install the package and all development dependencies
[group('setup')]
install:
    @uv sync --all-extras

# Install KLayout technology files for the PDK
[group('setup')]
install-tech:
    @uv run --dev {{ pdk }}/install_tech.py

# Clean up all build, test, coverage and Python artifacts
[confirm]
[group('setup')]
clean:
    @rm -rf dist build *.egg-info docs/_build docs/notebooks

# Update pre-commit hooks to the latest revisions
[group('lint')]
update-pre:
    @uvx prek autoupdate -j $(( {{ cpus }} / 2 + {{ cpus }} % 2 ))

# Run all pre-commit hooks on all files
[group('lint')]
run-pre:
    uvx prek run --all-files

# Build the Python package (install build tool and create dist)
[group('build')]
build:
    @rm -rf dist
    uv build

# Run all tests, pre-commit hooks, build wheel and documentation in parallel
[group('all')]
[parallel]
all: test run-pre build docs
