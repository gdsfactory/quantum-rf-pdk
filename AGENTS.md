# AI Agent Best Practices for Superconducting Quantum PDK

This is a Python-based superconducting microwave Process Design Kit (PDK) built on gdsfactory for designing quantum
devices and circuits. It provides components like transmons, resonators, couplers, and quantum layouts. Please follow
these guidelines when contributing:

## Code Standards

### Required Before Each Commit

- **ALWAYS** run pre-commit hooks before committing any changes by using `uvx prek run --all-files`
- Pre-commit hooks will automatically run code formatting (ruff), linting, YAML formatting, and other quality checks
- All pre-commit hooks must pass before any changes can be committed

### Development Flow

- **Install dependencies**: `make install` (uses `uv sync --all-extras`)
- **Test**: `make test` (runs `uv run pytest -n auto` with parallel execution)
- **Test GDS components**: `make test-gds` (runs only GDS regression tests in `tests/test_pdk.py`)
- **Regenerate GDS regression tests**: `make test-gds-force` (regenerates reference GDS files)
- **Test with fail-fast**: `make test-gds-fail-fast` (stops at first test failure for debugging)
- **Run pre-commit hooks**: `make run-pre` (runs all pre-commit hooks on all files)
- **Update pre-commit hooks**: `make update-pre` (updates pre-commit hooks to latest versions)
- **Build package**: `make build` (creates distribution packages)
- **Build documentation**: `make docs` (builds Jupyter Book documentation, this needs to succeed in order to be
  mergeable to main)

## Repository Structure

- `qpdk/`: Core Python package containing quantum device components and PDK configuration
- `qpdk/cells/`: Device cell, or gdsfactory component, definitions (transmons, resonators, couplers, etc.)
- `qpdk/klayout/`: KLayout technology files and layer definitions
- `qpdk/tech.py`: Layer stack and main technology cross sections etc.
- `tests/`: Test suite using pytest
- `docs/`: Documentation built with Jupyter Book
- `install_tech.py`: Script to symlink technology files to KLayout, rarely needed for AI agents

## Technology and Tools

- **Python versions**: Specified in `pyproject.toml` (currently >=3.11,\<3.14)
- **Package manager**: `uv` (preferred over pip/conda)
- **Main dependencies**: gdsfactory (>=9.15.0,\<9.21.0), doroutes (>=0.2.0)
- **Testing**: pytest with regression testing using `pytest_regressions`, hypothesis for property-based testing
- **Linting**: ruff for Python code formatting and linting, pyrefly for type checking
- **Layout tool**: KLayout for viewing and editing GDS layouts
- **Documentation**: Jupyter Book for building documentation, jupytext for notebook management
- **Simulation tools** (optional): sax, scikit-rf, scqubits, jaxellip (install with `uv sync --extra models`)

## Git and Version Control

- **Git LFS Required**: This repository uses Git LFS (Large File Storage) for test data files. Install Git LFS before
  cloning or testing: <https://git-lfs.github.com/>
- **Binary files**: GDS and OAS files are tracked as binary in `.gitattributes` to prevent merge conflicts
- **Test data with LFS**: CSV files in `tests/models/data/` are stored with Git LFS (`filter=lfs diff=lfs merge=lfs`)
- **Branching**: Work on feature branches, not directly on `main`. Pull requests are required for merging to `main`
- **Commit messages**: Write clear, concise commit messages. Use imperative mood (e.g., "Add component" not "Added
  component")

## CI/CD and Automation

### GitHub Actions Workflows

The repository uses several automated workflows:

- **test.yml**: Runs on every PR and push to main. Executes pre-commit hooks and pytest test suite
- **pages.yml**: Builds HTML and PDF documentation, deploys to GitHub Pages on push to main
- **build.yml**: Builds Python package distribution files
- **release.yml**: Publishes to PyPI when a version tag (e.g., `v0.0.3`) is pushed

### Automated Checks

All PRs must pass:

1. Pre-commit hooks (linting, formatting, type checking, etc.)
1. Full test suite (203+ tests including regression tests)
1. Documentation build (both HTML and PDF must build successfully)

### Release Process

1. Update version in `pyproject.toml`, `qpdk/__init__.py`, and `README.md` using tbump
1. Create and push a version tag: `git tag v0.0.X && git push origin v0.0.X`
1. GitHub Actions automatically builds and publishes to PyPI
1. Draft release is automatically published on GitHub

## Key Guidelines

1. **Follow quantum device design patterns**: This PDK is specifically for superconducting quantum circuits
1. **Maintain layer stack consistency**: All layer definitions must match between `qpdk/layers.yaml` and `qpdk/tech.py`
1. **Use regression testing**: New components should have regression tests for both settings and netlists, the files are
   generated automatically with `make test-force`
1. **Prefer Makefile commands**: Use `make` commands instead of direct tool invocation if possible
1. **Write comprehensive tests**: Add tests for new functionality following existing patterns in `tests/`
1. **Document quantum-specific behavior**: Include docstrings explaining the quantum physics and device characteristics,
   ideally with citations
1. **Expose new components to PDK**: Import new all components from new files with the form `from ... import *` in
   `qpdk/cells/__init__.py`
1. **Use predefined layers**: Prefer predefined layers from the `LAYER` enumerable in `qpdk/tech.py`. An agent rarely
   needs to create a new layer.

## Testing Guidelines

- **Component tests**: Each new component should be added to the cell registry with `@gf.cell`
- **Netlist validation**: Components must generate valid netlists that can be round-tripped (component -> netlist ->
  component). This is tested by `test_netlists` in `tests/test_pdk.py`.
- **Prefer using the `hypothesis` library**: Use the `hypothesis` library to generate tests with generic arguments with
  appropriate types

## Creating New Components

When adding new quantum device components:

1. Define the component in `qpdk/cells/`
1. Add it to the `cells` registry in `qpdk/cells/__init__.py`, with `from ... import *`
1. Create appropriate layer assignments using the defined `LAYER` enum from `qpdk/tech.py`
1. Add regression tests following the existing parametrized test pattern, generated by running `make test-gds-force`
   multiple times
1. Ensure the component works with the netlist extraction system
1. Document the quantum device physics and design parameters

## Models and Simulation

The `qpdk/models/` directory contains S-parameter and circuit models for quantum RF components:

- **Models available**: Resonators, couplers, waveguides, generic components
- **Simulation frameworks**: Uses `sax` for S-parameter simulations, `scikit-rf` for RF/microwave analysis, `scqubits`
  for quantum circuit analysis
- **Media definitions**: `qpdk/models/media.py` defines coplanar waveguide (CPW) media for different substrates
- **Installing simulation tools**: Run `uv sync --extra models` to install optional simulation dependencies
- **Model structure**: Each model typically provides functions that return S-parameters or network parameters as
  functions of frequency
- **Integration with components**: Models can be linked to layout components through the netlist system

### Example Model Usage

Models are typically used in notebooks and sample scripts (see `notebooks/` and `qpdk/samples/simulate_resonator.py`)
for examples of:

- Calculating resonator frequencies
- Simulating S-parameters for coupled systems
- Optimizing component parameters
- Comparing simulated results with measurements

## Notebooks and Samples

### Jupytext Workflow

- **Notebook source**: All notebook source files are in `notebooks/src/` as Python files (`.py`) using jupytext format
- **Conversion**: Run `make convert-notebooks` or `./.github/convert-notebooks.sh` to convert `.py` files to `.ipynb`
- **Pre-commit hook**: Notebooks are automatically converted when `.py` files in `notebooks/src/` are modified
- **Format**: Notebooks use the "percent" format with `# %%` cell markers and YAML metadata at the top
- **Documentation**: Converted notebooks are copied to `docs/notebooks/` during documentation build

### Samples Directory

- **Location**: `qpdk/samples/` contains example layout and simulation scripts
- **Purpose**: Demonstrates how to use PDK components to create complete designs
- **Examples include**:
  - Simple component demonstrations (`sample0.py` through `sample6.py`)
  - Resonator test chips (`resonator_test_chip.py`)
  - Filled test chips with multiple components (`filled_test_chip.py`)
  - Routing examples with airbridges (`route_with_airbridges.py`)
  - Simulation workflows (`simulate_resonator.py`)
- **YAML files**: Some samples use `.pic.yml` and `.scm.yml` for pictorial and schematic component definitions
- **Testing**: Samples are tested to ensure they generate valid GDS files (see `tests/test_pdk.py`)

### Creating New Notebooks or Samples

1. For notebooks: Create `.py` file in `notebooks/src/` using jupytext percent format with proper YAML header
1. For samples: Create `.py` file in `qpdk/samples/` following existing examples
1. Include docstrings and comments explaining the design and physics
1. Add citations where appropriate using Sphinx citation syntax
1. Test that the script runs without errors
1. For samples, ensure the generated component is added to the test suite

## Pre-commit Hooks Details

The repository uses extensive pre-commit hooks Including, but not limited to:

- `ruff`: Python code formatting and linting
- `yamlfmt`: YAML file formatting
- `codespell`: Spell checking
- `nbstripout`: Jupyter notebook cleaning
- `actionlint`: GitHub Actions validation
- `uv-lock`: Lock file validation
## Common Pitfalls and Troubleshooting

### Git LFS Issues

**Problem**: Tests fail with "File not found" errors for CSV data files, or GDS files appear as text pointers

**Solution**: Install Git LFS and run `git lfs pull` to download large files

```bash
# Install Git LFS (varies by OS)
# Ubuntu/Debian:
sudo apt-get install git-lfs
# RHEL/Rocky
sudo dnf install git-lfs
# macOS:
brew install git-lfs

# Initialize and pull
git lfs install
git lfs pull
```

### Pre-commit Hooks Not Running

**Problem**: CI fails with formatting or linting errors

**Solution**: Always run pre-commit hooks before committing

```bash
make run-pre
# or
uvx prek run --all-files
```

### Layer Consistency Issues

**Problem**: Components render incorrectly in KLayout or documentation

**Solution**: Ensure layers are defined in both `qpdk/layers.yaml` AND `qpdk/tech.py`. Layer numbers and datatypes must
match exactly.

### Documentation Build Failures

**Problem**: `make docs` fails with import errors or missing cells

**Solution**:

1. Ensure all dependencies are installed: `make install`
1. Run `make setup-ipython-config` to set up IPython configuration
1. Check that all components are properly registered in `qpdk/cells/__init__.py`
1. Verify that notebooks in `notebooks/src/` are valid Python files with correct jupytext format

### Regression Test Failures

**Problem**: GDS regression tests fail after making changes

**Solution**:

1. If changes are intentional, regenerate reference files: `make test-gds-force`
1. Review the diff to ensure changes are expected
1. Commit both code and updated reference files in `tests/gds_ref/`
1. Note: GDS reference files may change between gdsfactory versions

### Import Errors in Models

**Problem**: `ModuleNotFoundError` when importing simulation modules

**Solution**: Install optional model dependencies: `uv sync --extra models`

### Type Checking Errors

**Problem**: Pyrefly or mypy report type errors in pre-commit

**Solution**:

1. Review the specific error message
1. Add type hints where missing
1. Use `cast()` for type conversions only when absolutely necessary
1. Check `pyproject.toml` for any type checking configuration that may be ignoring specific errors
