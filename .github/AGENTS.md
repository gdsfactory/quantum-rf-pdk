# AI Agent Best Practices for Quantum RF PDK

This is a Python-based quantum RF Process Design Kit (PDK) built on gdsfactory for designing superconducting quantum devices. It provides components like transmons, resonators, couplers, and quantum layouts. Please follow these guidelines when contributing:

## Code Standards

### Required Before Each Commit
- **ALWAYS** run pre-commit hooks before committing any changes by using `pre-commit run --all-files`
- Pre-commit hooks will automatically run code formatting (ruff), linting, YAML formatting, and other quality checks
- All pre-commit hooks must pass before any changes can be committed

### Development Flow
- **Install dependencies**: `make install` (uses `uv sync --extra docs --extra dev`)
- **Test**: `make test` (runs `uv run pytest -s tests/test_pdk.py`)
- **Test with regeneration**: `make test-force` (includes `--force-regen` flag)
- **Fast-fail testing**: `make test-fail-fast` (includes `-x` flag to stop on first failure)
- **Build package**: `make build` (creates distribution packages)
- **Build documentation**: `make docs` (builds Jupyter Book documentation)

## Repository Structure
- `qpdk/`: Core Python package containing quantum device components and PDK configuration
- `qpdk/cells/`: Device cell definitions (transmons, resonators, couplers, etc.)
- `qpdk/klayout/`: KLayout technology files and layer definitions
- `qpdk/layers.yaml`: Layer stack configuration for the PDK
- `tests/`: Test suite using pytest with regression testing
- `docs/`: Documentation built with Jupyter Book
- `install_tech.py`: Script to symlink technology files to KLayout

## Technology and Tools
- **Python versions**: 3.11, 3.12, or 3.13 (specified in `pyproject.toml`)
- **Package manager**: `uv` (preferred over pip/conda)
- **Main dependencies**: gdsfactory, doroutes
- **Testing**: pytest with regression testing using `pytest_regressions`
- **Linting**: ruff for Python code formatting and linting
- **Layout tool**: KLayout for viewing and editing layouts

## Key Guidelines
1. **Follow quantum device design patterns**: This PDK is specifically for superconducting quantum circuits
2. **Maintain layer stack consistency**: All layer definitions must match between `qpdk/layers.yaml` and KLayout files
3. **Use regression testing**: New components should have regression tests for both settings and netlists
4. **Prefer Makefile commands**: Always use `make` commands instead of direct tool invocation
5. **Write comprehensive tests**: Add tests for new functionality following existing patterns in `tests/test_pdk.py`
6. **Document quantum-specific behavior**: Include docstrings explaining the quantum physics and device characteristics
7. **Validate netlists**: Ensure new components can generate and validate proper netlists for circuit simulation
8. **Test component settings**: New components should pass the `test_settings` regression test

## Testing Guidelines
- **Component tests**: Each new component should be added to the `cells` registry and tested
- **Netlist validation**: Components must generate valid netlists that can be round-tripped (component -> netlist -> component)
- **Settings regression**: Component settings must be stable across versions
- **Layer consistency**: The `test_yaml_matches_layers()` test ensures layer definitions are synchronized

## Creating New Components
When adding new quantum device components:
1. Define the component in `qpdk/cells/`
2. Add it to the `cells` registry in `qpdk/cells/__init__.py`
3. Create appropriate layer assignments using the defined `LAYER` enum
4. Add regression tests following the existing parametrized test pattern
5. Ensure the component works with the netlist extraction system
6. Document the quantum device physics and design parameters

## Pre-commit Hooks Details
The repository uses extensive pre-commit hooks including:
- `ruff`: Python code formatting and linting
- `yamlfmt`: YAML file formatting
- `codespell`: Spell checking
- `nbstripout`: Jupyter notebook cleaning
- `actionlint`: GitHub Actions validation
- `uv-lock`: Lock file validation

All hooks must pass before committing. Run `make update-pre` to update pre-commit hooks to latest versions.