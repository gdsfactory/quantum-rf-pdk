# Sample Generic Superconducting Quantum RF PDK

<!-- BADGES:START -->

[![Docs](https://github.com/gdsfactory/quantum-rf-pdk/actions/workflows/pages.yml/badge.svg)](https://github.com/gdsfactory/quantum-rf-pdk/actions/workflows/pages.yml)
[![Tests](https://github.com/gdsfactory/quantum-rf-pdk/actions/workflows/test.yml/badge.svg)](https://github.com/gdsfactory/quantum-rf-pdk/actions/workflows/test.yml)
[![DRC](https://github.com/gdsfactory/quantum-rf-pdk/raw/badges/drc.svg)](https://github.com/gdsfactory/quantum-rf-pdk/actions/workflows/drc.yml)
[![Model Regression](https://github.com/gdsfactory/quantum-rf-pdk/actions/workflows/model_regression.yml/badge.svg)](https://github.com/gdsfactory/quantum-rf-pdk/actions/workflows/model_regression.yml)
[![Test Coverage](https://github.com/gdsfactory/quantum-rf-pdk/raw/badges/coverage.svg)](https://github.com/gdsfactory/quantum-rf-pdk/actions/workflows/test_coverage.yml)
[![Model Coverage](https://github.com/gdsfactory/quantum-rf-pdk/raw/badges/model_coverage.svg)](https://github.com/gdsfactory/quantum-rf-pdk/actions/workflows/model_coverage.yml)
[![Issues](https://github.com/gdsfactory/quantum-rf-pdk/raw/badges/issues.svg)](https://github.com/gdsfactory/quantum-rf-pdk/issues)
[![PRs](https://github.com/gdsfactory/quantum-rf-pdk/raw/badges/prs.svg)](https://github.com/gdsfactory/quantum-rf-pdk/pulls)

<!-- BADGES:END -->

[![HTML Docs](https://img.shields.io/badge/%F0%9F%93%84_HTML-Docs-blue?style=flat)](https://gdsfactory.github.io/quantum-rf-pdk/)
[![PDF Docs](https://img.shields.io/badge/%F0%9F%93%84_PDF-Docs-blue?style=flat&logo=adobeacrobatreader)](https://gdsfactory.github.io/quantum-rf-pdk/qpdk.pdf)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gdsfactory/quantum-rf-pdk/HEAD)
[![PyPI - Version](https://img.shields.io/pypi/v/qpdk?color=blue)](https://pypi.org/p/qpdk)
[![MIT](https://img.shields.io/github/license/gdsfactory/quantum-rf-pdk)](https://choosealicense.com/licenses/mit/)

______________________________________________________________________

A generic process design kit (PDK) for superconducting quantum RF applications based on
[gdsfactory](https://gdsfactory.github.io/gdsfactory/).

## Examples

- [PDK cells in the documentation](https://gdsfactory.github.io/quantum-rf-pdk/cells.html): showcases available
  geometries.
- [`qpdk/samples/`](https://github.com/gdsfactory/quantum-rf-pdk/tree/main/qpdk/samples): contains example layouts and
  simulations.
- [`notebooks/`](https://github.com/gdsfactory/quantum-rf-pdk/tree/main/notebooks): contains notebooks demonstrating
  design and simulation workflows.
- [gsim example notebooks](https://gdsfactory.github.io/gsim/): electromagnetic simulation examples using Palace and
  Meep with gdsfactory.

## Installation

We recommend using [`uv`](https://astral.sh/uv/) for package management. [`just`](https://github.com/casey/just) is used
for project-specific recipes.

### Installation for Users

Install the package with:

```bash
uv pip install qpdk
```

Optional dependencies for the models and simulation tools can be installed with:

```bash
uv pip install qpdk[models]
```

### KLayout Technology Installation

To use the PDK in KLayout (for viewing GDS files with correct layers and technology settings), you should install the
technology files:

```bash
python -m qpdk.install_tech
```

> [!NOTE]
> After installation, restart KLayout to ensure the new technology appears.

### Installation for Contributors

For contributors, please follow the [installation and development workflow instructions](docs/CONTRIBUTING.md).

## Documentation

- [Quantum RF PDK documentation (HTML)](https://gdsfactory.github.io/quantum-rf-pdk/)
- [Quantum RF PDK documentation (PDF)](https://gdsfactory.github.io/quantum-rf-pdk/qpdk.pdf)
- [Quantum RF PDK test coverage report](https://gdsfactory.github.io/quantum-rf-pdk/reports/coverage/)
- [gdsfactory documentation](https://gdsfactory.github.io/gdsfactory/)
