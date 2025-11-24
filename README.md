# Sample Generic Superconducting Quantum RF PDK

[![Docs](https://github.com/gdsfactory/quantum-rf-pdk/actions/workflows/pages.yml/badge.svg)](https://gdsfactory.github.io/quantum-rf-pdk/)
[![Tests](https://github.com/gdsfactory/quantum-rf-pdk/actions/workflows/test.yml/badge.svg)](https://github.com/gdsfactory/quantum-rf-pdk/actions/workflows/test.yml)
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
- [`qpdk/samples/`](qpdk/samples): contains example layouts and simulations.
- [`notebooks/`](notebooks): contains notebooks demonstrating design and simulation workflows.

## Installation

We recommend using [`uv`](https://astral.sh/uv/) for package management.

### Installation for Users

Install the package with:

```bash
uv pip install qpdk
```

> [!NOTE]
> After installation, restart KLayout to ensure the new technology appears.

Optional dependencies for the models and simulation tools can be installed with:

```bash
uv pip install qpdk[models]
```

### Installation for Contributors

Clone the repository and install at least the development dependencies:

```bash
git clone https://github.com/gdsfactory/quantum-rf-pdk.git
cd quantum-rf-pdk
uv sync --group dev
```

> [!NOTE]
> [Git LFS](https://git-lfs.github.com/) must be installed to run all tests locally. Some test data files (e.g., CSV
> files in `tests/models/data/`) are tracked with Git LFS and will not be properly downloaded without it.

#### Testing and Building Documentation

Check out the commands for testing and building documentation with:

```bash
make help
```

## Documentation

- [Quantum RF PDK documentation (HTML)](https://gdsfactory.github.io/quantum-rf-pdk/)
- [Quantum RF PDK documentation (PDF)](https://gdsfactory.github.io/quantum-rf-pdk/qpdk.pdf)
- [gdsfactory documentation](https://gdsfactory.github.io/gdsfactory/)
