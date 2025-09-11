# Sample Generic Quantum RF PDK 0.0.2

[![Docs](https://github.com/gdsfactory/quantum-rf-pdk/actions/workflows/pages.yml/badge.svg)](https://gdsfactory.github.io/quantum-rf-pdk/)
[![Tests](https://github.com/gdsfactory/quantum-rf-pdk/actions/workflows/test.yml/badge.svg)](https://github.com/gdsfactory/quantum-rf-pdk/actions/workflows/test.yml)
[![HTML Docs](https://img.shields.io/badge/📄_HTML-Docs-blue?style=flat)](https://gdsfactory.github.io/quantum-rf-pdk/)
[![PDF Docs](https://img.shields.io/badge/📄_PDF-Docs-blue?style=flat&logo=adobeacrobatreader)](https://gdsfactory.github.io/quantum-rf-pdk/qpdk.pdf)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gdsfactory/quantum-rf-pdk/HEAD)
[![MIT](https://img.shields.io/github/license/gdsfactory/quantum-rf-pdk)](https://choosealicense.com/licenses/mit/)

A generic process design kit (PDK) for superconducting quantum RF applications based on [gdsfactory](https://gdsfactory.github.io/gdsfactory/).

## Installation

We recommend using [`uv`](https://astral.sh/uv/) for package management.

### Install `uv`

#### On macOS and Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### On Windows

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Installation for Users

- Use Python 3.11, 3.12, or 3.13.
- We recommend [VSCode](https://code.visualstudio.com/) as your IDE.

Install the package with:

```bash
uv pip install qpdk --upgrade
```

> **Note:** After installation, restart KLayout to ensure the new technology appears.

### Installation for Contributors

Clone the repository and install all dependencies:

```bash
git clone https://github.com/gdsfactory/quantum-rf-pdk.git
cd quantum-rf-pdk
uv sync --all-extras
```

Check out the commands for testing and building documentation with:

```bash
make help
```

## Documentation

- [Quantum RF PDK documentation](https://gdsfactory.github.io/quantum-rf-pdk/)
- [Quantum RF PDK documentation as a PDF](https://gdsfactory.github.io/quantum-rf-pdk/qpdk.pdf)
- [gdsfactory documentation](https://gdsfactory.github.io/gdsfactory/)
