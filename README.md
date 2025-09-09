# Sample Generic Quantum RF PDK 0.0.2

[![Docs](https://github.com/gdsfactory/quantum-rf-pdk/actions/workflows/pages.yml/badge.svg)](https://gdsfactory.github.io/quantum-rf-pdk/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gdsfactory/quantum-rf-pdk/HEAD)
[![Tests](https://github.com/gdsfactory/quantum-rf-pdk/actions/workflows/test_code.yml/badge.svg)](https://github.com/gdsfactory/quantum-rf-pdk/actions/workflows/test_code.yml)
[![MIT](https://img.shields.io/github/license/gdsfactory/quantum-rf-pdk)](https://choosealicense.com/licenses/mit/)


Based on a combination of the most popular Open source Quantum PDK packages

- **Qiskit Metal** keeps a layer stack as a dataframe/CSV (per-chip rows) with columns like `chip_name`, `layer`, `datatype`, `material`, `thickness`, `z_coord`, and `fill`. The `LayerStackHandler` enforces uniqueness of layer numbers across chips and is used by the renderers (HFSS/Gmsh).

- **KQCircuits** (KLayout-based) ships a default layer config (`default_layers`) plus an optional KLayout layer-props file (`.lyp`) and supports multi-face/multi-mask export. There’s also a utility to sync and display props from `default_layers`.

- **DeviceLayout.jl** (AWS CQC) targets GDS (for fabrication) and 3D models/meshes (for EM simulation, e.g., Palace). Teams version their “PDKs” as Julia packages/registries; a 17-qubit reference design and transmon/resonator examples are published.

## Installation

We recommend `uv`

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Installation for users

Use python 3.11, 3.12 or 3.13. We recommend [VSCode](https://code.visualstudio.com/) as an IDE.

```
uv pip install qpdk --upgrade
```

Then you need to restart Klayout to make sure the new technology installed appears.

### Installation for contributors


Then you can install with:

```bash
git clone https://github.com/gdsfactory/quantum-rf-pdk.git
cd quantum-rf-pdk
uv venv --python 3.12
uv sync --extra docs --extra dev
```

## Documentation

- [gdsfactory docs](https://gdsfactory.github.io/gdsfactory/)
