# QPDK — Superconducting Quantum RF Process Design Kit

<!-- BADGES:START -->

[![HTML Docs](https://img.shields.io/badge/%F0%9F%93%84_HTML-Docs-blue?style=flat)](https://gdsfactory.github.io/quantum-rf-pdk/)
[![PDF Docs](https://img.shields.io/badge/%F0%9F%93%84_PDF-Docs-blue?style=flat&logo=adobeacrobatreader)](https://gdsfactory.github.io/quantum-rf-pdk/qpdk.pdf)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gdsfactory/quantum-rf-pdk/HEAD)
[![PyPI - Version](https://img.shields.io/pypi/v/qpdk?color=blue)](https://pypi.org/p/qpdk)
[![MIT](https://img.shields.io/github/license/gdsfactory/quantum-rf-pdk)](https://choosealicense.com/licenses/mit/)
[![Docs](https://github.com/gdsfactory/quantum-rf-pdk/actions/workflows/pages.yml/badge.svg)](https://gdsfactory.github.io/quantum-rf-pdk/)
[![Tests](https://github.com/gdsfactory/quantum-rf-pdk/actions/workflows/test.yml/badge.svg)](https://github.com/gdsfactory/quantum-rf-pdk/actions/workflows/test.yml)
[![DRC](https://github.com/gdsfactory/quantum-rf-pdk/raw/badges/drc.svg)](https://github.com/gdsfactory/quantum-rf-pdk/actions/workflows/drc.yml)
[![Code Coverage](https://github.com/gdsfactory/quantum-rf-pdk/raw/badges/coverage.svg)](https://github.com/gdsfactory/quantum-rf-pdk/actions/workflows/test.yml)
[![Model Coverage](https://github.com/gdsfactory/quantum-rf-pdk/raw/badges/model_coverage.svg)](https://github.com/gdsfactory/quantum-rf-pdk/actions/workflows/test.yml)
[![Issues](https://github.com/gdsfactory/quantum-rf-pdk/raw/badges/issues.svg)](https://github.com/gdsfactory/quantum-rf-pdk/issues)
[![PRs](https://github.com/gdsfactory/quantum-rf-pdk/raw/badges/prs.svg)](https://github.com/gdsfactory/quantum-rf-pdk/pulls)

<!-- BADGES:END -->

______________________________________________________________________

**QPDK** is an open-source process design kit (PDK) for superconducting quantum RF applications built on
[gdsfactory](https://gdsfactory.github.io/gdsfactory/). It provides a library of parametric quantum circuit components
(transmon qubits, CPW resonators, Josephson junctions, etc.), analytical S-parameter models, routing utilities, and
test-chip examples.

QPDK gives researchers, engineers, and students a scriptable, version-controlled foundation to go from concept to GDSII
in minutes.

## Key Features

- **Rich component library** — Transmons, fluxonium, unimon qubits, CPW resonators, interdigital capacitors, SQUID
  junctions, launchers, bump bonds, TSVs, and more.
- **Parametric & composable** — Combine Python functions (`@gf.cell`) into hierarchical designs or define full chips in
  YAML.
- **Analytical circuit models** — Fast, differentiable S-parameter simulations powered by
  [SAX](https://flaport.github.io/sax/) and [JAX](https://github.com/jax-ml/jax).
- **Automated routing** — CPW-aware routing strategies with auto-tapers for complex layouts, and
  [DoRoutes](https://doplaydo.github.io/DoRoutes/).
- **KLayout integration** — Layer definitions, technology files, and cross sections for immediate visual inspection.
- **Regression-tested** — GDS regression tests, netlist checks, and model validation.
- **Notebook-driven workflows** — Jupyter notebooks for frequency modeling, tolerance analysis, parameter extraction,
  pulse-level simulation, and optimization.
- **[GDSFactory+](https://gdsfactory.com/plus/) integration** — Seamlessly design, verify, and validate chips with the
  enhanced commercial extension. Includes access to 43+ foundry PDKs, graphical layout and schematic editors directly in
  VSCode, an AI assistant, and comprehensive verification tools (DRC, LVS, Connectivity checks).

## Notable Components

QPDK ships with a broad set of ready-to-use superconducting circuit components. Browse all the components in the
[documentation](https://gdsfactory.github.io/quantum-rf-pdk/cells.html).

### Qubits

|                   Transmon                    |                    Fluxonium                    |
| :-------------------------------------------: | :---------------------------------------------: |
|     Double-pad capacitively shunted qubit     |          Superinductance-shunted qubit          |
| ![Transmon](docs/_static/images/transmon.png) | ![Fluxonium](docs/_static/images/fluxonium.png) |

|                  Unimon                   |                   SQUID Junction                    |
| :---------------------------------------: | :-------------------------------------------------: |
|     Resonator-embedded junction qubit     |     Superconducting quantum interference device     |
| ![Unimon](docs/_static/images/unimon.png) | ![SQUID Junction](docs/_static/images/junction.png) |

### Passive Components

|                    CPW Resonator                    |                    Interdigital Capacitor                    |
| :-------------------------------------------------: | :----------------------------------------------------------: |
|       Meandering coplanar waveguide resonator       |            Finger-style lumped-element capacitor             |
| ![CPW Resonator](docs/_static/images/resonator.png) | ![Interdigital Capacitor](docs/_static/images/capacitor.png) |

### Composite Components

**Transmon with Resonator & Probeline** — Qubit cell with coupled resonator and probeline section:

![Transmon with Resonator and Probeline](docs/_static/images/transmon_resonator.png)

## Sample Test Chips

QPDK includes some complete, tapeout-ready test chip examples that demonstrate real-world design workflows.

### Qubit Test Chip

A four-transmon test chip with coupled readout resonators, probeline routing, flux lines, and launchers. Defined
entirely in YAML.

![Qubit Test Chip](docs/_static/images/qubit_test_chip.png)

### Filled Qubit Test Chip

The same qubit test chip with magnetic vortex trapping holes filling the ground plane and chip edges.

![Filled Qubit Test Chip](docs/_static/images/filled_qubit_test_chip.png)

### Resonator Test Chip

A 16-resonator characterization chip with systematically varied CPW widths and gaps across two probelines. Ideal for
extracting loss tangents and kinetic inductance. Also check
[the notebook demonstrating network model simulations and fabrication tolerance Monte Carlo](https://gdsfactory.github.io/quantum-rf-pdk/notebooks/monte_carlo_fabrication_tolerance.html).

![Resonator Test Chip](docs/_static/images/resonator_test_chip.png)

## Quick Start

```python
import gdsfactory as gf
from qpdk import PDK

PDK.activate()

# Create a transmon qubit
from qpdk.cells.transmon import double_pad_transmon

qubit = double_pad_transmon(pad_size=(250, 400), pad_gap=15)
qubit.plot()
```

```python
# Build a complete test chip from YAML
from gdsfactory.read import from_yaml
from qpdk import tech

chip = from_yaml(
    "qpdk/samples/qubit_test_chip.pic.yml",
    routing_strategies=tech.routing_strategies,
)
chip.show()  # Opens in KLayout
```

## Examples & Notebooks

- **[PDK cells in the documentation](https://gdsfactory.github.io/quantum-rf-pdk/cells.html)** — Collection of all
  available geometries.
- **[`qpdk/samples/`](https://github.com/gdsfactory/quantum-rf-pdk/tree/main/qpdk/samples)** — Example layouts and
  simulations including qubit test chips, resonator arrays, routing demos, and 3D export.
- **[`notebooks/`](https://gdsfactory.github.io/quantum-rf-pdk/notebooks.html)** — Jupyter notebooks covering:
  - Resonator frequency modeling and S-parameter analysis
  - Circuit simulation with [SAX](https://github.com/flaport/sax)
  - Monte Carlo fabrication tolerance analysis
  - Hamiltonian parameter extraction with [scqubits](https://github.com/scqubits/scqubits)
  - Pulse-level quantum gate simulation with [QuTiP](https://qutip.org/)
  - Capacitor geometry optimization with [Optuna](https://optuna.org/)
  - Dispersive shift calculation with [Pymablock](https://pymablock.readthedocs.io/en/latest/)
  - Transmon design optimization with [NetKet](https://www.netket.org/)
- **[gsim example notebooks](https://gdsfactory.github.io/gsim/)** — Electromagnetic simulation examples using Palace
  and Meep with gdsfactory.

## Installation

We recommend using [`uv`](https://astral.sh/uv/) for package management. [`just`](https://github.com/casey/just) is used
for project-specific recipes.

### Installation for Users

Install the package with:

```bash
uv pip install qpdk
```

Or with pip:

```bash
pip install qpdk
```

Optional dependencies for the analytical models and simulation tools (SAX, scqubits, JAX) can be installed with:

```bash
uv pip install qpdk[models]
```

### KLayout Technology Installation

To use the PDK in KLayout (for viewing GDS files with correct layers and technology settings), install the technology
files:

```bash
python -m qpdk.install_tech
```

> [!NOTE]
> After installation, restart KLayout to ensure the new technology appears.

### Installation for Contributors

For contributors, please follow the [installation and development workflow instructions](docs/contributing.md).

## Project Structure

```text
qpdk/                   Core Python package
  cells/                Component definitions (transmons, resonators, capacitors, …)
  models/               Analytical models, mostly S-parameters
  samples/              Example layouts and complete test chips
  klayout/              KLayout technology files
  tech.py               Layer stack, cross sections, routing strategies
tests/                  Regression, integration and unit tests
notebooks/              Jupyter notebooks for design and simulation workflows
docs/                   Sphinx documentation (HTML + PDF)
```

## Documentation

- [Quantum RF PDK documentation (HTML)](https://gdsfactory.github.io/quantum-rf-pdk/)
- [Quantum RF PDK documentation (PDF)](https://gdsfactory.github.io/quantum-rf-pdk/qpdk.pdf)
- [gdsfactory documentation](https://gdsfactory.github.io/gdsfactory/)

## Contributing

We welcome contributions of all sizes: new components, improved models, bug fixes, documentation, and notebook
tutorials. Please see the [contributing guide](docs/contributing.md) to get started.

## Support

For commercial support, training, and custom PDK development, please visit
[gdsfactory+ (gdsfactory.com/plus)](https://gdsfactory.com/plus/).

## License

QPDK is released under the [MIT License](LICENSE).
