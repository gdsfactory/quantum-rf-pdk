# Models

This directory houses the models for frequency-dependent SPICE-like circuit simulation with scattering parameters
($`S`$-matrices).

## Libraries

- [Sax](https://flaport.github.io/sax/) is used as the circuit simulator backend
- [scikit-rf](https://scikit-rf.org/) is used for coplanar waveguide properties

## Installation

Install the dependencies by requesting the optional dependencies `models`. With a non-contributor install using pip,
this would be:

```bash
pip install qpdk[models]
```

### Contributors

Development install with `uv` would use:

```bash
uv sync --extra models
```
