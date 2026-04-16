# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.6] - 2026-04-14

### Fixed

- Initial pre-commit hook fixes.
- Resolved circular imports and implemented lazy-loading for model dependencies.
- Fixed import redundancy issues ("Module is imported with 'import' and 'import from'").

### Added

- Added fluxonium qubit layout with superinductor and Josephson junction.
- Introduced Monte Carlo CPW tolerance analysis for resonator simulations.
- Added component tags and improved svgbob ASCII art diagrams for docstrings.
- Enabled Google Colab support for notebooks.

### Changed

- Refactored waveguide models to use `sax[rf]`.
- Switched PyPI publishing to OIDC Trusted Publisher.

## [0.3.5] - 2026-03-29

### Added

- Implemented unimon qubit layout, SAX model, and Hamiltonian.
- Added `just show` command for interactive component visualization.
- Added lumped-element resonator with meander inductor.

### Fixed

- Fixed `M1_ETCH` overlap with `M1_DRAW` in `half_circle_coupler`.
- Resolved LaTeX PDF documentation build warnings and errors.

### Changed

- Increased test coverage from 86% to 93%.
- Upgraded `pyrefly` to 0.58.0 for stricter type checking.

## [0.3.4] - 2026-03-25

### Added

- Added pulse-level simulation notebook with JAX backend via `qutip-qip`.
- Introduced NetKet transmon qubit design notebook.
- Added a component designer agent for quantum device visualization.

### Changed

- Removed `scikit-rf` dependency in favor of consolidated CPW models.
- Refactored resonator length optimization to utilize Optax and JAX.
- Migrated documentation from Jupyter Book to pure Sphinx.

## [0.3.3] - 2026-03-12

### Added

- Added HFSS simulation support via PyAEDT and Q2D impedance extraction helpers.
- Enabled collision checking for CPW route bundles.

### Changed

- Standardized logging using `gf.logger` across the library.
- Refactored AEDT simulation utilities into a class-based structure.

## [0.3.0] - 2026-03-09

### Added

- Replaced `scikit-rf` backend with JAX-native transmission line models.
- Added `pymablock` dispersive shift notebook and perturbation theory helpers.

### Changed

- Enhanced models for airbridge physics and SQUID junctions.
- Optimized cell layouts using `gf.pack`.
