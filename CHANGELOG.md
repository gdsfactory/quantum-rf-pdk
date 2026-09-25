# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-09-18

### Added

- Added schematic-symbol capability to cells.
- Added Static/Dynamic layout viewer tabs to the cells page, served through the kwasm viewer.
- Added a Circulax transmon optimization notebook with harmonic-balance and transient analysis.
- Added `llms.txt` to the documentation site per llmstxt.org.
- Added Python 3.14 to the test matrix and Codecov upload of coverage and test results.

### Changed

- Built the PDF documentation with Typst (typsphinx) instead of LaTeX, and made PDF build failures fail CI.
- Modernized chip-level schematics and simulation for GDSFactory+ v2.
- Updated `gdsfactory` to `~=9.51.0`.
- Bounded the caches in models to prevent unbounded memory growth during sweeps.

### Removed

- Removed the `doroutes` dependency and the A\* routing strategies.

### Fixed

- Preserved RF resonances in recursive SAX simulations.
- Removed the silent gap fallback in the CPW coupling capacitance model.
- Fixed `JosephsonJunction` layer thickness units and added a `LAYER_STACK` thickness range test.
- Fixed the `snspd` cell to honour `port_type` and validate `num_meanders`.
- Made `lumped_element_resonator` port types consistent with the rest of the PDK.
- Fixed the unimon `meander_radius` metadata fallback and forwarded `cross_section` to the nxn junction in `tee`.
- Fixed Q3D assigning PEC to all renamed objects including the substrate.
- Fixed the resonator test chip schematic simulation.

## [0.3.8] - 2026-05-21

### Added

- Added a MATLAB integration notebook demonstrating `qpdk` usage from MATLAB.
- Added a PCells gallery section to the cells page with clickable card thumbnails.
- Added dielectric loss tangent to the CPW models.
- Added HFSS field plots and improved Q2D geometry visualization.
- Added SBOM generation to the release workflow and a `lychee` link-checking pre-commit hook.

### Changed

- Updated branding to QPDK across the codebase, with a new documentation theme and logo.
- Enabled `sbend="bend_s"` in `route_bundle` strategies.
- Updated `gdsfactory` to `~=9.43.0`, `gplugins` to `~=2.1.0`, and `doroutes` to `~=0.6.0`.
- Matched matplotlib figure fonts to the Sphinx theme and used Fira Math for math rendering.
- Added fzf caching and background loading to component selection.

### Fixed

- Fixed the "Edit on GitHub" links for notebook pages pointing to the wrong path.
- Fixed Sphinx image paths programmatically in `conf.py`.
- Resolved pyright errors and the `jnp.clip` `a_min` TypeError.
- Avoided an artifact name collision in the docs deployment job.

## [0.3.7] - 2026-04-28

### Added

- Added a DRC workflow and DRC configuration.
- Implemented model regression tests for SAX S-parameter models.
- Added metric badges, an auto-label workflow, and labeler coverage for all repository labels.
- Added a test for GDSFactory+ integration.

### Changed

- Centralized the pre-commit hooks and standardized the project configuration.
- Migrated CI to the centralized reusable PDK workflows and reduced the Makefile to a thin shim for `just`.
- Refactored the documentation index and expanded the README with a component gallery and test chip images.
- Lazy-loaded the `sample_functions` dictionary in `qpdk/__init__.py`.

### Fixed

- Fixed the meander inductor monotonicity test tolerance.
- Improved `install-tech` output and fixed its symlink logic.
- Fixed deprecated `jnp.clip` arguments and suppressed third-party warnings.
- Fixed math rendering in the API docs for `sax` re-exported functions.
- Removed the `virtual` flag from `launcher_top` in the qubit test chip YAML.

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
