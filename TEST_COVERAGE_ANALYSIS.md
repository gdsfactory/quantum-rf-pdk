# Test Coverage Analysis

**Date:** 2026-03-26
**Overall coverage:** 86% (2224 statements, 307 missed)

## Coverage Summary by Module

| Module | Stmts | Miss | Cover | Notes |
|--------|-------|------|-------|-------|
| `qpdk/install_tech.py` | 27 | 27 | **0%** | Completely untested |
| `qpdk/simulation/q3d.py` | 114 | 97 | **15%** | Only imports tested |
| `qpdk/simulation/hfss.py` | 90 | 58 | **36%** | Only `lumped_port_rectangle_from_cpw` tested offline |
| `qpdk/simulation/aedt_base.py` | 132 | 66 | **50%** | `layer_stack_to_gds_mapping` and `prepare_component_for_aedt` tested; class methods untested |
| `qpdk/helper.py` | 81 | 19 | **77%** | `show_components` (lines 100-137) untested |
| `qpdk/models/couplers.py` | 53 | 7 | **87%** | `coupler_straight` fallback path (lines 115-129) untested |
| `qpdk/models/perturbation.py` | 36 | 5 | **86%** | `transmon_resonator_hamiltonian` (lines 70-84) untested |
| `qpdk/cells/resonator.py` | 121 | 6 | **95%** | Edge case branches untested |
| `qpdk/cells/snspd.py` | 47 | 4 | **91%** | Validation error paths (lines 48-51) untested |

All other modules are at 95-100% coverage.

---

## Recommended Improvements

### 1. Simulation module (`simulation/`) - Priority: HIGH

**Current state:** 15-50% coverage. The `@pytest.mark.hfss` tests require a licensed Ansys installation, so they only run in specific CI environments. The non-HFSS helper functions are well tested, but the class methods (`HFSS.import_component`, `Q3D.assign_nets_from_ports`, `Q2D.create_2d_from_cross_section`, etc.) have no unit-level test coverage.

**Recommendation:** Add **mock-based unit tests** for the simulation wrapper classes:
- Mock `pyaedt` objects to test `HFSS.import_component`, `HFSS.add_lumped_ports`, `HFSS.add_air_region`, `HFSS.get_eigenmode_results`, `HFSS.get_sparameter_results`
- Mock `Q3d` to test `Q3D.import_component`, `Q3D.assign_nets_from_ports`, `Q3D.get_capacitance_matrix`
- Mock `Q2d` to test `Q2D.create_2d_from_cross_section`
- Test `add_materials_to_aedt` with a mock app
- Test `rename_imported_objects` with various object naming patterns
- Test `export_component_to_gds_temp` context manager (both with explicit path and temp dir)
- Test `AEDTBase.add_substrate` and `AEDTBase.save` with mocks

This would bring the simulation module from ~30% to 80%+ coverage without requiring Ansys licenses.

### 2. `install_tech.py` - Priority: MEDIUM

**Current state:** 0% coverage (27 statements).

**Recommendation:** Add tests using `tmp_path` fixtures:
- `test_remove_path_or_dir` - test removing files, directories, and non-existent paths (expect `FileNotFoundError`)
- `test_make_link` - test symlink creation, overwrite=True/False behavior, and fallback to `shutil.copytree` on `OSError`
- Test the `FileNotFoundError` when source doesn't exist

### 3. `helper.py` - `show_components` function - Priority: LOW

**Current state:** Lines 100-137 (`show_components`) are untested.

**Recommendation:** This function orchestrates layout display, so a lightweight integration test could:
- Verify it returns the correct number of components
- Verify the combined component has the right number of references
- Mock `c.show()` to avoid GUI dependency

### 4. `models/couplers.py` - fallback path - Priority: MEDIUM

**Current state:** Lines 115-129 in `coupler_straight` are untested. This is the fallback path when `get_cpw_dimensions` raises a `ValueError` because the cross-section doesn't have standard CPW sections.

**Recommendation:** Add a test that passes a cross-section without standard gap sections to trigger the fallback path and verify the warning is logged and default gap of 6.0 um is used.

### 5. `models/perturbation.py` - symbolic Hamiltonian - Priority: LOW

**Current state:** `transmon_resonator_hamiltonian()` (lines 70-84) is untested. This function constructs symbolic quantum operators using `sympy` and `sympy.physics.secondquant`.

**Recommendation:** Add a test that:
- Calls `transmon_resonator_hamiltonian()` and verifies it returns `(H_0, H_p, symbols)` tuple
- Checks that returned symbols have the correct names
- Verifies the Hamiltonian has the expected operator structure (e.g., contains creation/annihilation operators)

### 6. Edge cases in cell validation - Priority: MEDIUM

**Current state:** Several cell modules have untested validation branches:

- `cells/resonator.py` (lines 73, 111, 131, 137, 140, 145) - validation for meander parameters, length constraints
- `cells/snspd.py` (lines 48-51) - validation for SNSPD parameters
- `cells/waveguides.py` (lines 272-273) - edge case in waveguide generation

**Recommendation:** Add parametrized tests with invalid inputs to verify these raise appropriate `ValueError` exceptions. For example:
- Resonator with negative length, zero meanders, invalid bend radius
- SNSPD with out-of-range wire width or pitch
- Waveguide with incompatible cross-section parameters

### 7. Error handling in `test_compare_to_qucs.py` - Priority: MEDIUM

**Current state:** All 4 tests in `test_compare_to_qucs.py` are erroring (not failing, but erroring during collection/execution). These tests compare model outputs against Qucs-S reference data.

**Recommendation:** Investigate and fix these test errors. They may indicate a missing dependency or data file issue. Reference-based regression tests are high-value for validating model accuracy.

### 8. Cross-cutting improvements

- **Property-based testing expansion:** The codebase already uses Hypothesis effectively in `test_waveguides.py`, `test_couplers_analytical.py`, and `test_resonators.py`. Extend this pattern to:
  - Capacitor models (varying finger count, length, gap)
  - Junction models (varying critical current, area)
  - Qubit models (varying EC, EJ ranges)

- **Integration test for full PDK round-trip:** Add a test that creates a component, extracts its netlist, uses the model to compute S-parameters, and verifies physical constraints (passivity, reciprocity). This would test the cells-to-models pipeline end-to-end.

- **Negative/boundary testing:** Many cell constructors accept physical parameters (widths, gaps, lengths) but lack tests for boundary conditions like zero, negative, or extremely large values.

---

## Summary of Impact

| Improvement | Effort | Coverage Gain | Risk Reduction |
|------------|--------|---------------|----------------|
| Mock-based simulation tests | High | +15% overall | High - catches AEDT integration regressions |
| `install_tech.py` tests | Low | +1% overall | Medium - prevents broken KLayout installs |
| Cell validation edge cases | Medium | +1% overall | High - prevents silent geometry errors |
| Coupler fallback path test | Low | <1% overall | Medium - validates warning + default behavior |
| Fix Qucs-S comparison tests | Medium | N/A (already counted) | High - validates model accuracy |
| Property-based test expansion | Medium | Minimal line coverage | High - catches numerical edge cases |
