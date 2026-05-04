# Qucs-S Reference Data

This directory contains S-parameter reference data generated from [Qucs-S](https://ra3xdh.github.io/) simulations. These
CSV files are used to validate the accuracy of qpdk's S-parameter models.

## Overview

The data files contain frequency-domain S-parameters exported from Qucs-S circuit simulations. Each file corresponds to
a specific component or test case, and the S-parameters are stored in a format with real and imaginary parts in separate
columns.

## File Format

All CSV files follow a consistent format:

```csv
frequency,"r S[i,j]","i S[i,j]", ...
1000000000.0,0.00145942,0.037644, ...
```

- **`frequency`**: Frequency points in $`\mathrm{Hz}`$
- **`r S[i,j]`**: Real part of S-parameter $`S_{ij}`$
- **`i S[i,j]`**: Imaginary part of S-parameter $`S_{ij}`$

## Data Files

### Primary Test Data

These files are actively used in the test suite (`test_compare_to_qucs.py`).

### Supporting Files

- **`dut.sch`**: Example Qucs-S schematic file showing the circuit setup used to generate reference data
- **`merge_csvs.py`**: Python script to merge multiple single-parameter CSV files into a combined file

## Generating New Reference Data

To generate new reference data:

1. Create or modify a circuit in Qucs-S
1. Add an S-parameter simulation (`.SP` component)
1. Run the simulation
1. Export S-parameter data as CSV from the data display
1. Ensure the CSV follows the format described above
1. Add corresponding test class in `test_compare_to_qucs.py`

## Notes

- Only S-parameters where `i >= j` (lower triangular) are typically needed due to reciprocity
