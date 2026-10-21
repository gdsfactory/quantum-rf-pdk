# TWPA reference data

Classical small-signal results for the 2047-junction circuit in the TWPA notebook, computed with
[JosephsonCircuits.jl](https://github.com/kpobrien/JosephsonCircuits.jl). The uniform ladder and the resonantly
phase-matched ladder have the same pump current, junction parameters, port impedances and half-cell end capacitors. Each
circuit's pump is solved independently.

`julia-reference.json` holds one record per ladder. `response_re` / `response_im` have axes
`(input port, output port, output spur, frequency)`, with ports `(input, output)` and spurs `(0, -2)`. The frequency
grid contains 51 points from 4 to 6.5 GHz in 50 MHz steps. This is the signal/idler submatrix, not the entire Floquet
scattering matrix. Entries are voltage-wave ratios: `reference.jl` converts JosephsonCircuits' photon-normalized
coefficients with `sqrt(abs(fout)/fin)`, including the negative-frequency idler at `fin - 2 fp`. Each record also keeps
the JosephsonCircuits version, source tree, script hash and solver settings.

At 5 GHz the signal gains are 24.99978 dB (phase matched) and 5.25402 dB (uniform). These are results of one harmonic
basis; the data contain no basis-refinement check.

## Recompute with JosephsonCircuits

The calculation uses registered JosephsonCircuits 0.5.4, source tree `43fa323200866910a93dd06ab8c002ed393ed9bb`, with
Julia 1.10.3. The project and resolved dependency graph are in `julia/`. Instantiate in your own environment/depot;
shared cluster dependencies need not be changed.

```sh
julia --project=notebooks/data/vacask_twpa/julia -e 'using Pkg; Pkg.instantiate()'
export QPDK_JULIA_CELLS=2047 QPDK_JULIA_H=35 QPDK_JULIA_K=41 QPDK_JULIA_SAMPLES=1609
export QPDK_JULIA_FREQUENCIES=$(julia -e 'print(join(4e9:5e7:6.5e9, ","))')
QPDK_JULIA_MATCHED=1 QPDK_JULIA_OUTPUT=build/twpa-julia-rpm julia --project=notebooks/data/vacask_twpa/julia notebooks/data/vacask_twpa/julia/reference.jl
QPDK_JULIA_MATCHED=0 QPDK_JULIA_OUTPUT=build/twpa-julia-uniform julia --project=notebooks/data/vacask_twpa/julia notebooks/data/vacask_twpa/julia/reference.jl
```

The circuit uses the published RCSJ/LC values, odd pump harmonics through 35, even RF offsets through 40, 1609 samples,
and five pump-current continuation stages. The exporter rejects nonconvergence warnings and nonfinite results.
`node_phase.csv` contains ordinary one-sided Fourier coefficients. Julia source coefficients use half the cosine peak
current.

The critical current and source current are multiplied by `1.0000000949017485`. This matches the reduced flux quantum of
the NIST 1998 constants compiled into the QPDK Verilog-A junction, so the junction inductance and dimensionless drive
equal those of the VACASK models; R, C and L are unchanged.

`julia/checks/` is a separate environment that tests the circuit construction and photon conversion in `reference.jl`
with JET; CI runs it.
