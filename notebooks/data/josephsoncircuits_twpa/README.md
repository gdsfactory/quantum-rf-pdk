# JosephsonCircuits TWPA notebook environment

`notebooks/src/josephsoncircuits_twpa.jl` runs on a Julia 1.12 kernel. `julia/` holds its Julia environment, pinned to
JosephsonCircuits 0.5.4, and `julia/twpa.jl`, which builds the JTWPA netlist and wraps `hbsolve`. The notebook, the
checks and CI all include that file.

The notebook calls QPDK and matplotlib through PythonCall. Its first cell sets `JULIA_CONDAPKG_BACKEND=Null` and points
`JULIA_PYTHONCALL_EXE` at the checkout's `.venv` unless they are already set, so PythonCall uses the QPDK environment
instead of creating its own.

```sh
uv sync --extra models
cd notebooks/data/josephsoncircuits_twpa/julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using IJulia; installkernel("Julia", "--project=$(pwd())"; specname="julia-1.12")'
```

Both 2047-junction chains solve in well under a minute, so the notebook keeps no recorded results.

`julia/checks/` is a separate environment with JET. `checks.jl` tests the circuit construction, the photon balance and
quantum efficiency on a short chain, and runs `JET.test_call` on the netlist builder. CI runs it and executes the
notebook.
