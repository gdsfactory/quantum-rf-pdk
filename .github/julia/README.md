# Julia checks

The pre-commit hooks run JuliaFormatter 2.14 and JuliaWorkspaces 13.5 in this environment. Install Julia 1.12 and put it
on PATH. CI uses `version: "min"` and `project: .github/julia` to read the minimum supported release from
`[compat].julia`. The `~1.12.0` constraint keeps the tooling on Julia 1.12; the notebook environments retain their
broader compatibility.

The hooks instantiate their own dependencies and never execute the simulation script. `format.jl` calls `format_text`
with `SciMLStyle()` explicitly, bypassing home-directory and ancestor `.JuliaFormatter.toml` files that `format(path)`
would pick up. `lint.jl` adds only the paths pre-commit passes, plus `.github/julia/JuliaLint.toml`, to an in-memory
workspace. The hook presents the config at the workspace root so its settings apply to every checked source without a
root-level file. It reports diagnostics as `file:line:column: severity: message [rule]` and exits non-zero on errors,
including NaN comparisons and duplicate branch conditions. Unresolved external names stay excluded because the hook
environment does not index the simulation packages; `JuliaLint.toml` pins `missing_reference` and `unresolved_import`
off rather than relying on the moving `default` preset.

JuliaFormatter is the established configurable formatter; Runic is a smaller, fixed-style alternative. JuliaWorkspaces
powers the language server and static lint. JET analyzes inferred code and can interpret top-level statements, so it is
better suited to targeted checks inside the simulation environment. Aqua checks Julia package quality rather than
standalone scripts, so it does not apply here.

## Targeted JET checks

`notebooks/data/vacask_twpa/julia/checks/` is a separate bounded environment pinned to JET 0.12.3 and JosephsonCircuits
0.5.4. `checks.jl` includes `../reference.jl` for its definitions only, then exercises the circuit construction and
photon-to-voltage helpers with `Test`, and runs `JET.test_call` with `target_modules=(Main,)` so third-party inference
noise is filtered out. The reference script guards `main()` with a `PROGRAM_FILE` comparison, so including it never
starts the harmonic-balance solve. The checks need no `QPDK_JULIA_OUTPUT` environment variable.

Run the checks with:

```sh
julia --startup-file=no --project=notebooks/data/vacask_twpa/julia/checks \
    notebooks/data/vacask_twpa/julia/checks/checks.jl
```

Tool documentation: [JuliaFormatter](https://github.com/JuliaEditorSupport/JuliaFormatter.jl),
[JuliaWorkspaces](https://julia-vscode.github.io/JuliaWorkspaces.jl/dev/),
[Runic](https://github.com/fredrikekre/Runic.jl), [JET](https://aviatesk.github.io/JET.jl/stable/),
[Aqua](https://github.com/JuliaTesting/Aqua.jl).

## Limitations

The hook has no package index, so semantic checks that need external symbol resolution stay off. The JET checks cover
only the extracted helpers; they do not analyze the harmonic-balance solver or its convergence.
