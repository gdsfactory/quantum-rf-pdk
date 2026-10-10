---
applyTo: "qpdk/models/datasets/**"
---

# FEM dataset review instructions

`qpdk/models/datasets/` stores reusable FEM extraction results (capacitance, inductance, and S-parameter matrices swept
over geometry) and looks them up from JAX models. A dataset is nothing but its files: Parquet parts in
`qpdk/models/datasets/data/<name>/`, or a Delta Lake table, each carrying its `DatasetMetadata` as JSON under the
`qpdk.dataset` key. There is no separate manifest.

## What to check

- **Lean on the dependencies.** Polars reads and writes Parquet and its key-value metadata, `deltalake` provides atomic,
  versioned appends on object stores, and `jax.scipy.interpolate.RegularGridInterpolator` does the interpolation. Flag
  new code that reimplements any of these.
- **Metadata stays in the files.** Grid values and variant values are derived from the data, never declared twice. A
  change that adds a sidecar file or a hand-maintained list of grid points should be questioned.
- **Missing is never zero.** Failed or missing points are NaN, and a lookup outside the validated domain returns NaN
  unless the caller opts into `out_of_range="clip"`. Flag silent clamping, extrapolation, or zero-filling.
- **Physical meaning is explicit.** Maxwell and mutual capacitance matrices are different `QuantityKind`s and are never
  mixed; quantities are stored in their canonical SI unit with no conversion at lookup time.
- **Lookups stay jittable.** Everything after `Dataset.grid()` must work under `jax.jit`, `jax.vmap`, and `jax.grad`: no
  Polars, file I/O, or data-dependent Python control flow inside `GridInterpolator.__call__`. Prefer `jnp` over `np` in
  array code.
- **Scans stay lazy.** Filter quantity and variants before collecting a grid. Append validation should project key and
  run columns instead of loading historical values.
- **Lazy imports.** `polars` and `deltalake` are in ruff's `require-lazy` list; import them inside functions, with
  `if TYPE_CHECKING:` imports for annotations.
- **Append-only.** `Dataset.append` never rewrites stored parts or Delta commits. `ParquetParts` is local and
  single-writer; concurrent or remote writers use a Delta table. Only `generate.write` replaces a dataset, and only by
  swapping in a complete staged directory; it refuses directories holding anything but dataset parts.
- **Synthetic data is labelled.** Any dataset not produced by a real solver sets `synthetic=True` and says so in its
  description. Experiment scripts in `qpdk/models/datasets/data/` supply a `solve` callback to the reusable
  `qpdk.models.datasets.generate.sweep` and `write` functions.

## Data files

- `qpdk/models/datasets/data/**/*.parquet` is tracked by Git LFS (`.gitattributes`). Flag a Parquet file committed as a
  regular blob, and a regenerated dataset committed without the code change that produced it.
- Regenerate a shipped dataset with its standalone Python experiment, e.g.
  `uv run --script qpdk/models/datasets/data/plate_capacitor.py`. Inline dependencies define the generation environment;
  scripts stay outside wheels.
- Lookups must not import gsim or Gmsh. Script previews import the experiment dependencies but never run Palace.
- Provenance ships in wheels: flag absolute host paths, user names, or other machine-specific details in it.

## Tests

- Lookup and storage tests live in `tests/models/test_datasets.py` and `tests/models/test_datasets_store.py`.
  Interpolation changes need agreement with `scipy` and the generating formula, plus a `jit`/`vmap`/`grad` check.
- `tests/models/test_datasets_palace.py` checks fresh Palace solves, interpolation and cache reuse when
  `uv run --group palace --python 3.12 pytest -m palace`; default tests exclude this marker, and the `test-palace` CI
  job runs it with a real runtime. Keep mesh/domain sensitivity evidence separate from algebraic matrix checks.
