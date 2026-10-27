---
myst:
  html_meta:
    description: How to fork QPDK into a process-specific PDK and keep the fork current with upstream releases.
---

# Customizing QPDK for Your Process

QPDK describes a generic superconducting process. For a real fabrication process, you need your own PDK with your own
layers, layer stack, and cells. You also want to keep getting fixes and new features from QPDK. This page shows how to
do both.

In short:

1. Fork the repository and rename it in one commit.
1. Replace the process (`qpdk/layers.yaml` and `qpdk/tech.py`) in one more commit.
1. Remove or override cells, then regenerate the regression references.
1. Merge new QPDK release tags often. Use the [ownership map](#ownership-map) to resolve conflicts.

## Fork or compose

You can build on QPDK in two ways:

- **Fork.** Copy the repository, own the process, and merge new QPDK changes into it over time.
- **Compose.** Keep `qpdk` as a dependency. Build your own `gdsfactory.Pdk` in a separate package that reuses QPDK cells
  and models.

**Fork if you change the layers, the layer stack, or the materials.** Almost all QPDK code reads the process from
`qpdk.tech`. Cells draw on `LAYER.*`, cross-sections are defined there, and the CPW models read thicknesses from
`LAYER_STACK` (see `qpdk/models/cpw.py`). A separate package cannot change these without patching QPDK internals.

**Compose if you only add things.** If the QPDK process is close enough, depend on `qpdk` and add your cells to its
registries, for example `cells={**qpdk.PDK.cells, **my_cells}`. You never merge anything. You upgrade by changing the
version. Issue [#376](https://github.com/gdsfactory/quantum-rf-pdk/issues/376) tracks a project template for this.

```{mermaid}
:alt: If you change the layers, layer stack, or materials, fork. If you only add cells, samples, or chips, compose.
flowchart LR
    Q1["New layers, stack,<br/>or materials?"] -->|yes| F["Fork:<br/>own the process"]
    Q1 -->|no| Q2["Only adding cells,<br/>samples, or chips?"]
    Q2 -->|no| F
    Q2 -->|yes| C["Compose:<br/>depend on qpdk"]
```

The rest of this page is about forks.

(setting-up-the-fork)=

## Setting up the fork

Create the fork on your Git host. Then add QPDK as a second remote. Here `origin` is your fork and `upstream` is QPDK.

```bash
git clone <your-fork-url> <your-pdk>
cd <your-pdk>
git remote add upstream https://github.com/gdsfactory/quantum-rf-pdk.git

# Your releases also create v<version> tags.
# Fetch the QPDK tags as upstream/v<version> so the two do not collide.
git config remote.upstream.tagOpt --no-tags
git config --add remote.upstream.fetch '+refs/tags/*:refs/tags/upstream/*'
git fetch upstream
git tag -l 'upstream/*'   # upstream/v0.4.0, upstream/v0.4.1, ...

# Let Git reuse your earlier conflict resolutions.
# See https://git-scm.com/book/en/v2/Git-Tools-Rerere
git config rerere.enabled true
```

Work on branches and merge them into `main` through pull requests. Do not work on `main` directly. It is the branch you
merge QPDK into.

(identity-rename)=

## Identity rename

The places below contain the QPDK name. Change all of them in **one commit** and change nothing else in it. Later merges
then conflict here in a known way, and the diff of this commit is your checklist.

| What                | Where                                                                                      |
| ------------------- | ------------------------------------------------------------------------------------------ |
| Package directory   | `qpdk/`, and every `import qpdk` / `from qpdk` (`git grep -l qpdk`: most of the repo)      |
| Name and version    | `[project]` in `pyproject.toml`, `__version__` in `qpdk/__init__.py`, `[tool.tbump]`       |
| Tooling paths       | `pyproject.toml`: coverage `source`, pytest `testpaths`, ruff per-file ignores             |
| GDSFactory+ project | `pyproject.toml`: `[tool.gdsfactoryplus]` `name`, `pdk.name`, `sidebar` libraries          |
| Layer map class     | `LayerMapQPDK` in `qpdk/tech.py`                                                           |
| PDK object          | `name="qpdk"` in `get_pdk()`, and the `QPdk` / `_QPdkMeta` classes, `qpdk/__init__.py`     |
| KLayout technology  | `<name>` in `qpdk/klayout/tech.lyt`, `KLayoutTechnology(name=...)` in `qpdk/tech.py`       |
| KLayout install     | `src` and `dest` in `qpdk/install_tech.py`                                                 |
| Docs                | `docs/conf.py`: `project`, logos, URLs, and badges in README and index                     |
| Workflows           | `test.yml` (`--cov=qpdk`, imports)                                                         |
| Repo metadata       | Repository URLs in `.github/` (issue templates, release drafter, lychee), `AGENTS.md`      |
| Citation metadata   | `CITATION.cff`: title, authors, ORCIDs, repository and documentation URLs, DOI identifiers |

The `justfile` reads the package name from `pdk := env('pdk', 'qpdk')`. Set `pdk=<your_pdk>` in `.env` or change the
default. Only the `install-tech` and `show` recipes read this variable.

Internal names that users never see can stay, for example `docs/_ext/qpdk_*.py` and the `qpdk.mplstyle` figure style.
Each name you keep is one less line that can conflict.

## Process replacement

The process is defined in two files, and they must agree:

- `qpdk/layers.yaml`: layer names, GDS numbers, and how KLayout draws them.
- `qpdk/tech.py`: everything else.

`tests/test_pdk.py::test_yaml_matches_layers` fails if the two files have different layer names or `(layer, datatype)`
pairs. After you edit `layers.yaml`, regenerate the KLayout files and commit them too:

```bash
uv run python qpdk/tech.py   # rewrites qpdk/klayout/layers.lyp and tech.lyt
```

QPDK reads `LAYER_VIEWS` from the generated `layers.lyp`, not from the YAML. If you skip this step, the test passes but
KLayout and GDSFactory+ still show the old layers. The generated files differ from the committed ones even if you
changed nothing. For example, `<symbols>` entries change order and layer names lose their `1/0` suffix. Check the diff
and keep only your changes.

What a foundry PDK usually changes in `qpdk/tech.py`:

- **`LayerMapQPDK`**: the GDS layer numbers and any extra mask layers. Keep the non-fabrication layers (`SIM_AREA`,
  `SIM_ONLY`, `LABEL_*`, `WG`, `ERROR_PATH`). Cells, routing, and simulation use them.
- **`get_layer_stack()`**: thicknesses, `zmin`, and materials, for example Al instead of Nb. Keep the level names `M1`,
  `Substrate`, and `Vacuum`. The CPW models and `LAYER_STACK_NO_VACUUM` look them up by name.
- **`get_gatemon_layer_stack()`**: the separate hybrid process with thermal oxide, Nb/Au contacts, hBN and an Al top
  gate. Set the hBN thickness from your measured flake. `CNT` records a transferred nanotube's position and is not a
  metal mask or a PEC extrusion. `CNT_CONTACT`, `TOP_GATE` and `JJ_PATCH` define the conductors, and `HBN` records the
  transferred dielectric footprint. Keep these definitions and their layer views together when merging process changes.
- **`material_properties`**: permittivity and loss tangent of your substrate. The CPW models read these.
- **`LAYER_CONNECTIVITY`**: which layers connect through vias, bumps, and bridges.
- **Cross-sections** such as `coplanar_waveguide`: default width, gap, and bend radius for 50 Ω on your substrate.
- **`routing_strategies`**: the default cross-sections for routing.

The default layer stack also drives gdsfactory and GDSFactory+ 3D views. Background substrates use the layout bounds.
Drawn metal renders directly, while ground extraction uses `SIM_AREA` and excludes the draw and etch masks. These metal
regions are disjoint and preserve additive priority. Ground is not a background level, so viewing prepared solver
geometry does not fill its intentional gaps. `SIM_AREA` is hidden in the layer views so vacuum does not obscure the
device, while background substrates remain visible. Keep these rules in step when changing the process.

For standalone 3D viewing, run `uv run --extra graphics python qpdk/samples/sample_to_3d.py`. The sample uses
gdsfactory's `component.to_3d()` and `scene.show()` to open a trimesh window. The same call embeds the view in Jupyter.
The sample scales the camera clipping range with the scene size so micrometre chip coordinates stay visible.

QPDK has no table of design rules. Minimum widths and gaps are default parameter values in the cross-sections and cells.
Check the cells you keep against your rules. For hosted DRC, set `[tool.gdsfactoryplus.drc] pdk` in `pyproject.toml` to
your own rule deck.

The FEM datasets in `qpdk/models/datasets/` were simulated on the QPDK stack. The analytical models follow your stack,
but these datasets do not. Regenerate them with the scripts in `qpdk/models/datasets/recipes/`, or treat the models that
use them as approximate.

(cell-library)=

## Cell library

`qpdk/cells/__init__.py` star-imports every cell module. `get_pdk()` then collects every public function decorated with
`@gf.cell` or annotated to return a `Component`. A cell is registered under its **name in `qpdk.cells`**.

Sample layouts in `qpdk/samples/*.pic.yml` are also registered by filename when no Python cell already owns that name. A
paired `.gsch` and `.pic.yml` sample needs no Python wrapper. The docs include these samples with links to both source
files.

- **To add a cell**, write it in a new module in `qpdk/cells/`, decorate it with `@gf.cell`, and star-import the module
  in `qpdk/cells/__init__.py`. Give it a prefix, for example `acme_transmon`, so it never clashes with a future QPDK
  cell.

- **To override a cell**, define a function with the same name in a module that is star-imported **after** the QPDK one.
  The last import wins. YAML netlists, `gf.get_component("name")`, and GDSFactory+ then use your version. QPDK cells
  that import the original directly still use the original. For example, `qpdk/cells/capacitor.py` imports `straight`
  directly, so it ignores your `straight`.

  Your version and the original also produce different cells with the **same name**, and both can end up in one layout.
  So only override cells that no other cell imports. Otherwise give your version a new name, such as `acme_straight`,
  and update the callers. Use the same `tags=` as the original cell.

- **To remove a cell**, delete its import or its function. Then delete its tests and regression references. Several
  tests import cells directly.

Before you remove a cell, check which cells import it, for example with `git grep -n "import.*\bjosephson_junction\b"`.

PDK model registration is derived from the SAX descriptors in each cell's schematic. Declare the model once with
`sax_model(...)` in `qpdk/cells/_schematic.py` and pass the schematic through `@gf.cell(schematic_function=...)`. The
PDK imports that model directly and derives aliases from the declaration. `SAX_MODEL_ALIASES` in `qpdk/__init__.py`
reports these aliases and is no longer a list to maintain.

The boxed double-pad transmon declares `boxed_transmon_fem` and forwards its pad dimensions, ground clearance and
junction circuit settings to the stored QPDK capacitance matrix. A fork with another process needs a matching extraction
or a replacement model declaration.

Virtual cells and partial aliases use a matching `<cell_name>_schematic` declaration in `qpdk/cells/_schematic.py`,
since `gf.vcell` and partial factories cannot accept `schematic_function`. Discovery reads that declaration without
constructing the layout. Define the SAX descriptor there just as for a regular cell.

Analytical models without a same-named layout are registered as circuit primitives. Other cell models stay in the full
`qpdk.models.models` catalog until their schematic declares a SAX binding. Existing supported registrations are
preserved.

Schematic metadata is available through `gf.kcl.factories[name].get_schematic()`, so no separate attachment call is
needed. Keep the model and schematic descriptor aligned when changing a cell's circuit ports. Placement markers are not
circuit ports. Partial cell factories inherit the base schematic and bind any model settings they preset.

Packaged artwork lives in `qpdk/symbols/<symbol>.svg` and reaches GDSFactory+ through `display.symbol`. Draw on a canvas
of 100 units per grid cell, with leads ending on the canvas edge where each port sits, and the aspect ratio follows the
port box derived from the declared sides. Without a matching file the editor renders the layout geometry instead.

## Pruning

These parts describe the generic QPDK process. Remove or replace them:

- `qpdk/samples/` and the test chips imported at the top of `qpdk/cells/__init__.py`
- `notebooks/` and the docs pages that include them (`docs/notebooks.rst`, `docs/simulation.rst`)
- README sections such as the component list, sample chips, and simulation showcase
- Regression references in `tests/gds_ref/`, `tests/test_pdk/`, and `tests/models/test_models_regression/`

When your process and cells are ready, regenerate the references. Check the diff before you commit.

```bash
just test-gds-force      # GDS, settings, and netlist references
just test-models-force   # model references, if your stack changes model results
just test                # everything should pass now
```

## Staying current with upstream

A fork that stops merging QPDK stops getting model fixes, new cells, and tool updates. After a year it is very hard to
catch up. So merge often and keep each merge small.

### Merge release tags, not `main`

QPDK creates a `v<version>` tag each time the version changes on `main`. A tag is a tested state with a changelog. Merge
each new tag, or at least one tag per month.

Your `main` is your own work plus one merge for each QPDK tag:

```{mermaid}
:alt: The fork starts from an upstream tag, adds its own commits, and merges each later tag via a sync pull request.
flowchart LR
    U1["upstream<br/>v0.4.0"] --> U2["upstream<br/>v0.4.1"] --> U3["upstream<br/>v0.5.0"]
    U1 -->|fork| F1["Rename, process,<br/>pruning"] --> M1["Merge<br/>v0.4.1"]
    M1 --> F2["Your<br/>changes"] --> M2["Merge<br/>v0.5.0"]
    U2 -->|sync PR| M1
    U3 -->|sync PR| M2
```

Before you merge, check what is new and how far you have diverged:

```bash
git fetch upstream
git for-each-ref --sort=-v:refname --count=1 --format='%(refname:short)' 'refs/tags/upstream/v*'  # latest tag
git log --oneline upstream/v0.4.0..upstream/v0.4.1   # what the new tag adds
git log --oneline upstream/v0.4.1..main              # what your fork adds
git diff --stat upstream/v0.4.1 main                 # files that differ
```

Read the release notes and the [changelog](changelog.md) for every tag since your last merge, including tags you skip.
Look for renamed cells and changed defaults.

(ownership-map)=

### Ownership map

Each path belongs to your fork, to QPDK, or to both. The table shows what to do when it conflicts.

| Path                                                   | Owner | On merge                                         |
| ------------------------------------------------------ | ----- | ------------------------------------------------ |
| [Identity files](#identity-rename), version            | Fork  | Keep your names. Take the other QPDK changes     |
| `CITATION.cff`                                         | Fork  | Use your authors and your own Zenodo DOI series  |
| `qpdk/layers.yaml`, `qpdk/klayout/`                    | Fork  | Keep yours. Add QPDK layers you want, regenerate |
| `qpdk/tech.py`                                         | Both  | Keep your values. Take new QPDK definitions      |
| `qpdk/models/` (except datasets)                       | QPDK  | Take QPDK                                        |
| `qpdk/models/datasets/data/`                           | QPDK  | Take QPDK, unless you regenerated it             |
| `qpdk/models/datasets/recipes/`                        | QPDK  | Take QPDK                                        |
| QPDK modules in `qpdk/cells/`                          | QPDK  | Merge. New QPDK cells usually merge cleanly      |
| `qpdk/__init__.py`                                     | Both  | Keep model discovery and Python cell precedence  |
| Your cell modules, override modules, and test chips    | Fork  | QPDK never changes them                          |
| `qpdk/samples/`, `notebooks/`                          | Both  | Keep removed files removed. Take QPDK for others |
| `tests/gds_ref/`, `tests/test_pdk/`, model `.npz` refs | Fork  | Never merge. Keep yours, then regenerate         |
| Other tests, `justfile`, `*.just`                      | QPDK  | Take QPDK                                        |
| `docs/conf.py`, `docs/index.md`, `README.md`           | Both  | Keep your names. Take structure changes          |
| `.github/workflows/`, `.pre-commit-config.yaml`        | QPDK  | Take QPDK, then rename again                     |
| `uv.lock`                                              | QPDK  | Take QPDK, then run `uv lock`                    |

### Add files, do not edit QPDK files

Each line you change in a QPDK file can conflict again on every later merge. Add new files instead:

- To change a cell, override it in your own module (see [Cell library](#cell-library)). Do not edit
  `qpdk/cells/transmon.py`.
- To add cross-sections, define them in your own module and register them in `get_pdk()`, for example
  `cross_sections={**get_cross_sections(tech), **get_cross_sections(my_tech)}`.
- To add tests, add new test files. Do not extend the QPDK ones.

(resolving-conflicts)=

### Resolving conflicts

Merge on a branch, so the result gets a review and CI like any other change:

```bash
git switch -c sync/upstream-v0.4.1 main
git merge upstream/v0.4.1
```

In a merge, `--ours` is your fork and `--theirs` is QPDK. Resolve conflicts by file type:

- **Identity and version files.** Resolve by hand. Keep your names and version, and take all other QPDK changes. Do not
  use `git checkout --ours` on the whole file. Files such as `pyproject.toml` also contain QPDK dependency updates, and
  you would lose them. Then check if QPDK added a new line with its name that you must rename.
- **`qpdk/tech.py` and `qpdk/layers.yaml`.** Resolve by hand. Keep your process values and take the QPDK additions. If
  the layers changed, run `uv run python qpdk/tech.py` again.
- **Cells and models.** Merge as usual.
- **Regression references** (`tests/gds_ref/*.gds`, `tests/test_pdk/*.yml`, `.npz`). Do not merge them by hand. Keep
  yours with `git checkout --ours -- <path>` and `git add <path>`. After the merge, regenerate them with
  `just test-gds-force`, and with `just test-models-force` if models changed. Commit them separately so the diff is easy
  to review.
- **Other Git LFS files** (`tests/models/data/*.csv` and the `.parquet` datasets). Git merges the LFS pointer files as
  text, so a conflict puts markers inside the pointer. Never edit the pointer. Pick one side and run `git add <path>`.
  Usually take QPDK with `git checkout --theirs -- <path>`. Take `--ours` only if you regenerated the data. The `.npz`
  files are also in LFS, but they are regression references. Always keep yours.

```{mermaid}
:alt: Conflicts are resolved by file type, as listed above, and every merge ends with just test and just run-pre.
flowchart LR
    S["Conflicted<br/>path"] --> I["Identity,<br/>version"] --> A1["Keep your names,<br/>take the rest"]
    S --> P["tech.py,<br/>layers.yaml"] --> A2["Merge by hand,<br/>regen KLayout"]
    S --> C["Cells,<br/>models"] --> A3["Merge normally"]
    S --> R["Regression<br/>references"] --> A4["Keep ours,<br/>regenerate"]
    S --> L["Other Git<br/>LFS files"] --> A5["Take one side,<br/>never edit pointer"]
    A1 & A2 & A3 & A4 & A5 --> T["just test,<br/>just run-pre"]
```

Then run `just test` and `just run-pre`, and open a pull request for the merge. Merge it with a merge commit. Never
squash or rebase it, or the next merge conflicts on everything again.

### Taking a single change

To take one QPDK fix without a full merge, cherry-pick it. The `-x` flag records where it came from.

```bash
git cherry-pick -x <commit>
```

To reset one file to the QPDK version:

```bash
git checkout upstream/v0.4.1 -- qpdk/models/cpw.py
```

Neither of these causes a conflict at the next merge.

### Send generic fixes to QPDK

If you fix a model bug or write a cell that works in any process, open a pull request to QPDK. You then get it back with
the next merge and do not have to maintain it yourself. See the [contributing guide](contributing.md).

### Optional: automated sync pull requests

A scheduled CI job can prepare each merge for you. It pushes the latest QPDK tag to a `sync/upstream` branch and opens a
pull request from it. Below are examples for GitHub Actions, GitLab CI, and Forgejo Actions. The job does nothing if
`main` or `sync/upstream` already contains the tag.

If the pull request has conflicts, resolve them locally as in [Resolving conflicts](#resolving-conflicts). Then push the
result to the sync branch. The open pull request updates.

```bash
git push origin HEAD:sync/upstream
```

Merge that pull request before the next QPDK tag. The next run force-pushes the new tag to `sync/upstream` and replaces
your resolution. Merge the sync pull request with a merge commit. Never squash or rebase it.

#### GitHub Actions

Save this as `.github/workflows/sync-upstream.yml`:

```yaml
name: Sync upstream

on:
  schedule:
    - cron: "0 6 * * 1" # Mondays 06:00 UTC
  workflow_dispatch:

permissions: {}

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
        with:
          fetch-depth: 0
          token: ${{ secrets.SYNC_TOKEN }}
      - name: Push the latest upstream tag to sync/upstream and open a pull request
        env:
          GH_TOKEN: ${{ secrets.SYNC_TOKEN }}
        run: |
          git fetch --no-tags https://github.com/gdsfactory/quantum-rf-pdk.git '+refs/tags/v*:refs/tags/upstream/v*'
          tag=$(git for-each-ref --sort=-v:refname --count=1 --format='%(refname:short)' 'refs/tags/upstream/v*')
          if git merge-base --is-ancestor "$tag" origin/main; then
            echo "Already up to date with $tag"
            exit 0
          fi
          if git rev-parse -q --verify origin/sync/upstream >/dev/null &&
            git merge-base --is-ancestor "$tag" origin/sync/upstream; then
            echo "sync/upstream already has $tag"
            exit 0
          fi
          git push --force origin "$tag^{commit}:refs/heads/sync/upstream"
          if [ -z "$(gh pr list -R "$GITHUB_REPOSITORY" --head sync/upstream --json number --jq '.[].number')" ]; then
            gh pr create -R "$GITHUB_REPOSITORY" --base main --head sync/upstream \
              --title "Merge upstream ${tag#upstream/}" \
              --body "Automated sync with QPDK ${tag#upstream/}. Regenerate the regression references before merging."
          fi
```

Use a personal access token or GitHub App token as `SYNC_TOKEN`, not `GITHUB_TOKEN`. GitHub rejects `GITHUB_TOKEN`
pushes that change `.github/workflows/`, and QPDK releases often do. Also, pull requests opened with `GITHUB_TOKEN` do
not start your CI. A fine-grained token needs write access to contents, pull requests, and workflows. Pin the action to
a commit SHA, as QPDK does.

#### GitLab CI

Add this job to `.gitlab-ci.yml`. Then create a pipeline schedule for `main` in **Build > Pipeline schedules**.

```yaml
sync-upstream:
  image:
    name: alpine/git
    entrypoint: [""]
  rules:
    - if: $CI_PIPELINE_SOURCE == "schedule"
  variables:
    GIT_DEPTH: 0
  script:
    - git fetch --no-tags https://github.com/gdsfactory/quantum-rf-pdk.git '+refs/tags/v*:refs/tags/upstream/v*'
    - tag=$(git for-each-ref --sort=-v:refname --count=1 --format='%(refname:short)' 'refs/tags/upstream/v*')
    - if git merge-base --is-ancestor "$tag" HEAD; then echo "Already up to date with $tag"; exit 0; fi
    - remote="https://oauth2:${SYNC_TOKEN}@${CI_SERVER_HOST}/${CI_PROJECT_PATH}.git"
    - |
      if git fetch -q "$remote" sync/upstream 2>/dev/null && git merge-base --is-ancestor "$tag" FETCH_HEAD; then
        echo "sync/upstream already has $tag"
        exit 0
      fi
    - >-
      git push --force
      -o merge_request.create -o merge_request.target=main
      -o merge_request.title="Merge upstream ${tag#upstream/}"
      "$remote" "$tag^{commit}:refs/heads/sync/upstream"
```

The `merge_request.*` push options open a merge request on the first push and update it later. Store a project access
token with the `write_repository` scope as the masked CI/CD variable `SYNC_TOKEN`. By default `CI_JOB_TOKEN` cannot
push.

#### Forgejo Actions

Save this as `.forgejo/workflows/sync-upstream.yml`. Set `runs-on` to a label of your runner. The image needs `git` and
`curl`.

```yaml
on:
  schedule:
    - cron: "0 6 * * 1" # Mondays 06:00 UTC
  workflow_dispatch:

jobs:
  sync:
    runs-on: docker
    steps:
      - name: Push the latest upstream tag to sync/upstream and open a pull request
        env:
          SYNC_TOKEN: ${{ secrets.SYNC_TOKEN }}
          SERVER: ${{ github.server_url }}
          REPO: ${{ github.repository }}
        run: |
          host=${SERVER#*://}
          git clone -q "https://oauth2:${SYNC_TOKEN}@${host}/${REPO}.git" fork
          cd fork
          git fetch --no-tags https://github.com/gdsfactory/quantum-rf-pdk.git '+refs/tags/v*:refs/tags/upstream/v*'
          tag=$(git for-each-ref --sort=-v:refname --count=1 --format='%(refname:short)' 'refs/tags/upstream/v*')
          if git merge-base --is-ancestor "$tag" origin/main; then
            echo "Already up to date with $tag"
            exit 0
          fi
          if git rev-parse -q --verify origin/sync/upstream >/dev/null &&
            git merge-base --is-ancestor "$tag" origin/sync/upstream; then
            echo "sync/upstream already has $tag"
            exit 0
          fi
          git push --force origin "$tag^{commit}:refs/heads/sync/upstream"
          # 201: pull request opened. 409: one is already open for sync/upstream.
          code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$SERVER/api/v1/repos/$REPO/pulls" \
            -H "Authorization: token $SYNC_TOKEN" -H 'Content-Type: application/json' \
            -d "{\"base\":\"main\",\"head\":\"sync/upstream\",\"title\":\"Merge upstream ${tag#upstream/}\"}")
          [ "$code" = 201 ] || [ "$code" = 409 ]
```

Store an access token with write access to the repository as the Actions secret `SYNC_TOKEN`. The automatic token may
not start CI on the pull request.

## Your first merge

After your identity and process commits:

1. Set up the `upstream` remote ([Setting up the fork](#setting-up-the-fork)).
1. Write down which paths you own, starting from the [ownership map](#ownership-map).
1. Find the latest `upstream/v*` tag and look at `git diff --stat <tag> main`.
1. Merge it on a branch and resolve conflicts by file type.
1. Run `just test-gds-force` and `just test`, commit the new references separately, and open a pull request.
