---
myst:
  html_meta:
    description: How to fork QPDK into a process-specific PDK and keep the fork current with upstream releases.
---

# Customizing QPDK for Your Process

QPDK describes a generic superconducting process. If you have a real fabrication process, you will want your own PDK
with your layer map, layer stack, and cell library, while still picking up model, cell, and tooling improvements from
QPDK. This page describes the recommended way to do that and how to keep the result healthy over time.

The short version:

1. Fork the repository and rename its identity in one commit.
1. Replace the process definition (`qpdk/layers.yaml` and `qpdk/tech.py`) in one more commit.
1. Prune or override cells, then regenerate the regression references.
1. Merge upstream release tags regularly, using the [ownership map](#ownership-map) to resolve conflicts.

## Fork or compose

There are two ways to build on QPDK:

- **Fork**: copy the repository, own the process definition, and merge upstream changes into it over time.
- **Compose**: keep `qpdk` as a normal dependency, and build your own `gdsfactory.Pdk` in a separate package that reuses
  selected QPDK cells and models.

**Fork if you change the layer stack, layer numbers, or materials.** Almost everything in QPDK reads the process through
`qpdk.tech`: cells draw on `LAYER.*`, cross-sections are defined there, and the analytical CPW models read substrate and
metal thickness from `LAYER_STACK` (see `qpdk/models/cpw.py`). A composed package cannot swap those without
monkeypatching, so a fork is simpler and more honest.

**Compose if you only add things.** If the QPDK process is close enough and you want extra cells, samples, or chips on
top of it, depend on `qpdk` and merge its registries into your own `Pdk` (for example
`cells={**qpdk.PDK.cells, **my_cells}`). You never have to merge anything, and you upgrade by bumping a version. The
project-template work in [#376](https://github.com/gdsfactory/quantum-rf-pdk/issues/376) covers this path.

```{mermaid}
:alt: If you change the layers, layer stack, or materials, fork. If you only add cells, samples, or chips, compose.
flowchart LR
    Q1["New layers, stack,<br/>or materials?"] -->|yes| F["Fork:<br/>own the process"]
    Q1 -->|no| Q2["Only adding cells,<br/>samples, or chips?"]
    Q2 -->|no| F
    Q2 -->|yes| C["Compose:<br/>depend on qpdk"]
```

The rest of this page assumes a fork.

(setting-up-the-fork)=

## Setting up the fork

Create the fork on your Git host, then add QPDK as a second remote. `origin` is your fork; `upstream` is QPDK.

```bash
git clone <your-fork-url> <your-pdk>
cd <your-pdk>
git remote add upstream https://github.com/gdsfactory/quantum-rf-pdk.git

# Keep upstream tags out of your own tag namespace: your release workflow creates
# v<version> tags too, and the two series must not collide.
git config remote.upstream.tagOpt --no-tags
git config --add remote.upstream.fetch '+refs/tags/*:refs/tags/upstream/*'
git fetch upstream
git tag -l 'upstream/*'   # upstream/v0.4.0, upstream/v0.4.1, ...

# Remember how you resolved a conflict, so the next merge reuses it.
# See https://git-scm.com/book/en/v2/Git-Tools-Rerere
git config rerere.enabled true
```

Do your work on branches off your fork's `main` and merge them through pull requests. Do not develop directly on `main`;
it is the branch you merge upstream into.

(identity-rename)=

## Identity rename

These names are hardcoded to QPDK. Change all of them in **one commit** (or two at most) and touch nothing else in that
commit. It becomes the single known conflict point for every later upstream merge, and its diff is your checklist when a
conflict shows up.

| What                | Where                                                                                  |
| ------------------- | -------------------------------------------------------------------------------------- |
| Package directory   | `qpdk/`, and every `import qpdk` / `from qpdk` (`git grep -l qpdk`: most of the repo)  |
| Name and version    | `[project]` in `pyproject.toml`, `__version__` in `qpdk/__init__.py`, `[tool.tbump]`   |
| Tooling paths       | `pyproject.toml`: coverage `source`, pytest `testpaths`, ruff per-file ignores         |
| GDSFactory+ project | `pyproject.toml`: `[tool.gdsfactoryplus]` `name`, `pdk.name`, `sidebar` libraries      |
| Layer map class     | `LayerMapQPDK` in `qpdk/tech.py`                                                       |
| PDK object          | `name="qpdk"` in `get_pdk()`, and the `QPdk` / `_QPdkMeta` classes, `qpdk/__init__.py` |
| KLayout technology  | `<name>` in `qpdk/klayout/tech.lyt`; `KLayoutTechnology(name=...)` in `qpdk/tech.py`   |
| KLayout install     | `src` and `dest` in `qpdk/install_tech.py` (both spell out `qpdk`)                     |
| Docs                | `docs/conf.py`: `project`, logos, and URLs; badges in README and index                 |
| Workflows           | `test.yml` (`--cov=qpdk`, imports)                                                     |
| Repo metadata       | Repository URLs in `.github/` (issue templates, release drafter, lychee), `AGENTS.md`  |

The `justfile` reads the package name from `pdk := env('pdk', 'qpdk')`, so set `pdk=<your_pdk>` in `.env` or change the
default. Only the `install-tech` and `show` recipes read that variable.

Internal names that never reach a user, such as `docs/_ext/qpdk_*.py` and the `qpdk.mplstyle` figure style, can stay.
Every one you leave alone is one fewer line that can conflict.

## Process replacement

The process definition lives in two files that must agree:

- `qpdk/layers.yaml`: layer names, GDS numbers, and how KLayout draws them.
- `qpdk/tech.py`: everything else about the process.

`tests/test_pdk.py::test_yaml_matches_layers` fails if the layer names or `(layer, datatype)` pairs in the two files
differ. After editing `layers.yaml`, regenerate the KLayout files and commit them with your change:

```bash
uv run python qpdk/tech.py   # rewrites qpdk/klayout/layers.lyp and tech.lyt
```

At runtime `LAYER_VIEWS` is read from the generated `layers.lyp`, not from the YAML. If you skip this step, the YAML
test passes but KLayout and GDSFactory+ keep showing the old layers.

Even with no changes, the regenerated files do not match the committed ones exactly: the connectivity `<symbols>`
entries can come out in a different order, and the layer names can lose their `1/0` suffix. Review the diff and keep
only the changes you meant.

What a foundry PDK usually changes in `qpdk/tech.py`:

- **`LayerMapQPDK`**: the foundry's GDS layer numbers and any extra mask layers. Keep the non-fabrication layers
  (`SIM_AREA`, `SIM_ONLY`, `LABEL_*`, `WG`, `ERROR_PATH`): cells, routing, and the simulation tooling use them.
- **`get_layer_stack()`**: film thicknesses, `zmin`, and materials (for example Al instead of Nb, or sapphire instead of
  Si). Keep the level names `M1`, `Substrate`, and `Vacuum`, or update their users: the CPW models read
  `LAYER_STACK.layers["Substrate"]` and `["M1"]`, and `LAYER_STACK_NO_VACUUM` filters on `"Vacuum"`.
- **`material_properties`**: permittivity and loss tangent of your substrate. The analytical CPW models pick this up
  through `get_layer_material_properties`.
- **`LAYER_CONNECTIVITY`**: which layers connect through vias, bumps, and bridges.
- **Cross-sections** such as `coplanar_waveguide`: default width, gap, and bend radius for 50 Ω on your substrate.
- **`routing_strategies`**: the cross-sections your routers default to.

QPDK has no central design-rules table. Minimum widths, gaps, and spacings live as parameter defaults in the
cross-sections and in each cell's signature, so check the cells you keep against your rules. For hosted DRC, point
`[tool.gdsfactoryplus.drc] pdk` in `pyproject.toml` at your own rule deck.

The FEM datasets under `qpdk/models/datasets/` were simulated on the QPDK stack. Unlike the analytical models, they do
not follow your stack automatically. Regenerate them with the scripts in `qpdk/models/datasets/data/`, or treat the
models built on them as approximate.

(cell-library)=

## Cell library

`qpdk/cells/__init__.py` star-imports every cell module, and `get_pdk()` collects the cells from it with
`gdsfactory.get_factories.get_cells`, which keeps the public callables decorated with `@gf.cell` (or annotated to return
a `Component`). The registry key is the **attribute name in `qpdk.cells`**, so:

- **Add a cell**: put it in a new module under `qpdk/cells/`, decorate it with `@gf.cell`, and star-import it in
  `qpdk/cells/__init__.py`. Prefix fork-specific names (for example `acme_transmon`) so they never collide with a cell
  QPDK adds later.

- **Override a cell by name**: define a function with the same name in a module that is star-imported **after** the
  upstream one. The later binding wins. This changes what YAML netlists, `gf.get_component("name")`, and GDSFactory+
  resolve to. It does **not** change upstream cells that import the original directly. For example,
  `qpdk/cells/capacitor.py` does `from qpdk.cells.waveguides import straight`, so overriding `straight` leaves the
  capacitor using the upstream one.

  `@gf.cell` names a cell after its function name and arguments, so the override and the upstream function produce
  different cells with the **same name**, and both can end up in one layout. Either override only cells that no other
  cell imports, or give your version a new name (`acme_straight`) and change the callers in your fork. Pass the same
  `tags=` as the upstream cell so the override is grouped the same way.

- **Remove a cell you have no process for**: delete its star import or the function, then delete its tests (several
  tests import cells directly) and its regression references.

Cells call each other through these direct imports, so check what depends on a cell before removing it, for example with
`git grep -n "import.*\bjosephson_junction\b"`.

SAX models are matched to cells by name as well. `SAX_MODEL_ALIASES` in `qpdk/__init__.py` maps a few cells onto another
cell's model. If you remove a model that an alias points to, `import qpdk` raises a `KeyError` when the models load. If
you remove an aliased cell, delete its entry too.

## Pruning

Several parts of the repository describe the generic process, not yours:

- `qpdk/samples/` and the test chips imported at the top of `qpdk/cells/__init__.py`
- `notebooks/`, and the docs pages that include them (`docs/notebooks.rst`, `docs/simulation.rst`)
- README sections such as the component list, sample chips, and simulation showcase
- Regression references: `tests/gds_ref/*.gds`, `tests/test_pdk/*.yml`, and `tests/models/test_models_regression/*.npz`

After the process and cells are in place, regenerate the references and review the diff before you commit it:

```bash
just test-gds-force      # GDS, settings, and netlist references
just test-models-force   # model references, if your stack changes model outputs
just test                # everything should now pass
```

## Staying current with upstream

A fork that stops merging upstream stops getting model fixes, new cells, and tooling updates. After a year it cannot
catch up without a rewrite. Most of the advice below is about keeping each merge small and predictable.

### Merge release tags, not `main`

QPDK's `release.yml` creates a tag `v<version>` whenever the version in `pyproject.toml` changes on `main`. A tag is a
coherent snapshot with a changelog, which makes it easier to reason about than a moving `main`. Merge every new tag, or
at least monthly, so each merge stays small.

Your fork's `main` is a chain of your own work with one merge per upstream tag:

```{mermaid}
:alt: The fork starts from an upstream tag, adds its own commits, and merges each later tag via a sync pull request.
flowchart LR
    U1["upstream<br/>v0.4.0"] --> U2["upstream<br/>v0.4.1"] --> U3["upstream<br/>v0.5.0"]
    U1 -->|fork| F1["Rename, process,<br/>pruning"] --> M1["Merge<br/>v0.4.1"]
    M1 --> F2["Your<br/>changes"] --> M2["Merge<br/>v0.5.0"]
    U2 -->|sync PR| M1
    U3 -->|sync PR| M2
```

Before merging, see what came in and how far you have diverged:

```bash
git fetch upstream
git for-each-ref --sort=-v:refname --count=1 --format='%(refname:short)' 'refs/tags/upstream/v*'  # latest tag
git log --oneline upstream/v0.4.0..upstream/v0.4.1   # what the new tag adds
git log --oneline upstream/v0.4.1..main              # what your fork carries on top
git diff --stat upstream/v0.4.1 main                 # where you differ, file by file
```

Read the release notes and [changelog](changelog.md) for the tags you are skipping over, especially for renamed cells or
changed defaults.

(ownership-map)=

### Ownership map

Every path is owned by one side. This table tells you what will conflict and how to resolve it.

| Path                                                   | Owner    | On merge                                         |
| ------------------------------------------------------ | -------- | ------------------------------------------------ |
| [Identity files](#identity-rename), version            | Fork     | Keep your names; take upstream's other changes   |
| `qpdk/layers.yaml`, `qpdk/klayout/`                    | Fork     | Keep yours; add wanted upstream layers, regen    |
| `qpdk/tech.py`                                         | Shared   | Keep your values; take new upstream definitions  |
| `qpdk/models/` (except datasets)                       | Upstream | Take upstream                                    |
| `qpdk/models/datasets/data/`                           | Upstream | Take upstream unless you regenerated it          |
| Upstream modules in `qpdk/cells/`                      | Upstream | Merge; new upstream cells usually apply cleanly  |
| Fork cell and override modules, fork test chips        | Fork     | Upstream never touches them                      |
| `qpdk/samples/`, `notebooks/`                          | Depends  | Keep pruned files pruned; take upstream for rest |
| `tests/gds_ref/`, `tests/test_pdk/`, model `.npz` refs | Fork     | Never merge; keep yours, then regenerate         |
| Other tests, `justfile`, `*.just`                      | Upstream | Take upstream                                    |
| `docs/conf.py`, `docs/index.md`, `README.md`           | Shared   | Keep identity lines; take structural changes     |
| `.github/workflows/`, `.pre-commit-config.yaml`        | Upstream | Take upstream, reapply identity lines            |
| `uv.lock`                                              | Upstream | Take upstream, then `uv lock` to match yours     |

### Keep your edits out of upstream files

Every line you edit in an upstream-owned file is a conflict you may have to resolve again on every future merge. Prefer
adding files:

- To change a cell, write a fork module that defines a cell with the same name and star-import it last (see
  [Cell library](#cell-library)). Do not edit `qpdk/cells/transmon.py` in place.
- To add cross-sections, put them in a fork module and register them in `get_pdk()`, for example
  `cross_sections={**get_cross_sections(tech), **get_cross_sections(my_tech)}`, rather than adding them inside
  `qpdk/tech.py`.
- To add tests, add new test files instead of extending upstream ones.

(resolving-conflicts)=

### Resolving conflicts

Do the merge on a branch, so the result is reviewed and CI-tested like any other change:

```bash
git switch -c sync/upstream-v0.4.1 main
git merge upstream/v0.4.1
```

During the merge, `--ours` is your fork and `--theirs` is upstream. Handle conflicts by file class:

- **Identity and version files**: resolve by hand. In each conflict, keep your names and version and take everything
  else from upstream's side. Do not `git checkout --ours` the whole file: files such as `pyproject.toml` also carry
  upstream's dependency and tooling changes, and that would drop them. Then check whether upstream added a new identity
  line you also need to rename.
- **`qpdk/tech.py` and `qpdk/layers.yaml`**: resolve by hand. Keep your process values and take upstream's additions. If
  the layers changed, rerun `uv run python qpdk/tech.py`.
- **Cells and models**: merge normally.
- **Regression references** (`tests/gds_ref/*.gds`, `tests/test_pdk/*.yml`, `.npz`): do not resolve them by hand.
  `*.gds` is marked `binary` in `.gitattributes`, so any file both sides changed is a hard conflict. Keep yours
  (`git checkout --ours -- <path>`, then `git add` it), finish the merge, then regenerate with `just test-gds-force`
  (and `just test-models-force` if models changed). Commit the regenerated references separately and review that diff on
  its own.
- **Other Git LFS files** (`tests/models/data/*.csv`, `qpdk/models/datasets/data/**/*.parquet`, and
  `notebooks/data/jax_backend_comparison/*.parquet`): `.gitattributes` sets `merge=lfs`, but Git LFS does not define a
  merge driver by that name, so Git falls back to a text merge of the pointer files. If both sides changed a file, the
  conflict markers appear inside the LFS pointer. Never edit them; pick one side with `git checkout --theirs -- <path>`
  (upstream, the usual choice for these datasets unless you regenerated them) or `--ours`, then `git add` it. The `.npz`
  model references are LFS files too, but they are regression references: keep yours and regenerate them as above, never
  take upstream's.

```{mermaid}
:alt: Conflicts are resolved by file class, as listed above, and every merge ends with just test and just run-pre.
flowchart LR
    S["Conflicted<br/>path"] --> I["Identity,<br/>version"] --> A1["Keep your names,<br/>take the rest"]
    S --> P["tech.py,<br/>layers.yaml"] --> A2["Merge by hand,<br/>regen KLayout"]
    S --> C["Cells,<br/>models"] --> A3["Merge normally"]
    S --> R["Regression<br/>references"] --> A4["Keep ours,<br/>regenerate"]
    S --> L["Other Git<br/>LFS files"] --> A5["Take one side,<br/>never edit pointer"]
    A1 & A2 & A3 & A4 & A5 --> T["just test,<br/>just run-pre"]
```

Then run `just test` and `just run-pre`, and open a pull request for the merge.

### Taking a single change

To take one upstream fix without merging a whole tag, cherry-pick it (`-x` records where it came from):

```bash
git cherry-pick -x <commit>
```

To reset one file to upstream's version, for example a model you never meant to diverge:

```bash
git checkout upstream/v0.4.1 -- qpdk/models/cpw.py
```

The next tag merge sees the same content on both sides, so neither of these causes a conflict later.

### Contribute generic fixes back

If you fix a bug in a model or write a cell that would work in any process, open a pull request to QPDK first and let it
reach your fork with the next tag merge. Each fix carried only in your fork is a difference you have to maintain
forever. See the [contributing guide](contributing.md).

### Optional: automated sync pull requests

A scheduled CI job can do the bookkeeping. It pushes the latest upstream tag to a `sync/upstream` branch and opens a
pull (or merge) request from it, so you review a diff instead of starting a merge by hand. The examples below are for
GitHub Actions, GitLab CI, and Forgejo Actions. All three skip the run when your `main` already contains the tag, or
when `sync/upstream` already does, so a resolved merge you pushed there is kept until a newer upstream tag replaces it.

If the merge conflicts, the pull request shows it. Resolve it locally on a branch as described in
[Resolving conflicts](#resolving-conflicts), then push the result to the sync branch so the open pull request picks it
up:

```bash
git push origin HEAD:sync/upstream
```

The merge commit has the upstream tag as a parent, so this is a fast-forward of `sync/upstream`. Merge the pull request
before the next upstream tag: the next run force-pushes that tag over the branch.

Merge these pull requests with a merge commit, never squash or rebase. A squash discards the link to upstream history,
and the next sync then conflicts on everything again.

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

`SYNC_TOKEN` must be a personal access token or GitHub App token, not the default `GITHUB_TOKEN`, for two reasons:
GitHub rejects `GITHUB_TOKEN` pushes that add or change files under `.github/workflows/`, and upstream releases usually
do, and pull requests opened with `GITHUB_TOKEN` do not trigger your CI. A fine-grained token needs contents, pull
requests, and workflows write access on the fork. Pin the action to a commit SHA, as QPDK's own workflows do.
`.github/workflows/test-gfp-upstream.yml` uses the same scheduled-check pattern to test against the latest GDSFactory+
release.

#### GitLab CI

Add this job to `.gitlab-ci.yml`, and create a pipeline schedule for `main` under **Build > Pipeline schedules**:

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

The `merge_request.*` push options open the merge request on the first push; later pushes update the open one. Store a
project access token with the `write_repository` scope as the masked CI/CD variable `SYNC_TOKEN`; by default
`CI_JOB_TOKEN` cannot push.

#### Forgejo Actions

Save this as `.forgejo/workflows/sync-upstream.yml`. Set `runs-on` to a label your runner provides; the job needs `git`
and `curl` in its image.

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

`SYNC_TOKEN` is an access token with read and write access to the repository, stored as an Actions secret. Use it rather
than the automatic token, which may not be allowed to trigger CI on the pull request the job opens.

## Your first merge

Once your fork has its identity and process commits, you can do a complete first sync:

1. Configure the `upstream` remote with namespaced tags ([Setting up the fork](#setting-up-the-fork)).
1. Write down which paths you own, starting from the [ownership map](#ownership-map).
1. Find the latest `upstream/v*` tag, and check `git diff --stat <tag> main` for surprises.
1. Merge it on a branch and resolve conflicts by file class.
1. Run `just test-gds-force` and `just test`, commit the regenerated references separately, and open a pull request.
