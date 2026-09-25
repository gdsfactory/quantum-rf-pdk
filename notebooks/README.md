# Notebooks

This folder contains [Jupyter Notebooks](https://jupyter.org/) demonstrating different features of QPDK.

## Contributors

The source for the notebooks is the [`src`](src) folder that contains
[Jupytext](https://jupytext.readthedocs.io/en/latest/) `py:percent`.

> [!IMPORTANT]
> Keep the scripts here out of the import scope of the package.

The scripts may be used as-is or converted to the Jupyter Notebooks with:

```bash
uvx jupytext --to ipynb <script>.py
```

or all-at once with

```bash
just convert-notebooks
```

There is also a pre-commit hook checking that the notebooks are in-sync with the source files.

## Documentation figures

Figures in `figures/` are generated from `docs/figures/*.typ` with `just build-doc-figures`. Run it before opening the
notebooks locally. Documentation builds generate the figures automatically. The recipe uses the same pinned Inter and
Outfit fonts as the docs PDF, and the colors come from the HTML theme in `docs/_static/css/custom.css`. Only commit the
Typst source. Notebook Markdown uses `figures/<name>.svg`, and the docs build copies the generated figures into
`docs/notebooks/`. GitHub's notebook preview cannot display these uncommitted figures; use the built docs or generate
them locally.
