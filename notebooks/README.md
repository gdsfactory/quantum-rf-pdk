# Notebooks

This folder contains [Jupyter Notebooks](https://jupyter.org/) demonstrating different features of QPDK.

## Contributors

The source for the notebooks is the [`src`](notebooks/src) folder that contains
[Jupytext](https://jupytext.readthedocs.io/en/latest/) `py:percent`.

> [!IMPORTANT]
> Keep the scripts here out of the import scope of the package.

The scripts may be used as-is or converted to the Jupyter Notebooks with:

```bash
uvx jupytext --to ipynb <script>.py
```

or all-at once with

```bash
make convert-notebooks
```

There is also a pre-commit hook checking that the notebooks are in-sync with the source files
