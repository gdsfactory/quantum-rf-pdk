# Contributing

We welcome contributions of all sizes—bug fixes, documentation, new components, and improvements to existing layouts or
models. Please keep pull requests focused, documented, and easy to review.

## How to contribute

- Discuss substantial changes in an issue or draft PR before investing significant effort.
- Keep changeset size reasonable; split large efforts into smaller, reviewable chunks.
- Add or update tests and documentation alongside code changes.

## Installation for Contributors

Clone the repository and install the development dependencies:

```bash
git clone https://github.com/gdsfactory/quantum-rf-pdk.git
cd quantum-rf-pdk
just install
```

> [!NOTE]
> [Git LFS](https://git-lfs.github.com/) must be installed to run all tests locally. Some test data files (e.g., CSV
> files in `tests/models/data/`) are tracked with Git LFS and will not be properly downloaded without it.

### KLayout Technology Installation

For contributors, you can install the technology files directly from the repository:

```bash
just install-tech
```

> [!NOTE]
> After installation, restart KLayout to ensure the new technology appears.

## Development workflow

Check out the commands for testing and building documentation with:

```bash
just --list
```

- Run the full test suite: `just test`
- Run layout regression tests: `just test-gds` (or `just test-gds-fail-fast` while iterating)
- Run formatting/linting hooks: `just run-pre`
- Build docs (optional but encouraged when docs change): `just docs`

## AI usage policy

We encourage using AI tools to accelerate your workflow—for example, prototyping components and layouts, generating
model visualizations, writing tests, or drafting documentation. However, **do not submit "AI slop."** You are fully
responsible for the quality of your contributions. Any AI-generated code, layouts, or text must be reviewed,
functionally tested, and completely human-maintainable.

```{include} CODE_OF_CONDUCT.md
```
