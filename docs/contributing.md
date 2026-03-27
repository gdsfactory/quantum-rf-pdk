# Contributing

We welcome contributions of all sizes—bug fixes, documentation, new components, and improvements to existing layouts or
models. Please keep pull requests focused, documented, and easy to review.

## How to contribute

- Discuss substantial changes in an issue or draft PR before investing significant effort.
- Keep changeset size reasonable; split large efforts into smaller, reviewable chunks.
- Add or update tests and documentation alongside code changes.

## Development workflow

- Install dependencies: `just install`
- Run the full test suite: `just test`
- Run layout regression tests: `just test-gds` (or `just test-gds-fail-fast` while iterating)
- Run formatting/linting hooks: `just run-pre`
- Build docs (optional but encouraged when docs change): `just docs`

## AI usage policy

We encourage using AI tools to accelerate your workflow—for example, prototyping components and layouts, generating
model visualizations, writing tests, or drafting documentation. However, **do not submit "AI slop."** You are fully
responsible for the quality of your contributions. Any AI-generated code, layouts, or text must be reviewed,
functionally tested, and completely human-maintainable.

## Code of Conduct

We follow [gdsfactory](https://github.com/gdsfactory/gdsfactory)
[Code of Conduct](https://gdsfactory.github.io/gdsfactory/code_of_conduct.html).
