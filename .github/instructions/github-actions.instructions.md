---
applyTo: ".github/workflows/**,.github/actions/**"
---

# GitHub Actions review instructions

Workflows are linted by `actionlint`, security-audited by `zizmor`, schema-checked by `check-github-workflows`,
formatted by `yamlfmt` (`.github/.yamlfmt.yaml`) and linted by `yamllint --strict` (`.github/.yamllint.yaml`). The
`check-workflows` hook from `pdk-ci-workflow-public` enforces the shared PDK CI structure.

## Security (what zizmor looks for, and you should too)

- **Pin third-party actions to a full commit SHA**, not a tag. First-party `actions/*` may use a major tag where the
  repo already does so — match the surrounding style.
- Flag `pull_request_target` combined with a checkout of the PR head. That executes untrusted code with write
  permissions and secrets.
- Flag untrusted input (`github.event.issue.title`, `.body`, `.head_ref`, PR titles) interpolated directly into a `run:`
  block — that is script injection. Pass it through `env:` and reference `"$VAR"` instead.
- Every job needs least-privilege `permissions:`. Flag a new job that inherits broad write permissions it does not need,
  and `contents: write` / `id-token: write` granted without a clear reason.
- Secrets must not be echoed, written to artifacts, or passed to third-party actions that do not need them.
- Prefer `persist-credentials: false` on `actions/checkout` unless the job pushes.

## Correctness

- Keep the Python version matrix in sync with `requires-python` in `pyproject.toml` (3.12–3.14) and with the
  `get-python-versions.yml` reusable workflow — do not hard-code a version list that will drift.
- Add a `timeout-minutes` to new jobs; a hung simulation or docs build should not burn an hour.
- `concurrency` with `cancel-in-progress` belongs on PR-triggered workflows, not on release or publish workflows.
- Changes to `release.yml` / `build.yml` affect PyPI publishing. Treat them as high risk and ask for a maintainer to
  confirm, especially anything touching trusted publishing or version tags.
- `copilot-setup-steps.yml` provisions the Copilot cloud agent environment; if a change adds a required system
  dependency or extra, check it is installed there too.

## Style

- Name every job and every non-obvious step.
- Prefer an existing reusable workflow over a copy-pasted job body.
