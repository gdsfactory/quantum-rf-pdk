# Security Policy

## Supported Versions

We provide security fixes for the latest released version of this project. Older versions are not actively patched.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

If you discover a security vulnerability, we ask that you disclose it responsibly by contacting the maintainers
privately using one of the following methods:

### GitHub Security Advisories (preferred)

Use GitHub's private vulnerability reporting feature to submit a report directly to the maintainers:

1. Navigate to the [Security tab](https://github.com/gdsfactory/quantum-rf-pdk/security) of this repository.
1. Click **"Report a vulnerability"**.
1. Fill in the details of the vulnerability and submit.

This keeps the report confidential while allowing the maintainers to assess and address the issue before any public
disclosure.

### Direct Contact

If you prefer, you may contact the maintainers directly via the email addresses listed in the project's
[PyPI page](https://pypi.org/p/qpdk) or in `pyproject.toml`.

## What to Include

To help us triage your report quickly, please include:

- A description of the vulnerability and its potential impact.
- Steps to reproduce or a proof-of-concept.
- The affected version(s).
- Any suggested mitigations, if known.

## Response Process

- We will acknowledge receipt of your report within **5 business days**.
- We aim to provide an initial assessment within **10 business days**.
- We will coordinate a fix and release timeline with you and credit you in the release notes (unless you prefer to
  remain anonymous).

## Scope

This project is a process design kit (PDK) for superconducting quantum RF devices. Security concerns most likely to be
relevant include:

- Malicious code execution via crafted layout or netlist files.
- Supply chain issues in PDK components or dependencies.
- Sensitive fabrication or device parameters inadvertently exposed.

## Disclosure Policy

We follow a coordinated disclosure model. We ask that you give us a reasonable amount of time to address a vulnerability
before any public disclosure. We will work with you to agree on a disclosure timeline.
