PYTEST_COMMAND := "uv run --extra models --extra graphics --extra hfss --extra qutip --group dev pytest -n auto"

# Check if Git LFS is available and pull LFS files
[private]
[group('test')]
check-lfs:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! command -v git-lfs >/dev/null 2>&1; then
        echo ""
        echo "Error: Git LFS is not installed!"
        echo ""
        echo "This repository uses Git LFS to store test data files."
        echo "Please install Git LFS before running tests:"
        echo ""
        echo "  Ubuntu/Debian:  sudo apt-get install git-lfs"
        echo "  RHEL/Rocky:     sudo dnf install git-lfs"
        echo "  macOS:          brew install git-lfs"
        echo "  Windows:        Download from https://git-lfs.github.com/"
        echo ""
        echo "After installing, run: git lfs install && git lfs pull"
        echo ""
        exit 1
    fi
    echo "Git LFS is available. Pulling LFS files…"
    git lfs pull

# Run the full test suite in parallel using pytest
[group('test')]
test *args: check-lfs
    {{ PYTEST_COMMAND }} {{ args }}

# Run optical port position tests (tests/test_pdk.py::test_optical_port_positions)
[group('test')]
test-ports:
    {{ PYTEST_COMMAND }} -s tests/test_pdk.py::test_optical_port_positions

# Run GDS regressions tests (tests/test_pdk.py)
[group('test')]
test-gds:
    {{ PYTEST_COMMAND }} -s tests/test_pdk.py

# Run GDS regressions tests (tests/test_pdk.py) and regenerate
[group('test')]
test-gds-force:
    {{ PYTEST_COMMAND }} -s tests/test_pdk.py --force-regen

# Run GDS regressions tests (tests/test_pdk.py) and stop at first failure
[group('test')]
test-gds-fail-fast:
    {{ PYTEST_COMMAND }} -s tests/test_pdk.py -x

# Run model regression tests (tests/test_models_regression.py)
[group('test')]
test-models:
    {{ PYTEST_COMMAND }} -s tests/test_models_regression.py

# Regenerate model regression reference files
[group('test')]
test-models-force:
    {{ PYTEST_COMMAND }} -s tests/test_models_regression.py --force-regen

# Run HFSS simulation tests (requires HFSS to be installed)
[group('test')]
test-hfss *args: check-lfs
    uv run --all-extras --group dev pytest -m hfss {{ args }}

# Filter a JSON array of Python versions to only those supported by the latest version of gdsfactoryplus on PyPI
[group('test')]
[script('uv', 'run', '--script')]
get-gfp-versions versions:
    import json, urllib.request, sys, re
    v_in = json.loads('{{ versions }}')
    try:
        with urllib.request.urlopen("https://pypi.org/pypi/gdsfactoryplus/json") as r:
            data = json.loads(r.read().decode())

        latest = data.get("info", {}).get("version")
        info = data.get("info", {})
        req = info.get("requires_python", "")
        rels = data.get("releases", {}).get(latest, [])

        v_out = []
        for v in v_in:
            t = v.replace(".", "")
            # Check for wheels first as they are definitive
            if any(f"cp{t}" in rel.get("python_version","") or f"py{t}" in rel.get("python_version","") or v in rel.get("python_version","") for rel in rels):
                v_out.append(v)
            elif req:
                # Fallback to parsing requires_python string for the latest version
                vm = int(v.split('.')[1])
                # Match major.minor patterns like >=3.11, <3.14, ~=3.12
                m_ge = re.search(r'>=3\.(\d+)', req)
                m_lt = re.search(r'<3\.(\d+)', req)
                m_ti = re.search(r'~=3\.(\d+)', req)

                is_supported = True
                if m_ge and vm < int(m_ge.group(1)): is_supported = False
                if m_lt and vm >= int(m_lt.group(1)): is_supported = False
                if m_ti and vm != int(m_ti.group(1)): is_supported = False

                if is_supported:
                    v_out.append(v)
        # If filtering resulted in an empty list, fallback to input to avoid breaking the matrix
        print(json.dumps(v_out or v_in))
    except Exception:
        # Fallback to input versions on any error (network, parsing, etc.)
        print(json.dumps(v_in))

# Run GDSFactory+ tests
[group('test')]
test-gfp:
    uv run --extra gdsfactoryplus gfp test
