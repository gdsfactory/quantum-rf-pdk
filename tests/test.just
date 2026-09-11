PYTEST_COMMAND := "uv run --extra models --extra graphics --extra hfss --extra qutip --group dev pytest -n auto --dist loadgroup"

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

# Shared pytest invocation for internal callers passing pre-split argument strings (ensures LFS data first)
[private]
[group('test')]
_pytest *args: check-lfs
    {{ PYTEST_COMMAND }} {{ args }}

# Run the full test suite in parallel using pytest
# Uses positional arguments so quoted args survive, e.g. `just test -k "some name"`
[positional-arguments]
[group('test')]
test *args: check-lfs
    {{ PYTEST_COMMAND }} "$@"

# Run optical port position tests (tests/test_pdk.py::test_optical_port_positions)
[group('test')]
test-ports: (_pytest "-s tests/test_pdk.py::test_optical_port_positions")

# Run GDS regressions tests (tests/test_pdk.py)
[group('test')]
test-gds: (_pytest "-s tests/test_pdk.py")

# Run GDS regressions tests (tests/test_pdk.py) and regenerate
[group('test')]
test-gds-force: (_pytest "-s tests/test_pdk.py --force-regen")

# Run GDS regressions tests (tests/test_pdk.py) and stop at first failure
[group('test')]
test-gds-fail-fast: (_pytest "-s tests/test_pdk.py -x")

# Run model regression tests (tests/test_models_regression.py)
[group('test')]
test-models: (_pytest "-s tests/test_models_regression.py")

# Regenerate model regression reference files
[group('test')]
test-models-force: (_pytest "-s tests/test_models_regression.py --force-regen")

# Run HFSS simulation tests (requires HFSS to be installed)
[group('test')]
test-hfss *args: check-lfs
    uv run --all-extras --group dev pytest -m hfss {{ args }}

# Query the latest version of the public GDSFactory+ VS Code extension.
# Output contract (used by .github/workflows/test.yml and test-gfp-upstream.yml):
#   line 1: JSON array of Python versions for the gfp test matrix
#   line 2: latest=<latest extension version> (empty when unknown)
#   line 3: major=<major version of that release> (empty when unknown)
[group('test')]
[script('uv', 'run', '--script')]
get-gfp-versions:
    import json, urllib.request

    # The extension bundles a cp312-only SDK and runtime, so the gfp
    # matrix is pinned to Python 3.12 regardless of the main matrix.
    print(json.dumps(["3.12"]))

    def print_version_info(latest):
        major = ""
        if latest:
            head = latest.split(".")[0]
            if head.isdigit():
                major = head
        print(f"latest={latest or ''}")
        print(f"major={major}")

    def version_key(v):
        return [int(x) if x.isdigit() else 0 for x in v.split(".")]

    try:
        body = json.dumps({
            "filters": [{
                "criteria": [{"filterType": 7, "value": "gdsfactory.gdsfactoryplus"}],
                "pageNumber": 1,
                "pageSize": 1,
                "sortBy": 0,
                "sortOrder": 0,
            }],
            "assetTypes": [],
            "flags": 1,  # IncludeVersions
        }).encode()
        req = urllib.request.Request(
            "https://marketplace.visualstudio.com/_apis/public/gallery/extensionquery?api-version=7.2-preview.1",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json;api-version=7.2-preview.1",
                "User-Agent": "quantum-rf-pdk-ci",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
        extensions = data.get("results", [{}])[0].get("extensions", [])
        versions = (
            [v["version"] for v in extensions[0].get("versions", [])]
            if extensions
            else []
        )
        print_version_info(max(versions, key=version_key) if versions else None)
    except Exception:
        # Unknown version → empty latest/major so consumers skip the gfp job
        print_version_info(None)

# Download the public GDSFactory+ VS Code extension and extract the bundled
# gdsfactoryplus SDK and `gfp` binary into build/gfp-vsix/ (neither is
# pip-installable: gdsfactoryplus >= 2.0 ships only inside the extension).
# Writes build/gfp-vsix/env.sh exporting PYTHONPATH (SDK + bundled nyancad)
# and GFP_BIN (binary) for `just test-gfp` to source.
[group('test')]
[script('uv', 'run', '--script')]
fetch-gfp version="latest":
    import gzip, io, json, os, platform, shutil, stat, urllib.request, zipfile
    from pathlib import Path

    PUBLISHER = "gdsfactory"
    EXTENSION = "gdsfactoryplus"
    API = "https://marketplace.visualstudio.com/_apis/public/gallery"
    OUT = Path("build/gfp-vsix")
    IS_WINDOWS = platform.system() == "Windows"

    def resolve_latest():
        body = json.dumps({
            "filters": [{
                "criteria": [{"filterType": 7, "value": f"{PUBLISHER}.{EXTENSION}"}],
                "pageNumber": 1,
                "pageSize": 1,
                "sortBy": 0,
                "sortOrder": 0,
            }],
            "assetTypes": [],
            "flags": 1,  # IncludeVersions
        }).encode()
        req = urllib.request.Request(
            f"{API}/extensionquery?api-version=7.2-preview.1",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json;api-version=7.2-preview.1",
                "User-Agent": "quantum-rf-pdk-ci",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
        extensions = data.get("results", [{}])[0].get("extensions", [])
        versions = (
            [v["version"] for v in extensions[0].get("versions", [])]
            if extensions
            else []
        )
        if not versions:
            raise SystemExit(
                f"extension {PUBLISHER}.{EXTENSION} not found on the Marketplace"
            )

        def version_key(v):
            # Release versions must beat prereleases at equal numbers
            # ("2.1.0" > "2.1.0-beta"), so append a release flag; the
            # non-digit component of a prerelease sorts as 0 either way.
            return ([int(x) if x.isdigit() else 0 for x in v.split(".")], "-" not in v)

        return max(versions, key=version_key)

    def target_platform():
        arch = {
            "x86_64": "x64",
            "amd64": "x64",
            "arm64": "arm64",
            "aarch64": "arm64",
        }.get(platform.machine().lower())
        os_name = {"Darwin": "darwin", "Linux": "linux", "Windows": "win32"}.get(
            platform.system()
        )
        if not arch or not os_name:
            raise SystemExit(
                f"unsupported platform: {platform.system()} {platform.machine()}"
            )
        return f"{os_name}-{arch}"

    version = "{{ version }}"
    target = target_platform()
    if version == "latest":
        version = resolve_latest()
    stamp = OUT / "VERSION"
    wanted = f"{version} {target}"

    if stamp.exists() and stamp.read_text().strip() == wanted:
        print(f"gdsfactoryplus {wanted} already extracted at {OUT}")
    else:
        url = (
            f"{API}/publishers/{PUBLISHER}/vsextensions/{EXTENSION}"
            f"/{version}/vspackage?targetPlatform={target}"
        )
        print(f"downloading {PUBLISHER}.{EXTENSION} {version} ({target})")
        req = urllib.request.Request(url, headers={"User-Agent": "quantum-rf-pdk-ci"})
        with urllib.request.urlopen(req, timeout=600) as r:
            payload = r.read()
        # The Marketplace may serve the VSIX gzip-encoded.
        if payload[:2] == b"\x1f\x8b":
            payload = gzip.decompress(payload)

        if OUT.exists():
            shutil.rmtree(OUT)
        OUT.mkdir(parents=True)
        with zipfile.ZipFile(io.BytesIO(payload)) as vsix:
            for member in vsix.namelist():
                if member.startswith("extension/bin/") and not member.endswith("/"):
                    target_path = OUT / member.removeprefix("extension/")
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    target_path.write_bytes(vsix.read(member))

        stamp.write_text(wanted)

    # Validate on every run (not just fresh extraction) and chmod
    # unconditionally: zip extraction does not preserve the exec bit, so
    # a stamp hit on a re-cloned or partially cleaned checkout would
    # otherwise produce a non-executable gfp with no error.
    gfp_bin = OUT / "bin" / ("gfp.exe" if IS_WINDOWS else "gfp")
    sdk_dirs = [OUT / "bin" / "python" / "gdsfactoryplus", OUT / "bin" / "python" / "nyancad"]
    if not gfp_bin.is_file():
        raise SystemExit(f"extraction incomplete: {gfp_bin} is missing from {OUT}")
    missing_dirs = [str(d) for d in sdk_dirs if not d.is_dir()]
    if missing_dirs:
        raise SystemExit(f"extraction incomplete: SDK dirs missing: {missing_dirs}")
    if not IS_WINDOWS:
        gfp_bin.chmod(gfp_bin.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    sep = os.pathsep
    env_sh = OUT / "env.sh"
    env_sh.write_text(
        f'export PYTHONPATH="{sep.join(str(d.resolve()) for d in sdk_dirs)}"\n'
        f'export GFP_BIN="{gfp_bin.resolve()}"\n'
    )
    print(f"extracted to {OUT}; source {env_sh} for PYTHONPATH and GFP_BIN")

# Run GDSFactory+ tests (pytest-based suite marked `gfp`; replaces the `gfp test` CLI removed in v2).
# GFP_REQUIRED=1: gdsfactoryplus and its v2 API must be present (no silent skips).
# The SDK and the `gfp` binary are not pip-installable: gdsfactoryplus >= 2.0
# ships bundled in the public GDSFactory+ VS Code extension, so when the SDK is
# not importable this recipe first extracts both from the extension (fetch-gfp).
# The gfp-marked tests live in the three files listed below; naming them
# explicitly avoids collecting the whole repo under the minimal gfp extras.
# --dist loadgroup keeps the server tests (xdist_group "gfp-server", which
# share one gfp serve process and the repo's build/ dir) on a single worker.
[group('test')]
test-gfp *args:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -f build/gfp-vsix/env.sh ]]; then
        source build/gfp-vsix/env.sh
    fi
    if ! uv run --no-sync python -c "import gdsfactoryplus" 2>/dev/null; then
        just fetch-gfp
        source build/gfp-vsix/env.sh
    fi
    GFP_REQUIRED=1 uv run --extra gdsfactoryplus --group dev pytest -m gfp -n auto --dist loadgroup \
        tests/test_gfp_lvs.py tests/test_gfp_sax.py tests/test_gfp_server.py {{ args }}
