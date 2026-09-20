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
    if git rev-parse --git-dir >/dev/null 2>&1; then
        echo "Git LFS is available. Pulling LFS files…"
        git lfs pull
    elif command -v jj >/dev/null 2>&1 && jj git root >/dev/null 2>&1; then
        # jj does not run smudge filters, so create a private child change for
        # the fetched content rather than leaving LFS files dirty in real work.
        echo "Git LFS is available. Smudging LFS files in the jj working copy…"
        gitdir=$(jj git root)
        mapfile -t lfs_files < <(git --git-dir "$gitdir" lfs ls-files -n)
        if ((${#lfs_files[@]} == 0)); then
            echo "Error: Git LFS tracks no files in the backing repository." >&2
            exit 1
        fi
        pointer_files=()
        for f in "${lfs_files[@]}"; do
            if [[ ! -f "$f" ]]; then
                echo "Error: LFS-tracked file is missing from the workspace: $f" >&2
                exit 1
            fi
            # Skip files that already hold real content, not a pointer (cmp, not
            # $(head ...): the binary content would trip bash's null-byte check)
            if ! cmp -s <(head -c 42 "$f") \
                <(printf 'version https://git-lfs.github.com/spec/v1'); then
                continue
            fi
            pointer_files+=("$f")
        done
        if ((${#pointer_files[@]})); then
            jj new -m "private: lfs"
        fi
        for f in "${pointer_files[@]}"; do
            tmp=$(mktemp "$f.XXXXXX")
            GIT_LFS_SKIP_SMUDGE=0 git --git-dir "$gitdir" lfs smudge < "$f" > "$tmp"
            if cmp -s <(head -c 42 "$tmp") \
                <(printf 'version https://git-lfs.github.com/spec/v1'); then
                rm -f "$tmp"
                echo "Error: Git LFS did not smudge $f." >&2
                exit 1
            fi
            mv "$tmp" "$f"
        done
    else
        echo "Error: Not in a Git or Jujutsu repository." >&2
        exit 1
    fi

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
        # Split off any prerelease suffix and add a release flag so a
        # stable release beats prereleases at equal version numbers
        # ("2.1.0" over "2.1.0-beta.1"): without splitting, the "1" in
        # "2.1.0-beta.1" parses as an extra numeric component and sorts
        # after the stable release.
        nums = v.split("-")[0].split(".")
        return ([int(x) if x.isdigit() else 0 for x in nums], "-" not in v)

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

# Download the public GDSFactory+ VS Code extension, or accept a local VSIX
# path as the argument, and extract the bundled gdsfactoryplus SDK and `gfp`
# binary into build/gfp-vsix/ (neither is pip-installable: gdsfactoryplus >=
# 2.0 ships only inside the extension).
# Writes build/gfp-vsix/env.sh exporting PYTHONPATH (SDK + bundled nyancad)
# and GFP_BIN (binary) for `just test-gfp` to source.
[group('test')]
[script('uv', 'run', '--script')]
fetch-gfp $version="latest":
    import gzip, hashlib, io, json, os, platform, shlex, shutil, stat, subprocess, sys, tempfile, urllib.request, zipfile
    from pathlib import Path

    PUBLISHER = "gdsfactory"
    EXTENSION = "gdsfactoryplus"
    API = "https://marketplace.visualstudio.com/_apis/public/gallery"
    OUT = Path("build/gfp-vsix")
    IS_WINDOWS = platform.system() == "Windows"
    MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024
    MAX_EXTRACTED_BYTES = 1024 * 1024 * 1024

    def read_limited(stream, limit, label):
        payload = stream.read(limit + 1)
        if len(payload) > limit:
            raise SystemExit(f"{label} exceeds {limit} bytes")
        return payload

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
            data = json.loads(read_limited(r, 10 * 1024 * 1024, "Marketplace response"))
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
            # Split off any prerelease suffix and add a release flag so a
            # stable release beats prereleases at equal version numbers
            # ("2.1.0" over "2.1.0-beta.1"): without splitting, the "1" in
            # "2.1.0-beta.1" parses as an extra numeric component and sorts
            # after the stable release.
            nums = v.split("-")[0].split(".")
            return ([int(x) if x.isdigit() else 0 for x in nums], "-" not in v)

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

    requested = os.environ["version"]
    target = target_platform()
    local_vsix = None
    if requested.lower().endswith(".vsix"):
        local_vsix = Path(requested).expanduser().resolve()
        if not local_vsix.is_file():
            raise SystemExit(f"local VSIX does not exist: {local_vsix}")
        with local_vsix.open("rb") as stream:
            fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()[:16]
        wanted = f"local:{fingerprint} {target}"
    else:
        version = resolve_latest() if requested == "latest" else requested
        wanted = f"{version} {target}"
    stamp = OUT / "VERSION"
    extract_root = OUT
    staged = False
    staging_dir = None

    if stamp.exists() and stamp.read_text().strip() == wanted:
        print(f"gdsfactoryplus {wanted} already extracted at {OUT}")
    else:
        if local_vsix is not None:
            print(f"extracting local VSIX {local_vsix} ({target})")
            archive = local_vsix
        else:
            url = (
                f"{API}/publishers/{PUBLISHER}/vsextensions/{EXTENSION}"
                f"/{version}/vspackage?targetPlatform={target}"
            )
            print(f"downloading {PUBLISHER}.{EXTENSION} {version} ({target})")
            req = urllib.request.Request(url, headers={"User-Agent": "quantum-rf-pdk-ci"})
            with urllib.request.urlopen(req, timeout=600) as r:
                payload = read_limited(r, MAX_DOWNLOAD_BYTES, "VSIX download")
            if payload[:2] == b"\x1f\x8b":
                with gzip.GzipFile(fileobj=io.BytesIO(payload)) as compressed:
                    payload = read_limited(
                        compressed, MAX_DOWNLOAD_BYTES, "decompressed VSIX"
                    )
            archive = io.BytesIO(payload)

        OUT.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = tempfile.TemporaryDirectory(
            prefix=f".{OUT.name}-", dir=OUT.parent
        )
        extract_root = Path(staging_dir.name)
        staged = True
        bin_dir = (extract_root / "bin").resolve()
        with zipfile.ZipFile(archive) as vsix:
            members = [
                member
                for member in vsix.infolist()
                if member.filename.startswith("extension/bin/")
                and not member.is_dir()
            ]
            if sum(member.file_size for member in members) > MAX_EXTRACTED_BYTES:
                raise SystemExit("VSIX extension/bin contents exceed extraction limit")
            for member in members:
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise SystemExit(f"unsafe VSIX symlink member: {member.filename}")
                target_path = (
                    extract_root / member.filename.removeprefix("extension/")
                ).resolve()
                if not target_path.is_relative_to(bin_dir):
                    raise SystemExit(
                        "unsafe VSIX member escapes extension/bin: "
                        f"{member.filename}"
                    )
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with vsix.open(member) as source, target_path.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)

    executable_names = (
        ["gfp.exe", "uv.exe", "rg.exe"] if IS_WINDOWS else ["gfp", "uv", "rg"]
    )
    executables = [extract_root / "bin" / name for name in executable_names]
    sdk_dirs = [
        extract_root / "bin" / "python" / "gdsfactoryplus",
        extract_root / "bin" / "python" / "nyancad",
    ]
    missing_executables = [str(path) for path in executables if not path.is_file()]
    if missing_executables:
        raise SystemExit(
            f"extraction incomplete: executables missing: {missing_executables}"
        )
    missing_dirs = [str(d) for d in sdk_dirs if not d.is_dir()]
    if missing_dirs:
        raise SystemExit(f"extraction incomplete: SDK dirs missing: {missing_dirs}")
    if not IS_WINDOWS:
        for executable in executables:
            executable.chmod(
                executable.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
            )
    import_env = dict(os.environ)
    import_env["PYTHONPATH"] = os.pathsep.join(str(path) for path in sdk_dirs)
    import_check = subprocess.run(
        [sys.executable, "-c", "import gdsfactoryplus, nyancad"],
        capture_output=True,
        env=import_env,
        text=True,
    )
    if import_check.returncode:
        raise SystemExit(f"extracted SDK import failed: {import_check.stderr.strip()}")

    final_gfp_bin = OUT / "bin" / executable_names[0]
    final_sdk_dirs = [
        OUT / "bin" / "python" / "gdsfactoryplus",
        OUT / "bin" / "python" / "nyancad",
    ]
    (extract_root / "VERSION").write_text(wanted)
    sep = os.pathsep
    env_sh = extract_root / "env.sh"
    env_sh.write_text(
        f"export PYTHONPATH={shlex.quote(sep.join(str(d.resolve()) for d in final_sdk_dirs))}\n"
        f"export GFP_BIN={shlex.quote(str(final_gfp_bin.resolve()))}\n"
    )

    if staged:
        backup = OUT.with_name(f".{OUT.name}-previous-{os.getpid()}")
        if OUT.exists():
            OUT.replace(backup)
        try:
            extract_root.replace(OUT)
        except Exception:
            if backup.exists():
                backup.replace(OUT)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        staging_dir.cleanup()
    print(f"extracted to {OUT}; source {OUT / 'env.sh'} for PYTHONPATH and GFP_BIN")

# Run GDSFactory+ tests (pytest-based suite marked `gfp`; replaces the `gfp test` CLI removed in v2).
# GFP_REQUIRED=1: gdsfactoryplus and its v2 API must be present (no silent skips).
# The SDK and the `gfp` binary are not pip-installable: gdsfactoryplus >= 2.0
# ships bundled in the public GDSFactory+ VS Code extension, so when the SDK is
# not importable this recipe first extracts both from the extension (fetch-gfp).
# The gfp-marked tests live in the files listed below; naming them
# explicitly avoids collecting the whole repo under the minimal gfp extras.
# --dist loadgroup keeps the server tests (xdist_group "gfp-server", which
# share one gfp serve process and the repo's build/ dir) on a single worker.
[group('test')]
[positional-arguments]
test-gfp *args:
    #!/usr/bin/env bash
    set -euo pipefail
    uv sync --extra gdsfactoryplus --group dev
    probe_gfp_v2() {
        uv run --no-sync python -c \
            "from gdsfactoryplus.factory_metadata import resolve_factory_model; from gdsfactoryplus.sim.sax import simulate_mosaic_sax"
    }
    if [[ -f build/gfp-vsix/env.sh ]]; then
        source build/gfp-vsix/env.sh
    fi
    if ! probe_gfp_v2 >/dev/null 2>&1 || [[ -z "${GFP_BIN:-}" || ! -x "${GFP_BIN:-}" ]]; then
        just fetch-gfp
        source build/gfp-vsix/env.sh
    fi
    probe_gfp_v2
    GFP_REQUIRED=1 uv run --no-sync pytest -m gfp -n auto --dist loadgroup \
        tests/test_gfp_lvs.py tests/test_gfp_sax.py tests/test_gfp_server.py \
        tests/test_sample_schematic_geometry.py "$@"
