set dotenv-load := true

pdk := env('pdk', 'qpdk')
cpus := num_cpus()

import 'tests/test.just'
import 'docs/docs.just'

# List available commands
default:
    @just --list

# Install the package and all development dependencies
[group('setup')]
install extras="--all-extras":
    @uv sync {{ extras }}

# Install KLayout technology files for the PDK
[group('setup')]
install-tech:
    @uv run --dev {{ pdk }}/install_tech.py

# Clean up all build, test, coverage and Python artifacts
[confirm]
[group('setup')]
clean:
    @rm -rf dist build *.egg-info docs/_build docs/notebooks

# Update pre-commit hooks to the latest revisions
[group('lint')]
update-pre:
    @uvx prek autoupdate -j $(( {{ cpus }} / 2 + {{ cpus }} % 2 ))

# Run all pre-commit hooks on all files
[group('lint')]
run-pre:
    @uvx prek run --all-files

# Install pre-commit hooks to run on `git commit`
[group('lint')]
install-pre:
    @uvx prek install

# Build the Python package (install build tool and create dist)
[group('build')]
build:
    @rm -rf dist
    uv build

# Generate and show a PDK component by name (opens interactive chooser by default), saving its GDS to build/
[group('build')]
show component_name="":
    #!/usr/bin/env -S uv run python
    import shutil
    import subprocess
    import sys
    import threading
    from pathlib import Path

    component_name = "{{ component_name }}"
    cache_path = Path("build/cell_names.cache")

    if not component_name:
        if not shutil.which("fzf"):
            print("Error: 'fzf' is not installed.")
            print("Please install it (see https://github.com/junegunn/fzf#installation) or provide a component name: 'just show <name>'")
            sys.exit(1)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Use a descriptive header and a prompt that implies background activity
        process = subprocess.Popen(
            ["fzf", "--header=PDK Components", "--prompt=Select Component> "],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )
        sent_cells = set()

        def feed_fzf():
            fresh_process = None
            try:
                # 1. Feed from cache immediately
                if cache_path.exists():
                    with open(cache_path, "r") as f:
                        for line in f:
                            cell = line.strip()
                            if cell and cell not in sent_cells:
                                try:
                                    process.stdin.write(cell + "\n")
                                    sent_cells.add(cell)
                                except (BrokenPipeError, ValueError): return
                    try: process.stdin.flush()
                    except (BrokenPipeError, ValueError): return

                # 2. Feed fresh list in background (unbuffered)
                list_cmd = ["uv", "run", "python", "-u", "-c", "from qpdk import PDK; print('\\n'.join(sorted(PDK.cells.keys())))"]
                fresh_process = subprocess.Popen(list_cmd, stdout=subprocess.PIPE, text=True, stderr=subprocess.DEVNULL)
                new_cells = []
                if fresh_process.stdout:
                    for line in fresh_process.stdout:
                        cell = line.strip()
                        if cell:
                            new_cells.append(cell)
                            if cell not in sent_cells:
                                try:
                                    process.stdin.write(cell + "\n")
                                    sent_cells.add(cell)
                                    process.stdin.flush()
                                except (BrokenPipeError, ValueError): break

                if new_cells:
                    with open(cache_path, "w") as f: f.write("\n".join(new_cells) + "\n")
            except Exception as e:
                print(f"Error in background component loading: {e}", file=sys.stderr)
            finally:
                try:
                    if process.stdin: process.stdin.close()
                except Exception: pass
                if fresh_process:
                    try: fresh_process.terminate()
                    except Exception: pass

        # Start background feeder
        thread = threading.Thread(target=feed_fzf, daemon=True)
        thread.start()

        # Wait for selection WITHOUT using communicate() as it closes stdin immediately
        try:
            stdout_data = process.stdout.read()
            process.wait()
            if process.returncode not in (0, 130) and not stdout_data:
                print(f"Component selection process exited unexpectedly with code {process.returncode}.", file=sys.stderr)
                sys.exit(process.returncode)
            component_name = stdout_data.strip()
        except KeyboardInterrupt:
            process.terminate()
            print("Component selection cancelled by user. Exiting.", file=sys.stderr)
            sys.exit(1)

        if not component_name:
            print("No component selected; exiting without running any layout.", file=sys.stderr)
            sys.exit(0)

    import gdsfactory as gf
    from qpdk import PDK, logger
    from qpdk.config import PATH
    PDK.activate()
    (build_dir := PATH.gds).mkdir(parents=True, exist_ok=True)
    component = gf.get_component(component_name)
    gds_path = build_dir / f"{component.name}.gds"
    component.write_gds(gds_path)
    logger.info(f"Saved GDS for {component_name} to '{gds_path}'")
    component.show()

# Run all tests, pre-commit hooks, build wheel and documentation in parallel
[group('all')]
[parallel]
all: test run-pre build docs
