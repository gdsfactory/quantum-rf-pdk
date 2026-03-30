set dotenv-load := true

pdk := env('pdk', 'qpdk')
cpus := num_cpus()

python_version := env_var_or_default("UV_PYTHON", `python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'`)
hfss_extra := if python_version =~ '^3\.14.*' { "" } else { "--extra hfss" }

install_extras := if hfss_extra == "" { "--no-extra hfss" } else { "" }
all_extras_run := if hfss_extra == "" { "--extra models --extra graphics --extra qutip" } else { "--all-extras" }

import 'tests/test.just'
import 'docs/docs.just'

# List available commands
default:
    @just --list

# Install the package and all development dependencies
[group('setup')]
install *args:
    @uv sync --all-extras {{ if args =~ '--python 3\.14' { "--no-extra hfss" } else { install_extras } }} {{ args }}

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
    uvx prek run --all-files

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
    import gdsfactory as gf
    from qpdk import PDK, logger
    from qpdk.config import PATH

    PDK.activate()
    component_name = "{{ component_name }}"

    if not component_name:
        if not shutil.which("fzf"):
            print("Error: 'fzf' is not installed.")
            print(
                "Please install it (see https://github.com/junegunn/fzf#installation) or provide a component name: 'just show <name>'"
            )
            sys.exit(1)

        try:
            cell_names = sorted(PDK.cells.keys())
            process = subprocess.Popen(
                ["fzf", "--header=Select a PDK component to show"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )
            stdout, _ = process.communicate(input="\n".join(cell_names))
            component_name = stdout.strip()
            if not component_name:
                print("No component selected.")
                sys.exit(0)
        except Exception as e:
            logger.error(f"Error during selection: {e}")
            sys.exit(1)

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
