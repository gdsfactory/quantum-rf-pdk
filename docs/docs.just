ipython_config := home_directory() / ".ipython/profile_default/ipython_config.py"
mpl_style := home_directory() / ".config/matplotlib/stylelib/qpdk.mplstyle"

# Write cell outputs into documentation notebooks
[private]
[group('docs')]
write-cells:
    @uv run --group docs .github/write_cells.py

# Write model outputs into documentation notebooks
[private]
[group('docs')]
write-models:
    @uv run --extra models --group docs .github/write_models.py

# Write Justfile help output to documentation
[private]
[group('docs')]
[working-directory: 'docs']
[script('uv', 'run', '--script')]
write-justfile-help:
    import re
    import shutil
    import subprocess
    from pathlib import Path

    output_file = Path("justfile_help.txt")
    result = subprocess.run(  # noqa: S603
        [shutil.which("just"), "--list"],
        capture_output=True,
        text=True,
        check=True,
    )

    # Remove ANSI color codes
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    output = result.stdout
    output = ansi_escape.sub("", output)
    output = output.rstrip()
    output_file.write_text(output)

    print(f"Justfile help written to {output_file}")

# Convert jupytext scripts from notebooks/src to ipynb format in notebooks
[private]
[group('docs')]
convert-notebooks:
    @./.github/convert-notebooks.sh notebooks/src/*.py

# Copy all sample scripts to use as notebooks docs
[private]
[group('docs')]
copy-sample-notebooks:
    @mkdir -p docs/notebooks
    @cp notebooks/src/*.py docs/notebooks/

# Temporarily setup IPython configuration for documentation build
[private]
[group('docs')]
setup-ipython-config-temporary-before:
    @mkdir -p ~/.ipython/profile_default
    @{{ if path_exists(ipython_config) == "TRUE" { "mv " + ipython_config + " " + ipython_config + ".bak" } else { "true" } }}
    @cp docs/ipython_config.py {{ ipython_config }}
    @mkdir -p ~/.config/matplotlib/stylelib/
    @{{ if path_exists(mpl_style) == "TRUE" { "mv " + mpl_style + " " + mpl_style + ".bak" } else { "true" } }}
    @cp docs/qpdk.mplstyle {{ mpl_style }}

# Restore original IPython configuration
[private]
[group('docs')]
setup-ipython-config-temporary-after:
    @{{ if path_exists(ipython_config + ".bak") == "TRUE" { "mv " + ipython_config + ".bak " + ipython_config } else { "rm -f " + ipython_config } }}
    @{{ if path_exists(mpl_style + ".bak") == "TRUE" { "mv " + mpl_style + ".bak " + mpl_style } else { "rm -f " + mpl_style } }}

# Shared prerequisites for building documentation (runs in parallel)
[private]
[parallel]
[group('docs')]
docs-prerequisites: write-cells write-models write-justfile-help copy-sample-notebooks

# Build the HTML documentation
[group('docs')]
docs: docs-prerequisites setup-ipython-config-temporary-before
    @trap 'just setup-ipython-config-temporary-after' EXIT INT TERM; uv run {{ all_extras_run }} --group docs sphinx-build -b html docs docs/_build/html

# Setup LaTeX for PDF documentation
[group('docs')]
docs-latex: docs-prerequisites setup-ipython-config-temporary-before
    @trap 'just setup-ipython-config-temporary-after' EXIT INT TERM; uv run {{ all_extras_run }} --group docs sphinx-build -b latex docs docs/_build/latex

# Build PDF documentation (requires a TeXLive installation)
[working-directory: 'docs/_build/latex']
[group('docs')]
docs-pdf: docs-latex
    #!/usr/bin/env bash
    set -euo pipefail
    XINDYOPTS="-M sphinx.xdy" latexmk -pdfxe -xelatex -interaction=nonstopmode -f -file-line-error || {
        if [ -f qpdk.pdf ]; then
            echo "PDF generated despite warnings"
            exit 0
        else
            exit 1
        fi
    }
