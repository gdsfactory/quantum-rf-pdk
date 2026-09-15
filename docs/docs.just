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
write-justfile-help:
    @just --list --color never > justfile_help.txt

# Convert jupytext scripts from notebooks/src to ipynb format in notebooks
[private]
[group('docs')]
convert-notebooks:
    @./.github/convert-notebooks.sh notebooks/src/*.py

# Copy llms.txt (https://llmstxt.org/) into the built site root
[private]
[group('docs')]
write-llms-txt:
    @mkdir -p docs/_extra
    @cp docs/llms.txt docs/_extra/llms.txt

# Notebooks that should be copied as .ipynb because they have precalculated outputs
# (These will have their .py counterparts removed if they exist to force MyST-NB to use .ipynb)
precalculated_notebooks := "hfss_*.ipynb matlab_integration.ipynb"

# Copy all sample scripts and notebooks to use as documentation
[private]
[group('docs')]
copy-sample-notebooks:
    @mkdir -p docs/notebooks
    # Copy all source files from notebooks/src
    @cp -p notebooks/src/*.py notebooks/src/*.m docs/notebooks/
    # Copy notebooks with pre-calculated outputs and remove their source counterparts
    @cd notebooks && for f in {{precalculated_notebooks}}; do \
        cp -p "$f" ../docs/notebooks/; \
        base="${f%.ipynb}"; \
        rm -f "../docs/notebooks/$base.py" "../docs/notebooks/$base.m"; \
    done

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

# Local cache of the docs fonts (Inter, Outfit, Code New Roman, Fira Math).
# CI installs the same families system-wide (see .github/workflows/pages.yml),
# so this only exists to give local Typst builds identical rendering instead of
# DejaVu fallbacks. docs/conf.py picks the dir up as extra Typst font paths.
docs-fonts-dir := "build/docs-fonts"

# Download the docs fonts into build/docs-fonts/ (stamp-cached, reruns on bump)
[private]
[group('docs')]
[script('uv', 'run', '--script')]
fetch-docs-fonts:
    import io, shutil, urllib.request, zipfile
    from urllib.parse import unquote
    from pathlib import Path

    OUT = Path("build/docs-fonts")
    STAMP = OUT / "FONTS"
    VERSION = "1"

    FILES = [
        "https://github.com/firamath/firamath/releases/download/v0.3.4/FiraMath-Regular.otf",
        "https://github.com/Outfitio/Outfit-Fonts/raw/main/fonts/ttf/Outfit-Regular.ttf",
        "https://github.com/Outfitio/Outfit-Fonts/raw/main/fonts/ttf/Outfit-Bold.ttf",
        "https://github.com/Outfitio/Outfit-Fonts/raw/main/fonts/ttf/Outfit-SemiBold.ttf",
        "https://github.com/chrissimpkins/codeface/raw/master/fonts/code-new-roman/Code%20New%20Roman-Regular.otf",
        "https://github.com/chrissimpkins/codeface/raw/master/fonts/code-new-roman/Code%20New%20Roman-Bold.otf",
        "https://github.com/chrissimpkins/codeface/raw/master/fonts/code-new-roman/Code%20New%20Roman-Italic.otf",
    ]
    # CI gets Inter via apt, which is no option locally; the variable font
    # instances every weight the docs use from a single file.
    INTER_ZIP = "https://github.com/rsms/inter/releases/download/v4.1/Inter-4.1.zip"
    INTER_FILES = ["InterVariable.ttf", "InterVariable-Italic.ttf"]

    if STAMP.exists() and STAMP.read_text().strip() == VERSION:
        print(f"docs fonts already cached in {OUT}")
    else:
        print(f"downloading docs fonts into {OUT}")
        if OUT.exists():
            shutil.rmtree(OUT)
        OUT.mkdir(parents=True)

        def fetch(url):
            req = urllib.request.Request(url, headers={"User-Agent": "quantum-rf-pdk-docs"})
            with urllib.request.urlopen(req, timeout=600) as r:
                return r.read()

        for url in FILES:
            name = unquote(url.rsplit("/", 1)[-1])
            (OUT / name).write_bytes(fetch(url))
        with zipfile.ZipFile(io.BytesIO(fetch(INTER_ZIP))) as z:
            for name in INTER_FILES:
                (OUT / name).write_bytes(z.read(name))
        STAMP.write_text(VERSION)

# Shared prerequisites for building documentation (runs in parallel)
[private]
[parallel]
[group('docs')]
docs-prerequisites: write-cells write-models write-justfile-help copy-sample-notebooks write-llms-txt

# Build documentation with the given Sphinx builder into docs/_build/<builder>
[private]
[group('docs')]
_sphinx builder: check-lfs docs-prerequisites setup-ipython-config-temporary-before
    @trap 'just setup-ipython-config-temporary-after' EXIT INT TERM; \
    uv run --group docs sphinx-build -b {{ builder }} docs docs/_build/{{ builder }}

# Build the HTML documentation
[group('docs')]
docs: (_sphinx "html")

# Generate Typst sources for the PDF documentation
[group('docs')]
docs-typst: fetch-docs-fonts (_sphinx "typst")

# Build PDF documentation (typsphinx compiles it with the bundled Typst compiler)
[group('docs')]
docs-pdf: fetch-docs-fonts (_sphinx "typstpdf")
