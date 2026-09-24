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

# Build SVG figures from docs/figures/*.typ
[group('docs')]
[script('uv', 'run', '--script')]
build-doc-figures: fetch-docs-fonts
    # /// script
    # requires-python = ">=3.12"
    # dependencies = ["typst==0.15.0", "pymupdf==1.28.2", "fonttools==4.63.0"]
    # ///
    import base64
    from io import BytesIO
    from pathlib import Path
    from xml.etree import ElementTree

    import pymupdf
    import typst
    from fontTools import subset
    from fontTools.ttLib import TTFont

    sources = {p.stem: p for p in Path("docs/figures").glob("*.typ") if p.stem != "style"}
    output_dir = Path("notebooks/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    fonts_dir = Path("build/docs-fonts")
    font_faces = (
        ("Inter Variable", "InterVariable.ttf", "100 900"),
        ("Outfit", "Outfit-Regular.ttf", "400"),
        ("Outfit", "Outfit-Bold.ttf", "700"),
        ("FiraMath-Regular", "FiraMath-Regular.otf", "400"),
    )
    colors = {
        "#ffffff": "#14181e",
        "#f6f4ef": "#2a3340",
        "#0e1116": "#f0f2f6",
        "#5b6472": "#c5cbd5",
        "#dcdfe4": "#586272",
        "#2a6fb5": "#76a9ee",
        "#3979ba": "#5e9be0",
        "#1f5994": "#9ac4ff",
        "#a95a20": "#edb37c",
    }

    def color_rules(palette, prefix=""):
        return "".join(
            f'{prefix}svg.qpdk-figure [{attribute}="{color}"]'
            + "{" + f"{attribute}:{value}" + "}"
            for color, value in palette.items()
            for attribute in ("fill", "stroke")
        )

    dark = color_rules(colors)
    theme = (
        "@media(prefers-color-scheme:dark){" + dark + "}"
        + color_rules({color: color for color in colors}, "html[data-theme=light] ")
        + color_rules(colors, "html[data-theme=dark] ")
    )

    for name, source in sorted(sources.items()):
        pdf = typst.compile(
            str(source), root=".", font_paths=[str(fonts_dir)],
            ignore_system_fonts=True, format="pdf",
        )
        with pymupdf.open(stream=pdf, filetype="pdf") as document:
            svg = document[0].get_svg_image(text_as_path=False)

        characters = {}
        for element in ElementTree.fromstring(svg).iter("{http://www.w3.org/2000/svg}text"):
            family = element.get("font-family")
            characters.setdefault(family, set()).update("".join(element.itertext()))

        faces = []
        for family, filename, weight in font_faces:
            if family not in characters:
                continue
            font = TTFont(fonts_dir / filename)
            options = subset.Options()
            options.drop_tables.append("FFTM")
            subsetter = subset.Subsetter(options=options)
            subsetter.populate(text="".join(sorted(characters[family])))
            subsetter.subset(font)
            data = BytesIO()
            font.save(data)
            encoded = base64.b64encode(data.getvalue()).decode("ascii")
            kind = "opentype" if filename.endswith(".otf") else "truetype"
            extension = "otf" if filename.endswith(".otf") else "ttf"
            faces.append(
                '@font-face{font-family:"' + family + '";font-weight:' + weight + ';'
                + f'src:url(data:font/{extension};base64,{encoded}) format("{kind}")' + '}'
            )

        svg = svg.replace("<svg ", '<svg class="qpdk-figure" ', 1)
        opening_tag_end = svg.index(">") + 1
        svg = svg[:opening_tag_end] + "<style>" + "".join(faces) + theme + "</style>" + svg[opening_tag_end:]
        (output_dir / f"{name}.svg").write_text(svg + "\n")

    for stale in output_dir.glob("*.svg"):
        if stale.stem not in sources:
            stale.unlink()

# Copy llms.txt (https://llmstxt.org/) into the built site root
[private]
[group('docs')]
write-llms-txt:
    @mkdir -p docs/_extra
    @cp docs/llms.txt docs/_extra/llms.txt

# Notebooks that should be copied as .ipynb because they have precalculated outputs
# (These will have their .py counterparts removed if they exist to force MyST-NB to use .ipynb)
precalculated_notebooks := "hfss_*.ipynb comsol_*.ipynb matlab_integration.ipynb"

# Copy all sample scripts and notebooks to use as documentation
[private]
[group('docs')]
copy-sample-notebooks: build-doc-figures
    @mkdir -p docs/notebooks/data docs/_extra/notebooks/data
    @cp -p notebooks/data/comsol_* docs/notebooks/data/
    @cp -p notebooks/data/comsol_* docs/_extra/notebooks/data/
    # Copy all source files from notebooks/src
    @cp -p notebooks/src/*.py notebooks/src/*.m docs/notebooks/
    @mkdir -p docs/notebooks/figures
    @cp -p notebooks/figures/*.svg docs/notebooks/figures/
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

# Download the pinned docs fonts into build/docs-fonts/ (stamp-cached).
[private]
[group('docs')]
[script('uv', 'run', '--script')]
fetch-docs-fonts:
    import hashlib, io, shutil, urllib.request, zipfile
    from pathlib import Path

    OUT = Path("build/docs-fonts")
    STAMP = OUT / "FONTS"
    FILES = {
        "FiraMath-Regular.otf": (
            "https://github.com/firamath/firamath/releases/download/v0.3.4/FiraMath-Regular.otf",
            "2028cbd3dd4d8c0cf1608520eb4759956a83a67931d7b6d8e7c313520186e35b",
        ),
        "Outfit-Regular.ttf": (
            "https://github.com/Outfitio/Outfit-Fonts/raw/902773808eb372f70fb34e8946dd1ffe604efc79/fonts/ttf/Outfit-Regular.ttf",
            "3b64ac4f6ab6a8eebddd4b0bc03c811c43602e11e176382ab0ee6be615ab861b",
        ),
        "Outfit-Bold.ttf": (
            "https://github.com/Outfitio/Outfit-Fonts/raw/902773808eb372f70fb34e8946dd1ffe604efc79/fonts/ttf/Outfit-Bold.ttf",
            "f620b69582e06d7e1b3bbde74ed8c5876eadabb038390780db2a3414a1490197",
        ),
        "Outfit-SemiBold.ttf": (
            "https://github.com/Outfitio/Outfit-Fonts/raw/902773808eb372f70fb34e8946dd1ffe604efc79/fonts/ttf/Outfit-SemiBold.ttf",
            "bf2e1d2a6ec2a67952e8b36edd2b2bb9f340c0cdd10b0ad5145b4dbbc1339608",
        ),
        "Code New Roman-Regular.otf": (
            "https://github.com/chrissimpkins/codeface/raw/40538ae5ffee793c98355b2c03c30130aed51932/fonts/code-new-roman/Code%20New%20Roman-Regular.otf",
            "dfa19137778f3af70380f68452dc1c2ab364597cddda80a10819e46e35c1b984",
        ),
        "Code New Roman-Bold.otf": (
            "https://github.com/chrissimpkins/codeface/raw/40538ae5ffee793c98355b2c03c30130aed51932/fonts/code-new-roman/Code%20New%20Roman-Bold.otf",
            "42d7dd94691a24bc3aece8489c24c8050250b320a6ec1d250115dafbfa0314f3",
        ),
        "Code New Roman-Italic.otf": (
            "https://github.com/chrissimpkins/codeface/raw/40538ae5ffee793c98355b2c03c30130aed51932/fonts/code-new-roman/Code%20New%20Roman-Italic.otf",
            "3d2f971aa1829cca3df46cdf9f4385c57afba9e507b1821ba64d73dc1cc7ec08",
        ),
    }
    INTER_ZIP = (
        "https://github.com/rsms/inter/releases/download/v4.1/Inter-4.1.zip",
        "9883fdd4a49d4fb66bd8177ba6625ef9a64aa45899767dde3d36aa425756b11e",
    )
    INTER_FILES = {
        "InterVariable.ttf": "4989b125924991b90d05b2d16e0e388c48f7d5bb8b30539bbf9c755278d0ccaf",
        "InterVariable-Italic.ttf": "d6f1f6a172d9e588438db9f986fd5cfad7b30f644374080a8a9d4d91e344586f",
    }
    VERSION = hashlib.sha256(repr((FILES, INTER_ZIP, INTER_FILES)).encode()).hexdigest()

    if STAMP.exists() and STAMP.read_text().strip() == VERSION:
        print(f"docs fonts already cached in {OUT}")
    else:
        print(f"downloading docs fonts into {OUT}")
        if OUT.exists():
            shutil.rmtree(OUT)
        OUT.mkdir(parents=True)

        def verify(content, expected_hash, source):
            actual_hash = hashlib.sha256(content).hexdigest()
            if actual_hash != expected_hash:
                raise RuntimeError(f"SHA-256 mismatch for {source}: {actual_hash}")
            return content

        def fetch(url, expected_hash):
            req = urllib.request.Request(url, headers={"User-Agent": "quantum-rf-pdk-docs"})
            with urllib.request.urlopen(req, timeout=600) as r:
                content = r.read()
            return verify(content, expected_hash, url)

        for name, (url, expected_hash) in FILES.items():
            (OUT / name).write_bytes(fetch(url, expected_hash))
        with zipfile.ZipFile(io.BytesIO(fetch(*INTER_ZIP))) as z:
            for name, expected_hash in INTER_FILES.items():
                (OUT / name).write_bytes(
                    verify(z.read(name), expected_hash, f"{INTER_ZIP[0]}!{name}")
                )
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
