# Building PDF Documentation

This document describes how to build the PDF version of the documentation.

## Prerequisites

The following LaTeX packages must be installed on your system:

- `texlive-xetex` - XeTeX engine for PDF generation
- `texlive-fonts-extra` - Additional fonts including bbm.sty
- `texlive-fonts-recommended` - Recommended fonts
- `texlive-latex-extra` - Extra LaTeX packages
- `latexmk` - Build automation tool for LaTeX
- `xindy` - Index generation tool
- `imagemagick` - Image conversion (optional, for SVG badge conversion)

### Ubuntu/Debian Installation

```bash
sudo apt-get update
sudo apt-get install -y texlive-xetex texlive-fonts-recommended \
    texlive-fonts-extra texlive-latex-extra latexmk xindy imagemagick
```

### macOS Installation

```bash
brew install --cask mactex
brew install imagemagick
```

### Other Systems

Install a full TeXLive distribution from [tug.org/texlive](https://tug.org/texlive/).

## Building

Once prerequisites are installed:

```bash
make docs-pdf
```

The generated PDF will be located at `docs/_build/latex/qpdk.pdf`.

## Troubleshooting

### Missing xindy module

If you see errors about missing xindy modules, ensure the `XINDYOPTS` environment variable is set correctly in the Makefile (this should already be configured).

### Font warnings

Some Unicode characters (icons/glyphs) may not be available in the default fonts. These warnings are cosmetic and don't affect the PDF content.

### Build warnings

The build may produce warnings about:
- Overfull/underfull boxes (LaTeX typesetting warnings)
- Missing characters in fonts (icon glyphs)
- Duplicate citations (bibliography references)

These warnings don't prevent PDF generation. The build will succeed if the PDF file is created.
