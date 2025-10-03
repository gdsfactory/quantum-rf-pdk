# IPython Configuration for Documentation

This directory contains configuration files for customizing the appearance and output of Jupyter notebooks in the
documentation.

## Files

### `ipython_config.py`

IPython configuration file that sets up high-quality figure output for documentation generation.

**Features:**

- Exports figures in multiple formats (PDF, SVG, PNG) for flexibility
- Sets PNG resolution to 300 DPI for publication-quality output
- Configures proper font embedding for PDF/SVG (Type 42 TrueType fonts)
- Uses tight bounding boxes to minimize whitespace
- Suppresses fontTools warnings in notebook output
- Automatically loads the custom matplotlib style

**Settings:**

```python
c.InlineBackend.figure_formats = ["pdf", "svg", "png"]
c.InlineBackend.print_figure_kwargs = {"bbox_inches": "tight"}
c.InlineBackend.rc = {"figure.dpi": 300}
```

**Font Configuration:**

- SVG: Converts text to paths for maximum compatibility
- PDF: TrueType (Type 42) font embedding with maximum compression
- PS: TrueType (Type 42) font embedding

### `qpdk.mplstyle`

Custom matplotlib style for consistent, professional-looking plots in the documentation.

**Features:**

- Uses DejaVu Sans font family (widely available, modern sans-serif)
- Clean plot appearance with grid lines
- Removes top and right spines for a cleaner look
- High-resolution output (300 DPI for saved figures)
- Consistent sizing and spacing

**Font Options:** You can customize the font by editing the `font.sans-serif` line. Available options include:

- DejaVu Sans (default)
- Helvetica
- Arial
- Liberation Sans

## Usage

### During Documentation Build

The configuration is automatically applied when building documentation:

```bash
make docs        # Builds HTML documentation
make docs-latex  # Builds LaTeX/PDF documentation
```

The `setup-ipython-config` target (run as part of `make docs`) copies these files to `~/.ipython/profile_default/`.

### Manual Setup

To use the configuration for local development:

```bash
make setup-ipython-config
```

Then start Jupyter Lab or Jupyter Notebook, and the configuration will be applied automatically.

### In Notebooks

The configuration is applied automatically when IPython starts. You don't need to do anything special in your notebooks.

To verify the configuration is loaded:

```python
import matplotlib.pyplot as plt

# Check font settings
print(plt.rcParams['font.sans-serif'])  # Should show ['DejaVu Sans', 'Helvetica', ...]
print(plt.rcParams['svg.fonttype'])     # Should be 'path'
print(plt.rcParams['pdf.fonttype'])     # Should be 42
```

## Customization

### Changing the Font

Edit `qpdk.mplstyle` and modify:

```text
font.sans-serif: YourFont, DejaVu Sans, Helvetica, Arial, sans-serif
```

### Adjusting Figure Quality

Edit `ipython_config.py` and modify:

```python
c.InlineBackend.rc = {"figure.dpi": 300}  # Change 300 to your desired DPI
```

### Adding More Formats

Edit `ipython_config.py` and modify:

```python
c.InlineBackend.figure_formats = ["pdf", "svg", "png", "retina"]  # Add more formats
```

## GitHub Actions

The GitHub Actions workflow automatically sets up the IPython configuration when building documentation. The `pages.yml`
workflow runs `make docs` which includes the `setup-ipython-config` target.

## Troubleshooting

### Configuration Not Loading

1. Verify the config was copied:

   ```bash
   ls -la ~/.ipython/profile_default/ipython_config.py
   ```

1. Check for errors in the config file:

   ```bash
   python -c "exec(open('~/.ipython/profile_default/ipython_config.py').read())"
   ```

### Style Not Loading

1. Verify the style file exists:

   ```bash
   ls -la ~/.ipython/profile_default/qpdk.mplstyle
   ```

1. Manually load the style:

   ```python
   import matplotlib.pyplot as plt
   plt.style.use('~/.ipython/profile_default/qpdk.mplstyle')
   ```

## References

- [IPython Configuration](https://ipython.readthedocs.io/en/stable/config/intro.html)
- [Matplotlib Configuration](https://matplotlib.org/stable/users/explain/customizing.html)
- [Jupyter Book Configuration](https://jupyterbook.org/en/stable/customize/config.html)
