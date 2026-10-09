"""Sequential colormaps ``qpdk`` and ``qpdk-dark`` for heatmaps in the docs.

Both are straight lines in OKLab from the page colour to a QPDK blue, so lightness
changes linearly (perceptually uniform, readable under colour-vision deficiency)
and hue stays fixed.  The two maps share that hue and chroma path, so a figure reads
as the same series in the light and the dark docs theme.

Importing this module registers both (and their ``_r`` reversals) with matplotlib.
Import it before the first colormapped artist is drawn, e.g. before
``plt.style.use("qpdk")``, whose ``image.cmap`` names ``qpdk``.
"""

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb

#: Page colours of ``docs/qpdk.mplstyle`` and ``docs/qpdk-dark.mplstyle``.
LIGHT_PAPER = "#ffffff"
DARK_PAPER = "#14181e"
#: ``--qpdk-accent-hover`` (``#1f5994``) and ``--qpdk-accent-bright`` (``#5e9be0``)
#: from ``docs/_static/css/custom.css``, with OKLab chroma raised by 20% at the same
#: lightness and hue (about 252°).
LIGHT_ANCHOR = "#0358a0"
DARK_ANCHOR = "#4f9bee"

_N = 256
# OKLab, https://bottosson.github.io/posts/oklab/
_LMS_FROM_LINEAR_SRGB = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
])
_OKLAB_FROM_LMS = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
])


def _srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return np.cbrt(linear @ _LMS_FROM_LINEAR_SRGB.T) @ _OKLAB_FROM_LMS.T


def _oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    lms = (lab @ np.linalg.inv(_OKLAB_FROM_LMS).T) ** 3
    linear = np.clip(lms @ np.linalg.inv(_LMS_FROM_LINEAR_SRGB).T, 0, 1)
    return np.where(
        linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055
    )


def _ramp(name: str, paper: str, anchor: str) -> LinearSegmentedColormap:
    start, end = (_srgb_to_oklab(np.array(to_rgb(c))) for c in (paper, anchor))
    t = np.linspace(0, 1, _N)[:, None]
    return LinearSegmentedColormap.from_list(
        name, _oklab_to_srgb(start + t * (end - start)), N=_N
    )


QPDK = _ramp("qpdk", LIGHT_PAPER, LIGHT_ANCHOR)
QPDK_DARK = _ramp("qpdk-dark", DARK_PAPER, DARK_ANCHOR)

for _cmap in (QPDK, QPDK_DARK):
    for _c in (_cmap, _cmap.reversed()):
        if _c.name not in mpl.colormaps:
            mpl.colormaps.register(_c)
