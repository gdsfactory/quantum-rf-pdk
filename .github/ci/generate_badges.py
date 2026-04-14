"""Generate SVG metric badges for the repository.

Reads coverage.xml (pytest-cov) and the qpdk PDK registry to produce
lightweight SVG badge files in the ``badges/`` directory.
"""

import importlib
import xml.etree.ElementTree as ET
from pathlib import Path

SVG_TEMPLATE = """\
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="20">
  <linearGradient id="b" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="a">
    <rect width="{width}" height="20" rx="3" fill="#fff"/>
  </clipPath>
  <g clip-path="url(#a)">
    <rect width="{label_w}" height="20" fill="#555"/>
    <rect x="{label_w}" width="{value_w}" height="20" fill="{color}"/>
    <rect width="{width}" height="20" fill="url(#b)"/>
  </g>
  <g fill="#fff" text-anchor="middle" \
font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <text x="{label_x}" y="15" fill="#010101" fill-opacity=".3">{label}</text>
    <text x="{label_x}" y="14">{label}</text>
    <text x="{value_x}" y="15" fill="#010101" fill-opacity=".3">{value}</text>
    <text x="{value_x}" y="14">{value}</text>
  </g>
</svg>"""

BADGES_DIR = Path("badges")


def _make_badge(label: str, value: str, color: str, filename: str) -> None:
    """Write a single SVG badge to *badges/{filename}*."""
    label_w = len(label) * 6.5 + 12
    value_w = len(value) * 6.5 + 12
    width = label_w + value_w
    svg = SVG_TEMPLATE.format(
        width=int(width),
        label_w=int(label_w),
        value_w=int(value_w),
        label_x=int(label_w / 2),
        value_x=int(label_w + value_w / 2),
        color=color,
        label=label,
        value=value,
    )
    BADGES_DIR.mkdir(parents=True, exist_ok=True)
    (BADGES_DIR / filename).write_text(svg)
    print(f"  {filename}: {label} = {value}")


def _color_for_pct(pct: float, *, good: float = 80, warn: float = 60) -> str:
    """Return a hex colour string (green/yellow/red) for a percentage value."""
    if pct >= good:
        return "#4c1"
    return "#dfb317" if pct >= warn else "#e05d44"


def generate_coverage_badge() -> None:
    """Generate a test-coverage badge from *coverage.xml*."""
    try:
        tree = ET.parse("coverage.xml")
        rate = float(tree.getroot().attrib.get("line-rate", 0)) * 100
    except Exception:
        rate = 0
    _make_badge("coverage", f"{rate:.0f}%", _color_for_pct(rate), "coverage.svg")


def generate_model_coverage_badge() -> None:
    """Generate a model-coverage badge from the PDK cell/model registries."""
    try:
        mod = importlib.import_module("qpdk")
        pdk = mod.PDK
        pdk.activate()
        cells = pdk.cells
        models = getattr(pdk, "models", {}) or {}
        total = len(cells)
        with_model = len(set(cells) & set(models))
        pct = (with_model / total * 100) if total else 0
    except Exception as exc:
        print(f"  model coverage error: {exc}")
        pct = 0
    _make_badge(
        "models",
        f"{pct:.0f}%",
        _color_for_pct(pct, good=80, warn=40),
        "model_coverage.svg",
    )


if __name__ == "__main__":
    generate_coverage_badge()
    generate_model_coverage_badge()
