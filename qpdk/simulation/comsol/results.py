"""Read COMSOL result files and judge the curves they hold.

Standalone, license-free helpers shared by the COMSOL notebooks: locating an
exported file, parsing the frequency COMSOL annotates an export header with,
rebuilding a requested sweep grid, and the physical checks a driven curve is
read against. Nothing here imports MPh or touches a model, so the result cells
still draw saved exports where no solver is installed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

# Frequency units COMSOL may annotate an export header with, to GHz.
FREQUENCY_UNITS_GHZ = {"GHz": 1.0, "MHz": 1.0e-3, "kHz": 1.0e-6, "Hz": 1.0e-9}
# ``@ freq=7.5`` as written by some exports, or a bare ``@ 7.2921 GHz``, or the
# complex ``@ 7.3266+5.3458E-4i GHz`` a ported eigenfield export carries.
_FREQUENCY_ANNOTATION = re.compile(
    r"@\s*freq\s*=\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
)
_FREQUENCY_HEADER = re.compile(
    r"@\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\s*(GHz|MHz|kHz|Hz)\b"
)
_COMPLEX_FREQUENCY_HEADER = re.compile(
    r"@\s*([-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)"
    r"\s*([-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)\s*i"
    r"\s*(GHz|MHz|kHz|Hz)\b"
)


def result_file(results_dir: Path | None, name: str) -> Path | None:
    """Return the path of an exported solver result, if one is available.

    Args:
        results_dir: Directory holding exported results, or ``None`` when the
            notebook has no results directory configured.
        name: File name to look for inside ``results_dir``.

    Returns:
        The path, or ``None`` when ``results_dir`` is unset or holds no such
        file.
    """
    if results_dir is None:
        return None
    path = results_dir / name
    return path if path.exists() else None


def resolve_record_path(results_dir: Path | None, base: Path, name: str) -> Path:
    """Resolve a file name stored in a result record.

    Records store a curve by bare file name so a saved notebook never embeds an
    absolute path from the machine that solved. A bare name resolves under
    ``results_dir`` when that directory holds the file, otherwise next to the
    record itself. An absolute path in an older record is still honoured.

    Args:
        results_dir: Directory holding exported results, or ``None``.
        base: Directory the record itself lives in.
        name: The stored path or file name.

    Returns:
        The path to read, which may not exist yet.
    """
    stored = Path(name)
    if stored.is_absolute():
        return stored
    under_results = results_dir / stored if results_dir is not None else None
    if under_results is not None and under_results.exists():
        return under_results
    return base / stored


def explain_missing_results(results_dir: Path | None, name: str) -> str:
    """Explain how to supply a result file that is not on disk.

    Args:
        results_dir: Directory the file was looked for in.
        name: File name that was looked for inside ``results_dir``.

    Returns:
        A one-paragraph message naming the missing file and the two ways to
        supply it.
    """
    return (
        f"No {name} in RESULTS_DIR ({results_dir}). The figures on the "
        "documentation page are saved outputs of a licensed solve, and the cells "
        "here replot only from files on disk. To supply them, run with "
        "RUN_COMSOL = True on a licensed machine, which exports into MODEL_DIR, "
        "or set RESULTS_DIR to a directory that already holds an exported "
        f"{name}."
    )


def write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON through a sibling temporary file, then replace in place.

    A series writes on every row, so an interrupt partway through still leaves
    the rows already solved on disk. The temporary file is a sibling so the
    replace stays a same-filesystem rename, which is what makes it atomic.

    Args:
        path: The JSON file to write.
        payload: The object to serialise.
    """
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def exported_frequency_ghz(file: Path) -> float | None:
    """Read the frequency annotation COMSOL writes into a data export header.

    COMSOL writes either ``@ freq=<value>`` or a bare ``@ <value> <unit>`` line,
    and the live field exports use the second form, so both are parsed and the
    unit is applied explicitly rather than assumed to be GHz.

    Args:
        file: Exported text file to scan.

    Returns:
        The annotated frequency in GHz, or ``None`` when the export carries no
        frequency.
    """
    with file.open(encoding="utf-8") as handle:
        for line in handle:
            if (match := _FREQUENCY_ANNOTATION.search(line)) is not None:
                return float(match.group(1))
            if (match := _FREQUENCY_HEADER.search(line)) is not None:
                return float(match.group(1)) * FREQUENCY_UNITS_GHZ[match.group(2)]
    return None


def complex_frequency_ghz(file: Path) -> tuple[float, float] | None:
    """Read a complex frequency annotation from a COMSOL export header.

    A ported eigenfield export writes its frequency with an imaginary part,
    e.g. ``@ 7.3266+5.3458E-4i GHz``, which :func:`exported_frequency_ghz`
    does not parse.

    Args:
        file: Exported text file to scan.

    Returns:
        ``(real, imag)`` in GHz, or ``None`` when the export carries no complex
        frequency annotation.
    """
    with file.open(encoding="utf-8") as handle:
        for line in handle:
            if (match := _COMPLEX_FREQUENCY_HEADER.search(line)) is not None:
                scale = FREQUENCY_UNITS_GHZ[match.group(3)]
                return float(match.group(1)) * scale, float(match.group(2)) * scale
    return None


def requested_frequency_grid(
    low_ghz: float, high_ghz: float, points: int
) -> tuple[str, np.ndarray]:
    """Return the COMSOL ``range`` for a requested grid, and that exact grid.

    The start and step are formatted once and the grid is rebuilt from those
    formatted tokens, so the grid checked here is the grid COMSOL will build.
    Formatting all three tokens independently is what goes wrong: a rounded step
    can make ``start + (points - 1) * step`` exceed a separately rounded stop by a
    few millihertz, and an inclusive range then returns only ``points - 1`` rows.
    The stop here is the intended last frequency plus half a step, so rounding
    cannot push the last point past it, while the point one step further is still
    beyond it, so the range returns exactly ``points`` frequencies.

    Args:
        low_ghz: Low end of the physical window, in GHz.
        high_ghz: High end of the physical window, in GHz.
        points: Number of grid points.

    Returns:
        The ``range`` expression, and the ``points``-long grid it should return.
    """
    step_ghz = (high_ghz - low_ghz) / (points - 1)
    start_token = f"{low_ghz:.12g}"
    step_token = f"{step_ghz:.12g}"
    start_ghz = float(start_token)
    step_rounded_ghz = float(step_token)
    grid_ghz = start_ghz + step_rounded_ghz * np.arange(points)
    stop_ghz = grid_ghz[-1] + 0.5 * step_rounded_ghz
    expression = f"range({start_token}[GHz],{step_token}[GHz],{stop_ghz:.12g}[GHz])"
    return expression, grid_ghz


def curve_value_db(
    frequencies_ghz: np.ndarray,
    s21_db: np.ndarray,
    at_ghz: float,
    *,
    endpoint_tolerance_ghz: float,
) -> float | None:
    """Interpolate a curve at one directly solved frequency.

    A direct point is asked for at the same finite precision as the grid tokens,
    but the two still differ by a fraction of a hertz from formatting, so a
    frequency a hair outside the curve is clamped to the nearest endpoint.
    Materially outside the window it is ``None``, so an out-of-window direct
    point is never accepted silently.

    Args:
        frequencies_ghz: The curve's frequency column, increasing.
        s21_db: The curve's ``S21`` levels in dB.
        at_ghz: The frequency to read, in GHz.
        endpoint_tolerance_ghz: How far outside the curve a frequency may sit and
            still be clamped to its endpoint.

    Returns:
        The interpolated ``S21`` level in dB, or ``None`` when the frequency is
        outside the curve by more than the endpoint tolerance.
    """
    if frequencies_ghz.size == 0:
        return None
    if (
        at_ghz < frequencies_ghz[0] - endpoint_tolerance_ghz
        or at_ghz > frequencies_ghz[-1] + endpoint_tolerance_ghz
    ):
        return None
    clamped = float(np.clip(at_ghz, frequencies_ghz[0], frequencies_ghz[-1]))
    return float(np.interp(clamped, frequencies_ghz, s21_db))


def power_balance(power_sum: float, *, low: float, high: float) -> dict[str, Any]:
    """Judge one frequency's two-port power balance.

    Args:
        power_sum: ``|S11|^2 + |S21|^2`` at that frequency.
        low: Lower edge of the accepted band.
        high: Upper edge of the accepted band.

    Returns:
        The sum, its deficit from unity, and whether it sits inside the band.
    """
    deficit = 1.0 - power_sum
    return {
        "power_sum": float(power_sum),
        "power_deficit": float(deficit),
        "power_within_band": bool(low <= power_sum <= high),
    }


def notch_verdict(
    centre_db: float, flank_levels_db: list[float], *, min_depth_db: float
) -> dict[str, Any]:
    """Decide from directly solved points whether the centre is a notch.

    Only direct solves count: the minimum of an adaptive curve says where to
    look, not that a notch is there.

    Args:
        centre_db: Directly solved ``S21`` at the centre, in dB.
        flank_levels_db: Directly solved ``S21`` at the two flanks, in dB.
        min_depth_db: Depth below the lower flank the centre must clear.

    Returns:
        The depth below the lower flank and whether that clears the threshold.
    """
    lower = min(flank_levels_db)
    depth = lower - centre_db
    return {
        "depth_below_lower_flank_db": depth,
        "minimum_depth_db": min_depth_db,
        "is_a_notch": bool(depth >= min_depth_db),
    }
