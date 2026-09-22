# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "gmsh",
#   "gsim>=0.5",
#   "qpdk[models]",
# ]
#
# The dependency block makes the script runnable standalone with
# `uv run .github/palace_smoke.py ...`; the CI workflow instead runs it
# through the repository environment (`uv run python .github/palace_smoke.py`)
# so that the local checkout of qpdk is what gets tested.
# ///
"""Palace CI smoke test: generate a tiny eigenmode simulation and check its results.

The `palace.yml` workflow uses this to keep the qpdk-to-Palace pipeline (layout,
etch-to-conductor conversion, gsim stack and ports, meshing) exercised in CI.
The settings are deliberately small: a shortened resonator, a thin vacuum and
substrate, a coarse mesh, first-order elements and two modes, so a runner
finishes in minutes.

Usage:
    uv run python .github/palace_smoke.py generate [--out DIR]
    uv run python .github/palace_smoke.py check [--sim DIR]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import gdsfactory as gf
import klayout.db as kdb

from qpdk import PDK
from qpdk.cells import straight, transmon_with_resonator_and_probeline
from qpdk.models.resonator import resonator_frequency
from qpdk.simulation import to_fem_regions
from qpdk.tech import LAYER

MEANDER_LENGTH = 2000.0  # µm, shortened from the notebook's 5000
RESONATOR_MEANDERS = 3
SUBSTRATE_THICKNESS = 200.0  # µm
VACUUM_THICKNESS = 200.0  # µm
NUM_MODES = 2


def _drawn_meander_length(resonator_length: float, meanders: int) -> float:
    """Return the meander length the cell draws for a given parameter."""
    probe = transmon_with_resonator_and_probeline(
        qubit="double_pad_transmon_with_bbox",
        resonator_length=resonator_length,
        resonator_meanders=meanders,
        qubit_rotation=90,
    )
    return next(
        inst.cell for inst in probe.insts if inst.cell.name.startswith("resonator_")
    ).info["length"]


def build_sim():
    """Return (configured eigenmode simulation, analytical estimate in Hz)."""
    from gsim.palace import EigenmodeSim

    from qpdk.simulation import single_chip_stack

    PDK.activate()

    @gf.cell
    def sim_component() -> gf.Component:
        """Small transmon-resonator layout with the notebook's geometry patches."""
        c = gf.Component()
        probe_length = 5000.0
        overhead = probe_length - _drawn_meander_length(
            probe_length, RESONATOR_MEANDERS
        )
        ref = c << transmon_with_resonator_and_probeline(
            qubit="double_pad_transmon_with_bbox",
            resonator_length=MEANDER_LENGTH + overhead,
            resonator_meanders=RESONATOR_MEANDERS,
            qubit_rotation=90,
        )
        c.add_ports(ref.ports)
        # Geometry patches identical to the notebook: the quarter-wave short
        # bridge at the resonator start and open probeline feed ends.
        c.kdb_cell.shapes(LAYER.M1_DRAW).insert(
            kdb.DBox(-975.0, -1201.0, -920.0, -1197.0)
        )
        for name in ("coupling_o1", "coupling_o2"):
            ext = c << straight(length=10.0, cross_section="etch")
            ext.connect("o1", ref.ports[name], allow_layer_mismatch=True)
        c.kdb_cell.shapes(LAYER.SIM_AREA).insert(c.bbox().enlarged(100, 100))
        return c

    etched = to_fem_regions(sim_component())
    etched.ports["junction"].orientation = 270.0

    analytical_freq = resonator_frequency(
        length=MEANDER_LENGTH, cross_section="cpw", is_quarter_wave=True
    )

    sim = EigenmodeSim()
    sim.set_geometry(etched)
    sim.set_stack(
        single_chip_stack(
            substrate_thickness=SUBSTRATE_THICKNESS, vacuum_thickness=VACUUM_THICKNESS
        )
    )
    sim.set_numerical(order=1, solver_type="MUMPS")
    sim.add_port(
        "junction",
        layer="SUPERCONDUCTOR",
        length=25.0,
        inductance=10e-9,
        resistance=0.0,
    )
    sim.add_cpw_port(
        "coupling_o1", layer="SUPERCONDUCTOR", s_width=10.0, gap_width=6.0, length=5.0
    )
    sim.add_cpw_port(
        "coupling_o2", layer="SUPERCONDUCTOR", s_width=10.0, gap_width=6.0, length=5.0
    )
    sim.set_eigenmode(target=2e9, num_modes=NUM_MODES)
    return sim, analytical_freq


def cmd_generate(out_dir: Path) -> None:
    """Mesh and write the Palace configuration for the small model."""
    sim, analytical_freq = build_sim()
    sim.set_output_dir(out_dir)
    sim.mesh(preset="coarse")
    sim.write_config()
    print(f"analytical estimate: {analytical_freq / 1e9:.4f} GHz")
    print(f"simulation written to {out_dir}")


def cmd_check(sim_dir: Path) -> None:
    """Sanity-check the eigenmode CSV of a finished Palace run."""
    eig_csv = sim_dir / "output" / "palace" / "eig.csv"
    with eig_csv.open() as f:
        rows = [
            {k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(f)
        ]
    assert len(rows) == NUM_MODES, f"expected {NUM_MODES} modes, got {len(rows)}"
    freqs = [float(row["Re{f} (GHz)"]) for row in rows]
    qs = [float(row["Q"]) for row in rows]
    assert all(0.1 < f < 100.0 for f in freqs), f"unphysical frequencies: {freqs}"
    assert all(q > 1.0 for q in qs), f"unphysical quality factors: {qs}"
    # The readout mode must sit near the analytical quarter-wave estimate for
    # the drawn meander: this catches a solver or layout that is off by an
    # order of magnitude, which the per-mode physicality checks above would
    # happily pass.
    analytical_ghz = (
        resonator_frequency(
            length=MEANDER_LENGTH, cross_section="cpw", is_quarter_wave=True
        )
        / 1e9
    )
    best = min(freqs, key=lambda f: abs(f - analytical_ghz))
    assert abs(best - analytical_ghz) / analytical_ghz < 0.25, (
        f"no mode within 25% of the analytical estimate {analytical_ghz:.4f} GHz,"
        f" closest is {best:.4f} GHz in {freqs}"
    )
    print(f"eigenmodes (GHz): {freqs}, Q: {qs}")
    print(f"mode closest to the analytical {analytical_ghz:.4f} GHz: {best:.4f} GHz")


def main() -> None:
    """Parse the command line and dispatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--out", type=Path, default=Path("build/palace_ci_sim"))
    chk = sub.add_parser("check")
    chk.add_argument("--sim", type=Path, default=Path("build/palace_ci_sim"))
    args = parser.parse_args()
    if args.command == "generate":
        cmd_generate(args.out)
    else:
        cmd_check(args.sim)


if __name__ == "__main__":
    main()
