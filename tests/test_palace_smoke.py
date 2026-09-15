"""Smoke test for the Palace driven-resonator notebook pipeline.

Mirrors notebooks/src/palace_driven_resonator.py at minimal settings: coarse
mesh, a single drive frequency, no field saves, and the prebuilt Palace CPU
runtime from the palace-toolkit package. Skips unless gsim is available and the
host is Linux x86_64, so the regular test suite is unaffected.
"""

import subprocess

import gdsfactory as gf
import klayout.db as kdb
import numpy as np
import pytest

from qpdk import cells
from qpdk.cells.airbridge import cpw_with_airbridges
from qpdk.tech import LAYER, route_bundle_sbend_cpw

pytest.importorskip("gsim")

SUBSTRATE_THICKNESS = 500
VACUUM_THICKNESS = 500
DRIVE_FREQ = 7.78e9
CPW_LAYERS = {"SUBSTRATE": (1, 0), "SUPERCONDUCTOR": (2, 0), "VACUUM": (3, 0)}


@pytest.fixture
def resonator_compact():
    """Coupled resonator with two CPW feeds routed close to the coupling section."""

    @gf.cell
    def _resonator_compact(coupling_gap: float = 20.0) -> gf.Component:
        c = gf.Component()
        res = c << cells.resonator_coupled(
            coupling_straight_length=300, coupling_gap=coupling_gap
        )
        res.movex(-res.size_info.width / 4)
        left = c << cells.straight()
        right = c << cells.straight()
        w = res.size_info.width + 100
        left.move((-w, 0))
        right.move((w, 0))
        route_bundle_sbend_cpw(
            c,
            [left["o2"], right["o1"]],
            [res["coupling_o1"], res["coupling_o2"]],
            cross_section=cpw_with_airbridges(
                airbridge_spacing=250.0, airbridge_padding=20.0
            ),
        )
        c.kdb_cell.shapes(LAYER.SIM_AREA).insert(c.bbox().enlarged(0, 100))
        c.add_port(name="o1", port=left["o1"])
        c.add_port(name="o2", port=right["o2"])
        return c

    return _resonator_compact


def etched_component(component: gf.Component) -> gf.Component:
    """Copy the simulation-area region minus the metal etch onto gsim layers."""
    sim_area_layer = (LAYER.SIM_AREA[0], LAYER.SIM_AREA[1])
    etch_layer = (LAYER.M1_ETCH[0], LAYER.M1_ETCH[1])

    layout = component.kdb_cell.layout()
    sim_region = kdb.Region(
        component.kdb_cell.begin_shapes_rec(layout.layer(*sim_area_layer))
    )
    etch_region = kdb.Region(
        component.kdb_cell.begin_shapes_rec(layout.layer(*etch_layer))
    )
    conductor_region = sim_region - etch_region

    etched = gf.Component("etched_component")
    el = etched.kdb_cell.layout()
    for name, region in [
        ("SUPERCONDUCTOR", conductor_region),
        ("SUBSTRATE", sim_region),
        ("VACUUM", sim_region),
    ]:
        idx = el.layer(*CPW_LAYERS[name])
        etched.kdb_cell.shapes(idx).insert(region)
    for port in component.ports:
        etched.add_port(name=port.name, port=port)
    return etched


def test_palace_driven_resonator_smoke(resonator_compact, tmp_path):
    """Run the notebook pipeline end to end and check the port S-parameters."""
    from gsim.common.stack import (  # ruff: ignore[import-outside-top-level]
        Layer,
        LayerStack,
    )
    from gsim.common.stack.materials import (  # ruff: ignore[import-outside-top-level]
        MATERIALS_DB,
    )
    from gsim.palace import DrivenSim  # ruff: ignore[import-outside-top-level]
    from palacetoolkit.palace_runtime import (  # ruff: ignore[import-outside-top-level]
        install_palace_runtime,
    )
    from palacetoolkit.simulation import (  # ruff: ignore[import-outside-top-level]
        get_palace_runtime_env,
    )

    # install_palace_runtime rather than resolve_palace_binary: the resolver
    # validates with "palace --version", which the launcher wrapper treats as a
    # config path and rejects because no MPI launcher is bundled.
    try:
        palace = install_palace_runtime()
    except RuntimeError:  # non-Linux x86_64 host
        pytest.skip("prebuilt Palace runtime is Linux x86_64 only")
        # Unreachable; satisfies the type checker since ``pytest.skip`` only raises
        raise AssertionError from None

    stack = LayerStack(pdk_name="qpdk")
    stack.layers["SUBSTRATE"] = Layer(
        name="SUBSTRATE",
        gds_layer=CPW_LAYERS["SUBSTRATE"],
        zmin=0.0,
        zmax=SUBSTRATE_THICKNESS,
        thickness=SUBSTRATE_THICKNESS,
        material="sapphire",
        layer_type="dielectric",
    )
    stack.layers["SUPERCONDUCTOR"] = Layer(
        name="SUPERCONDUCTOR",
        gds_layer=CPW_LAYERS["SUPERCONDUCTOR"],
        zmin=SUBSTRATE_THICKNESS,
        zmax=SUBSTRATE_THICKNESS,
        thickness=0,
        material="aluminum",
        layer_type="conductor",
    )
    stack.layers["VACUUM"] = Layer(
        name="VACUUM",
        gds_layer=CPW_LAYERS["VACUUM"],
        zmin=SUBSTRATE_THICKNESS,
        zmax=SUBSTRATE_THICKNESS + VACUUM_THICKNESS,
        thickness=VACUUM_THICKNESS,
        material="vacuum",
        layer_type="dielectric",
    )
    stack.dielectrics = [
        {
            "name": "substrate",
            "zmin": 0.0,
            "zmax": SUBSTRATE_THICKNESS,
            "material": "sapphire",
        },
        {
            "name": "vacuum",
            "zmin": SUBSTRATE_THICKNESS,
            "zmax": SUBSTRATE_THICKNESS + VACUUM_THICKNESS,
            "material": "vacuum",
        },
    ]
    stack.materials = {
        "sapphire": MATERIALS_DB["sapphire"].to_dict(),
        "aluminum": MATERIALS_DB["aluminum"].to_dict(),
        "vacuum": MATERIALS_DB["vacuum"].to_dict(),
    }

    etched = etched_component(resonator_compact(coupling_gap=15.0))

    sim = DrivenSim()
    sim.set_geometry(etched)
    sim.set_stack(stack)
    sim.add_cpw_port(
        "o1", layer="SUPERCONDUCTOR", s_width=10.0, gap_width=6.0, offset=2.5
    )
    sim.add_cpw_port(
        "o2", layer="SUPERCONDUCTOR", s_width=10.0, gap_width=6.0, offset=2.5
    )
    sim.set_driven(f=DRIVE_FREQ)

    out_dir = tmp_path / "sim"
    out_dir.mkdir()
    sim.set_output_dir(str(out_dir))
    sim.mesh(preset="coarse")
    sim.write_config()
    assert (out_dir / "config.json").exists()

    # The wrapper script expands the config path unquoted, so run it from the
    # output directory with a relative path.
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [str(palace), "--serial", "config.json"],
        cwd=out_dir,
        env=get_palace_runtime_env(palace),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode == 0, (
        f"Palace exited with code {result.returncode}\n"
        f"stdout tail:\n{result.stdout[-4000:]}\n"
        f"stderr tail:\n{result.stderr[-4000:]}"
    )

    port_csv = out_dir / "output" / "palace" / "port-S.csv"
    assert port_csv.exists(), (
        f"Palace wrote no S-parameters to {port_csv}\n"
        f"stdout tail:\n{result.stdout[-4000:]}\n"
        f"stderr tail:\n{result.stderr[-4000:]}\n"
        f"output tree:\n"
        + "\n".join(str(p.relative_to(out_dir)) for p in out_dir.rglob("*"))
    )
    s_params = np.loadtxt(port_csv, delimiter=",", skiprows=1, ndmin=2)
    assert s_params.shape[0] >= 1
    assert np.all(np.isfinite(s_params))
    assert np.isclose(s_params[0, 0], DRIVE_FREQ / 1e9)
    # Passive device: |S11| <= 1, so the dB value cannot be positive
    assert s_params[0, 1] <= 0.0
