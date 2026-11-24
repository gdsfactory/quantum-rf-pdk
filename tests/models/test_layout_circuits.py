"""Every cell with a SAX model builds a circuit from its own layout netlist."""

from __future__ import annotations

import gdsfactory as gf
import jax.numpy as jnp
import pytest
import sax
from jax.errors import JaxRuntimeError
from sax.netlists import remove_unused_instances

from qpdk import PDK
from qpdk.helper import get_circuit_netlist
from qpdk.models import models as qpdk_models

PDK.activate()

# Models that exist nowhere but are needed by cells that open issues track.
KNOWN_MISSING_MODELS = {
    "plate_capacitor_single": "#299",
    "transmon_with_resonator_and_probeline": "#299",
}
# Cells whose netlists have no circuit ports, tracked by an open issue.
KNOWN_PORTLESS = {"xmon_transmon": "#1173"}
# Cells whose circuit fails to build or evaluate although every model exists.
EVALUATION_FAILURES = {
    "hyperbolic_lattice": (
        JaxRuntimeError,
        "sax's default KLU backend fails with an n_nz mismatch",
    ),
    "single_josephson_junction_wire": (
        ValueError,
        (
            "the straight model needs an etch section, which the junction"
            " cross-section of its straight instances lacks"
        ),
    ),
}


def _wrapper(cell: gf.Component | gf.ComponentAllAngle) -> gf.Component:
    """Return a one-instance component exposing all ports of ``cell``."""
    c = gf.Component()
    ref = (
        c.add_ref_off_grid(cell)
        if isinstance(cell, gf.ComponentAllAngle)
        else c << cell
    )
    c.add_ports(ref.ports)
    return c


def _missing_models(netlist: dict) -> set[str]:
    """Return the models ``sax.circuit`` needs for ``netlist`` but PDK lacks."""
    instances = remove_unused_instances(netlist)["instances"]
    return {i["component"] for i in instances.values()} - PDK.models.keys()


def _mark_unsimulatable(
    name: str,
    cell: gf.Component | gf.ComponentAllAngle,
    netlist: dict,
    request: pytest.FixtureRequest,
) -> None:
    """Xfail or skip a cell that is not in ``PDK.models`` and cannot use ``netlist``.

    ``netlist`` is the netlist the test will simulate.
    """
    if not cell.ports:
        pytest.skip("cell has no ports")
    if not netlist["ports"]:
        reason = (
            "cell's circuit ports are on no instance; netlist has no circuit ports"
            " to simulate"
            if any(p.port_type != "placement" for p in cell.ports)
            else "cell exposes only placement ports; no circuit ports to simulate"
        )
        if name in KNOWN_PORTLESS:
            reason += f" ({KNOWN_PORTLESS[name]})"
        request.applymarker(
            pytest.mark.xfail(reason=reason, strict=True, raises=AssertionError)
        )
        return
    missing = _missing_models(netlist)
    unregistered = sorted(missing & qpdk_models.keys())
    absent = sorted(missing - qpdk_models.keys())
    if any(m not in KNOWN_MISSING_MODELS for m in absent):
        pytest.skip(f"no SAX model for {absent}")
    reasons = []
    if unregistered:
        reasons.append(
            f"models for {unregistered} exist in qpdk.models but are not"
            " registered in PDK.models (#827)"
        )
    if absent:
        issues = ", ".join(sorted({KNOWN_MISSING_MODELS[m] for m in absent}))
        reasons.append(f"no SAX model for {absent} ({issues})")
    request.applymarker(
        pytest.mark.xfail(reason="; ".join(reasons), strict=True, raises=ValueError)
    )


@pytest.mark.parametrize(
    "name",
    sorted(PDK.cells.keys() - {"transform_component"}),
)
def test_cell_builds_sax_circuit(name: str, request: pytest.FixtureRequest) -> None:
    """Build and evaluate SAX circuits from the cell's own netlist and its wrapper."""
    cell = gf.get_component(name)
    own = get_circuit_netlist(cell)
    missing = _missing_models(own)
    netlists = []
    if name in PDK.models:
        netlists.append(get_circuit_netlist(_wrapper(cell)))
    if own["ports"] and not missing:
        netlists.append(own)
    elif name not in PDK.models:
        # A leaf cell is simulated through its wrapper, which needs its model.
        netlist = own if own["instances"] else get_circuit_netlist(_wrapper(cell))
        _mark_unsimulatable(name, cell, netlist, request)
        netlists.append(netlist)

    if name in EVALUATION_FAILURES:
        raises, reason = EVALUATION_FAILURES[name]
        request.applymarker(
            pytest.mark.xfail(reason=reason, strict=True, raises=raises)
        )

    for netlist in netlists:
        assert netlist["ports"], "netlist has no circuit ports to simulate"
        circuit, _ = sax.circuit(netlist, models=PDK.models)
        result = circuit(f=jnp.array([5e9]))
        assert set(sax.get_ports(result)) == set(netlist["ports"])
