"""Tests for the chip-aware through-silicon via cells."""

import json
from collections.abc import Callable
from pathlib import Path

import gdsfactory as gf
import jax
import jax.numpy as jnp
import klayout.db as kdb
import numpy as np
import pytest
import sax
from gdsfactory.read import from_yaml
from gdsfactory.technology import LayerLevel, LogicalLayer

import qpdk
from qpdk import PDK
from qpdk.cells import (
    bump_transition,
    tsv,
    tsv_transition,
    tsv_transition_double_sided,
)
from qpdk.models import models
from qpdk.tech import (
    CHIP_LAYERS,
    LAYER_CONNECTIVITY,
    LAYER_STACK_FLIP_CHIP_BACKSIDE,
    Chip,
)

PDK.activate()

TSV_CELLS = [tsv, tsv_transition, tsv_transition_double_sided]
CHIPS: list[Chip] = list(CHIP_LAYERS)


def _region(c: gf.Component, layer: tuple[int, int]) -> kdb.Region:
    return kdb.Region(c.kdb_cell.begin_shapes_rec(gf.get_layer(layer))).merged()


def _is_layer(spec: object, layer: tuple[int, int]) -> bool:
    return isinstance(spec, LogicalLayer) and gf.get_layer(spec.layer) == gf.get_layer(
        layer
    )


def _level(layer: tuple[int, int]) -> LayerLevel:
    """Stack level drawn on, or whose derived conductor follows, ``layer``."""
    for level in LAYER_STACK_FLIP_CHIP_BACKSIDE.layers.values():
        if _is_layer(level.layer, layer) or _is_layer(level.derived_layer, layer):
            return level
    raise KeyError(layer)


@pytest.mark.parametrize("chip", CHIPS)
@pytest.mark.parametrize("cell", TSV_CELLS, ids=lambda f: f.__name__)
def test_cell_stays_on_its_chip(cell: Callable[..., gf.Component], chip: Chip) -> None:
    """Every TSV lands on both faces of its own chip and touches no other chip."""
    c = cell(chip=chip)
    layers = CHIP_LAYERS[chip]
    vias = _region(c, layers.tsv)
    assert not vias.is_empty()
    assert (vias - _region(c, layers.front_draw)).is_empty()
    assert (vias - _region(c, layers.back_draw)).is_empty()
    for other_chip, other in CHIP_LAYERS.items():
        if other_chip == chip:
            continue
        for layer in (other.front_draw, other.back_draw, other.tsv):
            assert _region(c, layer).is_empty(), (other_chip, layer)


@pytest.mark.parametrize("chip", CHIPS)
def test_tsv_spans_its_substrate(chip: Chip) -> None:
    """The via level runs exactly from the front metal to the backside metal."""
    layers = CHIP_LAYERS[chip]
    via = _level(layers.tsv)
    front = _level(layers.front_draw)
    back = _level(layers.back_draw)
    # Front metal faces the flip-chip gap, so the chips mirror each other.
    if front.zmin < back.zmin:
        via_ends = (front.zmin + front.thickness, back.zmin)
    else:
        via_ends = (back.zmin + back.thickness, front.zmin)
    assert (via.zmin, via.zmin + via.thickness) == pytest.approx(via_ends)


@pytest.mark.parametrize("chip", CHIPS)
def test_tsv_connects_its_faces(chip: Chip) -> None:
    layers = CHIP_LAYERS[chip]
    names = tuple(
        str(layer) for layer in (layers.front_draw, layers.tsv, layers.back_draw)
    )
    assert names in LAYER_CONNECTIVITY


@pytest.mark.parametrize("chip", CHIPS)
def test_chip_cross_sections_use_chip_layers(chip: Chip) -> None:
    layers = CHIP_LAYERS[chip]
    for name, draw in (
        (layers.front_cross_section, layers.front_draw),
        (layers.back_cross_section, layers.back_draw),
    ):
        assert gf.get_layer(gf.get_cross_section(name).layer) == gf.get_layer(draw)


@pytest.mark.parametrize(
    "cell",
    [tsv_transition, tsv_transition_double_sided, bump_transition],
    ids=lambda f: f.__name__,
)
def test_transition_tapers_resolve_after_serialization(
    cell: Callable[..., gf.Component],
) -> None:
    """Taper cross-sections survive a JSON round trip, as GDSFactory+ sends them.

    An unregistered ``CrossSection`` is recorded as an ``xs_<hash>`` name that
    the ``taper_cross_section`` model cannot look up.
    """
    netlist = json.loads(json.dumps(cell().get_netlist(), default=str))
    tapers = [
        inst["settings"]
        for inst in netlist["instances"].values()
        if inst["component"] == "taper_cross_section"
    ]
    assert tapers
    for settings in tapers:
        models["taper_cross_section"](
            length=settings["length"],
            cross_section1=settings["cross_section1"],
            cross_section2=settings["cross_section2"],
        )


def test_flip_chip_sample_sweeps_through_sax() -> None:
    """Both of the sample's lines transmit through a jitted SAX circuit.

    GDSFactory+ builds the circuit from a JSON netlist and passes numeric
    settings as arrays, and ``jax.jit`` traces them all, so the transition
    models must not turn them into Python ints or cross-sections. The
    transition cells' own netlists have no path between their ports, so they
    need their own models.
    """
    sample = Path(qpdk.__file__).parent / "samples" / "tsv_flip_chip_test_chip.pic.yml"
    netlist = json.loads(json.dumps(from_yaml(sample).get_netlist(), default=str))
    with jax.enable_x64():
        circuit, _ = sax.circuit(netlist, models=models)
        s = jax.jit(circuit)(f=jnp.linspace(4e9, 8e9, 3))
    np.testing.assert_allclose(jnp.abs(s["o1", "o2"]), 1.0, atol=1e-3)
    np.testing.assert_allclose(jnp.abs(s["o3", "o4"]), 1.0, atol=1e-3)


def test_taper_accepts_array_npoints() -> None:
    """SAX hands ``npoints`` over as a float array, not an int."""
    f = jnp.linspace(4e9, 8e9, 3)
    kwargs = {
        "f": f,
        "cross_section2": {"cross_section": "cpw", "settings": {"width": 20}},
    }
    expected = models["taper_cross_section"](**kwargs, npoints=7)
    actual = models["taper_cross_section"](**kwargs, npoints=jnp.asarray(7.0))
    np.testing.assert_allclose(actual["o1", "o2"], expected["o1", "o2"])


def test_bump_transition_has_single_signal_bump() -> None:
    """One indium bump joins the two pads, as in the published indium transition."""
    c = bump_transition()
    bumps = _region(c, qpdk.tech.LAYER.IND)
    assert bumps.count() == 1
    for pad_layer in ("M1_DRAW", "M2_DRAW"):
        pad = _region(c, getattr(qpdk.tech.LAYER, pad_layer))
        assert (bumps - pad).is_empty(), f"bump overhangs the {pad_layer} pad"
