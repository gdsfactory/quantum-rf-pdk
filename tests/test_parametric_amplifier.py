"""Tests for the flux-driven CPW JPA cells and their SAX hierarchy."""

from functools import partial
from itertools import pairwise

import gdsfactory as gf
import jax.numpy as jnp
import pytest
import sax

from qpdk.cells.junction import squid_junction
from qpdk.cells.parametric_amplifier import cpw_jpa, squid_termination
from qpdk.models import models
from qpdk.models.capacitor import interdigital_capacitor_capacitance_analytical
from qpdk.models.constants import Φ_0, c_0
from qpdk.models.cpw import cpw_ep_r_from_cross_section, cpw_parameters
from qpdk.samples.cpw_jpa_chip import cpw_jpa_chip
from qpdk.tech import LAYER, coplanar_waveguide, get_etch_section

IC_TOT = 4.8e-6


def _resonance(f: jnp.ndarray, **squid_settings: float) -> float:
    """Return the frequency where the JPA reflection phase turns fastest."""
    circuit, _ = sax.circuit(cpw_jpa().get_netlist(recursive=True), models=models)
    s11 = circuit(f=f, squid_termination=squid_settings)["o1", "o1"]
    phase_slope = jnp.abs(jnp.diff(jnp.unwrap(jnp.angle(s11))))
    return float(f[phase_slope.argmax()])


def test_squid_shorts_the_lead_to_ground() -> None:
    """Every SQUID contact patch lands on either the lead or the ground plane."""
    c = squid_termination()

    def region(layer: LAYER) -> gf.kdb.Region:
        return gf.kdb.Region(c.begin_shapes_rec(gf.get_layer(layer))).merged()

    lead, etch = region(LAYER.M1_DRAW), region(LAYER.M1_ETCH)
    on_lead, on_ground = 0, 0
    for patch in region(LAYER.JJ_PATCH).each():
        patch_region = gf.kdb.Region(patch)
        touches_lead = not (patch_region & lead).is_empty()
        touches_ground = not (patch_region - etch - lead).is_empty()
        assert touches_lead != touches_ground, patch
        on_lead += touches_lead
        on_ground += touches_ground
    assert on_lead > 0
    assert on_ground > 0


def test_squid_wider_than_the_lead_is_rejected() -> None:
    """A SQUID whose lead-side contacts overhang the centre conductor would float."""
    with pytest.raises(ValueError, match="centre conductor width"):
        squid_termination(junction_spec=squid_junction)


def _asymmetric_squid() -> gf.Component:
    """Return a SQUID stand-in whose lead-side contacts are unevenly spaced.

    The contacts are drawn at -45° so that they line up along x once
    :func:`squid_termination` rotates the SQUID by 45°.
    """
    c = gf.Component()
    patches = gf.Component()
    for x in (-6.0, -1.0, 3.5):
        patches.add_ref(
            gf.c.rectangle(size=(1.0, 1.0), layer=LAYER.JJ_PATCH, centered=True)
        ).dmove((x, 0.0))
    c.add_ref(patches).drotate(-45)
    return c


def test_floating_contact_is_rejected() -> None:
    """A contact that misses both the lead and the ground plane is caught."""
    with pytest.raises(ValueError, match="wholly over the etched pocket"):
        squid_termination(junction_spec=_asymmetric_squid)


def test_pump_runs_past_the_ground_side_of_the_squid() -> None:
    """The pump slot is ``pump_gap`` past the SQUID, its short ``pump_offset`` up."""
    pump_length, pump_gap, pump_offset = 200.0, 2.0, 20.0
    c = cpw_jpa(pump_length=pump_length, pump_gap=pump_gap, pump_offset=pump_offset)
    termination = next(i for i in c.insts if i.cell.name.startswith("squid_term"))
    squid = termination.ports["squid"]
    xs = gf.get_cross_section("cpw")
    pump_slot_x = c.ports["o2"].dx - xs.width / 2 - get_etch_section(xs).width
    assert pump_slot_x == pytest.approx(termination.dxmax + pump_gap, abs=1e-3)
    assert c.ports["o2"].dy == pytest.approx(
        squid.dy + pump_offset - pump_length, abs=1e-3
    )
    assert c.ports["o2"].orientation == pytest.approx(270)


def test_cross_section_reaches_every_child() -> None:
    """A custom cross-section is used by the capacitor and termination too."""
    xs = partial(coplanar_waveguide, width=12.0, gap=7.0)
    c = cpw_jpa(cross_section=xs)
    width = gf.get_cross_section(xs).width
    for name in ("interdigital_capacitor", "squid_termination"):
        instance = next(i for i in c.insts if i.cell.name.startswith(name))
        assert instance.ports["o1"].dwidth == pytest.approx(width)


def test_length_must_exceed_the_termination_lead() -> None:
    """A JPA shorter than its SQUID lead has no room for the resonator."""
    with pytest.raises(ValueError, match="must exceed"):
        cpw_jpa(length=5.0)


def test_info_exposes_model_parameters() -> None:
    """The geometry the model needs is available on the cell."""
    info = cpw_jpa().info
    for key in (
        "length",
        "cross_section",
        "coupling_fingers",
        "coupling_finger_length",
        "squid_loop_area",
        "pump_gap",
        "pump_offset",
    ):
        assert key in info


def test_flux_pulls_resonance_as_lumped_estimate() -> None:
    r"""The SAX shift matches :math:`1 / (1 + L_\text{s} / L_\text{r})`.

    A large-:math:`I_\text{c}` SQUID reduces the JPA to a shorted
    :math:`\lambda/4` line; the ratio to it isolates the SQUID loading.
    """
    f = jnp.linspace(5.8e9, 6.6e9, 16001)
    f_squid = _resonance(f, ic_tot=IC_TOT)
    f_short = _resonance(f, ic_tot=1.0)

    ep_eff, z0 = cpw_parameters(width=10.0, gap=6.0)
    l_r = jnp.real(z0 * jnp.sqrt(ep_eff)) / c_0 * cpw_jpa().info["length"] * 1e-6
    l_s = Φ_0 / (2 * jnp.pi * IC_TOT)
    assert f_squid / f_short == pytest.approx(float(1 / (1 + l_s / l_r)), rel=5e-3)


def test_resonance_falls_with_flux() -> None:
    """Flux raises the SQUID inductance, so the resonance moves down."""
    f = jnp.linspace(5.4e9, 6.4e9, 10001)
    f0 = [_resonance(f, flux=flux * Φ_0) for flux in (0.0, 0.2, 0.3, 0.4)]
    assert all(a > b for a, b in pairwise(f0))


def test_chip_simulates_through_launchers() -> None:
    """The sample chip netlist resolves to SAX models end to end."""
    chip = cpw_jpa_chip()
    circuit, _ = sax.circuit(chip.get_netlist(recursive=True), models=models)
    s = circuit(f=jnp.linspace(5.5e9, 6.5e9, 101))
    for key in (("o1", "o1"), ("o2", "o2"), ("o1", "o2")):
        assert jnp.all(jnp.isfinite(s[key])), key
    assert jnp.all(jnp.abs(s["o1", "o1"]) <= 1 + 1e-9)


def test_external_quality_factor_matches_coupling_capacitor() -> None:
    r"""The reflection phase slope gives :math:`Q_\text{ext}` of the IDC coupling.

    A lossless reflection resonator turns its phase by :math:`2\pi` with
    :math:`\mathrm{d}\varphi / \mathrm{d}f = 4 Q_\text{ext} / f_0` at resonance.
    The :math:`\lambda/4` estimate :math:`Q_\text{ext} = \pi / (4 (\omega_0 C_\text{c} Z_0)^2)`
    holds to a few percent for the default :math:`28\,\text{fF}` capacitor.
    """
    f = jnp.linspace(5.8e9, 6.3e9, 50001)
    circuit, _ = sax.circuit(cpw_jpa().get_netlist(recursive=True), models=models)
    s11 = circuit(f=f, squid_termination={"resistance": 1e18})["o1", "o1"]
    phase_slope = jnp.abs(jnp.diff(jnp.unwrap(jnp.angle(s11)))) / (f[1] - f[0])
    i = int(phase_slope.argmax())
    q_ext = float(phase_slope[i] * f[i] / 4)

    info = cpw_jpa().info
    capacitance = interdigital_capacitor_capacitance_analytical(
        fingers=info["coupling_fingers"],
        finger_length=info["coupling_finger_length"],
        finger_gap=info["coupling_finger_gap"],
        thickness=info["coupling_thickness"],
        ep_r=cpw_ep_r_from_cross_section("cpw"),
    )
    _, z0 = cpw_parameters(width=10.0, gap=6.0)
    q_estimate = jnp.pi / (4 * (2 * jnp.pi * f[i] * capacitance * jnp.real(z0)) ** 2)
    assert q_ext == pytest.approx(float(q_estimate), rel=0.1)
