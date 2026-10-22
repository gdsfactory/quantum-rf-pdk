# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "qpdk[models] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git",
# ]
# ///

r"""Flux-driven CPW JPA chip with signal and pump launchers.

A flux-driven :math:`\lambda/4` CPW Josephson parametric amplifier following
:cite:`yamamotoFluxdrivenJosephsonParametric2008`: an interdigital capacitor
couples the signal into a meandered quarter-wave line that is shorted to ground
through a DC SQUID, and a shorted CPW pump line next to the SQUID loop
modulates the flux. The static flux bias comes from an off-chip coil. The
coupling and critical current follow the WMI devices of
:cite:`pogorzalekHystereticFluxResponse2017`.

Running this script writes the chip GDS and prints the SAX resonance of the
full chip netlist against flux.
"""

import gdsfactory as gf

from qpdk.cells.launcher import launcher
from qpdk.cells.parametric_amplifier import cpw_jpa
from qpdk.tech import route_bundle_cpw


@gf.cell(tags=["samples", "resonators", "junctions"])
def cpw_jpa_chip(
    signal_route_length: float = 800.0,
    pump_route_length: float = 800.0,
) -> gf.Component:
    """Flux-driven CPW JPA with launchers on the signal and pump lines.

    Args:
        signal_route_length: Distance in µm between the signal launcher and
            the JPA signal port.
        pump_route_length: Distance in µm between the pump launcher and the
            JPA pump port.

    Returns:
        Component: The chip with the signal launcher waveport ``o1`` and the
        pump launcher waveport ``o2``.
    """
    c = gf.Component()

    jpa_ref = c.add_ref(cpw_jpa(), name="jpa")
    signal_port = jpa_ref.ports["o1"]
    pump_port = jpa_ref.ports["o2"]

    signal_launcher_ref = c.add_ref(launcher(), name="signal_launcher")
    signal_launcher_ref.dmove((
        signal_port.dx - signal_route_length - signal_launcher_ref.ports["o1"].dx,
        signal_port.dy - signal_launcher_ref.ports["o1"].dy,
    ))

    pump_launcher_ref = c.add_ref(launcher(), name="pump_launcher")
    pump_launcher_ref.drotate(90)
    pump_launcher_ref.dmove((
        pump_port.dx - pump_launcher_ref.ports["o1"].dx,
        pump_port.dy - pump_route_length - pump_launcher_ref.ports["o1"].dy,
    ))

    for name, launcher_ref, port in (
        ("signal_route", signal_launcher_ref, signal_port),
        ("pump_route", pump_launcher_ref, pump_port),
    ):
        existing_names = {instance.name for instance in c.insts}
        route_bundle_cpw(c, ports1=[launcher_ref.ports["o1"]], ports2=[port])
        (route_ref,) = (i for i in c.insts if i.name not in existing_names)
        route_ref.name = name

    c.add_port("o1", port=signal_launcher_ref.ports["waveport"])
    c.add_port("o2", port=pump_launcher_ref.ports["waveport"])
    c.info.update({f"jpa_{k}": v for k, v in jpa_ref.cell.info.model_dump().items()})
    return c


if __name__ == "__main__":
    import jax.numpy as jnp
    import sax

    from qpdk import PATH, PDK, logger
    from qpdk.models import models
    from qpdk.models.constants import Φ_0

    PDK.activate()

    chip = cpw_jpa_chip()
    gds_path = chip.write_gds(PATH.gds / "cpw_jpa_chip.gds")
    logger.info(f"Wrote {gds_path}")

    # The SQUID flux is an instance override of the JPA's termination model.
    circuit, _ = sax.circuit(chip.get_netlist(recursive=True), models=models)
    f = jnp.linspace(5.5e9, 6.5e9, 20001)
    for flux in (0.0, 0.2, 0.3, 0.4):
        s11 = circuit(f=f, jpa={"squid_termination": {"flux": flux * Φ_0}})["o1", "o1"]
        # The reflection phase turns fastest at resonance.
        phase_slope = jnp.abs(jnp.diff(jnp.unwrap(jnp.angle(s11))))
        logger.info(f"Φ = {flux:.1f} Φ0: f0 = {f[phase_slope.argmax()] / 1e9:.3f} GHz")

    chip.show()
