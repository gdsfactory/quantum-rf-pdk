"""Gatemon test chip sample, materialized from its ``.pic.yml`` netlist."""

from pathlib import Path

import gdsfactory as gf
from gdsfactory.read import from_yaml

YAML_SAMPLE = Path(__file__).parent / "gatemon_test_chip.pic.yml"


@gf.cell(tags=["samples", "qubits"])
def gatemon_test_chip() -> gf.Component:
    r"""Layout of the gatemon test chip, read from its ``.pic.yml`` netlist.

    Two independent gatemons, each at the input end of a :math:`\lambda/2`
    meandered readout resonator placed in series with a probe line through
    interdigital coupling capacitors, with a gate line routed to its own
    launcher. Follows the chip layout of :cite:`riechertCarbonNanotubeGatemon2025`
    (Fig. 1c). Device ``a`` targets :math:`f_r = 5.056` GHz (device A of the
    reference) and device ``b`` 5.3 GHz; the resonator lengths are
    :math:`c / (2 f_r \sqrt{\varepsilon_\text{eff}})` for the default CPW,
    ignoring the loading by the coupling capacitors.

    The materialized name matches the ``.pic.yml``/GDS stem because elvis
    resolves the top cell from the GDS file name.

    Returns:
        The materialized sample component.
    """
    return from_yaml(YAML_SAMPLE, name="gatemon_test_chip")
