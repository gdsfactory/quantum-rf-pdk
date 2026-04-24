"""Model definitions for qpdk."""

# ruff: noqa: E402

import jax
import sax

from qpdk.models.capacitor import (
    interdigital_capacitor,
    plate_capacitor,
)
from qpdk.models.inductor import (
    lumped_element_resonator,
    meander_inductor,
    meander_inductor_inductance_analytical,
)
from qpdk.models.junction import josephson_junction, squid_junction

sax.set_port_naming_strategy("optical")

from qpdk.models.constants import (
    DEFAULT_FREQUENCY,
)
from qpdk.models.couplers import (
    coupler_ring,
    coupler_straight,
    cpw_cpw_coupling_capacitance,
)
from qpdk.models.cpw import (
    cpw_epsilon_eff,
    cpw_parameters,
    cpw_thickness_correction,
    cpw_z0,
    cpw_z0_from_cross_section,
    microstrip_epsilon_eff,
    microstrip_thickness_correction,
    microstrip_z0,
    propagation_constant,
    transmission_line_s_params,
)
from qpdk.models.generic import (
    admittance,
    capacitor,
    electrical_open,
    electrical_short,
    electrical_short_2_port,
    gamma_0_load,
    impedance,
    inductor,
    lc_resonator,
    lc_resonator_coupled,
    open,
    series_impedance,
    short,
    short_2_port,
    shunt_admittance,
)
from qpdk.models.perturbation import (
    dispersive_shift,
    dispersive_shift_to_coupling,
    ej_ec_to_frequency_and_anharmonicity,
    measurement_induced_dephasing,
    purcell_decay_rate,
    resonator_linewidth_from_q,
)
from qpdk.models.qubit import (
    coupling_strength_to_capacitance,
    double_island_transmon,
    double_island_transmon_with_bbox,
    double_island_transmon_with_resonator,
    double_pad_transmon,
    double_pad_transmon_with_bbox,
    double_pad_transmon_with_resonator,
    ec_to_capacitance,
    ej_to_inductance,
    el_to_inductance,
    flipmon,
    flipmon_with_bbox,
    flipmon_with_resonator,
    fluxonium,
    fluxonium_coupled,
    fluxonium_with_bbox,
    fluxonium_with_resonator,
    qubit_with_resonator,
    shunted_transmon,
    transmon_coupled,
    transmon_with_resonator,
    xmon_transmon,
)
from qpdk.models.resonator import (
    quarter_wave_resonator_coupled,
    resonator,
    resonator_coupled,
    resonator_frequency,
    resonator_half_wave,
    resonator_quarter_wave,
)
from qpdk.models.unimon import (
    el_to_arm_inductance,
    unimon_coupled,
    unimon_energies,
    unimon_frequency_and_anharmonicity,
    unimon_hamiltonian,
)
from qpdk.models.waveguides import (
    airbridge,
    bend_circular,
    bend_circular_all_angle,
    bend_euler,
    bend_euler_all_angle,
    bend_s,
    indium_bump,
    launcher,
    nxn,
    rectangle,
    straight,
    straight_all_angle,
    straight_double_open,
    straight_microstrip,
    straight_open,
    straight_shorted,
    taper_cross_section,
    tee,
    tsv,
)

__all__ = [
    "DEFAULT_FREQUENCY",
    "admittance",
    "airbridge",
    "bend_circular",
    "bend_circular_all_angle",
    "bend_euler",
    "bend_euler_all_angle",
    "bend_s",
    "capacitor",
    "coupler_ring",
    "coupler_straight",
    "coupling_strength_to_capacitance",
    "cpw_cpw_coupling_capacitance",
    "cpw_epsilon_eff",
    "cpw_parameters",
    "cpw_thickness_correction",
    "cpw_z0",
    "cpw_z0_from_cross_section",
    "dispersive_shift",
    "dispersive_shift_to_coupling",
    "double_island_transmon",
    "double_island_transmon_with_bbox",
    "double_island_transmon_with_resonator",
    "double_pad_transmon",
    "double_pad_transmon_with_bbox",
    "double_pad_transmon_with_resonator",
    "ec_to_capacitance",
    "ej_ec_to_frequency_and_anharmonicity",
    "ej_to_inductance",
    "el_to_arm_inductance",
    "el_to_inductance",
    "electrical_open",
    "electrical_short",
    "electrical_short_2_port",
    "flipmon",
    "flipmon_with_bbox",
    "flipmon_with_resonator",
    "fluxonium",
    "fluxonium_coupled",
    "fluxonium_with_bbox",
    "fluxonium_with_resonator",
    "gamma_0_load",
    "impedance",
    "indium_bump",
    "inductor",
    "interdigital_capacitor",
    "josephson_junction",
    "launcher",
    "lc_resonator",
    "lc_resonator_coupled",
    "lumped_element_resonator",
    "meander_inductor",
    "meander_inductor_inductance_analytical",
    "measurement_induced_dephasing",
    "microstrip_epsilon_eff",
    "microstrip_thickness_correction",
    "microstrip_z0",
    "models",
    "nxn",
    "open",
    "plate_capacitor",
    "propagation_constant",
    "purcell_decay_rate",
    "quarter_wave_resonator_coupled",
    "qubit_with_resonator",
    "rectangle",
    "resonator",
    "resonator_coupled",
    "resonator_frequency",
    "resonator_half_wave",
    "resonator_linewidth_from_q",
    "resonator_quarter_wave",
    "series_impedance",
    "short",
    "short_2_port",
    "shunt_admittance",
    "shunted_transmon",
    "squid_junction",
    "straight",
    "straight_all_angle",
    "straight_double_open",
    "straight_microstrip",
    "straight_open",
    "straight_shorted",
    "taper_cross_section",
    "tee",
    "transmission_line_s_params",
    "transmon_coupled",
    "transmon_with_resonator",
    "tsv",
    "unimon_coupled",
    "unimon_energies",
    "unimon_frequency_and_anharmonicity",
    "unimon_hamiltonian",
    "xmon_transmon",
]


def _is_sax_model(obj: object) -> bool:
    """Check if an object is a SAX model function."""
    if not callable(obj):
        return False

    # Skip functions that return jax.Array or tuple (these are Hamiltonian models)
    if hasattr(obj, "__annotations__") and "return" in obj.__annotations__:
        ret = obj.__annotations__["return"]
        # If it returns jax.Array or a tuple of floats, it's not a SAX S-parameter model
        if ret in {jax.Array, "jax.Array"}:
            return False
        # Match names like tuple[float, float]
        if hasattr(ret, "__name__") and ret.__name__ == "tuple":
            return False
        if str(ret).startswith("tuple") or str(ret).startswith("Tuple"):
            return False

    # Check if return type is sax.SType or sax.SDict
    for target in [obj, getattr(obj, "__wrapped__", None)]:
        if target is None:
            continue
        if hasattr(target, "__annotations__") and "return" in target.__annotations__:
            ret = target.__annotations__["return"]
            if hasattr(ret, "__name__") and ret.__name__ in {"SType", "SDict"}:
                return True
            # Also check identity for standard cases
            if ret in {sax.SType, sax.SDict}:
                return True
    return sax.try_into[sax.Model](obj) is not None


models = {
    k: v
    for k, v in globals().items()
    if k in __all__ and k != "models" and _is_sax_model(v)
}
