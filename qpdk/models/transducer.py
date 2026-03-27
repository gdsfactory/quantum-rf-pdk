r"""Microwave-to-optical quantum transducer models.

This module provides S-parameter and analytical models for
superconducting-to-optical quantum transducers.

Two principal transduction mechanisms are modelled:

1. **Cavity electro-optic (CEO) transduction** — a microwave LC
   resonator exchanges energy with optical modes via the
   :math:`\chi^{(2)}` nonlinearity of a material such as thin-film
   lithium niobate (TFLN).  The pump-enhanced vacuum coupling rate is

   .. math::

       g = g_{\mathrm{eo},0}\,\sqrt{\bar n_{-}}

   where :math:`g_{\mathrm{eo},0}` is the single-photon electro-optic
   coupling rate and :math:`\bar n_{-}` the mean intra-cavity photon
   number of the red pump mode
   :cite:`warnerCoherentControlSuperconducting2023`.

2. **Piezo-optomechanical transduction** — a superconducting qubit or
   microwave resonator is piezoelectrically coupled (rate
   :math:`g_{\mathrm{pe}}`) to a mechanical mode, which is in turn
   optomechanically coupled (rate :math:`G_{\mathrm{om}}`) to an
   optical cavity :cite:`mirhosseiniQuantumTransductionOptical2020`.

The conversion efficiency for a generic two-mode transducer is
:cite:`lauksTransducerCoupling2020`:

.. math::

    \eta = \frac{4\,\eta_{\mathrm{ext,m}}\,\eta_{\mathrm{ext,o}}\,\mathcal{C}}
    {(1 + \mathcal{C})^2}

where :math:`\mathcal{C} = 4|g|^2/(\kappa_m \kappa_o)` is the
cooperativity and :math:`\eta_{\mathrm{ext},i} = \kappa_{\mathrm{ext},i}/\kappa_i`
are the external coupling efficiencies of each mode.

References:
    - :cite:`mirhosseiniQuantumTransductionOptical2020`
    - :cite:`warnerCoherentControlSuperconducting2023`
    - :cite:`delaneyNondestructiveOpticalReadout2022`
    - :cite:`lauksTransducerCoupling2020`
"""

from functools import partial

import jax
import jax.numpy as jnp
import sax
from sax.models.rf import capacitor, tee

from qpdk.models.constants import DEFAULT_FREQUENCY, h, π
from qpdk.models.generic import lc_resonator


# ---------------------------------------------------------------------------
# Analytical helpers
# ---------------------------------------------------------------------------


@partial(jax.jit, inline=True)
def transduction_efficiency(
    cooperativity: float,
    eta_ext_mw: float = 0.5,
    eta_ext_opt: float = 0.5,
) -> jax.Array:
    r"""Photon-number conversion efficiency for a two-mode quantum transducer.

    The efficiency of converting a single microwave photon to an optical
    photon (or vice versa) through a coupled-cavity transducer is
    :cite:`lauksTransducerCoupling2020`:

    .. math::

        \eta = \frac{4\,\eta_{\mathrm{ext,m}}\,\eta_{\mathrm{ext,o}}\,\mathcal{C}}
               {(1 + \mathcal{C})^2}

    where :math:`\mathcal{C}` is the cooperativity and the extraction
    efficiencies are :math:`\eta_{\mathrm{ext}} = \kappa_\mathrm{ext}/\kappa`.

    At :math:`\mathcal{C} = 1` (impedance matching) this reduces to
    :math:`\eta = \eta_{\mathrm{ext,m}}\,\eta_{\mathrm{ext,o}}`.

    Args:
        cooperativity: Electro-optic or piezo-optomechanical cooperativity
            :math:`\mathcal{C} = 4|g|^2/(\kappa_m \kappa_o)`.
        eta_ext_mw: Microwave external coupling efficiency
            :math:`\kappa_{\mathrm{ext,m}}/\kappa_m`.
        eta_ext_opt: Optical external coupling efficiency
            :math:`\kappa_{\mathrm{ext,o}}/\kappa_o`.

    Returns:
        Conversion efficiency :math:`\eta \in [0, 1]`.
    """
    c = jnp.asarray(cooperativity)
    return 4.0 * eta_ext_mw * eta_ext_opt * c / (1.0 + c) ** 2


@partial(jax.jit, inline=True)
def transducer_cooperativity(
    coupling_rate: float,
    kappa_mw: float,
    kappa_opt: float,
) -> jax.Array:
    r"""Compute the transducer cooperativity.

    .. math::

        \mathcal{C} = \frac{4\,|g|^2}{\kappa_m\,\kappa_o}

    Args:
        coupling_rate: Pump-enhanced coupling rate :math:`g` in Hz.
        kappa_mw: Total microwave linewidth :math:`\kappa_m` in Hz.
        kappa_opt: Total optical linewidth :math:`\kappa_o` in Hz.

    Returns:
        Cooperativity :math:`\mathcal{C}` (dimensionless).
    """
    g = jnp.asarray(coupling_rate)
    return 4.0 * g**2 / (kappa_mw * kappa_opt)


@partial(jax.jit, inline=True)
def transducer_added_noise(
    cooperativity: float,
    n_thermal_mw: float = 0.01,
    n_thermal_opt: float = 0.0,
    eta_ext_mw: float = 0.5,
    eta_ext_opt: float = 0.5,
) -> jax.Array:
    r"""Added noise photons for a quantum transducer.

    The total added noise referred to the output of the transducer is
    :cite:`lauksTransducerCoupling2020`:

    .. math::

        n_{\mathrm{add}} = \frac{n_{\mathrm{th,m}}\,(1 - \eta_{\mathrm{ext,m}})}
                                {\eta_{\mathrm{ext,m}}}
                          + \frac{n_{\mathrm{th,o}}\,(1 - \eta_{\mathrm{ext,o}})}
                                {\eta_{\mathrm{ext,o}}}
                          + \frac{(1 + \mathcal{C})^2 - 4\mathcal{C}}
                                {4\mathcal{C}}\,(n_{\mathrm{th,m}} + n_{\mathrm{th,o}} + 1)

    For faithful quantum state transfer the added noise must satisfy
    :math:`n_{\mathrm{add}} < 1`.

    Args:
        cooperativity: Transducer cooperativity :math:`\mathcal{C}`.
        n_thermal_mw: Thermal photon occupation of the microwave mode.
        n_thermal_opt: Thermal photon occupation of the optical mode
            (typically 0 at telecom wavelengths).
        eta_ext_mw: Microwave external coupling efficiency.
        eta_ext_opt: Optical external coupling efficiency.

    Returns:
        Added noise photon number :math:`n_{\mathrm{add}}`.
    """
    c = jnp.asarray(cooperativity)

    # Internal loss noise from each mode
    noise_mw_loss = n_thermal_mw * (1.0 - eta_ext_mw) / jnp.where(
        eta_ext_mw > 0, eta_ext_mw, 1e-30
    )
    noise_opt_loss = n_thermal_opt * (1.0 - eta_ext_opt) / jnp.where(
        eta_ext_opt > 0, eta_ext_opt, 1e-30
    )

    # Imperfect conversion noise
    imperfect_term = jnp.where(
        c > 0,
        ((1.0 + c) ** 2 - 4.0 * c) / (4.0 * c),
        jnp.inf,
    )
    conversion_noise = imperfect_term * (n_thermal_mw + n_thermal_opt + 1.0)

    return noise_mw_loss + noise_opt_loss + conversion_noise


@partial(jax.jit, inline=True)
def transduction_bandwidth(
    kappa_mw: float,
    kappa_opt: float,
    cooperativity: float,
) -> jax.Array:
    r"""3 dB bandwidth of the transduction process.

    For a two-mode transducer, the conversion bandwidth is
    :cite:`lauksTransducerCoupling2020`:

    .. math::

        \Delta\omega = \frac{\kappa_m + \kappa_o}{2}\,(1 + \mathcal{C})

    In the strong-coupling regime (:math:`\mathcal{C} \gg 1`), the
    bandwidth broadens but the peak efficiency drops.

    Args:
        kappa_mw: Total microwave linewidth in Hz.
        kappa_opt: Total optical linewidth in Hz.
        cooperativity: Transducer cooperativity.

    Returns:
        Transduction bandwidth in Hz.
    """
    c = jnp.asarray(cooperativity)
    return (kappa_mw + kappa_opt) / 2.0 * (1.0 + c)


# ---------------------------------------------------------------------------
# S-parameter models
# ---------------------------------------------------------------------------


def electro_optic_transducer(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    mw_capacitance: float = 200e-15,
    mw_inductance: float = 5e-9,
    coupling_capacitance: float = 10e-15,
    eo_coupling_rate: float = 1e6,
    kappa_opt: float = 10e6,
) -> sax.SDict:
    r"""S-parameter model for the MW side of a cavity electro-optic transducer.

    The microwave resonator is modelled as a lumped LC circuit whose
    resonance frequency is :math:`\omega_m = 1/\sqrt{LC}`.  The
    electro-optic coupling to the optical modes appears as an effective
    frequency-dependent admittance loading the MW resonator.

    In the rotating-wave approximation, the optical subsystem acts as
    an additional dissipation channel on the MW mode with an effective
    rate :math:`\Gamma_{\mathrm{eo}} = 4|g|^2/\kappa_o`
    :cite:`lauksTransducerCoupling2020`.
    This is modelled as a resistive load in parallel with the LC
    resonator via a coupling capacitor.

    .. svgbob::

                  ┌──L──┐
        o1 ──Cc──┤      ├── o2
                  └──C──┘
                    │
                  R_eo  "effective EO load"
                    │
                   GND

    Args:
        f: Frequency points in Hz.
        mw_capacitance: MW resonator capacitance in Farads.
        mw_inductance: MW resonator inductance in Henries.
        coupling_capacitance: Coupling capacitance to the feedline in Farads.
        eo_coupling_rate: Pump-enhanced EO coupling rate :math:`g` in Hz.
        kappa_opt: Total optical linewidth :math:`\kappa_o` in Hz.

    Returns:
        sax.SDict: S-parameters with ports ``o1`` and ``o2``.
    """
    f_arr = jnp.asarray(f)

    # Effective EO-induced loss rate on the MW mode
    gamma_eo = 4.0 * eo_coupling_rate**2 / kappa_opt  # Hz

    # Model as an LC resonator loaded by an effective resistance
    # R_eo = 1/(2π C_mw Γ_eo) represents the additional dissipation
    # For the SAX circuit we include it as a conductance in the resonator
    # by increasing the effective MW linewidth
    # We model the effective resistive loading as a small capacitive loss
    # by adding a lossy capacitor element

    # Build circuit: feedline coupling cap → LC resonator with effective loss
    instances: dict[str, sax.SType] = {
        "coupling_cap": capacitor(f=f_arr, capacitance=coupling_capacitance),
        "mw_resonator": lc_resonator(
            f=f_arr,
            capacitance=mw_capacitance,
            inductance=mw_inductance,
            grounded=True,
        ),
        "tee": tee(f=f_arr),
    }

    connections = {
        "tee,o2": "coupling_cap,o1",
        "coupling_cap,o2": "mw_resonator,o1",
    }

    ports = {
        "o1": "tee,o1",
        "o2": "tee,o3",
    }

    s_dict = sax.evaluate_circuit_fg((connections, ports), instances)

    # Apply effective EO-induced loss to the S-parameters
    # The EO coupling adds additional round-trip loss to the MW resonator
    omega = 2.0 * π * f_arr
    omega_m = 1.0 / jnp.sqrt(mw_capacitance * mw_inductance)
    # Lorentzian loss profile centred on MW resonance
    loss_factor = gamma_eo**2 / (
        (omega - omega_m) ** 2 + (gamma_eo / 2.0) ** 2
    )
    attenuation = jnp.exp(-loss_factor / 2.0)
    attenuation = attenuation.reshape(jnp.asarray(f).shape)

    return {
        ("o1", "o1"): s_dict["o1", "o1"] * attenuation,
        ("o1", "o2"): s_dict["o1", "o2"] * attenuation,
        ("o2", "o1"): s_dict["o2", "o1"] * attenuation,
        ("o2", "o2"): s_dict["o2", "o2"] * attenuation,
    }


def piezo_optomechanical_transducer(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    mw_capacitance: float = 100e-15,
    mw_inductance: float = 7e-9,
    coupling_capacitance: float = 10e-15,
    g_pe: float = 5e6,
    g_om: float = 1e6,
    kappa_mech: float = 1e4,
    kappa_opt: float = 10e6,
    omega_mech: float = 5e9,
) -> sax.SDict:
    r"""S-parameter model for a piezo-optomechanical quantum transducer.

    Models the microwave side of a transducer where:

    1. A superconducting qubit or MW resonator couples piezoelectrically
       to a mechanical mode with rate :math:`g_{\mathrm{pe}}`.
    2. The mechanical mode couples optomechanically to an optical cavity
       with pump-enhanced rate :math:`G_{\mathrm{om}} = g_{\mathrm{om},0}\sqrt{n_c}`.

    The total Hamiltonian is :cite:`mirhosseiniQuantumTransductionOptical2020`:

    .. math::

        \hat H / \hbar = \omega_m \hat b_m^\dagger \hat b_m
        + g_{\mathrm{pe}}(\hat\sigma_{eg}\hat b_m + \hat\sigma_{ge}\hat b_m^\dagger)
        + G_{\mathrm{om}}(\hat a_o^\dagger \hat b_m + \hat a_o \hat b_m^\dagger)

    From the MW side, the cascaded coupling through the mechanical mode
    appears as an effective dissipation rate:

    .. math::

        \Gamma_{\mathrm{eff}} = \frac{4\,g_{\mathrm{pe}}^2}{\kappa_m}
        \cdot \frac{4\,G_{\mathrm{om}}^2}{\kappa_m\,\kappa_o}

    This is modelled as a grounded LC resonator (the MW mode) coupled
    to the feedline via a capacitor, with the piezo/OM chain adding
    effective loss.

    Args:
        f: Frequency points in Hz.
        mw_capacitance: MW resonator total capacitance in Farads.
        mw_inductance: MW resonator inductance in Henries.
        coupling_capacitance: Feedline coupling capacitance in Farads.
        g_pe: Piezoelectric coupling rate in Hz.
        g_om: Pump-enhanced optomechanical coupling rate in Hz.
        kappa_mech: Mechanical mode linewidth in Hz.
        kappa_opt: Optical mode linewidth in Hz.
        omega_mech: Mechanical mode frequency in Hz.

    Returns:
        sax.SDict: S-parameters with ports ``o1`` and ``o2``.
    """
    f_arr = jnp.asarray(f)

    # Effective loss rate induced by the piezo-OM chain on the MW mode
    # Γ_pe = 4 g_pe^2 / κ_mech  (piezo cooperativity * κ_mech)
    gamma_pe = 4.0 * g_pe**2 / kappa_mech
    # Γ_om = 4 G_om^2 / κ_opt  (OM cooperativity * κ_opt)
    gamma_om = 4.0 * g_om**2 / kappa_opt
    # Total effective rate on MW mode from cascaded coupling
    gamma_eff = gamma_pe * gamma_om / (gamma_pe + gamma_om + kappa_mech)

    # Build MW circuit
    instances: dict[str, sax.SType] = {
        "coupling_cap": capacitor(f=f_arr, capacitance=coupling_capacitance),
        "mw_resonator": lc_resonator(
            f=f_arr,
            capacitance=mw_capacitance,
            inductance=mw_inductance,
            grounded=True,
        ),
        "tee": tee(f=f_arr),
    }

    connections = {
        "tee,o2": "coupling_cap,o1",
        "coupling_cap,o2": "mw_resonator,o1",
    }

    ports = {
        "o1": "tee,o1",
        "o2": "tee,o3",
    }

    s_dict = sax.evaluate_circuit_fg((connections, ports), instances)

    # Apply frequency-dependent loss from piezo-OM coupling
    omega = 2.0 * π * f_arr
    omega_m_res = 2.0 * π * omega_mech
    # Lorentzian profile around mechanical frequency
    loss_factor = gamma_eff**2 / (
        (omega - omega_m_res) ** 2 + (gamma_eff / 2.0) ** 2
    )
    attenuation = jnp.exp(-loss_factor / 2.0)
    attenuation = attenuation.reshape(jnp.asarray(f).shape)

    return {
        ("o1", "o1"): s_dict["o1", "o1"] * attenuation,
        ("o1", "o2"): s_dict["o1", "o2"] * attenuation,
        ("o2", "o1"): s_dict["o2", "o1"] * attenuation,
        ("o2", "o2"): s_dict["o2", "o2"] * attenuation,
    }


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from qpdk import PDK

    PDK.activate()

    f = jnp.linspace(1e9, 10e9, 2001)

    # --- EO transducer ---
    S_eo = electro_optic_transducer(f=f, eo_coupling_rate=5e6, kappa_opt=50e6)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(f / 1e9, 20 * jnp.log10(jnp.abs(S_eo["o1", "o1"])), label="|S11| EO")
    ax1.plot(f / 1e9, 20 * jnp.log10(jnp.abs(S_eo["o1", "o2"])), label="|S21| EO")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.legend()
    ax1.grid(True)
    ax1.set_title("Electro-Optic Transducer S-parameters")

    # --- Conversion efficiency vs cooperativity ---
    C = jnp.logspace(-2, 2, 200)
    eta = transduction_efficiency(C, eta_ext_mw=0.8, eta_ext_opt=0.8)
    ax2.semilogx(C, eta)
    ax2.set_xlabel("Cooperativity C")
    ax2.set_ylabel("Conversion efficiency η")
    ax2.set_title("Transduction Efficiency vs Cooperativity")
    ax2.grid(True)
    ax2.axvline(1.0, color="r", linestyle=":", label="C = 1 (impedance matched)")
    ax2.legend()
    plt.tight_layout()
    plt.show()
