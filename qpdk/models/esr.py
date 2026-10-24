"""Model of a planar LC resonator for inductive-detection electron spin resonance."""

import jax
import jax.numpy as jnp
import sax
from gdsfactory.typings import CrossSectionSpec
from jax.typing import ArrayLike

from qpdk.models.capacitor import interdigital_capacitor_capacitance_analytical
from qpdk.models.constants import DEFAULT_FREQUENCY, μ_0, π
from qpdk.models.cpw import cpw_ep_r_from_cross_section, cpw_z0_from_cross_section
from qpdk.models.generic import shunt_admittance
from qpdk.models.inductor import self_inductance_strip

__all__ = [
    "esr_lc_resonator",
    "esr_lc_resonator_parameters",
    "esr_mode_volume",
    "thin_strip_magnetic_field",
]


def thin_strip_magnetic_field(
    current: ArrayLike,
    width: ArrayLike,
    height: ArrayLike,
    x: ArrayLike = 0.0,
) -> tuple[jax.Array, jax.Array]:
    r"""Magnetic field of an infinite thin strip carrying a uniform current.

    The strip lies in the plane :math:`y = 0` between :math:`x = \pm w/2` and
    carries the current :math:`I` along :math:`+z`, spread uniformly over its
    width. Integrating the field of a line current, :math:`\mu_0 I/(2\pi r)`,
    across the strip with the Biot–Savart law
    :cite:`griffithsIntroductionElectrodynamics2017` gives, for :math:`y > 0`,

    .. math::

        B_x &= -\frac{\mu_0 I}{2\pi w}\left[
            \arctan\frac{x + w/2}{y} - \arctan\frac{x - w/2}{y}\right] \\
        B_y &= \frac{\mu_0 I}{4\pi w}
            \ln\frac{(x + w/2)^2 + y^2}{(x - w/2)^2 + y^2}

    Above the centre of the strip, :math:`B_x = -\mu_0 I \arctan(w/2y)/(\pi w)`,
    which tends to :math:`-\mu_0 I/(2w)` at the surface and to the line-current
    field far away. By mirror symmetry the field at depth :math:`y` below the
    strip, where the spins of an ESR sample sit, has :math:`B_x` reversed and
    :math:`B_y` unchanged.

    Uniform-current approximation: a superconducting strip carries most of its
    current near the edges :cite:`bienfaitReachingQuantumLimit2016`, so close
    to the strip the real field is larger near the edges and smaller above the
    centre than this formula gives. The strip thickness is neglected.

    Args:
        current: Current :math:`I` in A.
        width: Strip width :math:`w` in µm.
        height: Position :math:`y` relative to the strip plane in µm, nonzero.
            Negative values are below the strip.
        x: Lateral position measured from the strip centre in µm.

    Returns:
        Tuple ``(B_x, B_y)`` in T.
    """
    w = jnp.asarray(width) * 1e-6
    y_signed = jnp.asarray(height) * 1e-6
    # Evaluate at |y| and mirror B_x, so both half-spaces use the same branch.
    y = jnp.abs(y_signed)
    x_m = jnp.asarray(x) * 1e-6
    prefactor = μ_0 * jnp.asarray(current) / (2 * π * w)
    b_x = (
        -jnp.sign(y_signed)
        * prefactor
        * (jnp.arctan2(x_m + w / 2, y) - jnp.arctan2(x_m - w / 2, y))
    )
    b_y = (
        prefactor
        / 2
        * jnp.log(((x_m + w / 2) ** 2 + y**2) / ((x_m - w / 2) ** 2 + y**2))
    )
    return b_x, b_y


def esr_mode_volume(wire_width: ArrayLike, inductance: ArrayLike) -> jax.Array:
    r"""Magnetic mode volume of an LC resonator with a thin-strip inductor.

    The magnetic mode volume is the field energy over the energy density at the
    spins, :math:`V_\text{m} = \int |\mathbf{B}|^2\,\mathrm{d}V / |\delta B_1|^2`.
    The field energy of a current :math:`\delta i` in the inductance :math:`L` is
    :math:`\int |\mathbf{B}|^2\,\mathrm{d}V = \mu_0 L\,\delta i^2`. Taking
    :math:`\delta B_1` at the wire surface from
    :func:`thin_strip_magnetic_field`, :math:`\delta B_1 = \mu_0 \delta i/(2w)`,
    gives

    .. math::

        V_\text{m} = \frac{\mu_0 L\,\delta i^2}{\delta B_1^2} = \frac{4 w^2 L}{\mu_0},

    independent of :math:`\delta i`. With the single-photon current
    :math:`\delta i = \omega_0\sqrt{\hbar/2Z_\text{c}}` of
    :cite:`ranjanElectronSpinResonance2020` this is the volume in
    :math:`\delta B_1 = \sqrt{\mu_0 \hbar \omega_0 / 2 V_\text{m}}`.

    Neither design reference states a closed-form mode volume; both quote
    volumes from EM field simulations
    :cite:`bienfaitReachingQuantumLimit2016,ranjanElectronSpinResonance2020`.
    With their own :math:`L = Z_\text{c}/\omega_0` this estimate gives 0.077 nL
    against "~0.02 nl" and 10.5 fL against "~6 fL": the same order of
    magnitude, not a reproduction. The derivation counts :math:`L` as
    magnetic-field energy, so pass the geometric inductance: kinetic inductance
    stores energy in the electrons, not the field. The comparison with the
    papers' :math:`Z_\text{c}/\omega_0` therefore overstates the volume where
    kinetic inductance contributes, and for Ranjan's 10 µm wire most of that
    :math:`L` is stray inductance of the capacitor rather than of the wire.
    The uniform-current approximation of :func:`thin_strip_magnetic_field`
    applies.

    Args:
        wire_width: Wire width :math:`w` in µm.
        inductance: Inductance :math:`L` in H whose field energy is counted.

    Returns:
        Mode volume in µm³ (1 µm³ = 1 fL).
    """
    return 4 * jnp.asarray(wire_width) ** 2 * jnp.asarray(inductance) / μ_0 * 1e6


def esr_lc_resonator_parameters(
    wire_width: float = 5.0,
    wire_length: float = 730.0,
    fingers: int = 12,
    finger_length: float = 800.0,
    finger_gap: float = 50.0,
    finger_thickness: float = 50.0,
    film_thickness: float = 0.05,
    sheet_inductance: float = 0.4e-12,
    coupling_capacitance: float = 4e-15,
    cross_section: CrossSectionSpec = "cpw",
) -> dict[str, jax.Array | float]:
    r"""Circuit parameters of :func:`qpdk.cells.esr_lc_resonator`.

    The resonator is a linear LC to ground, coupled to the feedline through
    :math:`C_\text{c}`:

    .. math::

        L &= L_\text{geo} + L_\square N_\square \\
        \omega_\text{r} &= \frac{1}{\sqrt{L (C + C_\text{c})}} \\
        Q_\text{c} &= \frac{2 (C + C_\text{c})}{\omega_\text{r} C_\text{c}^2 Z_0}

    :math:`L_\text{geo}` is the self-inductance of the wire as a rectangular
    strip from :func:`~qpdk.models.inductor.self_inductance_strip`,
    :math:`N_\square = \ell/w` its square count, and :math:`C` the interdigital
    capacitance from
    :func:`~qpdk.models.capacitor.interdigital_capacitor_capacitance_analytical`.
    :math:`Q_\text{c}` follows from the feedline loading the resonator with
    :math:`Z_0/2` through :math:`C_\text{c}`, valid for
    :math:`\omega_\text{r} C_\text{c} Z_0 \ll 1`.

    For the wire of :cite:`bienfaitReachingQuantumLimit2016` (5 µm × 730 µm,
    50 nm film) :math:`L_\text{geo} = 0.90` nH, against the
    :math:`Z_0/\omega_0 = 0.97` nH that the paper's 44 Ω and 7.26 GHz imply.
    The paper does not give the finger length; the default 800 µm puts
    :math:`f_\text{r}` near its 7.26 GHz design value.

    Assumptions:

    * The resonator is linear and lossless: no kinetic-inductance
      nonlinearity, and the internal quality factor is infinite.
    * The stray inductance of the capacitor electrodes and leads is not
      included. For the bowtie capacitor of
      :cite:`ranjanElectronSpinResonance2020`, :math:`Z_\text{c} \approx 15` Ω at
      7.25 GHz implies :math:`L \approx 0.33` nH, far above the ~11 pH geometric
      inductance of its 10 µm wire, so this model underestimates :math:`L` for
      such short wires.
    * The lower electrode is grounded and the resonator is coupled to an
      on-chip CPW. Both references instead float the resonator in a 3D cavity
      driven by antennas.
    * Capacitance from the electrodes to the side ground planes is neglected.
    * :math:`C_\text{c}` is a parameter, not derived from ``coupling_gap``. The
      4 fF default is chosen to give :math:`Q_\text{c} \approx 3 \times 10^4`,
      the external quality factor of :cite:`ranjanElectronSpinResonance2020`;
      it does not describe the default cell. That cell runs its 1.15 mm bus
      bar 50 µm from the feedline, which a coplanar-strip estimate puts at
      order 0.1 pF, giving :math:`Q_\text{c}` of order 10–100. Set
      :math:`C_\text{c}` from a capacitance extraction of the layout.
    * The 0.4 pH/□ sheet inductance is the PDK default for the base metal; the
      references do not state one for their 50 nm Al films.

    Args:
        wire_width: Wire width in µm.
        wire_length: Wire length in µm.
        fingers: Number of capacitor fingers.
        finger_length: Length of each capacitor finger in µm.
        finger_gap: Gap between adjacent capacitor fingers in µm.
        finger_thickness: Width of each capacitor finger in µm.
        film_thickness: Film thickness in µm, used for :math:`L_\text{geo}`.
        sheet_inductance: Kinetic sheet inductance :math:`L_\square` in H/□.
        coupling_capacitance: Coupling capacitance :math:`C_\text{c}` in F.
        cross_section: Feedline cross-section, used for the substrate
            permittivity and :math:`Z_0`.

    Returns:
        Dictionary with ``wire_squares``, ``geometric_inductance``,
        ``kinetic_inductance``, ``inductance`` (H), ``capacitance`` (F),
        ``coupling_capacitance`` (F), ``z0`` (Ω), ``resonance_frequency`` (Hz)
        and ``coupling_quality_factor``.
    """
    wire_squares = wire_length / wire_width
    geometric = self_inductance_strip(
        wire_length * 1e-6, wire_width * 1e-6, film_thickness * 1e-6
    )
    kinetic = sheet_inductance * wire_squares
    inductance = geometric + kinetic
    ep_r = cpw_ep_r_from_cross_section(cross_section)
    capacitance = interdigital_capacitor_capacitance_analytical(
        fingers, finger_length, finger_gap, finger_thickness, ep_r
    )
    z0 = jnp.real(cpw_z0_from_cross_section(cross_section))
    c_total = capacitance + coupling_capacitance
    omega_r = 1 / jnp.sqrt(inductance * c_total)
    q_c = 2 * c_total / (omega_r * coupling_capacitance**2 * z0)
    return {
        "wire_squares": wire_squares,
        "geometric_inductance": geometric,
        "kinetic_inductance": kinetic,
        "inductance": inductance,
        "capacitance": capacitance,
        "coupling_capacitance": coupling_capacitance,
        "z0": z0,
        "resonance_frequency": omega_r / (2 * π),
        "coupling_quality_factor": q_c,
    }


def esr_lc_resonator(
    *,
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    wire_width: float = 5.0,
    wire_length: float = 730.0,
    fingers: int = 12,
    finger_length: float = 800.0,
    finger_gap: float = 50.0,
    finger_thickness: float = 50.0,
    film_thickness: float = 0.05,
    sheet_inductance: float = 0.4e-12,
    coupling_capacitance: float = 4e-15,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SDict:
    r"""SAX model of :func:`qpdk.cells.esr_lc_resonator`.

    The resonator branch, :math:`C_\text{c}` in series with :math:`L \parallel C`,
    shunts the feedline at a single reference plane:

    .. math::

        Y(\omega) = \left(\frac{1}{\text{j}\omega C_\text{c}}
            + \frac{1}{\text{j}\omega C + 1/(\text{j}\omega L)}\right)^{-1}

    and the two-port follows from
    :func:`~qpdk.models.generic.shunt_admittance`
    :cite:`m.pozarMicrowaveEngineering2012`. :math:`|S_{21}|` vanishes at
    :math:`\omega_\text{r} = 1/\sqrt{L (C + C_\text{c})}`. Circuit values and
    assumptions are those of
    :func:`~qpdk.models.esr.esr_lc_resonator_parameters`; the feedline itself
    is treated as lossless and of zero length.

    Design references are
    :cite:`bienfaitReachingQuantumLimit2016,ranjanElectronSpinResonance2020`.

    Args:
        f: Array of frequency points in Hz.
        wire_width: Wire width in µm.
        wire_length: Wire length in µm.
        fingers: Number of capacitor fingers.
        finger_length: Length of each capacitor finger in µm.
        finger_gap: Gap between adjacent capacitor fingers in µm.
        finger_thickness: Width of each capacitor finger in µm.
        film_thickness: Film thickness in µm.
        sheet_inductance: Kinetic sheet inductance in H/□.
        coupling_capacitance: Coupling capacitance in F.
        cross_section: Feedline cross-section.

    Returns:
        sax.SDict: S-parameters with ports ``o1`` and ``o2``.
    """
    p = esr_lc_resonator_parameters(
        wire_width=wire_width,
        wire_length=wire_length,
        fingers=fingers,
        finger_length=finger_length,
        finger_gap=finger_gap,
        finger_thickness=finger_thickness,
        film_thickness=film_thickness,
        sheet_inductance=sheet_inductance,
        coupling_capacitance=coupling_capacitance,
        cross_section=cross_section,
    )
    jw = 1j * 2 * π * jnp.asarray(f)
    z_lc = 1 / (jw * p["capacitance"] + 1 / (jw * p["inductance"]))
    y = 1 / (1 / (jw * p["coupling_capacitance"]) + z_lc)
    # Normalise to the feedline impedance, which is traced under jax.jit.
    return shunt_admittance(f=f, y=y * p["z0"], z0=1.0)
