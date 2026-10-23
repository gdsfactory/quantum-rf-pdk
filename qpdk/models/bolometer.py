r"""Normal-metal hot-electron nanobolometer and calorimeter models.

The absorber is a normal-metal (N) nanowire between superconducting (S) leads.
Absorbed power heats its electrons, which relax to the phonon bath through
electron-phonon coupling. The models here describe that thermal response and
the RF load the absorber presents to its input line.

**Hot-electron heat balance** :cite:`giazottoOpportunitiesMesoscopicsThermometry2006`.
An electron system of volume :math:`V` at temperature :math:`T` has the heat
capacity and electron-phonon conductance

.. math::

    C = \gamma V T, \qquad
    G_\text{e-ph} = \frac{\partial}{\partial T}\,\Sigma V (T^5 - T_\text{ph}^5)
                  = 5 \Sigma V T^4,

so a single reservoir relaxes with
:math:`\tau = C / G_\text{e-ph} = \gamma / (5 \Sigma T^3)`, independent of
:math:`V`. Heat flows along the normal metal as given by the Wiedemann-Franz law,
:math:`G = \mathcal{L}_0 T / R`. Andreev reflection blocks subgap heat
transport into clean NS contacts, leaving only a quasiparticle leak suppressed
roughly as :math:`e^{-\Delta / k_\text{B} T}` :cite:`bardeenTheoryThermalConductivity1959`.

**Two-reservoir model.** Measured relaxation traces of both NIS calorimeters
:cite:`gasparinettiFastElectronThermometry2015` and SNS nanobolometers
:cite:`gunyhoZeptojouleCalorimetry2026` fit a double exponential, with no
established physical origin for the second time constant. The layout-resolved
model asks what the drawn geometry alone predicts. The second reservoir is the
normal metal under the superconducting leads and beyond them. It is coupled to
the absorber through the Wiedemann-Franz conductance of the connecting
segment, and to the bath through its own electron-phonon coupling. The
linearised network

.. math::

    \mathbf{C} \, \dot{\boldsymbol{\theta}} = -\mathbf{K} \boldsymbol{\theta},
    \qquad
    \mathbf{K} = \begin{pmatrix}
        G_1 + G_{12} & -G_{12} \\ -G_{12} & G_2 + G_{12}
    \end{pmatrix}

gives, after a heat pulse into the absorber, the response at the absorber
:math:`\theta_1(t) \propto a_1 e^{-t/\tau_1} + a_2 e^{-t/\tau_2}`. The
two-reservoir equations are the linearised form of the model used in
:cite:`wangDynamicThermalRelaxation2026` (Eq. 2). An optional third reservoir
coupled only to the absorber electrons stands in for an internal material heat
capacity, as proposed by :cite:`goveniusDetectionZeptojouleMicrowave2016` and
:cite:`wangDynamicThermalRelaxation2026`. It is off by default.

Proximity suppression of the heat flow under the leads is not modelled. Phonon
temperature gradients and the tunnel contacts of the thermometer are also
neglected.

Default material parameters are those of AuPd nanowires in
:cite:`goveniusDetectionZeptojouleMicrowave2016` and
:cite:`girardCryogenicSensorEnabling2023`. The gap of Al is taken from
:cite:`gasparinettiFastElectronThermometry2015`.
"""

import jax
import jax.numpy as jnp
import sax
from jax.typing import ArrayLike

from qpdk.models.constants import DEFAULT_FREQUENCY, e, k_B, π
from qpdk.models.jit import jit

__all__ = [
    "AL_GAP",
    "AUPD_ELECTRON_PHONON_COUPLING",
    "AUPD_HEAT_CAPACITY_COEFFICIENT",
    "AUPD_RESISTIVITY",
    "CU_RESISTIVITY",
    "LORENZ_NUMBER",
    "andreev_heat_suppression",
    "electron_heat_capacity",
    "electron_phonon_conductance",
    "nanobolometer",
    "nanobolometer_thermal_modes",
    "nanocalorimeter",
    "thermal_relaxation_modes",
    "two_reservoir_relaxation",
    "wiedemann_franz_conductance",
]

LORENZ_NUMBER: float = π**2 * k_B**2 / (3 * e**2)
r"""Sommerfeld value of the Lorenz number :math:`\mathcal{L}_0` in W Ω K⁻²."""

AUPD_HEAT_CAPACITY_COEFFICIENT: float = 580.0
r"""Electronic heat-capacity coefficient :math:`\gamma` of AuPd in J m⁻³ K⁻².

Derived from :math:`\gamma V_0 = 8\,\text{aJ/K}^2` with
:math:`V_0 \approx (240\,\text{nm})^3` in :cite:`goveniusDetectionZeptojouleMicrowave2016`.
"""

AUPD_ELECTRON_PHONON_COUPLING: float = 3e9
r"""Electron-phonon coupling constant :math:`\Sigma` of AuPd in W m⁻³ K⁻⁵.

From :cite:`goveniusDetectionZeptojouleMicrowave2016`.
"""

AUPD_RESISTIVITY: float = 1e-7
r"""Low-temperature resistivity of AuPd in Ω m.

Rounded from the :math:`24.4\,\Omega` per :math:`1\,\text{µm}` heater of
:cite:`girardCryogenicSensorEnabling2023`, which gives :math:`9.9 \times 10^{-8}`
for the :math:`135\,\text{nm} \times 30\,\text{nm}` wire cross-section of the
default cell.
"""

CU_RESISTIVITY: float = 4.5e-8
r"""Low-temperature resistivity of the thin Cu calorimeter wire in Ω m.

Upper bound from the :math:`360\,\Omega` wire resistance of
:cite:`gasparinettiFastElectronThermometry2015` and its
:math:`20\,\text{µm} \times 100\,\text{nm} \times 25\,\text{nm}` geometry.
"""

AL_GAP: float = 213e-6 * e
r"""Superconducting gap :math:`\Delta` of the Al leads in J.

From :cite:`gasparinettiFastElectronThermometry2015`.
"""

_UM = 1e-6
_UM3 = 1e-18
_LI2_TERMS = 64


@jit(inline=True)
def electron_heat_capacity(
    volume: ArrayLike,
    temperature: ArrayLike,
    heat_capacity_coefficient: ArrayLike = AUPD_HEAT_CAPACITY_COEFFICIENT,
) -> jax.Array:
    r"""Heat capacity :math:`C = \gamma V T` of a degenerate electron gas.

    See :cite:`giazottoOpportunitiesMesoscopicsThermometry2006`.

    Args:
        volume: Electron-system volume :math:`V` in m³.
        temperature: Electron temperature :math:`T` in K.
        heat_capacity_coefficient: Sommerfeld coefficient :math:`\gamma` in J m⁻³ K⁻².

    Returns:
        Heat capacity in J/K.
    """
    return jnp.asarray(heat_capacity_coefficient) * volume * temperature


@jit(inline=True)
def electron_phonon_conductance(
    volume: ArrayLike,
    temperature: ArrayLike,
    electron_phonon_coupling: ArrayLike = AUPD_ELECTRON_PHONON_COUPLING,
) -> jax.Array:
    r"""Linearised electron-phonon thermal conductance :math:`G = 5 \Sigma V T^4`.

    The derivative of :math:`P = \Sigma V (T^5 - T_\text{ph}^5)` with respect to
    the electron temperature :cite:`giazottoOpportunitiesMesoscopicsThermometry2006`.

    Args:
        volume: Electron-system volume :math:`V` in m³.
        temperature: Electron temperature :math:`T` in K.
        electron_phonon_coupling: Coupling constant :math:`\Sigma` in W m⁻³ K⁻⁵.

    Returns:
        Thermal conductance in W/K.
    """
    return 5 * jnp.asarray(electron_phonon_coupling) * volume * temperature**4


@jit(inline=True)
def wiedemann_franz_conductance(
    resistance: ArrayLike,
    temperature: ArrayLike,
) -> jax.Array:
    r"""Electronic thermal conductance :math:`G = \mathcal{L}_0 T / R` of a normal metal.

    See :cite:`giazottoOpportunitiesMesoscopicsThermometry2006`.

    Args:
        resistance: Electrical resistance :math:`R` in Ω.
        temperature: Temperature :math:`T` in K.

    Returns:
        Thermal conductance in W/K.
    """
    return LORENZ_NUMBER * jnp.asarray(temperature) / resistance


@jit(inline=True)
def andreev_heat_suppression(
    temperature: ArrayLike,
    superconducting_gap: ArrayLike = AL_GAP,
) -> jax.Array:
    r"""Ratio :math:`\kappa_\text{s} / \kappa_\text{n}` of quasiparticle heat conduction.

    Bardeen-Rickayzen-Tewordt result for impurity scattering
    :cite:`bardeenTheoryThermalConductivity1959`:

    .. math::

        \frac{\kappa_\text{s}}{\kappa_\text{n}} =
        \frac{2 F_1(-y) + 2 y \ln(1 + e^{-y}) + y^2 / (1 + e^{y})}{2 F_1(0)},
        \qquad y = \frac{\Delta}{k_\text{B} T},

    with :math:`F_1(-y) = -\operatorname{Li}_2(-e^{-y})` and
    :math:`F_1(0) = \pi^2 / 12`. Subgap electrons are Andreev reflected, so the
    ratio falls roughly as :math:`e^{-y}`. The gap is taken at its
    zero-temperature value, valid for :math:`T \ll T_\text{c}`.

    Args:
        temperature: Temperature :math:`T` in K.
        superconducting_gap: Gap :math:`\Delta` in J.

    Returns:
        Dimensionless suppression factor in :math:`[0, 1]`.
    """
    y = jnp.asarray(superconducting_gap) / (k_B * jnp.asarray(temperature))
    k = jnp.arange(1, _LI2_TERMS + 1)
    u = jnp.exp(-y)[..., None]
    f1 = jnp.sum((-1.0) ** (k + 1) * u**k / k**2, axis=-1)
    numerator = 2 * f1 + 2 * y * jnp.log1p(jnp.exp(-y)) + y**2 * jax.nn.sigmoid(-y)
    return numerator / (π**2 / 6)


@jit(inline=True)
def two_reservoir_relaxation(
    c1: ArrayLike,
    c2: ArrayLike,
    g1: ArrayLike,
    g2: ArrayLike,
    g12: ArrayLike,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""Analytic relaxation modes of two thermally coupled reservoirs.

    Solves :math:`\mathbf{C} \dot{\boldsymbol{\theta}} = -\mathbf{K}
    \boldsymbol{\theta}` for a heat pulse into reservoir 1. With
    :math:`p = (G_1 + G_{12}) / C_1`, :math:`r = (G_2 + G_{12}) / C_2` and
    :math:`q^2 = G_{12}^2 / (C_1 C_2)`, the rates are

    .. math::

        \lambda_\pm = \frac{p + r}{2} \pm D, \qquad
        D = \sqrt{\left(\frac{p - r}{2}\right)^2 + q^2},

    and the temperature of reservoir 1 decays as
    :math:`a_1 e^{-\lambda_+ t} + a_2 e^{-\lambda_- t}` with
    :math:`a_1 = (\lambda_+ - r) / 2D` and :math:`a_2 = (r - \lambda_-) / 2D`,
    normalised so that :math:`a_1 + a_2 = 1`. The slow rate uses
    :math:`\lambda_- = \det \mathbf{K} / (C_1 C_2 \lambda_+)` to avoid cancellation.

    Args:
        c1: Heat capacity of the heated reservoir in J/K.
        c2: Heat capacity of the second reservoir in J/K.
        g1: Conductance from reservoir 1 to the bath in W/K.
        g2: Conductance from reservoir 2 to the bath in W/K.
        g12: Conductance between the reservoirs in W/K.

    Returns:
        ``(tau_1, tau_2, a_1, a_2)``: fast and slow time constants in s and
        their normalised amplitudes at reservoir 1.
    """
    c1, c2, g1, g2, g12 = (jnp.asarray(x) for x in (c1, c2, g1, g2, g12))
    p = (g1 + g12) / c1
    r = (g2 + g12) / c2
    q2 = g12**2 / (c1 * c2)
    half = (p - r) / 2
    d = jnp.sqrt(half**2 + q2)
    d = jnp.maximum(d, jnp.finfo(d.dtype).tiny)
    lambda_fast = (p + r) / 2 + d
    lambda_slow = (g1 * g2 + g1 * g12 + g2 * g12) / (c1 * c2 * lambda_fast)
    # a_1 = (half + d) / 2d, rewritten without cancellation when half < 0
    a_fast = jnp.where(
        half >= 0,
        (half + d) / (2 * d),
        q2 / (2 * d * (d - half)),
    )
    a_slow = jnp.where(
        half >= 0,
        q2 / (2 * d * (d + half)),
        (d - half) / (2 * d),
    )
    # Uncoupled reservoirs with equal rates: one degenerate mode at reservoir 1
    degenerate = (q2 == 0) & (half == 0)
    a_fast = jnp.where(degenerate, 1.0, a_fast)
    a_slow = jnp.where(degenerate, 0.0, a_slow)
    return 1 / lambda_fast, 1 / lambda_slow, a_fast, a_slow


@jit(inline=True)
def thermal_relaxation_modes(
    heat_capacities: ArrayLike,
    conductance_matrix: ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    r"""Relaxation modes of a thermal network after a heat pulse into node 0.

    Diagonalises the symmetric matrix
    :math:`\mathbf{A} = \mathbf{C}^{-1/2} \mathbf{K} \mathbf{C}^{-1/2}`. Node 0
    then decays as :math:`\sum_k a_k e^{-t / \tau_k}` with
    :math:`\tau_k = 1 / \lambda_k` and :math:`a_k = v_{0k}^2`, where
    :math:`\sum_k a_k = 1`.

    Args:
        heat_capacities: Heat capacities :math:`C_i` in J/K, shape ``(n,)``.
        conductance_matrix: Symmetric conductance matrix :math:`\mathbf{K}` in W/K,
            shape ``(n, n)``, including each node's link to the bath on the diagonal.

    Returns:
        ``(taus, amplitudes)``, both of shape ``(n,)`` and ordered from the
        fastest to the slowest mode.
    """
    s = 1 / jnp.sqrt(jnp.asarray(heat_capacities))
    a = s[:, None] * jnp.asarray(conductance_matrix) * s[None, :]
    eigenvalues, eigenvectors = jnp.linalg.eigh(a)
    order = jnp.argsort(eigenvalues)[::-1]
    return 1 / eigenvalues[order], eigenvectors[0, order] ** 2


def nanobolometer_thermal_modes(
    absorber_length: float = 1.0,
    absorber_width: float = 0.135,
    normal_metal_thickness: float = 0.03,
    contact_overlap: float = 0.3,
    extension_length: float = 0.2,
    temperature: float = 20e-3,
    heat_capacity_coefficient: float = AUPD_HEAT_CAPACITY_COEFFICIENT,
    electron_phonon_coupling: float = AUPD_ELECTRON_PHONON_COUPLING,
    resistivity: float = AUPD_RESISTIVITY,
    superconducting_gap: float = AL_GAP,
    lead_resistance: float = 1.0,
    material_heat_capacity: float | None = None,
    material_conductance: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    r"""Thermal relaxation modes predicted from the nanobolometer layout.

    Uses the geometry of :func:`~qpdk.cells.nanobolometer` (lengths in µm):

    - Reservoir 1 is the absorber between the two leads, of volume
      :math:`V_1 = l\,w\,t`.
    - Reservoir 2 lumps the normal metal under both leads and beyond them,
      :math:`V_2 = 2 (l_\text{o} + l_\text{e})\,w\,t`.
    - The two are linked by the Wiedemann-Franz conductance of two parallel
      segments of length :math:`(l_\text{o} + l_\text{e}) / 2`. That is the
      distance from the absorber edge to the centroid of each side's reservoir.
    - Each reservoir couples to the bath through :math:`5 \Sigma V T^4`.
      Reservoir 2 also leaks through its two Andreev-blocked NS contacts, with
      conductance :math:`(\kappa_\text{s} / \kappa_\text{n}) \mathcal{L}_0 T /
      R_\text{lead}`.

    With ``material_heat_capacity`` set, a third reservoir coupled only to the
    absorber electrons through ``material_conductance`` is added. Its modes are
    found numerically with :func:`thermal_relaxation_modes`.

    Args:
        absorber_length: Absorber length :math:`l` between the leads in µm.
        absorber_width: Normal-metal wire width :math:`w` in µm.
        normal_metal_thickness: Normal-metal film thickness :math:`t` in µm.
        contact_overlap: Length :math:`l_\text{o}` of normal metal under each lead in µm.
        extension_length: Length :math:`l_\text{e}` of normal metal beyond each lead in µm.
        temperature: Electron temperature :math:`T` of the operating point in K.
        heat_capacity_coefficient: Sommerfeld coefficient :math:`\gamma` in J m⁻³ K⁻².
        electron_phonon_coupling: Coupling constant :math:`\Sigma` in W m⁻³ K⁻⁵.
        resistivity: Normal-metal resistivity in Ω m.
        superconducting_gap: Gap :math:`\Delta` of the leads in J.
        lead_resistance: Normal-state resistance :math:`R_\text{lead}` of each
            superconducting lead in Ω, used for the Andreev-blocked leak.
        material_heat_capacity: Heat capacity of an optional internal material
            reservoir in J/K. ``None`` disables it.
        material_conductance: Conductance between the absorber electrons and the
            material reservoir in W/K.

    Returns:
        ``(taus, amplitudes)`` from the fastest to the slowest mode: two modes, or
        three with the material reservoir. The amplitudes are normalised to sum to
        one at the absorber.

    Raises:
        ValueError: If ``contact_overlap + extension_length`` is not positive.
    """
    side_length = contact_overlap + extension_length
    # Only concrete inputs can be checked; a traced zero still gives NaN
    if isinstance(side_length, int | float) and side_length <= 0:
        msg = (
            f"contact_overlap + extension_length = {side_length} µm must be "
            "positive; the second reservoir needs a volume."
        )
        raise ValueError(msg)
    area = absorber_width * normal_metal_thickness * _UM**2
    v1 = absorber_length * absorber_width * normal_metal_thickness * _UM3
    v2 = 2 * side_length * absorber_width * normal_metal_thickness * _UM3
    c1 = electron_heat_capacity(v1, temperature, heat_capacity_coefficient)
    c2 = electron_heat_capacity(v2, temperature, heat_capacity_coefficient)
    g1 = electron_phonon_conductance(v1, temperature, electron_phonon_coupling)
    g_ns = andreev_heat_suppression(
        temperature, superconducting_gap
    ) * wiedemann_franz_conductance(lead_resistance, temperature)
    g2 = electron_phonon_conductance(v2, temperature, electron_phonon_coupling) + (
        2 * g_ns
    )
    link_resistance = resistivity * (side_length / 2 * _UM) / area
    g12 = 2 * wiedemann_franz_conductance(link_resistance, temperature)

    if material_heat_capacity is None:
        tau_1, tau_2, a_1, a_2 = two_reservoir_relaxation(c1, c2, g1, g2, g12)
        return jnp.stack([tau_1, tau_2]), jnp.stack([a_1, a_2])

    g13 = jnp.asarray(material_conductance)
    conductance_matrix = jnp.array([
        [g1 + g12 + g13, -g12, -g13],
        [-g12, g2 + g12, 0.0],
        [-g13, 0.0, g13],
    ])
    heat_capacities = jnp.stack([c1, c2, jnp.asarray(material_heat_capacity)])
    return thermal_relaxation_modes(heat_capacities, conductance_matrix)


@jit
def nanobolometer(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    absorber_length: sax.Float = 1.0,
    absorber_width: sax.Float = 0.135,
    normal_metal_thickness: sax.Float = 0.03,
    resistivity: sax.Float = AUPD_RESISTIVITY,
    z0: sax.Float = 50.0,
) -> sax.SDict:
    r"""One-port SAX model of the nanobolometer RF input.

    The absorber is a lumped resistor
    :math:`R = \rho\, l / (w\, t)` from the input line to ground, so
    :math:`S_{11} = (R - Z_0) / (R + Z_0)`. The superconducting leads are
    treated as lossless shorts, and the reactance of the nanowire is
    neglected. That holds while its kinetic and geometric inductance is small
    compared with :math:`R / \omega`. Impedance matching of a direct-input
    nanobolometer to its line follows :cite:`kokkoniemiNanobolometerUltralowNoise2019`.

    Args:
        f: Array of frequency points in Hz.
        absorber_length: Absorber length :math:`l` between the leads in µm.
        absorber_width: Normal-metal wire width :math:`w` in µm.
        normal_metal_thickness: Normal-metal film thickness :math:`t` in µm.
        resistivity: Normal-metal resistivity :math:`\rho` in Ω m.
        z0: Reference impedance :math:`Z_0` in Ω.

    Returns:
        sax.SDict: S-parameters dictionary with port ``o1``.
    """
    f = jnp.asarray(f)
    resistance = (
        resistivity * absorber_length / (absorber_width * normal_metal_thickness * _UM)
    )
    s11 = (resistance - z0) / (resistance + z0) * jnp.ones_like(f)
    sdict: sax.SDict = {("o1", "o1"): s11.astype(complex)}
    return sdict


@jit
def nanocalorimeter(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    absorber_length: sax.Float = 20.0,
    absorber_width: sax.Float = 0.1,
    normal_metal_thickness: sax.Float = 0.025,
    resistivity: sax.Float = CU_RESISTIVITY,
    z0: sax.Float = 50.0,
) -> sax.SDict:
    r"""One-port SAX model of the nanocalorimeter RF input.

    The :func:`nanobolometer` model with the Cu wire of
    :cite:`gasparinettiFastElectronThermometry2015` as defaults.

    Args:
        f: Array of frequency points in Hz.
        absorber_length: Absorber length :math:`l` between the leads in µm.
        absorber_width: Normal-metal wire width :math:`w` in µm.
        normal_metal_thickness: Normal-metal film thickness :math:`t` in µm.
        resistivity: Normal-metal resistivity :math:`\rho` in Ω m.
        z0: Reference impedance :math:`Z_0` in Ω.

    Returns:
        sax.SDict: S-parameters dictionary with port ``o1``.
    """
    return nanobolometer(
        f,
        absorber_length=absorber_length,
        absorber_width=absorber_width,
        normal_metal_thickness=normal_metal_thickness,
        resistivity=resistivity,
        z0=z0,
    )
