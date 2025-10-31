"""S-parameter models for generic components."""

from functools import partial
from pprint import pprint

import jax
import jax.numpy as jnp
import sax
from jax.typing import ArrayLike


@partial(jax.jit, inline=True, static_argnames=("n_ports"))
def gamma_0_load(
    f: ArrayLike = jnp.array([5e9]),
    gamma_0: int | float | complex = 0,
    n_ports: int = 1,
) -> sax.SType:
    r"""Connection with given reflection coefficient.

    Args:
        f: Array of frequency points in Hz
        gamma_0: Reflection coefficient Γ₀ of connection
        n_ports: Number of ports in component. The diagonal ports of the matrix
            are set to Γ₀ and the off-diagonal ports to 0.

    Returns:
        sax.SType: S-parameters dictionary where :math:`S = \Gamma_0I_\text{n\_ports}`

    """
    sdict = {
        (f"o{i}", f"o{i}"): jnp.full(len(f), gamma_0) for i in range(1, n_ports + 1)
    }
    sdict |= {
        (f"o{i}", f"o{j}"): jnp.zeros(len(f), dtype=complex)
        for i in range(1, n_ports + 1)
        for j in range(i + 1, n_ports + 1)
    }
    return sax.reciprocal(sdict)


@partial(jax.jit, inline=True, static_argnames=("n_ports"))
def short(
    f: ArrayLike = jnp.array([5e9]),
    n_ports: int = 1,
) -> sax.SType:
    r"""Electrical short connections Sax model.

    Args:
        f: Array of frequency points in Hz
        n_ports: Number of ports to set as shorted

    Returns:
        sax.SType: S-parameters dictionary where :math:`S = -I_\text{n\_ports}`
    """
    return gamma_0_load(f=f, gamma_0=-1, n_ports=n_ports)


def short_2_port(f: ArrayLike = jnp.array([5e9])) -> sax.SType:
    """Electrical short 2-port connection Sax model."""
    return short(f=f, n_ports=2)


@partial(jax.jit, inline=True, static_argnames=("n_ports"))
def open(
    f: ArrayLike = jnp.array([5e9]),
    n_ports: int = 1,
) -> sax.SType:
    r"""Electrical open connection Sax model.

    Args:
        f: Array of frequency points in Hz
        n_ports: Number of ports to set as opened

    Returns:
        sax.SType: S-parameters dictionary where :math:`S = I_\text{n\_ports}`
    """
    return gamma_0_load(f=f, gamma_0=1, n_ports=n_ports)


@partial(jax.jit, inline=True)
def tee(f: ArrayLike = jnp.array([5e9])) -> sax.SType:
    """Ideal 3-port power divider/combiner (T-junction).

    Args:
        f: Array of frequency points in Hz

    Returns:
        sax.SType: S-parameters dictionary
    """
    sdict = {(f"o{i}", f"o{i}"): jnp.full(len(f), -1 / 3) for i in range(1, 4)}
    sdict |= {
        (f"o{i}", f"o{j}"): jnp.full(len(f), 2 / 3)
        for i in range(1, 4)
        for j in range(i + 1, 4)
    }
    return sax.reciprocal(sdict)
    # return sax.models.splitters.splitter_ideal(wl=f)


@partial(jax.jit, inline=True)
def single_impedance_element(
    z: int | float | complex = 50,
    z0: int | float | complex = 50,
) -> sax.SType:
    r"""Single impedance element Sax model.

    See :cite:`m.pozarMicrowaveEngineering2012` for details.

    Args:
        z: Impedance in Ω
        z0: Reference impedance in Ω. This may be retrieved from a scikit-rf
            Media object using `z0 = media.z0`.

    Returns:
        sax.SType: S-parameters dictionary
    """
    sdict = {
        ("o1", "o1"): z / (z + 2 * z0),
        ("o1", "o2"): 2 * z0 / (2 * z0 + z),
        ("o2", "o2"): z / (z + 2 * z0),
    }
    return sax.reciprocal(sdict)


@partial(jax.jit, inline=True)
def single_admittance_element(
    y: int | float | complex = 1 / 50,
) -> sax.SType:
    r"""Single admittance element Sax model.

    See :cite:`m.pozarMicrowaveEngineering2012` for details.

    Args:
        y: Admittance

    Returns:
        sax.SType: S-parameters dictionary
    """
    sdict = {
        ("o1", "o1"): 1 / (1 + y),
        ("o1", "o2"): y / (1 + y),
        ("o2", "o2"): 1 / (1 + y),
    }
    return sax.reciprocal(sdict)


@partial(jax.jit, inline=True)
def capacitor(
    f: ArrayLike = jnp.array([5e9]),
    capacitance: float = 1e-15,
    z0: int | float | complex = 50,
) -> sax.SType:
    r"""Ideal capacitor () Sax model.

    See :cite:`m.pozarMicrowaveEngineering2012` for details.

    Args:
        f: Array of frequency points in Hz
        capacitance: Capacitance in Farads
        z0: Reference impedance in Ω. This may be retrieved from a scikit-rf
            Media object using `z0 = media.z0`.

    Returns:
        sax.SType: S-parameters dictionary
    """
    ω = 2 * jnp.pi * jnp.asarray(f)
    # Y = 2 * (1j * ω * capacitance * z0)
    # return single_admittance_element(y=Y)
    Z𞁞 = 1 / (1j * ω * capacitance)
    return single_impedance_element(z=Z𞁞, z0=z0)


@partial(jax.jit, inline=True)
def inductor(
    f: ArrayLike = jnp.array([5e9]),
    inductance: float = 1e-12,
    z0: int | float | complex = 50,
) -> sax.SType:
    r"""Ideal inductor (󱡌) Sax model.

    See :cite:`m.pozarMicrowaveEngineering2012` for details.

    Args:
        f: Array of frequency points in Hz
        inductance: Inductance in Henries
        z0: Reference impedance in Ω. This may be retrieved from a scikit-rf
            Media object using `z0 = media.z0`.

    Returns:
        sax.SType: S-parameters dictionary
    """
    ω = 2 * jnp.pi * jnp.asarray(f)
    Zᵢ = 1j * ω * inductance
    return single_impedance_element(z=Zᵢ, z0=z0)


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    f = jnp.linspace(1e9, 25e9, 201)
    S = gamma_0_load(f=f, gamma_0=0.5 + 0.5j, n_ports=2)
    for key in S:
        plt.plot(f / 1e9, abs(S[key]) ** 2, label=key)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("S")
    plt.grid(True)
    plt.legend()
    plt.show(block=False)

    S_cap = capacitor(f, capacitance=(capacitance := 100e-15))
    pprint(S_cap)
    plt.figure()
    # Polar plot of S21 and S11
    plt.subplot(121, projection="polar")
    plt.plot(jnp.angle(S_cap[("o1", "o1")]), abs(S_cap[("o1", "o1")]), label="$S_{11}$")
    plt.plot(jnp.angle(S_cap[("o1", "o2")]), abs(S_cap[("o2", "o1")]), label="$S_{21}$")
    plt.title("S-parameters capacitor")
    plt.legend()
    # Magnitude and phase vs frequency
    ax1 = plt.subplot(122)
    ax1.plot(f / 1e9, abs(S_cap[("o1", "o1")]), label="|S11|", color="C0")
    ax1.plot(f / 1e9, abs(S_cap[("o1", "o2")]), label="|S21|", color="C1")
    ax1.set_xlabel("Frequency [GHz]")
    ax1.set_ylabel("Magnitude [unitless]")
    ax1.grid(True)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(
        f / 1e9,
        jnp.angle(S_cap[("o1", "o1")]),
        label="∠S11",
        color="C0",
        linestyle="--",
    )
    ax2.plot(
        f / 1e9,
        jnp.angle(S_cap[("o1", "o2")]),
        label="∠S21",
        color="C1",
        linestyle="--",
    )
    ax2.set_ylabel("Phase [rad]")
    ax2.legend(loc="upper right")

    plt.title(f"Capacitor $S$-parameters ($C={capacitance * 1e15}\\,$fF)")
    plt.show(block=False)

    S_ind = inductor(f, inductance=(inductance := 1e-9))
    pprint(S_ind)
    plt.figure()
    plt.subplot(121, projection="polar")
    plt.plot(jnp.angle(S_ind[("o1", "o1")]), abs(S_ind[("o1", "o1")]), label="$S_{11}$")
    plt.plot(jnp.angle(S_ind[("o1", "o2")]), abs(S_ind[("o2", "o1")]), label="$S_{21}$")
    plt.title("S-parameters inductor")
    plt.legend()
    ax1 = plt.subplot(122)
    ax1.plot(f / 1e9, abs(S_ind[("o1", "o1")]), label="|S11|", color="C0")
    ax1.plot(f / 1e9, abs(S_ind[("o1", "o2")]), label="|S21|", color="C1")
    ax1.set_xlabel("Frequency [GHz]")
    ax1.set_ylabel("Magnitude [unitless]")
    ax1.grid(True)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(
        f / 1e9,
        jnp.angle(S_ind[("o1", "o1")]),
        label="∠S11",
        color="C0",
        linestyle="--",
    )
    ax2.plot(
        f / 1e9,
        jnp.angle(S_ind[("o1", "o2")]),
        label="∠S21",
        color="C1",
        linestyle="--",
    )
    ax2.set_ylabel("Phase [rad]")
    ax2.legend(loc="upper right")

    plt.title(f"Inductor $S$-parameters ($L={inductance * 1e9}\\,$nH)")
    plt.show()
