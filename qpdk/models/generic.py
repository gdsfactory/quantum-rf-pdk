"""S-parameter models for generic components."""

import jax.numpy as jnp
import sax
from jax.typing import ArrayLike


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


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    f = jnp.linspace(1e9, 10e9, 201)
    S = gamma_0_load(f=f, gamma_0=0.5 + 0.5j, n_ports=2)

    for key in S:
        plt.plot(f / 1e9, abs(S[key]) ** 2, label=key)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Transmittance")
    plt.grid(True)
    plt.legend()
    plt.show()
