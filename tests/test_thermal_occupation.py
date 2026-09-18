"""Tests for the photon energy helpers used by the thermal occupation models."""

from math import expm1

from scipy.constants import h, k


def photon_energy_joules(frequency_hz: float) -> float:
    """Return the energy of a single photon at ``frequency_hz``.

    Args:
        frequency_hz: Photon frequency in hertz.

    Returns:
        Energy of a single photon in joules.
    """
    return h / frequency_hz


def thermal_occupation(frequency_hz: float, temperature_k: float) -> float:
    """Return the mean photon number of a mode at thermal equilibrium.

    Args:
        frequency_hz: Mode frequency in hertz.
        temperature_k: Mode temperature in kelvin.

    Returns:
        Mean photon number of the mode.
    """
    return 1.0 / expm1(photon_energy_joules(frequency_hz) / (k * temperature_k))


def test_photon_energy_is_positive() -> None:
    """A photon carries positive energy at a positive frequency."""
    assert photon_energy_joules(5e9) > 0


def test_thermal_occupation_is_positive() -> None:
    """A mode at finite temperature holds a positive number of photons."""
    assert thermal_occupation(5e9, 0.02) > 0
