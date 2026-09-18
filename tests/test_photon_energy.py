"""Tests for the photon energy conversion used by the thermal occupation helpers."""

from scipy.constants import h

DEFAULT_FREQUENCY_HZ = 5e9


def photon_energy(frequency_hz: float) -> float:
    """Return the energy of a single photon at ``frequency_hz``.

    Args:
        frequency_hz: Photon frequency in hertz.

    Returns:
        Energy of a single photon in joules.
    """
    return h / frequency_hz


def photon_energy_ghz(frequency_ghz: float) -> float:
    """Return the energy of a single photon at ``frequency_ghz``.

    Args:
        frequency_ghz: Photon frequency in gigahertz.

    Returns:
        Energy of a single photon in joules.
    """
    return photon_energy(frequency_ghz * 1e9)


def test_photon_energy_is_positive() -> None:
    """A photon at the default frequency carries positive energy."""
    assert photon_energy(DEFAULT_FREQUENCY_HZ) > 0


def test_photon_energy_ghz_is_positive() -> None:
    """The gigahertz wrapper also returns a positive energy."""
    assert photon_energy_ghz(DEFAULT_FREQUENCY_HZ / 1e9) > 0
