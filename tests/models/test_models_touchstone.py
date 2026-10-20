"""Exercise the SAX Touchstone bridge with QPDK RF models."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import sax
import skrf

from qpdk import PDK
from qpdk.models.resonator import quarter_wave_resonator_coupled
from qpdk.models.waveguides import straight

PDK.activate()


@pytest.mark.parametrize("n_ports", [2, 3])
def test_model_touchstone_round_trip(n_ports: int, tmp_path: Path) -> None:
    frequency = jnp.linspace(4e9, 8e9, 21)
    sdict = (
        straight(f=frequency, length=1000)
        if n_ports == 2
        else quarter_wave_resonator_coupled(f=frequency)
    )
    path = sax.write_sdict_touchstone(sdict, frequency, tmp_path / f"model.s{n_ports}p")
    recovered_frequency, recovered = sax.read_sdict_touchstone(path)
    np.testing.assert_allclose(recovered_frequency, frequency)
    assert set(recovered) == set(sdict)
    for ports, values in sdict.items():
        np.testing.assert_allclose(recovered[ports], values, atol=1e-11)

    matrix, _ = sax.sdense(sdict)
    np.testing.assert_allclose(skrf.Network(str(path)).s, matrix, atol=1e-11)
