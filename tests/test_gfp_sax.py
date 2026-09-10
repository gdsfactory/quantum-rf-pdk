"""gdsfactoryplus v2 SAX simulation and model-contract tests.

Covers the v2 (>= 2.0.0) SDK SAX surface against this PDK:

- Layout-driven simulation: ``simulate_layout_sax`` materializes a
  schematic (``.gsch`` nyancir file) to a layout, runs KLayout
  LayoutToNetlist extraction, and feeds the resulting netlist to SAX.
  The v1 ``gdsfactoryplus.serve.sax._run_simulation`` server API no
  longer exists; this module replaces the test that used it.
- Model contract: every model in ``PDK.models`` must accept ``f`` in Hz
  (the v2 frontend sweeps frequency, see ``[tool.gdsfactoryplus.sim.x]``
  ``name = "f"`` in ``pyproject.toml``).
- Model coverage: ``inspect_layout_sax_models`` must not ask for SAX
  models beyond the exemptions in ``pyproject.toml``
  (``[tool.gdsfactoryplus.pdk] cells_no_model_expected``).

The nyancir (``.gsch``) fixture is authored directly as JSON in the
Mosaic dialect. The file holds one ``type: "ckt"`` instance of the
sample factory plus four ``type: "port"`` markers exposing its
``o1``..``o4`` ports.

The PDK qualified name is ``"qpdk.PDK"``: the v2 API resolves
``pdk_qn`` as ``<module>.<attribute>`` (like ``ihp.PDK``), not a bare
package name.
"""

from __future__ import annotations

import inspect
import json
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from conftest import import_gfp_module

from qpdk import PDK
from qpdk.models.resonator import (
    resonator_test_chip_python as resonator_test_chip_python_model,
)

sax_sim = import_gfp_module("gdsfactoryplus.sim.sax")

if TYPE_CHECKING:
    from collections.abc import Callable

#: Speed of light in µm/s — converts between the API's wavelengths (µm)
#: and the models' native frequencies (Hz), matching gdsfactoryplus.
_SPEED_OF_LIGHT_UM_PER_S = 299_792_458_000_000.0

#: Fully qualified factory id of the resonator test chip sample.
_CHIP_QUALNAME = "qpdk.samples.resonator_test_chip.resonator_test_chip_python"

#: Chip ports exposed by the nyancir port markers.
_PORT_NAMES = ("o1", "o2", "o3", "o4")

#: RF band under test, in Hz.
_F_MIN_HZ = 4e9
_F_MAX_HZ = 10e9

#: ``PDK.models`` entries, sorted so parametrized failures identify the model.
_MODEL_PARAMS: list[tuple[str, Callable[..., Any]]] = sorted(PDK.models.items())


def _resonator_test_chip_nyancir() -> dict[str, Any]:
    """Build a nyancir (``.gsch``) document for the resonator test chip.

    One ``type: "ckt"`` instance of the sample factory, with each of its
    four ports wired to a ``type: "port"`` marker through a shared net
    name — the minimal Mosaic-dialect schematic that exposes ``o1``..``o4``
    as top-level ports.

    Returns:
        Nyancir document ready to be serialized to a ``.gsch`` file.
    """
    document: dict[str, Any] = {
        "chip:X1": {
            "type": "ckt",
            "model": _CHIP_QUALNAME,
            "name": "X1",
            "transform": [1, 0, 0, 1, 0, 0],
            "x": 0,
            "y": 0,
            "props": {},
            "nets": {port: f"net_{port}" for port in _PORT_NAMES},
        }
    }
    for port in _PORT_NAMES:
        document[f"chip:{port}"] = {
            "type": "port",
            "name": port,
            "x": 0,
            "y": 0,
            "nets": {"P": f"net_{port}"},
        }
    return document


@pytest.fixture(scope="module")
def nyancir_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write the resonator-test-chip nyancir to a temporary ``.gsch`` file.

    The stem is deliberately distinct from any PDK cell name so the
    materialized top cell does not collide in the shared kfactory
    registry when several tests run in one process.

    Returns:
        Path to the written ``.gsch`` nyancir file.
    """
    path = tmp_path_factory.mktemp("nyancir") / "resonator_test_chip_gfp.gsch"
    path.write_text(json.dumps(_resonator_test_chip_nyancir(), indent=2))
    return path


@pytest.mark.gfp
def test_resonator_test_chip_layout_sax_simulation(nyancir_path: Path) -> None:
    """Simulate the chip through the v2 layout-driven SAX pipeline.

    Replaces the removed v1
    ``test_resonator_test_chip_runs_through_layout_simulation_server``:
    the layout netlist is extracted from materialized geometry, the chip
    resolves to its registered SAX model, and the frequency sweep must
    agree with the reference model evaluated at the same frequencies.
    """
    result = sax_sim.simulate_layout_sax(
        str(nyancir_path),
        "qpdk.PDK",
        # The API represents the 4--10 GHz band as wavelengths in µm.
        wl_min=_SPEED_OF_LIGHT_UM_PER_S / 10e9,
        wl_max=_SPEED_OF_LIGHT_UM_PER_S / 4e9,
        wl_num=3,
        sweep_frequency=True,
    )

    wavelengths = np.asarray(result["wavelengths"], dtype=float)
    assert len(wavelengths) == 3
    assert wavelengths[0] == pytest.approx(299_792.458 / 4)
    assert wavelengths[-1] == pytest.approx(299_792.458 / 10)

    sdict = result["sdict"]
    ports = {name for key in sdict for name in key.split(",")}
    assert ports == {"o1", "o2", "o3", "o4"}
    for entry in sdict.values():
        assert len(entry["real"]) == len(entry["imag"]) == 3

    # The sweep is called with ascending frequencies in Hz (f = c / λ);
    # derive them back from the returned canonical wavelengths.
    frequencies = _SPEED_OF_LIGHT_UM_PER_S / wavelengths
    expected = resonator_test_chip_python_model(f=frequencies)
    zero = np.zeros(len(wavelengths), dtype=complex)
    simulated_keys = {tuple(key.split(",")) for key in sdict}

    for key in set(expected) | simulated_keys:
        entry = sdict.get(f"{key[0]},{key[1]}")
        actual = (
            zero
            if entry is None  # near-zero terms may be dropped
            else np.asarray(entry["real"]) + 1j * np.asarray(entry["imag"])
        )
        np.testing.assert_allclose(
            actual,
            expected.get(key, zero),
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"S-parameters disagree for {key}",
        )


def _cells_no_model_expected() -> set[str]:
    """Read the SAX-model exemptions from ``pyproject.toml``."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as stream:
        data = tomllib.load(stream)
    return set(data["tool"]["gdsfactoryplus"]["pdk"]["cells_no_model_expected"])


@pytest.mark.gfp
def test_layout_sax_model_coverage(nyancir_path: Path) -> None:
    """Model resolution over the extracted layout honors the exemptions.

    Components without a SAX model are acceptable only when they are
    listed in ``[tool.gdsfactoryplus.pdk] cells_no_model_expected``;
    anything else is an upstream regression to fix with a model.
    """
    info = sax_sim.inspect_layout_sax_models(str(nyancir_path), "qpdk.PDK")

    unexpected = set(info["missing_models"]) - _cells_no_model_expected()
    assert not unexpected, (
        f"Layout pipeline reported missing SAX models {sorted(unexpected)}; "
        "add a model or extend cells_no_model_expected in pyproject.toml"
    )
    assert {"resonator_test_chip_python", _CHIP_QUALNAME} & set(
        info["resolved_models"]
    ), f"chip model unresolved: {info}"


@pytest.mark.gfp
@pytest.mark.parametrize(("model_name", "model"), _MODEL_PARAMS)
def test_pdk_model_accepts_f_in_hz(model_name: str, model: Callable) -> None:
    """Every PDK model is callable with ``f`` alone, in Hz.

    The v2 frontend sweeps frequency (``[tool.gdsfactoryplus.sim.x]``
    ``name = "f"``), so ``sax.circuit`` calls each model with ``f=`` in
    Hz and no other arguments beyond instance settings. Models that
    cannot be called this way break every v2 SAX simulation that uses
    them.
    """
    try:
        parameters = inspect.signature(model).parameters
    except (TypeError, ValueError) as exc:
        pytest.skip(f"{model_name} has no inspectable signature: {exc}")
    else:
        if "f" not in parameters:
            pytest.skip(f"{model_name} does not take an 'f' parameter")
        required_beyond_f = [
            name
            for name, parameter in parameters.items()
            if name != "f"
            and parameter.default is inspect.Parameter.empty
            and parameter.kind
            in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        ]
        if required_beyond_f:
            pytest.skip(f"{model_name} requires {required_beyond_f} without defaults")

        s_params = model(f=np.linspace(_F_MIN_HZ, _F_MAX_HZ, 3))

        assert isinstance(s_params, dict), f"{model_name} returned {type(s_params)}"
        assert s_params, f"{model_name} returned an empty S-parameter dict"
        for key, value in s_params.items():
            arr = np.broadcast_to(np.asarray(value, dtype=complex), (3,))
            assert np.isfinite(arr).all(), f"{model_name} S[{key}] is not finite"
