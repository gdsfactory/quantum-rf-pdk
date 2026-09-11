"""gdsfactoryplus v2 SAX simulation and model-contract tests.

Covers the v2 (>= 2.0.0) SDK SAX surface against this PDK:

- Layout-driven simulation: ``simulate_layout_sax`` materializes a
  schematic (``.gsch`` nyancir file) to a layout, runs KLayout
  LayoutToNetlist extraction, and feeds the resulting netlist to SAX.
  The v1 ``gdsfactoryplus.serve.sax._run_simulation`` server API no
  longer exists; this module replaces the test that used it.
- Schematic-driven simulation: ``simulate_mosaic_sax`` builds the
  netlist directly from the same nyancir schematic (no layout
  materialization) and runs SAX on it — both paths must agree with the
  reference model. The schematic-driven tests also cover the two
  features only that path exercises: hierarchical subcircuits (a
  ``.gsch`` instantiated from another ``.gsch``) and instance ``props``
  reaching the model.
- Model contract: every model in ``PDK.models`` must accept ``f`` in Hz
  (the v2 frontend sweeps frequency, see ``[tool.gdsfactoryplus.sim.x]``
  ``name = "f"`` in ``pyproject.toml``).
- Model coverage: ``inspect_layout_sax_models`` and
  ``inspect_mosaic_sax_models`` must not ask for SAX models beyond the
  exemptions in ``pyproject.toml``
  (``[tool.gdsfactoryplus.pdk] cells_no_model_expected``).

The nyancir (``.gsch``) fixtures are authored directly as JSON in the
Mosaic dialect. The file holds one ``type: "ckt"`` instance of the
sample factory plus four ``type: "port"`` markers exposing its
``o1``..``o4`` ports. The Mosaic (schematic-driven) pipeline resolves
both the ``.gsch`` and ``models.nyanlib`` through nyancad's ``FileAPI``
relative to ``project_root``, so its fixture builds a self-contained
project root: the nyancir plus a hand-authored ``models.nyanlib`` with
the single entry the ``gfp`` binary generates for the sample factory.

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

#: File name of the resonator-test-chip nyancir inside both fixture roots.
_GSCH_NAME = "resonator_test_chip_gfp.gsch"

#: A non-default ``resonator_length`` (µm) used to prove instance ``props``
#: reach the model; the default in ``resonator_test_chip_python`` is 4000.
_TUNED_RESONATOR_LENGTH = 6000.0

#: File name of the nyancir whose chip instance overrides ``resonator_length``.
_TUNED_GSCH_NAME = "resonator_test_chip_tuned.gsch"

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

#: Models exempt from the ``f``-in-Hz call contract (e.g. models that
#: legitimately require instance settings with no defaults). Extend only
#: with a justification, never to silence a failure.
_F_CONTRACT_EXEMPT: frozenset[str] = frozenset()


def _resonator_test_chip_nyancir(
    props: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a nyancir (``.gsch``) document for the resonator test chip.

    One ``type: "ckt"`` instance of the sample factory, with each of its
    four ports wired to a ``type: "port"`` marker through a shared net
    name — the minimal Mosaic-dialect schematic that exposes ``o1``..``o4``
    as top-level ports.

    Args:
        props: Instance settings forwarded to the factory/model; defaults
            to none.

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
            "props": dict(props or {}),
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


def _resonator_test_chip_nyanlib(
    subcircuit_ids: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a minimal ``models.nyanlib`` for the resonator test chip.

    Mirrors the entries the ``gfp`` binary generates (see
    ``build/models.nyanlib``): for a Python-factory leaf, the Mosaic
    netlist builder only reads the component ``name`` and ``type``
    (``nyancad.netlist.kf_component_name``).

    Args:
        subcircuit_ids: Extra ``{schem_id: component_name}`` entries for
            ``.gsch`` files the top schematic instantiates as
            subcircuits; each needs its own nyanlib entry naming the
            SAX subcircuit.

    Returns:
        Nyanlib document ready to be serialized to a ``.nyanlib`` file.
    """
    lib: dict[str, Any] = {
        f"models:{_CHIP_QUALNAME}": {
            "name": "resonator_test_chip_python",
            "type": "ckt",
            "tags": ["qpdk"],
        }
    }
    for schem_id, component_name in (subcircuit_ids or {}).items():
        lib[f"models:{schem_id}"] = {
            "name": component_name,
            "type": "ckt",
            "tags": ["qpdk"],
        }
    return lib


@pytest.fixture(scope="module")
def nyancir_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write the resonator-test-chip nyancir to a temporary ``.gsch`` file.

    The stem is deliberately distinct from any PDK cell name so the
    materialized top cell does not collide in the shared kfactory
    registry when several tests run in one process.

    Returns:
        Path to the written ``.gsch`` nyancir file.
    """
    path = tmp_path_factory.mktemp("nyancir") / _GSCH_NAME
    path.write_text(json.dumps(_resonator_test_chip_nyancir(), indent=2))
    return path


@pytest.fixture(scope="module")
def mosaic_project_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a self-contained project root for the Mosaic pipeline.

    The schematic-driven pipeline resolves the ``.gsch`` and
    ``models.nyanlib`` relative to ``project_root`` through nyancad's
    ``FileAPI``, so both must live in one directory; the real app uses
    the repository root with a ``gfp``-generated nyanlib. A throwaway
    root with a hand-authored nyanlib exercises the same SDK path
    without cross-test ordering dependencies. Holds a default-props and
    a tuned-props nyancir (see ``_TUNED_GSCH_NAME``).

    Returns:
        Path to the temporary project root.
    """
    root = tmp_path_factory.mktemp("mosaic_project")
    (root / _GSCH_NAME).write_text(json.dumps(_resonator_test_chip_nyancir(), indent=2))
    (root / _TUNED_GSCH_NAME).write_text(
        json.dumps(
            _resonator_test_chip_nyancir({"resonator_length": _TUNED_RESONATOR_LENGTH}),
            indent=2,
        )
    )
    (root / "models.nyanlib").write_text(
        json.dumps(_resonator_test_chip_nyanlib(), indent=2)
    )
    return root


@pytest.fixture(scope="module")
def mosaic_hierarchy_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a two-level Mosaic project: top ``.gsch`` instantiating a sub ``.gsch``.

    The top schematic's single instance references ``sub.gsch`` by its
    project-relative id (the Mosaic subcircuit mechanism); the sub
    schematic holds the chip leaf. The subcircuit is a transparent port
    feed-through, so the top-level S-parameters must equal the chip
    model.

    Returns:
        Path to the temporary project root.
    """
    root = tmp_path_factory.mktemp("mosaic_hierarchy")
    (root / "sub.gsch").write_text(json.dumps(_resonator_test_chip_nyancir(), indent=2))
    top_document: dict[str, Any] = {
        "sub:X1": {
            "type": "ckt",
            "model": "sub.gsch",
            "name": "X1",
            "transform": [1, 0, 0, 1, 0, 0],
            "x": 0,
            "y": 0,
            "props": {},
            "nets": {port: f"net_{port}" for port in _PORT_NAMES},
        }
    }
    for port in _PORT_NAMES:
        top_document[f"top:{port}"] = {
            "type": "port",
            "name": port,
            "x": 0,
            "y": 0,
            "nets": {"P": f"net_{port}"},
        }
    (root / "top.gsch").write_text(json.dumps(top_document, indent=2))
    (root / "models.nyanlib").write_text(
        json.dumps(_resonator_test_chip_nyanlib({"sub.gsch": "sub"}), indent=2)
    )
    return root


def _assert_sweep_matches_reference(
    result: dict[str, Any],
    expected_model: Callable[..., Any] | None = None,
) -> None:
    """Assert a simulation result is the reference model over the 4--10 GHz band.

    Checks the returned sweep grid, the exposed chip ports, and that every
    S-parameter entry agrees with ``resonator_test_chip_python`` evaluated at
    the same frequencies (near-zero terms may be dropped from the sdict).

    Args:
        result: Return value of ``simulate_layout_sax``/``simulate_mosaic_sax``.
        expected_model: Model to compare against; defaults to the reference
            model with default settings.
    """
    if expected_model is None:
        expected_model = resonator_test_chip_python_model
    # The API canonicalizes to wavelength-descending (frequency-ascending)
    # order regardless of the wl_min/wl_max argument order — verified against
    # the public extension bundle.
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
    expected = expected_model(f=frequencies)
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
    _assert_sweep_matches_reference(result)


@pytest.mark.gfp
def test_resonator_test_chip_mosaic_sax_simulation(
    mosaic_project_root: Path,
) -> None:
    """Simulate the chip through the v2 schematic-driven (Mosaic) SAX pipeline.

    ``simulate_mosaic_sax`` builds the netlist directly from the nyancir
    schematic (resolving the schematic and ``models.nyanlib`` from
    ``project_root``) without materializing any layout geometry — the
    path used when simulating straight from a schematic. It must agree
    with the reference model exactly like the layout-driven pipeline.
    """
    result = sax_sim.simulate_mosaic_sax(
        str(mosaic_project_root / _GSCH_NAME),
        "qpdk.PDK",
        wl_min=_SPEED_OF_LIGHT_UM_PER_S / 10e9,
        wl_max=_SPEED_OF_LIGHT_UM_PER_S / 4e9,
        wl_num=3,
        project_root=str(mosaic_project_root),
        sweep_frequency=True,
    )
    _assert_sweep_matches_reference(result)


@pytest.mark.gfp
def test_mosaic_hierarchical_subcircuit_simulation(
    mosaic_hierarchy_root: Path,
) -> None:
    """Simulate a ``.gsch`` instantiated from another ``.gsch`` (subcircuit).

    The Mosaic netlist builder resolves subcircuits recursively: a
    device's ``model`` may name another schematic's project-relative id,
    whose own netlist is composed into the parent by SAX. The
    subcircuit here is a transparent port feed-through around the chip,
    so the top-level S-parameters must still equal the reference model.
    """
    result = sax_sim.simulate_mosaic_sax(
        str(mosaic_hierarchy_root / "top.gsch"),
        "qpdk.PDK",
        wl_min=_SPEED_OF_LIGHT_UM_PER_S / 10e9,
        wl_max=_SPEED_OF_LIGHT_UM_PER_S / 4e9,
        wl_num=3,
        project_root=str(mosaic_hierarchy_root),
        sweep_frequency=True,
    )
    _assert_sweep_matches_reference(result)


@pytest.mark.gfp
def test_mosaic_instance_props_reach_model(mosaic_project_root: Path) -> None:
    """Instance ``props`` override model settings in the schematic pipeline.

    A schematic-tuned chip (``resonator_length`` set on the instance)
    must simulate as the reference model called with that setting, and
    must differ from the default-settings result — the second assertion
    keeps this test from passing vacuously if the pipeline ever drops
    instance props.
    """
    result = sax_sim.simulate_mosaic_sax(
        str(mosaic_project_root / _TUNED_GSCH_NAME),
        "qpdk.PDK",
        wl_min=_SPEED_OF_LIGHT_UM_PER_S / 10e9,
        wl_max=_SPEED_OF_LIGHT_UM_PER_S / 4e9,
        wl_num=3,
        project_root=str(mosaic_project_root),
        sweep_frequency=True,
    )
    _assert_sweep_matches_reference(
        result,
        expected_model=lambda f: resonator_test_chip_python_model(
            resonator_length=_TUNED_RESONATOR_LENGTH, f=f
        ),
    )

    # Anti-vacuity: the tuned sweep must not equal the default sweep.
    wavelengths = np.asarray(result["wavelengths"], dtype=float)
    frequencies = _SPEED_OF_LIGHT_UM_PER_S / wavelengths
    default = resonator_test_chip_python_model(f=frequencies)
    zero = np.zeros(len(wavelengths), dtype=complex)
    differs_from_default = False
    for key in set(default):
        entry = result["sdict"].get(f"{key[0]},{key[1]}")
        actual = (
            zero
            if entry is None  # near-zero terms may be dropped
            else np.asarray(entry["real"]) + 1j * np.asarray(entry["imag"])
        )
        if not np.allclose(actual, default[key], rtol=1e-6, atol=1e-6):
            differs_from_default = True
            break
    assert differs_from_default, (
        "tuned instance simulated identically to the default settings — "
        "instance props may have been dropped by the pipeline"
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
def test_mosaic_sax_model_coverage(mosaic_project_root: Path) -> None:
    """Model resolution over the schematic honors the exemptions.

    Same contract as the layout pipeline, but for
    ``inspect_mosaic_sax_models`` (schematic-driven resolution from
    ``project_root``).
    """
    info = sax_sim.inspect_mosaic_sax_models(
        str(mosaic_project_root / _GSCH_NAME),
        "qpdk.PDK",
        project_root=str(mosaic_project_root),
    )

    unexpected = set(info["missing_models"]) - _cells_no_model_expected()
    assert not unexpected, (
        f"Schematic pipeline reported missing SAX models {sorted(unexpected)}; "
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
    them. A model failing this contract must be fixed or explicitly
    added to ``_F_CONTRACT_EXEMPT`` — skipping on contract violations
    would let regressions pass silently.

    Raises:
        AssertionError: Unreachable; ``pytest.fail`` always raises first.
    """
    if model_name in _F_CONTRACT_EXEMPT:
        pytest.skip(f"{model_name} is exempt from the f-in-Hz contract")

    try:
        parameters = inspect.signature(model).parameters
    except (TypeError, ValueError) as exc:
        pytest.fail(f"{model_name} has no inspectable signature: {exc}")
        raise AssertionError("unreachable") from None  # pytest.fail always raises
    if "f" not in parameters:
        pytest.fail(f"{model_name} does not take an 'f' parameter")
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
        pytest.fail(
            f"{model_name} requires {required_beyond_f} without defaults; "
            "it cannot be swept by sax.circuit — fix the model or exempt it "
            "in _F_CONTRACT_EXEMPT"
        )

    s_params = model(f=np.linspace(_F_MIN_HZ, _F_MAX_HZ, 3))

    assert isinstance(s_params, dict), f"{model_name} returned {type(s_params)}"
    assert s_params, f"{model_name} returned an empty S-parameter dict"
    for key, value in s_params.items():
        arr = np.broadcast_to(np.asarray(value, dtype=complex), (3,))
        assert np.isfinite(arr).all(), f"{model_name} S[{key}] is not finite"
