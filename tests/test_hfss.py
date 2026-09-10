"""Tests for HFSS and Q3D simulation integration."""

import contextlib
import os
import shutil
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gdsfactory as gf
import pytest
from gdsfactory.component import Component
from gdsfactory.technology import LayerLevel, LayerStack
from numpy.testing import assert_allclose

from qpdk import LAYER, LAYER_STACK, PDK, logger
from qpdk.cells.capacitor import interdigital_capacitor
from qpdk.cells.resonator import resonator
from qpdk.simulation import (
    HFSS,
    Q2D,
    Q3D,
    layer_stack_to_gds_mapping,
    lumped_port_rectangle_from_cpw,
    object_names_to_materials,
    prepare_component_for_aedt,
)
from qpdk.simulation.aedt_base import _get_layer_number_from_level
from qpdk.tech import LAYER_STACK_FLIP_CHIP, coplanar_waveguide

# Maximum wall-clock time for a single AEDT-bound call. AEDT startup can
# legitimately take minutes, so this is generous. The budget is per call, not
# per test: every guarded call gets its own, so a run in which they all hang is
# bounded by (number of guarded calls) x this rather than hanging forever.
#
# Thread-affinity caveat: each guarded call runs on its own worker thread, so an
# AEDT object is constructed on one thread and then used from the test body and
# from a fresh thread per subsequent guarded call. gRPC stubs are thread-safe
# and ``use_grpc_uds = False`` plus ``non_graphical=True`` force TCP gRPC, but
# this has not been verified on real AEDT hardware.
_AEDT_TIMEOUT_SECONDS = 900

# ``release_desktop()`` on a wedged session can also block; cut it off quickly.
_RELEASE_TIMEOUT_SECONDS = 120


def _run_with_timeout(
    func: Callable[[], Any],
    timeout: float = _AEDT_TIMEOUT_SECONDS,
    *,
    label: str | None = None,
) -> Any:
    """Run ``func`` and raise :class:`TimeoutError` if it exceeds ``timeout``.

    AEDT calls communicate over gRPC and can hang indefinitely, which would
    block the whole ``-n auto`` suite with no diagnostic. The call runs in a
    daemon thread so the test fails with a ``TimeoutError`` instead of hanging.
    Note that a genuinely hung call cannot be killed; the abandoned thread
    keeps running until the process exits. Any exception ``func`` itself raises
    is re-raised on the calling thread.

    Args:
        func: Zero-argument callable to run (e.g. a lambda constructing an
            AEDT application, updating a setup, or releasing the desktop).
        timeout: Maximum seconds to wait for ``func`` to complete.
        label: Name to report in the timeout message. Pass this for lambdas:
            their ``__name__`` is always ``"<lambda>"``, so a timed-out
            constructor would otherwise not say which AEDT app hung.

    Returns:
        Whatever ``func`` returns.

    Raises:
        TimeoutError: If ``func`` does not finish within ``timeout``.
    """
    name = label or getattr(func, "__name__", repr(func))
    result: dict[str, Any] = {}

    def target() -> None:
        try:
            result["value"] = func()
        except BaseException as error:
            result["error"] = error

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    # Sample the result dict before the thread state: a call that finishes
    # near the timeout may already have stored its result while the thread
    # object is still tearing down, which makes ``is_alive()`` spuriously
    # true and would discard a successfully produced value.
    if "error" in result:
        raise result["error"]
    if "value" in result:
        return result["value"]
    msg = f"AEDT call {name!r} timed out after {timeout}s"
    raise TimeoutError(msg)


@contextlib.contextmanager
def _aedt_project(
    app_factory: Callable[[], Any],
    project_dir: Path,
    timeout: float = _AEDT_TIMEOUT_SECONDS,
    *,
    release_timeout: float = _RELEASE_TIMEOUT_SECONDS,
    label: str | None = None,
) -> Iterator[Any]:
    """Construct an AEDT app and guarantee release plus project cleanup.

    The constructor runs inside the guarded region so that a hung startup
    (``TimeoutError``) or a raising constructor still reaches the ``finally``:
    the project directory is removed either way, and ``release_desktop()`` is
    attempted whenever an app object exists.

    A release that fails or times out is logged rather than raised, so it
    cannot mask a failure from the test body. A genuinely hung constructor
    cannot be released at all; its daemon thread keeps running until process
    exit, and may write project files back into ``project_dir`` after the
    removal below.

    Args:
        app_factory: Zero-argument callable constructing the AEDT application
            (e.g. ``lambda: Hfss(...)``).
        project_dir: Directory holding the AEDT project; removed on exit.
        timeout: Maximum seconds for the constructor itself.
        release_timeout: Maximum seconds for ``release_desktop()``; a release
            that exceeds it is logged and swallowed.
        label: Application name reported if the constructor times out (e.g.
            ``"Hfss"``), since ``app_factory`` is normally a lambda.

    Yields:
        The constructed AEDT application object.
    """
    app = None
    try:
        app = _run_with_timeout(app_factory, timeout=timeout, label=label)
        yield app
    finally:
        try:
            if app is not None:
                # Cut off a wedged session instead of hanging the xdist worker.
                _run_with_timeout(app.release_desktop, timeout=release_timeout)
        except Exception as error:
            # A wedged or already-dead session is expected here. Logging rather
            # than raising keeps it from masking (and from failing a test over)
            # the error that brought us into this ``finally``.
            logger.warning(f"Failed to release AEDT desktop: {error}")
        finally:
            try:
                # ``ignore_errors`` is deliberately avoided: a failure to clean
                # up should be visible, but it must not mask the error that
                # brought us into this ``finally``.
                shutil.rmtree(project_dir)
            except OSError as error:
                logger.warning(
                    f"Failed to remove AEDT project directory {project_dir}: {error}"
                )


class TestRunWithTimeout:
    """Tests for ``_run_with_timeout`` and ``_aedt_project``; run in ordinary CI without Ansys."""

    @staticmethod
    def test_returns_value():
        """A fast callable's return value passes through the helper."""
        assert _run_with_timeout(lambda: 42) == 42

    @staticmethod
    def test_timeout_raises():
        """A callable that outlives the timeout raises ``TimeoutError``."""
        with pytest.raises(TimeoutError, match="timed out"):
            _run_with_timeout(lambda: time.sleep(0.5), timeout=0.1)

    @staticmethod
    def test_exception_propagates():
        """An exception raised by the callable propagates through the helper."""

        def fail() -> None:
            msg = "boom"
            raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match="boom"):
            _run_with_timeout(fail)

    @staticmethod
    def test_aedt_project_releases_and_cleans_up(tmp_path):
        """A successful run releases the app and removes the project directory."""
        released = []

        class FakeApp:
            @staticmethod
            def release_desktop():
                released.append(True)

        project_dir = tmp_path / "aedt"
        project_dir.mkdir()
        (project_dir / "proj.aedt").write_text("fake project")

        with _aedt_project(FakeApp, project_dir) as app:
            assert isinstance(app, FakeApp)

        assert released
        assert not project_dir.exists()

    @staticmethod
    def test_aedt_project_constructor_timeout_still_cleans_up(tmp_path):
        """A hung constructor still removes the project directory."""
        project_dir = tmp_path / "aedt"
        project_dir.mkdir()
        (project_dir / "proj.aedt").write_text("fake project")

        with (
            pytest.raises(TimeoutError, match="timed out"),
            _aedt_project(lambda: time.sleep(0.5), project_dir, timeout=0.1),
        ):
            pass

        assert not project_dir.exists()

    @staticmethod
    def test_timeout_names_the_labelled_call():
        """A labelled lambda is named in the message instead of showing ``<lambda>``."""
        with pytest.raises(TimeoutError, match=r"'Q3d' timed out"):
            _run_with_timeout(lambda: time.sleep(0.5), timeout=0.1, label="Q3d")

    @staticmethod
    def test_aedt_project_constructor_error_still_cleans_up(tmp_path):
        """A constructor that raises still removes the project directory."""

        def raising_factory():
            msg = "no license"
            raise RuntimeError(msg)

        project_dir = tmp_path / "aedt"
        project_dir.mkdir()

        with (
            pytest.raises(RuntimeError, match="no license"),
            _aedt_project(raising_factory, project_dir),
        ):
            pass

        assert not project_dir.exists()

    @staticmethod
    def test_aedt_project_release_timeout_is_swallowed(tmp_path):
        """A wedged ``release_desktop`` is logged, not raised, and cleanup still runs."""

        class SlowReleaseApp:
            @staticmethod
            def release_desktop() -> None:
                time.sleep(0.5)

        project_dir = tmp_path / "aedt"
        project_dir.mkdir()

        with _aedt_project(SlowReleaseApp, project_dir, release_timeout=0.1):
            pass

        assert not project_dir.exists()

    @staticmethod
    def test_aedt_project_body_error_survives_failing_release(tmp_path):
        """A failing release does not mask the test body's own error."""

        class FailingReleaseApp:
            @staticmethod
            def release_desktop() -> None:
                msg = "release failed"
                raise RuntimeError(msg)

        project_dir = tmp_path / "aedt"
        project_dir.mkdir()

        def boom() -> None:
            msg = "boom"
            raise ValueError(msg)

        with (
            pytest.raises(ValueError, match="boom"),
            _aedt_project(FailingReleaseApp, project_dir),
        ):
            boom()

        assert not project_dir.exists()

    @staticmethod
    def test_aedt_project_releases_on_body_exception(tmp_path):
        """An exception in the test body still releases the app and cleans up."""
        released = []

        class FakeApp:
            @staticmethod
            def release_desktop():
                released.append(True)

        def boom() -> None:
            msg = "boom"
            raise RuntimeError(msg)

        project_dir = tmp_path / "aedt"
        project_dir.mkdir()

        with (
            pytest.raises(RuntimeError, match="boom"),
            _aedt_project(FakeApp, project_dir),
        ):
            boom()

        assert released
        assert not project_dir.exists()


@pytest.fixture
def custom_ansys_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set a non-default ``ANSYSEM_ROOT252`` before ``ansys_install_path`` runs."""
    monkeypatch.setenv("ANSYSEM_ROOT252", "/custom/ansys")


@pytest.fixture
def ansys_install_path(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Ensure ``ANSYSEM_ROOT252`` is set so PyAEDT can find Ansys, auto-restored.

    Scoped via ``monkeypatch`` so the variable does not leak to other tests.
    A pre-existing ``ANSYSEM_ROOT252`` (e.g. a developer's own installation)
    is left untouched; the default is only set when unset. Nothing needs the
    variable at import/collection time (pyaedt is imported lazily inside the
    tests), so a fixture is sufficient.

    Returns:
        Path to the Ansys installation directory named by ``ANSYSEM_ROOT252``.
    """
    ansys_default_path = "/usr/ansys_inc/v252/AnsysEM"
    if "ANSYSEM_ROOT252" not in os.environ:
        monkeypatch.setenv("ANSYSEM_ROOT252", ansys_default_path)
    return Path(os.environ["ANSYSEM_ROOT252"])


# ``usefixtures`` fixtures are set up before the ones named in the signature,
# so ``custom_ansys_env`` lands before ``ansys_install_path`` reads it.
@pytest.mark.usefixtures("custom_ansys_env")
def test_ansys_install_path_preserves_existing(ansys_install_path: Path):
    """A pre-existing ``ANSYSEM_ROOT252`` value wins over the default."""
    assert ansys_install_path == Path("/custom/ansys")


def test_layer_stack_to_gds_mapping():
    """Test generating GDS mapping from a LayerStack."""
    mapping = layer_stack_to_gds_mapping(LAYER_STACK)

    # Check that it returns a dictionary
    assert isinstance(mapping, dict)

    # Check a known layer from qpdk
    # For example, layer 1 should be in the mapping
    # The structure is {layer_number: (elevation, thickness)}
    assert 1 in mapping
    assert isinstance(mapping[1], tuple)
    assert len(mapping[1]) == 2

    elevation, thickness = mapping[1]
    assert isinstance(elevation, float)
    assert isinstance(thickness, float)


def test_layer_stack_to_gds_mapping_default_stack_collision_free():
    """Test that the default LAYER_STACK maps every level to a distinct entry.

    pyaedt's ``import_gds_3d`` keys on GDS layer number only, so the Vacuum
    level (which shares ``LAYER.SIM_AREA`` (98, 0) with Substrate) must be
    skipped, and the contiguous Nb spans of Airbridge (10, 0) and
    Airbridge_Via (10, 1) must merge into a single box.
    """
    mapping = layer_stack_to_gds_mapping(LAYER_STACK)

    # Vacuum (98, 0) is skipped, so Substrate keeps layer 98 at its own
    # elevation/thickness instead of being overwritten by vacuum values.
    assert mapping[98] == pytest.approx((-500.0, 500.0))

    # Airbridge (0.3-0.5 um) and Airbridge_Via (0.2-0.3 um) merge to one box.
    assert mapping[10] == pytest.approx((0.2, 0.3))

    # Every non-vacuum level with a resolvable layer number is mapped exactly
    # once (Airbridge and Airbridge_Via share one merged entry on layer 10).
    expected_layers = set()
    for level in LAYER_STACK.layers.values():
        if level.material == "vacuum":
            continue
        layer_number = _get_layer_number_from_level(level)
        if layer_number is not None:
            expected_layers.add(layer_number)
    assert set(mapping) == expected_layers

    # Sheets mode keeps the lowest elevation for colliding levels.
    sheets_mapping = layer_stack_to_gds_mapping(LAYER_STACK, thickness_override=0.0)
    assert sheets_mapping[10] == pytest.approx((0.2, 0.0))
    assert sheets_mapping[98] == pytest.approx((-500.0, 0.0))


def test_layer_stack_to_gds_mapping_unmergeable_collision_raises():
    """Test that unmergeable layer-number collisions raise a clear error."""
    stack = LayerStack(
        layers={
            "A": LayerLevel(
                name="A", layer=(7, 0), thickness=1.0, zmin=0.0, material="Nb"
            ),
            # Gap between spans: union is not a single box.
            "B": LayerLevel(
                name="B", layer=(7, 1), thickness=1.0, zmin=5.0, material="Nb"
            ),
        }
    )
    with pytest.raises(ValueError, match="do not join into a single box"):
        layer_stack_to_gds_mapping(stack)

    stack_different_material = LayerStack(
        layers={
            "A": LayerLevel(
                name="A", layer=(7, 0), thickness=1.0, zmin=0.0, material="Nb"
            ),
            "C": LayerLevel(
                name="C", layer=(7, 1), thickness=1.0, zmin=1.0, material="SiO2"
            ),
        }
    )
    with pytest.raises(ValueError, match="different materials"):
        layer_stack_to_gds_mapping(stack_different_material)

    # A material mismatch raises even in sheets mode (thickness_override set):
    # the colliding levels would still be imported as one object with a single
    # material.
    with pytest.raises(ValueError, match="different materials"):
        layer_stack_to_gds_mapping(stack_different_material, thickness_override=0.0)


def test_layer_stack_to_gds_mapping_order_independent():
    """Test that merged spans do not depend on layer stack insertion order.

    Three Nb levels on the same layer number whose z-spans chain into one
    contiguous box must merge to the same entry regardless of the order the
    levels appear in the ``LayerStack``.
    """

    def make_stack(names: Sequence[str]) -> LayerStack:
        spans = {"A": (0.0, 1.0), "B": (1.0, 3.0), "C": (3.0, 4.0)}
        return LayerStack(
            layers={
                name: LayerLevel(
                    name=name,
                    layer=(7, index),
                    thickness=spans[name][1] - spans[name][0],
                    zmin=spans[name][0],
                    material="Nb",
                )
                for index, name in enumerate(names)
            }
        )

    expected = layer_stack_to_gds_mapping(make_stack(("A", "B", "C")))
    assert expected[7] == pytest.approx((0.0, 4.0))
    assert layer_stack_to_gds_mapping(make_stack(("A", "C", "B"))) == expected
    assert layer_stack_to_gds_mapping(make_stack(("C", "B", "A"))) == expected


def test_layer_stack_to_gds_mapping_flip_chip_raises():
    """Test that the flip-chip stack raises on its Substrate collision.

    ``LAYER_STACK_FLIP_CHIP`` has ``Substrate`` (Si, span [-500, 0] um) and
    ``Substrate_top`` (Si, span [10.2, 510.2] um) sharing GDS layer 98 with a
    10.2 um vacuum gap between the chips. The two disjoint bodies cannot be
    represented by the single (elevation, thickness) box pyaedt's
    ``import_gds_3d`` allows per layer number, and merging would fill the
    inter-chip gap with silicon — so the mapping raises instead of silently
    producing wrong geometry. Flip-chip EM import is not currently supported;
    this test documents that contract.
    """
    with pytest.raises(
        ValueError, match=r"'Substrate' and 'Substrate_top' share GDS layer 98"
    ):
        layer_stack_to_gds_mapping(LAYER_STACK_FLIP_CHIP)


def test_prepare_component_for_aedt():
    """Test component preparation for AEDT."""
    comp = resonator()
    prepared = prepare_component_for_aedt(comp)

    assert isinstance(prepared, Component)
    # The prepared component should be different if additive metals are applied
    # or at least it should return a valid component
    assert prepared.name is not None


def test_prepare_component_for_aedt_margin():
    """Test component preparation for AEDT with margin."""
    comp = resonator(length=3000)

    # Original bbox
    bbox_orig = comp.bbox()
    prepared = prepare_component_for_aedt(comp, margin_draw=200)

    assert isinstance(prepared, Component)
    assert prepared.name is not None

    # Prepared bbox might be larger
    bbox_new = prepared.bbox()
    assert bbox_new.left <= bbox_orig.left
    assert bbox_new.right >= bbox_orig.right
    assert bbox_new.bottom <= bbox_orig.bottom
    assert bbox_new.top >= bbox_orig.top


def test_get_layer_number_from_level():
    """Test layer number extraction from various layer definitions."""

    # Test with a regular LayerLevel that has a direct tuple
    class MockLevelTuple:
        layer = (1, 0)

    assert _get_layer_number_from_level(MockLevelTuple()) == 1

    # Test with a derived layer structure
    class MockLogicalLayerInner:
        layer = (2, 0)

    class MockLogicalLayer:
        layer = MockLogicalLayerInner()

    class MockDerivedLevel:
        layer = None
        derived_layer = MockLogicalLayer()

    assert _get_layer_number_from_level(MockDerivedLevel()) == 2


@dataclass
class MockAEDTObject:
    """Minimal stand-in for a PyAEDT modeler object."""

    name: str


class MockModeler:
    """Minimal stand-in for a PyAEDT modeler."""

    def __init__(self, object_names: list[str]):
        """Initialize with the names of objects created by the GDS import."""
        self.object_names = list(object_names)
        self._objects = {name: MockAEDTObject(name) for name in object_names}

    def __getitem__(self, name: str) -> MockAEDTObject:
        """Return the modeler object with the given name."""
        return self._objects[name]


@dataclass
class MockMaterial:
    """Minimal stand-in for a PyAEDT material."""

    permittivity: float | None = None
    conductivity: float | None = None


class MockMaterials:
    """Minimal stand-in for a PyAEDT materials manager."""

    def __init__(self) -> None:
        """Initialize an empty materials database."""
        self.material_names: list[str] = []

    def exists_material(self, name: str) -> bool:
        """Return whether the material exists in the project."""
        return name in self.material_names

    def add_material(self, name: str) -> MockMaterial:
        """Add a material to the project."""
        self.material_names.append(name)
        return MockMaterial()


class MockQ3dApp:
    """Minimal stand-in for a PyAEDT Q3d application."""

    def __init__(self, object_names: list[str]):
        """Initialize with the names of objects created by the GDS import."""
        self.imported_object_names = list(object_names)
        self.modeler = MockModeler([])
        self.materials = MockMaterials()
        self.assigned_materials: dict[str, str] = {}

    def import_gds_3d(self, **kwargs: object) -> bool:
        """Simulate a successful 3D GDS import creating the objects."""
        self.import_kwargs = kwargs
        self.modeler.object_names.extend(self.imported_object_names)
        for name in self.imported_object_names:
            self.modeler._objects[name] = MockAEDTObject(name)
        return True

    def assign_material(self, assignment: list, material: str) -> None:
        for obj in assignment:
            self.assigned_materials[str(obj)] = material


def test_object_names_to_materials():
    """Test mapping imported object names to materials from the layer stack."""
    # Layer 1 -> M1 (Nb, a metal -> pec); layer 98 -> Substrate (Si); layer 31 -> TSV (TiN)
    result = object_names_to_materials(
        ["signal1", "signal25", "signal98", "signal31", "Substrate", "M1_offset"],
        LAYER_STACK,
    )

    # Metals are PEC
    assert result["signal1"] == "pec"
    assert result["signal25"] == "pec"  # NbTiN
    assert result["signal31"] == "pec"
    assert result["M1_offset"] == "pec"
    # Substrate and other dielectrics get their real material, not pec
    assert result["signal98"] == "Si"
    assert result["Substrate"] == "Si"
    assert result["Substrate"] != "pec"

    # Objects that cannot be resolved to a layer level fail closed
    with pytest.raises(ValueError, match="Could not resolve"):
        object_names_to_materials(["signal7", "unknown_object"], LAYER_STACK)


def test_object_names_to_materials_unregistered_material():
    """Levels with materials missing from material_properties fail closed."""
    stack = LayerStack(
        layers={
            "Custom": LayerLevel(
                name="Custom",
                layer=(50, 0),
                thickness=1,
                zmin=0.0,
                material="unobtainium",
            )
        }
    )

    with pytest.raises(ValueError, match="not registered in material_properties"):
        object_names_to_materials(["signal50"], stack)


def test_q3d_import_assigns_materials_from_layer_stack(tmp_path):
    """Q3D import must not blanket-assign pec: substrate gets its real material."""
    comp = gf.components.rectangle(size=(10, 10), layer=LAYER.M1_DRAW)

    # Simulate a GDS import that creates the M1 metal and the Substrate objects
    app = MockQ3dApp(["signal1", "signal98"])
    sim = Q3D(app)

    renamed = sim.import_component(comp, gds_path=tmp_path / "comp.gds")

    # Only conductor objects are returned for downstream net assignment
    assert renamed == ["M1"]
    # Metal level becomes PEC, substrate gets silicon from the layer stack
    assert app.assigned_materials["M1"] == "pec"
    assert app.assigned_materials["Substrate"] == "Si"
    assert app.assigned_materials["Substrate"] != "pec"


@pytest.mark.hfss
def test_hfss_import_and_draw(tmp_path: Path, ansys_install_path: Path):
    """Test creating an HFSS project and drawing a component."""
    if not ansys_install_path.exists():
        pytest.skip(f"HFSS installation not found at {ansys_install_path}")

    from ansys.aedt.core import Hfss, settings  # ruff: ignore[import-outside-top-level]

    settings.use_grpc_uds = False

    PDK.activate()
    comp = resonator(length=1000, meanders=1)

    project_dir = tmp_path / "aedt"
    project_dir.mkdir(exist_ok=True)
    project_name = f"test_draw_{uuid.uuid4().hex}"
    with _aedt_project(
        lambda: Hfss(
            project=str(project_dir / project_name),
            solution_type="Eigenmode",
            non_graphical=True,
        ),
        project_dir,
        label="Hfss",
    ) as hfss_app:
        hfss_sim = HFSS(hfss_app)
        # Use direct draw which we know is stable on Linux
        success = _run_with_timeout(
            lambda: hfss_sim.import_component(comp), label="HFSS.import_component"
        )
        assert success, "Failed to draw component in HFSS"


@pytest.mark.hfss
def test_hfss_eigenmode_setup(tmp_path: Path, ansys_install_path: Path):
    """Test setting up an eigenmode simulation."""
    if not ansys_install_path.exists():
        pytest.skip(f"HFSS installation not found at {ansys_install_path}")

    from ansys.aedt.core import Hfss, settings  # ruff: ignore[import-outside-top-level]

    settings.use_grpc_uds = False

    PDK.activate()
    comp = resonator(length=1000, meanders=1)

    project_dir = tmp_path / "aedt"
    project_dir.mkdir(exist_ok=True)
    project_name = f"test_eigenmode_{uuid.uuid4().hex}"
    with _aedt_project(
        lambda: Hfss(
            project=str(project_dir / project_name),
            solution_type="Eigenmode",
            non_graphical=True,
        ),
        project_dir,
        label="Hfss",
    ) as hfss_app:
        hfss_sim = HFSS(hfss_app)
        _run_with_timeout(
            lambda: hfss_sim.import_component(comp), label="HFSS.import_component"
        )
        _run_with_timeout(
            lambda: hfss_sim.add_substrate(comp), label="HFSS.add_substrate"
        )
        _run_with_timeout(
            lambda: hfss_sim.add_air_region(comp), label="HFSS.add_air_region"
        )

        setup = _run_with_timeout(
            lambda: hfss_app.create_setup(name="EigenmodeSetup"),
            label="Hfss.create_setup",
        )
        setup.props["MinimumFrequency"] = "1.0GHz"
        setup.props["NumModes"] = 1
        setup.props["MaximumPasses"] = 2
        setup.props["MinimumPasses"] = 2
        setup.props["PercentRefinement"] = 30
        setup.props["MaxDeltaFreq"] = 2.0
        setup.props["ConvergeOnRealFreq"] = True
        _run_with_timeout(setup.update, label="EigenmodeSetup.update")

        assert setup is not None, "Failed to setup eigenmode simulation"


@pytest.fixture
def mock_port_dimensions():
    """Provides a standardized set of dimensions for testing."""
    return {"center": [10.0, 20.0, 0.0], "cpw_gap": 6.0, "cpw_width": 2.0}


@pytest.mark.parametrize(
    ("orientation", "expected_origin", "expected_sizes", "expected_int_line"),
    [
        (0, [10.0, 19.0, 0.0], [6.0, 2.0], [[16.0, 20.0, 0.0], [10.0, 20.0, 0.0]]),
        (90, [9.0, 20.0, 0.0], [2.0, 6.0], [[10.0, 26.0, 0.0], [10.0, 20.0, 0.0]]),
        (180, [4.0, 19.0, 0.0], [6.0, 2.0], [[4.0, 20.0, 0.0], [10.0, 20.0, 0.0]]),
        (270, [9.0, 14.0, 0.0], [2.0, 6.0], [[10.0, 14.0, 0.0], [10.0, 20.0, 0.0]]),
    ],
)
def test_lumped_port_rectangle_from_cpw_valid_angles(
    mock_port_dimensions,
    orientation,
    expected_origin,
    expected_sizes,
    expected_int_line,
):
    """Verifies that the vectorized geometry perfectly matches the expected dictionary values."""
    result = lumped_port_rectangle_from_cpw(
        center=mock_port_dimensions["center"],
        orientation=orientation,
        cpw_gap=mock_port_dimensions["cpw_gap"],
        cpw_width=mock_port_dimensions["cpw_width"],
    )

    assert_allclose(result["origin"], expected_origin)
    assert_allclose(result["sizes"], expected_sizes)
    assert_allclose(result["integration_line"], expected_int_line)


def test_lumped_port_rectangle_from_cpw_invalid_angle(mock_port_dimensions):
    """Ensures the function throws a ValueError if passed an unaligned angle."""
    with pytest.raises(ValueError, match="Unsupported port orientation: 45°"):
        lumped_port_rectangle_from_cpw(
            center=mock_port_dimensions["center"],
            orientation=45,
            cpw_gap=mock_port_dimensions["cpw_gap"],
            cpw_width=mock_port_dimensions["cpw_width"],
        )


@pytest.mark.hfss
def test_q3d_import_and_net_assignment(tmp_path: Path, ansys_install_path: Path):
    """Test creating a Q3D project, importing a component, and assigning nets."""
    if not ansys_install_path.exists():
        pytest.skip(f"HFSS/Q3D installation not found at {ansys_install_path}")

    from ansys.aedt.core import Q3d, settings  # ruff: ignore[import-outside-top-level]

    settings.use_grpc_uds = False

    PDK.activate()
    comp = interdigital_capacitor(fingers=4, finger_length=20)

    project_dir = tmp_path / "aedt"
    project_dir.mkdir(exist_ok=True)
    project_name = f"test_q3d_{uuid.uuid4().hex}"
    with _aedt_project(
        lambda: Q3d(
            project=str(project_dir / project_name),
            solution_type="Q3DExtractor",
            non_graphical=True,
        ),
        project_dir,
        label="Q3d",
    ) as q3d_app:
        q3d_sim = Q3D(q3d_app)
        conductor_objects = _run_with_timeout(
            lambda: q3d_sim.import_component(comp), label="Q3D.import_component"
        )
        assert len(conductor_objects) > 0, "No conductor objects imported"

        signal_nets = _run_with_timeout(
            lambda: q3d_sim.assign_nets_from_ports(comp.ports, conductor_objects),
            label="Q3D.assign_nets_from_ports",
        )
        assert len(signal_nets) > 0, "No signal nets assigned"


@pytest.mark.hfss
def test_create_2d_from_cross_section(tmp_path: Path, ansys_install_path: Path):
    """Test creating a Q2D model from a CPW cross-section."""
    if not ansys_install_path.exists():
        pytest.skip(f"HFSS installation not found at {ansys_install_path}")

    from ansys.aedt.core import Q2d, settings  # ruff: ignore[import-outside-top-level]

    settings.use_grpc_uds = False

    PDK.activate()
    cross_section = coplanar_waveguide(width=10, gap=6)

    project_dir = tmp_path / "aedt"
    project_dir.mkdir(exist_ok=True)
    project_name = f"test_q2d_{uuid.uuid4().hex}"
    with _aedt_project(
        lambda: Q2d(
            project=str(project_dir / project_name),
            design="CPW_Test",
            non_graphical=True,
        ),
        project_dir,
        label="Q2d",
    ) as q2d_app:
        q2d_sim = Q2D(q2d_app)
        result = _run_with_timeout(
            lambda: q2d_sim.create_2d_from_cross_section(cross_section),
            label="Q2D.create_2d_from_cross_section",
        )

        assert isinstance(result, dict)
        assert "signal" in result
        assert "gnd_left" in result
        assert "gnd_right" in result
        assert "substrate" in result
