"""Tests for HFSS and Q3D simulation integration."""

import contextlib
import os
import shutil
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from gdsfactory.component import Component
from numpy.testing import assert_allclose

from qpdk import LAYER_STACK, PDK, logger
from qpdk.cells.capacitor import interdigital_capacitor
from qpdk.cells.resonator import resonator
from qpdk.simulation import (
    HFSS,
    Q2D,
    Q3D,
    layer_stack_to_gds_mapping,
    lumped_port_rectangle_from_cpw,
    prepare_component_for_aedt,
)
from qpdk.simulation.aedt_base import _get_layer_number_from_level
from qpdk.tech import coplanar_waveguide

# Maximum wall-clock time for a single AEDT-bound call (constructor, setup
# update). AEDT startup can legitimately take minutes, so this is generous.
# The timeout is per-call, not per-test: the file wraps up to ~5 calls, so the
# worst (all-timed-out) case is ~83 min across the file (five 900s AEDT calls
# plus four 120s release timeouts).
#
# Thread-affinity caveat: AEDT objects created inside the worker thread (e.g.
# the ``Hfss`` app) are used from the main thread afterwards. gRPC stubs are
# thread-safe and ``use_grpc_uds = False`` plus ``non_graphical=True`` force
# TCP gRPC, but this has not been verified on real AEDT hardware.
_AEDT_TIMEOUT_SECONDS = 900

# ``release_desktop()`` on a wedged session can also block; cut it off quickly.
_RELEASE_TIMEOUT_SECONDS = 120


def _run_with_timeout(
    func: Callable[[], Any], timeout: float = _AEDT_TIMEOUT_SECONDS
) -> Any:
    """Run ``func`` and raise :class:`TimeoutError` if it exceeds ``timeout``.

    AEDT calls communicate over gRPC and can hang indefinitely, which would
    block the whole ``-n auto`` suite with no diagnostic. The call runs in a
    daemon thread so the test fails with a ``TimeoutError`` instead of hanging.
    Note that a genuinely hung call cannot be killed; the abandoned thread
    keeps running until the process exits.

    Args:
        func: Zero-argument callable to run (e.g. a lambda constructing an
            AEDT application, updating a setup, or releasing the desktop).
        timeout: Maximum seconds to wait for ``func`` to complete.

    Returns:
        Whatever ``func`` returns, or ``None`` if it returns ``None``.

    Raises:
        TimeoutError: If ``func`` does not finish within ``timeout``.
    """
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
    msg = f"AEDT call {getattr(func, '__name__', '<lambda>')!r} timed out after {timeout}s"
    raise TimeoutError(msg)


@contextlib.contextmanager
def _aedt_project(
    app_factory: Callable[[], Any],
    project_dir: Path,
    timeout: float = _AEDT_TIMEOUT_SECONDS,
) -> Iterator[Any]:
    """Construct an AEDT app and guarantee release plus project cleanup.

    The constructor runs inside the guarded region so that a hung startup
    (``TimeoutError``) or a raising constructor still reaches the ``finally``:
    the project directory is removed either way, and ``release_desktop()`` is
    attempted whenever an app object exists. A genuinely hung constructor
    cannot be released; its daemon thread keeps running until process exit.

    Args:
        app_factory: Zero-argument callable constructing the AEDT application
            (e.g. ``lambda: Hfss(...)``).
        project_dir: Directory holding the AEDT project; removed on exit.
        timeout: Maximum seconds for the constructor itself.

    Yields:
        The constructed AEDT application object.
    """
    app = None
    try:
        app = _run_with_timeout(app_factory, timeout=timeout)
        yield app
    finally:
        try:
            if app is not None:
                # Cut off a wedged session instead of hanging the xdist worker.
                _run_with_timeout(app.release_desktop, timeout=_RELEASE_TIMEOUT_SECONDS)
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
    ) as hfss_app:
        hfss_sim = HFSS(hfss_app)
        # Use direct draw which we know is stable on Linux
        success = hfss_sim.import_component(comp)
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
    ) as hfss_app:
        hfss_sim = HFSS(hfss_app)
        hfss_sim.import_component(comp)
        hfss_sim.add_substrate(comp)
        hfss_sim.add_air_region(comp)

        setup = hfss_app.create_setup(name="EigenmodeSetup")
        setup.props["MinimumFrequency"] = "1.0GHz"
        setup.props["NumModes"] = 1
        setup.props["MaximumPasses"] = 2
        setup.props["MinimumPasses"] = 2
        setup.props["PercentRefinement"] = 30
        setup.props["MaxDeltaFreq"] = 2.0
        setup.props["ConvergeOnRealFreq"] = True
        _run_with_timeout(setup.update)

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
    ) as q3d_app:
        q3d_sim = Q3D(q3d_app)
        conductor_objects = q3d_sim.import_component(comp)
        assert len(conductor_objects) > 0, "No conductor objects imported"

        signal_nets = q3d_sim.assign_nets_from_ports(comp.ports, conductor_objects)
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
    ) as q2d_app:
        q2d_sim = Q2D(q2d_app)
        result = q2d_sim.create_2d_from_cross_section(cross_section)

        assert isinstance(result, dict)
        assert "signal" in result
        assert "gnd_left" in result
        assert "gnd_right" in result
        assert "substrate" in result
