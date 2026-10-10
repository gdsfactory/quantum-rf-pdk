"""Tests for QPDK's AEDT test helpers, ``prepare_component_for_aedt`` and live AEDT.

Everything here except the ``hfss``-marked live tests runs without the ``hfss``
extra. The wrapper tests that need gplugins are in ``test_aedt_wrappers.py``.
"""

import contextlib
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from gdsfactory.component import Component

from qpdk import PDK, logger
from qpdk.cells.capacitor import interdigital_capacitor
from qpdk.cells.resonator import resonator
from qpdk.simulation.aedt_base import prepare_component_for_aedt
from qpdk.tech import coplanar_waveguide

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


def test_prepare_component_for_aedt_imports_without_gplugins():
    """``prepare_component_for_aedt`` must not need gplugins at import time."""
    code = (
        "import sys; sys.modules['gplugins'] = None; "
        "from qpdk.simulation.aedt_base import prepare_component_for_aedt; "
        "from qpdk.simulation import prepare_component_for_aedt as wrapped; "
        "assert wrapped is prepare_component_for_aedt"
    )
    subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-c", code], check=True
    )


def test_missing_extra_raises_actionable_import_error():
    """Without gplugins, the AEDT names raise ``ImportError`` naming the extra."""
    code = """
import sys
sys.modules["gplugins"] = None
import qpdk.simulation.aedt_base as aedt_base
from qpdk.simulation.aedt_base import MISSING_HFSS_EXTRA

def check(get):
    try:
        get()
    except ModuleNotFoundError:
        raise AssertionError("bare ModuleNotFoundError escaped")
    except ImportError as error:
        assert str(error) == MISSING_HFSS_EXTRA, error
        assert "uv sync --extra hfss" in str(error)
    else:
        raise AssertionError("no ImportError")

check(lambda: aedt_base.AEDTBase)
check(lambda: aedt_base.fit_view)
check(aedt_base.layer_stack_to_gds_mapping)
check(lambda: __import__("qpdk.simulation.hfss"))
check(lambda: __import__("qpdk.simulation.q3d"))
"""
    subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-c", code], check=True
    )


@pytest.mark.hfss
def test_hfss_import_and_draw(tmp_path: Path, ansys_install_path: Path):
    """Test creating an HFSS project and drawing a component."""
    if not ansys_install_path.exists():
        pytest.skip(f"HFSS installation not found at {ansys_install_path}")

    from ansys.aedt.core import Hfss, settings  # ruff: ignore[import-outside-top-level]

    from qpdk.simulation import (  # ruff: ignore[import-outside-top-level]
        HFSS,
        detach_desktop_logging,
        fit_view,
    )

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
        # The helpers run against a real session here: the draw below is what proves
        # they left it alive, and the assertion catches a PyAEDT that renamed the
        # attribute the logger keeps the desktop in (the stub tests cannot).
        detach_desktop_logging(hfss_app)
        fit_view(hfss_app)
        assert hfss_app.logger._desktop_class is None
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

    from qpdk.simulation import HFSS  # ruff: ignore[import-outside-top-level]

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


@pytest.mark.hfss
def test_q3d_import_and_net_assignment(tmp_path: Path, ansys_install_path: Path):
    """Test creating a Q3D project, importing a component, and assigning nets."""
    if not ansys_install_path.exists():
        pytest.skip(f"HFSS/Q3D installation not found at {ansys_install_path}")

    from ansys.aedt.core import Q3d, settings  # ruff: ignore[import-outside-top-level]

    from qpdk.simulation import Q3D  # ruff: ignore[import-outside-top-level]

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

    from qpdk.simulation import Q2D  # ruff: ignore[import-outside-top-level]

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
