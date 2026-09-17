"""Tests for singleton classes (PATH, PDK, AEDT wrappers)."""

import threading
from itertools import combinations

import pytest
from gdsfactory.pdk import Pdk
from hypothesis import HealthCheck, given, settings, strategies as st
from pydantic import BaseModel

from qpdk import PDK, QPdk, get_pdk
from qpdk.config import PATH, Path
from qpdk.simulation import HFSS, Q2D, Q3D
from qpdk.singleton import SingletonMeta

WRAPPERS = (HFSS, Q3D, Q2D)


@pytest.fixture
def isolated_wrapper_cache():
    """Reset AEDT-wrapper singleton entries around each test.

    The cache is process-global, so without a reset wrapper tests would
    receive instances leaked by earlier tests or leak their own into
    later ones. PATH and PDK entries are deliberately never touched.
    """
    for cls in WRAPPERS:
        SingletonMeta._instances.pop(cls, None)
    yield
    for cls in WRAPPERS:
        SingletonMeta._instances.pop(cls, None)


def test_path_is_singleton() -> None:
    assert Path() is PATH
    assert Path() is Path()
    assert isinstance(Path, SingletonMeta)


@pytest.mark.parametrize("n_threads", [2, 8, 32])
def test_path_singleton_thread_safety(n_threads: int) -> None:
    instances: list[Path] = []
    barrier = threading.Barrier(n_threads)

    def construct() -> None:
        barrier.wait()
        instances.append(Path())

    threads = [threading.Thread(target=construct) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(instances) == n_threads
    assert all(inst is PATH for inst in instances)


@given(n=st.integers(min_value=1, max_value=50))
def test_path_repeated_construction_returns_same_instance(n: int) -> None:
    assert all(Path() is PATH for _ in range(n))


def test_pdk_is_singleton() -> None:
    assert get_pdk() is PDK
    assert get_pdk() is get_pdk()
    assert isinstance(PDK, Pdk)


def test_pdk_uses_singleton_metaclass() -> None:
    assert issubclass(type(QPdk), SingletonMeta)


def test_pdk_remains_pydantic_model() -> None:
    assert isinstance(PDK, BaseModel)
    assert PDK.name == "qpdk"


def test_pdk_ignores_later_arguments() -> None:
    # The metaclass never re-runs __init__, so later kwargs are discarded
    assert QPdk(name="ignored") is PDK
    assert PDK.name == "qpdk"


@given(n=st.integers(min_value=1, max_value=20))
def test_pdk_repeated_construction_returns_same_instance(n: int) -> None:
    assert all(get_pdk() is PDK for _ in range(n))


@pytest.mark.usefixtures("isolated_wrapper_cache")
@pytest.mark.parametrize("cls", WRAPPERS)
def test_wrapper_class_is_singleton(cls: type) -> None:
    app_first, app_later = object(), object()
    sim = cls(app_first)
    assert cls(app_later) is sim
    assert type(sim) is cls
    # Documents the hazard: the app passed later is silently ignored
    assert sim.app is app_first


@pytest.mark.parametrize("cls", WRAPPERS)
def test_wrappers_inherit_singleton_metaclass(cls: type) -> None:
    assert type(cls) is SingletonMeta


@pytest.mark.usefixtures("isolated_wrapper_cache")
@pytest.mark.parametrize(("cls_a", "cls_b"), list(combinations(WRAPPERS, 2)))
def test_wrapper_classes_have_distinct_singletons(cls_a: type, cls_b: type) -> None:
    assert cls_a(object()) is not cls_b(object())


@pytest.mark.usefixtures("isolated_wrapper_cache")
@pytest.mark.parametrize("cls", WRAPPERS)
def test_wrapper_singleton_thread_safety(cls: type) -> None:
    instances: list[object] = []
    barrier = threading.Barrier(8)

    def construct() -> None:
        barrier.wait()
        instances.append(cls(0))

    threads = [threading.Thread(target=construct) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(instances) == 8
    assert all(sim is instances[0] for sim in instances)
    assert type(instances[0]) is cls


@pytest.mark.usefixtures("isolated_wrapper_cache")
@given(apps=st.lists(st.integers(), min_size=1, max_size=20))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_hfss_construction_is_app_independent(apps: list[int]) -> None:
    # The identity property holds from any starting cache state, so the
    # fixture only cleans up afterwards; per-example resets would weaken it
    sims = [HFSS(app) for app in apps]
    assert all(sim is sims[0] for sim in sims)
