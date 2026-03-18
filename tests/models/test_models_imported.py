"""Tests for imports of qpdk.models module."""

import importlib
import inspect
import pkgutil
from collections.abc import Callable

import sax

import qpdk.models


def has_sax_stype_return(func: Callable) -> bool:
    """Check if a function has sax.SType return type annotation.

    Args:
        func: Function to check for sax.SType return type

    Returns:
        bool: True if function has sax.SType or sax.SDict return type, False otherwise
    """
    if hasattr(func, "__annotations__") and "return" in func.__annotations__:
        return_type = func.__annotations__["return"]

        # Check if return type is sax.SType or sax.SDict (which is equivalent)
        if (
            (hasattr(sax, "SType") and return_type is sax.SType)
            or (hasattr(sax, "SDict") and return_type is sax.SDict)
            or (
                isinstance(return_type, type)
                and return_type.__name__ in ["SType", "SDict"]
            )
            or (
                hasattr(return_type, "__name__")
                and return_type.__name__ in ["SType", "SDict"]
            )
        ):
            return True
    return False


def test_has_sax_stype_return():
    """Test the has_sax_stype_return function behavior."""

    # Test with a function that has sax.SType return type
    def func_with_stype() -> sax.SType:
        return {}

    assert has_sax_stype_return(func_with_stype), "Should detect sax.SType return type"

    # Test with a function that has sax.SDict return type
    def func_with_sdict() -> sax.SDict:
        return {}

    assert has_sax_stype_return(func_with_sdict), "Should detect sax.SDict return type"

    # Test with a function that has no return type annotation
    def func_without_annotation():
        return {}

    assert not has_sax_stype_return(func_without_annotation), (
        "Should not detect function without annotation"
    )

    # Test with a function that has different return type
    def func_with_other_type() -> dict:
        return {}

    assert not has_sax_stype_return(func_with_other_type), (
        "Should not detect function with different return type"
    )

    # Test with a function that has no annotations at all
    def func_no_annotations():
        return {}

    # Remove annotations to be sure
    if hasattr(func_no_annotations, "__annotations__"):
        delattr(func_no_annotations, "__annotations__")

    assert not has_sax_stype_return(func_no_annotations), (
        "Should not detect function with no annotations"
    )

    # Test __wrapped__ scenario
    def original_func() -> sax.SType:
        return {}

    def wrapper_func(*args, **kwargs):
        return original_func(*args, **kwargs)

    # Manually set __wrapped__ to simulate a decorator that doesn't preserve annotations
    wrapper_func.__wrapped__ = original_func  # pyrefly: ignore

    # Direct detection should fail
    assert not has_sax_stype_return(wrapper_func), (
        "Wrapper without annotations should not be detected directly"
    )

    # But __wrapped__ should be detectable
    assert has_sax_stype_return(wrapper_func.__wrapped__), (  # pyrefly: ignore
        "Should detect __wrapped__ function with sax.SType"
    )


def test_sax_stype_functions_in_models_dict():
    """Test that all functions with sax.SType return type are included in models dictionary.

    This test checks both direct function annotations and __wrapped__ attributes
    (commonly used by decorators like @jax.jit, @functools.wraps, etc.).
    """
    sax_stype_functions = set()

    for _, modname, ispkg in pkgutil.iter_modules(qpdk.models.__path__):
        if not ispkg:  # Only look at modules, not packages
            try:
                module = importlib.import_module(f"qpdk.models.{modname}")

                for name, obj in inspect.getmembers(module, callable):
                    # Skip private functions
                    if name.startswith("_"):
                        continue

                    # Only consider functions defined in this module OR specifically allowed sax models
                    # sax.models.rf functions are allowed because we re-export them as part of our API
                    is_local = (
                        hasattr(obj, "__module__") and obj.__module__ == module.__name__
                    )
                    is_sax_rf_model = (
                        hasattr(obj, "__module__") and obj.__module__ == "sax.models.rf"
                    )

                    if not (is_local or is_sax_rf_model):
                        continue

                    # Check if function or its __wrapped__ version has sax.SType return type
                    if any(
                        (
                            has_sax_stype_return(obj),
                            hasattr(obj, "__wrapped__")
                            and has_sax_stype_return(obj.__wrapped__),
                        )
                    ):
                        sax_stype_functions.add(name)

            except ImportError:
                continue

    models_names = set(qpdk.models.models.keys())

    assert sax_stype_functions == models_names, (
        "There are functions with sax.SType return type that are missing from the qpdk.models.models dictionary"
    )


if __name__ == "__main__":
    test_has_sax_stype_return()
    test_sax_stype_functions_in_models_dict()
    print("All tests passed!")
