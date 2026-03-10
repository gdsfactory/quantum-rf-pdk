"""Tests for qpdk.helper module - deprecated decorator and show_components."""

import warnings

from qpdk.helper import deprecated


class TestDeprecatedDecorator:
    """Tests for the deprecated() decorator."""

    @staticmethod
    def test_deprecated_with_custom_message() -> None:
        """Test @deprecated("custom message") produces custom warning text."""

        @deprecated("my custom deprecation message")
        def old_function():
            return 42

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = old_function()
            assert result == 42
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "my custom deprecation message" in str(w[0].message)

    @staticmethod
    def test_deprecated_without_message() -> None:
        """Test @deprecated (no args) produces default warning text."""

        @deprecated
        def old_function():
            return 99

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = old_function()
            assert result == 99
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "old_function is deprecated" in str(w[0].message)

    @staticmethod
    def test_deprecated_with_none_message() -> None:
        """Test @deprecated(None) produces default warning text."""

        @deprecated(None)
        def old_function():
            return 7

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = old_function()
            assert result == 7
            assert len(w) == 1
            assert "old_function is deprecated" in str(w[0].message)

    @staticmethod
    def test_deprecated_preserves_function_name() -> None:
        """Test that @deprecated preserves __name__ via wraps."""

        @deprecated("msg")
        def my_func():
            pass

        assert my_func.__name__ == "my_func"

    @staticmethod
    def test_deprecated_preserves_arguments() -> None:
        """Test that @deprecated correctly passes arguments."""

        @deprecated
        def add(a, b):
            return a + b

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            assert add(3, 4) == 7

    @staticmethod
    def test_deprecated_preserves_kwargs() -> None:
        """Test that @deprecated correctly passes keyword arguments."""

        @deprecated("old")
        def greet(name="world"):
            return f"hello {name}"

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            assert greet(name="test") == "hello test"
