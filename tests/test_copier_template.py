"""Test copier template generation."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SLUG = "my_quantum_pdk_project"

EXPECTED_FILES = [
    "pyproject.toml",
    "README.md",
    f"{DEFAULT_SLUG}/__init__.py",
    f"{DEFAULT_SLUG}/cells/__init__.py",
    f"{DEFAULT_SLUG}/cells/cross_mark.py",
    f"{DEFAULT_SLUG}/cells/launched_cpw.py",
    f"{DEFAULT_SLUG}/models/__init__.py",
    f"{DEFAULT_SLUG}/models/my_resonator.py",
]


@pytest.fixture
def generated_project(tmp_path: Path) -> Path:
    """Generate a project from the copier template with defaults."""
    from copier import run_copy

    output = tmp_path / "test-project"
    run_copy(
        str(REPO_ROOT),
        str(output),
        defaults=True,
        vcs_ref="HEAD",
    )
    return output


def test_project_files_exist(generated_project: Path) -> None:
    """Test that generated project contains all expected files."""
    for rel_path in EXPECTED_FILES:
        assert (generated_project / rel_path).exists(), f"Missing: {rel_path}"


def test_python_files_valid_syntax(generated_project: Path) -> None:
    """Test that all generated Python files have valid syntax."""
    for py_file in generated_project.rglob("*.py"):
        source = py_file.read_text()
        ast.parse(source, filename=str(py_file))


def test_pyproject_toml_valid(generated_project: Path) -> None:
    """Test that generated pyproject.toml is valid TOML."""
    import tomllib

    toml_path = generated_project / "pyproject.toml"
    data = tomllib.loads(toml_path.read_text())
    assert data["project"]["name"] == DEFAULT_SLUG
    assert "qpdk" in data["project"]["dependencies"]


def test_template_variables_replaced(generated_project: Path) -> None:
    """Test that Jinja2 template variables are fully resolved."""
    for path in generated_project.rglob("*"):
        if path.is_file():
            content = path.read_text()
            assert "{{ " not in content, f"Unresolved template variable in {path}"
            assert " }}" not in content, f"Unresolved template variable in {path}"
