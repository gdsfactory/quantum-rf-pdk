"""Generation recipes, previews and generator dispatch."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from qpdk.models.datasets import recipe

RECIPE = """generator = "qpdk.models.datasets.plate_capacitor:generate"
output = "results"
workdir = "runs"

[grid]
length = [40.0, 80.0]
width = [10.0]
gap = [4.0, 7.0, 10.0]

[palace]
command = ["apptainer", "exec", "image with spaces.sif", "palace", "-np", "4"]

[palace.settings]
near_mesh = 0.8
"""


@pytest.fixture
def config(tmp_path: Path) -> Path:
    path = tmp_path / "recipe.toml"
    path.write_text(RECIPE, encoding="utf-8")
    return path


def test_recipe_paths_and_grid(config: Path, tmp_path: Path) -> None:
    loaded = recipe.load_recipe(config)
    assert loaded.points == 6
    assert loaded.output == tmp_path / "results"
    assert loaded.workdir == tmp_path / "runs"
    assert loaded.settings.near_mesh == pytest.approx(0.8)
    assert loaded.command[2] == "image with spaces.sif"


@pytest.mark.parametrize(
    ("before", "after", "error"),
    [
        ('output = "results"', "", "Missing recipe fields"),
        ('output = "results"', 'ouptut = "results"', "Missing recipe fields"),
        (
            'output = "results"',
            'output = "results"\nunknown = 1',
            "Unknown recipe fields",
        ),
        ("length = [40.0, 80.0]", "length = []", "unique finite"),
        ("length = [40.0, 80.0]", "length = [40.0, 40.0]", "unique finite"),
        ("length = [40.0, 80.0]", "length = [nan]", "unique finite"),
        ("length = [40.0, 80.0]", "length = 40.0", "unique finite"),
        ("length = [40.0, 80.0]", "length = [true]", "unique finite"),
        (
            'command = ["apptainer", "exec", "image with spaces.sif", "palace", "-np", "4"]',
            'command = "palace -np 4"',
            "array of strings",
        ),
        ("near_mesh = 0.8", "near_mesh = -1.0", "Mesh sizes"),
        (
            "qpdk.models.datasets.plate_capacitor:generate",
            "not a function",
            "module:function",
        ),
    ],
)
def test_invalid_recipe(config: Path, before: str, after: str, error: str) -> None:
    config.write_text(RECIPE.replace(before, after), encoding="utf-8")
    with pytest.raises((ValueError, TypeError), match=error):
        recipe.load_recipe(config)


def test_dispatch_custom_generator(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = replace(recipe.load_recipe(config), generator="custom_dataset:generate")
    generate = Mock(return_value=object())
    module = Mock(generate=generate)
    imported = Mock(return_value=module)
    monkeypatch.setattr(recipe.importlib, "import_module", imported)
    assert loaded.run() is generate.return_value
    imported.assert_called_once_with("custom_dataset")
    kwargs = generate.call_args.kwargs
    assert kwargs["runner"].command == loaded.command
    assert kwargs["runner"].settings == loaded.settings
    assert kwargs["grid"] == loaded.grid
    assert kwargs["output"] == loaded.output


def test_cli_preview_does_not_run_and_preserves_quoted_args(
    config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "recipe",
            str(config),
            "--dry-run",
            "--output",
            "output with spaces",
            "--palace-command",
            "apptainer exec 'image with spaces.sif' palace",
        ],
    )
    run = Mock(side_effect=AssertionError("Preview must not solve"))
    log = Mock()
    monkeypatch.setattr(recipe.GenerationRecipe, "run", run)
    monkeypatch.setattr(recipe.logger, "info", log)
    recipe.main()
    run.assert_not_called()
    message = log.call_args.args[0]
    assert "6 geometries" in message
    assert str(tmp_path / "output with spaces") in message
    assert "'image with spaces.sif'" in message
