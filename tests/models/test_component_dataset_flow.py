"""Component sweeps retain labelled matrices across the native solver boundary."""

import importlib
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Self
from unittest.mock import Mock, call

import gdsfactory as gf
import jax.numpy as jnp
import numpy as np
import pytest
import typer
from typer.testing import CliRunner

from qpdk.models.datasets.generate import merge, partition_grid
from qpdk.tech import LAYER


@pytest.fixture
def experiment(monkeypatch: pytest.MonkeyPatch):
    scripts = Path(__file__).resolve().parents[2] / "qpdk/models/datasets/recipes"
    monkeypatch.syspath_prepend(str(scripts))
    return importlib.import_module("_experiment")


@pytest.fixture
def recipe(experiment):
    def geometry(point: dict, settings, _processes: int):
        component = gf.Component()
        for name, x in (("a", 0.0), ("b", 10.0)):
            component.add_polygon(
                [(x, 0), (x + point["width"], 0), (x + point["width"], 4), (x, 4)],
                layer=LAYER.M1_DRAW,
            )
            component.add_port(name, center=(x, 2), width=1, layer=LAYER.M1_DRAW)
        bounds = experiment.ground_frame(
            component, clearance=5, width=10, domain_pad=settings.domain_pad
        )
        return experiment.Geometry(
            component=component, bounds=bounds, terminal_ports={"a": "a", "b": "b"}
        )

    return experiment.Recipe(
        name="algebra",
        script=Path(__file__),
        grid={"width": [2.0, 4.0]},
        variants={"scale": ["1", "2"]},
        terminals=("a", "b"),
        geometry=geometry,
        description="Algebra fixture for the table workflow; no FEM calculation.",
        provenance={"test_fixture": True},
    )


def test_sweep_retains_quantities_and_variant_grids(
    experiment, recipe, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    helper = importlib.import_module("_palace")
    monkeypatch.setattr(helper, "runtime", lambda *_args: ({}, {}))
    monkeypatch.setattr(helper, "fingerprint", lambda _path: {})

    def solve(*_args, **kwargs) -> dict:
        point = kwargs["inputs"]["point"]
        return {
            "maxwell_capacitance": jnp.array([[3.0, -1.0], [-1.0, 3.0]])
            * point["width"]
            * float(point["scale"])
            * 1e-15,
            "fem_error_indicator_norm": 0.1,
            "solver_relative_residual": 1e-10,
            "solver_iterations": 3.0,
        }

    extract = Mock(side_effect=solve)
    monkeypatch.setattr(helper, "extract", extract)
    data = experiment.generate(
        recipe, grid=recipe.grid, output=tmp_path / "data", workdir=tmp_path / "runs"
    )
    assert extract.call_count == 4
    inputs = extract.call_args.kwargs["inputs"]
    assert "experiment_sha256" not in inputs["provenance"]
    assert "experiment_sha256" in data.metadata.provenance
    assert data.metadata.terminals == recipe.terminals
    assert {q.name for q in data.metadata.quantities} == {
        "maxwell_capacitance",
        "mutual_capacitance",
        "fem_error_indicator_norm",
        "solver_relative_residual",
        "solver_iterations",
    }
    for scale in recipe.variants["scale"]:
        maxwell = data.grid("maxwell_capacitance", allow_missing=False, scale=scale)
        mutual = data.grid("mutual_capacitance", allow_missing=False, scale=scale)
        np.testing.assert_allclose(mutual.values[:, 0, 1], -maxwell.values[:, 0, 1])
        np.testing.assert_allclose(
            mutual.values[:, 0, 0], maxwell.values[:, 0, :].sum(-1)
        )

    def reversed_ports(*args):
        result = recipe.geometry(*args)
        return replace(result, terminal_ports={"b": "b", "a": "a"})

    extract.reset_mock()
    with pytest.raises(ValueError, match="terminal order"):
        experiment.generate(
            replace(recipe, geometry=reversed_ports),
            grid=recipe.grid,
            output=tmp_path / "invalid",
            workdir=tmp_path / "invalid-runs",
        )
    extract.assert_not_called()


def test_shards_merge_with_explicit_axis_units(
    experiment, recipe, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    helper = importlib.import_module("_palace")
    monkeypatch.setattr(helper, "runtime", lambda *_args: ({}, {}))
    monkeypatch.setattr(helper, "fingerprint", lambda _path: {})
    monkeypatch.setattr(
        helper,
        "extract",
        lambda *_args, **_kwargs: {
            "maxwell_capacitance": jnp.array([[3.0, -1.0], [-1.0, 3.0]]) * 1e-15,
            "fem_error_indicator_norm": 0.1,
            "solver_relative_residual": 1e-10,
            "solver_iterations": 3.0,
        },
    )
    recipe = replace(
        recipe,
        grid={**recipe.grid, "rotation": [0.0, 90.0]},
        axis_units={"rotation": "deg"},
    )
    paths = [tmp_path / str(shard) for shard in range(2)]
    for shard, path in enumerate(paths):
        experiment.generate(
            recipe,
            grid=partition_grid(recipe.grid, shard=shard, shards=2),
            output=path,
            workdir=tmp_path / "runs",
        )
    merged = merge(
        paths, tmp_path / "merged", grid=recipe.grid, variants=recipe.variants
    )
    assert [(axis.name, axis.unit) for axis in merged.metadata.axes] == [
        ("width", "um"),
        ("rotation", "deg"),
    ]
    assert merged.grid(
        "maxwell_capacitance", allow_missing=False, scale="1"
    ).values.shape == (2, 2, 2, 2)


@pytest.mark.parametrize("mode", ["preview", "generate", "merge", "invalid-runtime"])
def test_cli_dispatch(
    experiment, recipe, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    args = {
        "preview": ["--dry-run"],
        "generate": ["--shard", "1", "--shards", "2"],
        "merge": ["--merge-shards", "first", "--merge-shards", "second"],
        "invalid-runtime": ["--sif", "image.sif", "--executable", "palace"],
    }[mode]
    generate, merge = Mock(), Mock()
    monkeypatch.setattr(experiment, "generate", generate)
    monkeypatch.setattr(experiment, "merge", merge)
    results = []

    def invoke(main) -> None:
        app = typer.Typer()
        app.command()(main)
        results.append(CliRunner().invoke(app, args))

    monkeypatch.setattr(typer, "run", invoke)
    experiment.run_cli(recipe)
    assert results[0].exit_code == (2 if mode == "invalid-runtime" else 0), results[
        0
    ].output
    if mode == "generate":
        assert generate.call_args.kwargs["grid"] == {"width": [4.0]}
    else:
        generate.assert_not_called()
    if mode == "merge":
        assert merge.call_args.kwargs["grid"] == recipe.grid
        assert merge.call_args.kwargs["variants"] == recipe.variants
    else:
        merge.assert_not_called()


@pytest.mark.parametrize(
    "settings",
    [
        {"near_mesh": 0.0},
        {"domain_pad": 0.0},
        {"permittivity": 0.0},
        {"order": 0},
        {"tolerance": 0.0},
    ],
)
def test_invalid_mesh_settings(experiment, settings: dict) -> None:
    with pytest.raises(ValueError, match=r"positive domain|Invalid permittivity"):
        experiment.Settings(**settings)


def test_stack_maps_electrodes_without_exciting_ground(
    experiment, monkeypatch: pytest.MonkeyPatch
) -> None:
    common = ModuleType("gsim.common")

    class Stack(SimpleNamespace):
        @classmethod
        def model_validate(cls, values: dict) -> Self:
            return cls(**{
                **values,
                "layers": {
                    name: SimpleNamespace(**layer)
                    for name, layer in values["layers"].items()
                },
            })

    monkeypatch.setattr(common, "LayerStack", Stack, raising=False)
    monkeypatch.setitem(sys.modules, "gsim", ModuleType("gsim"))
    monkeypatch.setitem(sys.modules, "gsim.common", common)
    rectangle = gf.kdb.Region(gf.kdb.Box(0, 0, 1000, 1000))
    electrodes = {
        "a": (rectangle, 0.0, 0.2),
        "b": (rectangle.moved(2000, 0), 1.0, 1.2),
        "ground": (rectangle.moved(4000, 0), 0.0, 0.2),
    }
    geometry = experiment.stack_geometry(
        electrodes,
        terminals=("a", "b"),
        dielectrics=[],
        settings=experiment.Settings(),
        processes=2,
    )
    sim = Mock()
    geometry.mesh(sim)
    stack = sim.set_stack.call_args.args[0]
    assert set(stack.layers) == set(electrodes)
    assert len({layer.gds_layer for layer in stack.layers.values()}) == 3
    assert tuple(geometry.terminal_ports) == ("a", "b")
    assert sim.add_terminal.call_args_list == [
        call("a", layer="a"),
        call("b", layer="b"),
    ]
    assert sim.mesh.call_args.kwargs["planar_conductors"] is False
    assert stack.layers["b"].zmin == pytest.approx(1.0)
    assert stack.layers["b"].zmax == pytest.approx(1.2)
    assert geometry.mesh_inputs["stack"]["layers"]["b"]["zmin"] == pytest.approx(1.0)
    assert geometry.mesh_inputs["mesh"] == sim.mesh.call_args.kwargs

    for region, bottom, top in [
        (gf.kdb.Region(), 0, 1),
        (rectangle, 1, 1),
        (rectangle, 2, 1),
    ]:
        with pytest.raises(ValueError, match="Electrode 'bad'"):
            experiment.stack_geometry(
                {"bad": (region, bottom, top)},
                terminals=("bad",),
                dielectrics=[],
                settings=experiment.Settings(),
                processes=1,
            )
