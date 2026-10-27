"""Full sweeps retain every labelled port and solve each geometry only once."""

import importlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import typer
from click import unstyle
from typer.testing import CliRunner

from qpdk.models.datasets import s_parameters_model
from qpdk.models.datasets.generate import merge, partition_grid


@pytest.fixture
def driven(monkeypatch: pytest.MonkeyPatch):
    scripts = Path(__file__).resolve().parents[2] / "qpdk/models/datasets/data"
    monkeypatch.syspath_prepend(str(scripts))
    return importlib.import_module("_driven")


@pytest.mark.parametrize(
    "frequencies",
    [[], [0.0], [-1.0], [float("nan")], [float("inf")], [2e9, 1e9], [1e9, 1e9]],
)
def test_invalid_frequency_grids_fail_before_preparation(
    driven, frequencies: list[float]
) -> None:
    prepare = Mock()
    with pytest.raises(ValueError, match="Frequencies"):
        driven.Recipe(
            name="fixture",
            script=Path(__file__),
            grid={"width": [2.0]},
            frequencies=frequencies,
            terminals=("o1",),
            prepare=prepare,
            description="Preflight fixture",
            provenance={},
        )
    prepare.assert_not_called()


def test_cli_preview_generation_and_merge(
    driven, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recipe = driven.Recipe(
        name="fixture",
        script=Path(__file__),
        grid={"width": [2.0, 4.0]},
        frequencies=[1e9, 2e9],
        terminals=("o1",),
        prepare=Mock(),
        description="CLI fixture",
        provenance={},
    )
    app = typer.Typer()
    monkeypatch.setattr(driven.typer, "run", app.command())
    generate, merge = Mock(), Mock()
    monkeypatch.setattr(driven, "generate", generate)
    monkeypatch.setattr(driven, "merge", merge)
    driven.run_cli(recipe)
    runner = CliRunner()
    assert runner.invoke(app, ["--dry-run"]).exit_code == 0
    generate.assert_not_called()
    result = runner.invoke(app, ["--shard", "1", "--shards", "2", "--processes", "8"])
    assert result.exit_code == 0, result.output
    assert generate.call_args.kwargs["grid"] == {"width": [4.0]}
    assert generate.call_args.kwargs["processes"] == 8
    assert generate.call_args.kwargs["output"] == Path("build/datasets/fixture")
    result = runner.invoke(app, ["--executable", "palace", "--sif", "palace.sif"])
    assert result.exit_code != 0
    assert "Choose --executable or --sif" in unstyle(result.output)
    result = runner.invoke(
        app,
        ["--merge-shards", str(tmp_path / "a"), "--merge-shards", str(tmp_path / "b")],
    )
    assert result.exit_code == 0, result.output
    assert merge.call_args.args[0] == [tmp_path / "a", tmp_path / "b"]
    assert merge.call_args.kwargs["grid"] == {
        "width": [2.0, 4.0],
        "frequency": [1e9, 2e9],
    }


@pytest.mark.parametrize("ports", [1, 3, 5])
def test_full_sweep_shards_and_resume(
    driven, ports: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    terminals = tuple(f"terminal_{i}" for i in reversed(range(ports)))
    frequencies = [1e9, 3e9, 5e9]
    prepare = Mock(return_value=Mock())
    recipe = driven.Recipe(
        name="fixture",
        script=Path(__file__),
        grid={"width": [2.0, 4.0]},
        frequencies=frequencies,
        variants={"turns": ["3", "5"]},
        terminals=terminals,
        prepare=prepare,
        description="Algebra fixture, not simulated data",
        provenance={"test_fixture": True, "reference_impedance_ohm": 50.0},
    )
    monkeypatch.setattr(driven, "runtime", lambda *_args: ({}, {}))
    monkeypatch.setattr(driven, "fingerprint", lambda _path: {})

    def response(run: Path, *_args, **_kwargs):
        point = json.loads((run / "inputs.json").read_text())["point"]
        values = (
            (jnp.asarray(frequencies) / 1e10 + 1j * point["width"] / 100)
            * int(point["turns"])
            / 10
        )
        return values[:, None, None] * jnp.eye(ports)

    monkeypatch.setattr(driven, "response", response)
    paths = [tmp_path / str(shard) for shard in range(2)]
    for shard, path in enumerate(paths):
        driven.generate(
            recipe,
            grid=partition_grid(recipe.grid, shard=shard, shards=2),
            output=path,
            workdir=tmp_path / "runs",
        )
    assert prepare.call_count == 4
    assert prepare.return_value.run_local.call_count == 4
    data = merge(
        paths,
        tmp_path / "merged",
        grid={**recipe.grid, "frequency": frequencies},
        variants=recipe.variants,
    )
    grid = data.grid("s_parameters", turns="3", allow_missing=False)
    assert grid.values.shape == (2, 3, ports, ports)
    assert grid.terminals == terminals
    model = s_parameters_model(data, turns="3")
    values = jax.jit(model)(f=jnp.asarray([2e9, 4e9]), width=3.0)
    np.testing.assert_allclose(
        values[terminals[0], terminals[0]], jnp.asarray([0.06 + 0.009j, 0.12 + 0.009j])
    )
    assert len(values) == ports**2
    assert jnp.isnan(model(f=6e9, width=3.0)[terminals[0], terminals[0]])

    prepare.reset_mock()
    driven.generate(
        recipe, grid=recipe.grid, output=tmp_path / "resumed", workdir=tmp_path / "runs"
    )
    prepare.assert_not_called()
    driven.generate(
        replace(recipe, provenance={**recipe.provenance, "film": "changed"}),
        grid=recipe.grid,
        output=tmp_path / "changed",
        workdir=tmp_path / "runs",
    )
    assert prepare.call_count == 4


def test_interrupted_sweep_resumes_only_unfinished_geometry(
    driven, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sim = Mock()
    sim.run_local.side_effect = [None, RuntimeError("solver interrupted"), None]
    prepare = Mock(return_value=sim)
    recipe = driven.Recipe(
        name="interrupted-fixture",
        script=Path(__file__),
        grid={"width": [2.0, 4.0]},
        frequencies=[1e9, 2e9],
        terminals=("o1",),
        prepare=prepare,
        description="Recovery fixture, not simulated data",
        provenance={"test_fixture": True, "linear_initial_guess": False},
    )
    monkeypatch.setattr(driven, "runtime", lambda *_args: ({}, {}))
    monkeypatch.setattr(driven, "fingerprint", lambda _path: {})
    monkeypatch.setattr(
        driven,
        "response",
        lambda *_args, **_kwargs: jnp.zeros((2, 1, 1), dtype=complex),
    )
    arguments = dict(
        grid=recipe.grid, output=tmp_path / "data", workdir=tmp_path / "runs"
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        driven.generate(recipe, **arguments)
    assert len(list((tmp_path / "runs").glob("*/complete.json"))) == 1
    data = driven.generate(recipe, **arguments)
    assert prepare.call_count == 3
    assert len(list((tmp_path / "runs").glob("*/complete.json"))) == 2
    assert data.grid("s_parameters", allow_missing=False).values.shape == (2, 2, 1, 1)


@pytest.mark.palace
@pytest.mark.parametrize("ports", [1, 3, 5])
def test_native_matrix_order_and_quality(
    driven, ports: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checks = pytest.importorskip("gsim.palace._validation_log")
    monkeypatch.setattr(
        checks,
        "solver_checks",
        lambda *_args, **_kwargs: (SimpleNamespace(status="passed"), None),
    )
    terminals = tuple(f"p{i}" for i in range(ports))
    (tmp_path / "solver.log").write_text("convergence is supplied by the test fixture")
    (tmp_path / "port_information.json").write_text(
        json.dumps({
            "ports": [
                {"portnumber": i + 1, "name": name, "Z0": 50.0}
                for i, name in enumerate(terminals)
            ]
        })
    )
    expected = jnp.diag(jnp.arange(1, ports + 1) / 10).astype(complex) + 0.01j

    def write_csv(matrix) -> None:
        columns = ["f (GHz)"]
        row = ["1.0"]
        for i in range(ports):
            for j in range(ports):
                columns.extend((
                    f"|S[{i + 1}][{j + 1}]| (dB)",
                    f"arg(S[{i + 1}][{j + 1}]) (deg.)",
                ))
                row.extend((
                    str(float(20 * jnp.log10(jnp.abs(matrix[i, j])))),
                    str(float(jnp.angle(matrix[i, j], deg=True))),
                ))
        (tmp_path / "port-S.csv").write_text(
            ",".join(columns) + "\n" + ",".join(row) + "\n"
        )

    write_csv(expected)
    matrix = driven.response(tmp_path, terminals[::-1], [1e9])
    np.testing.assert_allclose(matrix[0], expected[::-1, ::-1], atol=1e-12)
    for missing_column in (1, 2):
        write_csv(expected)
        path = tmp_path / "port-S.csv"
        rows = [line.split(",") for line in path.read_text().splitlines()]
        path.write_text(
            "\n".join(
                ",".join(row[:missing_column] + row[missing_column + 1 :])
                for row in rows
            )
            + "\n"
        )
        with pytest.raises(ValueError, match="Incomplete complex"):
            driven.response(tmp_path, terminals, [1e9])
    write_csv(expected)
    with pytest.raises(ValueError, match="ports"):
        driven.response(tmp_path, (*terminals, "absent"), [1e9])
    with pytest.raises(AssertionError):
        driven.response(tmp_path, terminals, [2e9])
    write_csv(jnp.eye(ports) * 1.1)
    with pytest.raises(ValueError, match="passivity"):
        driven.response(tmp_path, terminals, [1e9])
    write_csv(expected.at[0, 0].set(jnp.nan))
    with pytest.raises(ValueError, match="Invalid numeric value"):
        driven.response(tmp_path, terminals, [1e9])
    if ports > 1:
        asymmetric = expected.at[0, 1].set(0.03j)
        write_csv(asymmetric)
        with pytest.raises(ValueError, match="reciprocity"):
            driven.response(tmp_path, terminals, [1e9])
        driven.response(tmp_path, terminals, [1e9], reciprocal=False)
    monkeypatch.setattr(
        checks,
        "solver_checks",
        lambda *_args, **_kwargs: (SimpleNamespace(status="failed"), None),
    )
    with pytest.raises(ValueError, match="did not converge"):
        driven.response(tmp_path, terminals, [1e9])
