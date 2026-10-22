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

from qpdk.models.datasets import s_parameters_model
from qpdk.models.datasets.generate import merge, partition_grid


@pytest.fixture
def driven(monkeypatch: pytest.MonkeyPatch):
    scripts = Path(__file__).resolve().parents[2] / "qpdk/models/datasets/data"
    monkeypatch.syspath_prepend(str(scripts))
    return importlib.import_module("_driven")


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
    with pytest.raises(ValueError, match="ports"):
        driven.response(tmp_path, (*terminals, "absent"), [1e9])
    with pytest.raises(AssertionError):
        driven.response(tmp_path, terminals, [2e9])
    write_csv(jnp.eye(ports) * 1.1)
    with pytest.raises(ValueError, match="passivity"):
        driven.response(tmp_path, terminals, [1e9])
    if ports > 1:
        asymmetric = expected.at[0, 1].set(0.03j)
        write_csv(asymmetric)
        with pytest.raises(ValueError, match="reciprocity"):
            driven.response(tmp_path, terminals, [1e9])
        driven.response(tmp_path, terminals, [1e9], reciprocal=False)
