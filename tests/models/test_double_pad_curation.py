"""Portable native-proof and admission tests use procedural values, not FEM claims."""

import hashlib
import json
from copy import deepcopy
from itertools import pairwise, product
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from qpdk.models.datasets import Dataset
from qpdk.models.datasets.capacitance import maxwell_to_mutual
from qpdk.models.datasets.generate import sweep, write
from qpdk.models.datasets.metadata import (
    METADATA_KEY,
    Axis,
    DatasetMetadata,
    Quantity,
    QuantityKind,
)
from qpdk.models.datasets.recipes import curate_double_pad as curator
from qpdk.models.datasets.validation import validate_native_capacitances

from .test_double_pad_geometry import native_config, native_inputs


def save(path: Path, value: Any) -> None:
    """Write a disposable procedural artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def table() -> tuple[pl.DataFrame, DatasetMetadata]:
    """Provide a complete procedural 135-point table."""
    metadata = DatasetMetadata(
        name="procedural_double_pad",
        description="Procedural fixture, not FEM evidence",
        synthetic=True,
        axes=tuple(Axis(name=k, unit="um") for k in curator.GRID),
        terminals=curator.TERMINALS,
        reference_ground="coplanar_chip_ground_frame",
        quantities=(
            Quantity(
                name="maxwell_capacitance",
                kind=QuantityKind.MAXWELL_CAPACITANCE,
                unit="F",
            ),
            Quantity(
                name="mutual_capacitance",
                kind=QuantityKind.MUTUAL_CAPACITANCE,
                unit="F",
            ),
            *(
                Quantity(name=n, kind=QuantityKind.CIRCUIT_PARAMETER, unit="1")
                for n in (
                    "fem_error_indicator_norm",
                    "solver_iterations",
                    "solver_relative_residual",
                )
            ),
        ),
    )

    def solve(**_point: float) -> dict:
        matrix = jnp.asarray([[1e-13, -3e-14], [-3e-14, 1e-13]])
        return {
            "maxwell_capacitance": matrix,
            "mutual_capacitance": maxwell_to_mutual(matrix),
            "fem_error_indicator_norm": 0.1,
            "solver_iterations": 2.0,
            "solver_relative_residual": 1e-10,
        }

    with jax.enable_x64():
        return sweep(metadata, solve, curator.GRID), metadata


@pytest.fixture
def packet(tmp_path: Path) -> dict:
    """Build two-terminal native files in the original production schema."""
    point = dict(zip(curator.GRID, [250.0, 400.0, 15.0, 200.0], strict=True))
    settings = curator.SETTINGS.copy()
    provenance = {
        "settings": settings,
        "processes": 1,
        "evidence_kind": "procedural_fixture_not_FEM",
        "generator_sha256": "fixture-generator",
        "experiment_sha256": "fixture-experiment",
        "production_driver_sha256": "fixture-driver",
        "helper_sha256": "fixture-helper",
    }
    inputs = native_inputs(point, settings) | {
        "provenance": {k: v for k, v in provenance.items() if k != "generator_sha256"},
        "minimum_feature_elements": 4,
        "mesh_threads": 1,
    }
    run = (
        tmp_path
        / hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:20]
    )
    config = native_config(settings)
    save(run / "inputs.json", inputs)
    save(run / "complete.json", {"terminals": list(curator.TERMINALS)})
    save(run / "config.json", config)
    save(run / "output/palace/config_resolved.json", config)
    metadata = {
        "solver": "palace",
        "simulation": {
            "problem_type": "Electrostatic",
            "field_order": 4,
            "linear_solver": config["Solver"]["Linear"],
            "refinement": config["Model"]["Refinement"],
        },
        "mesh": {
            "tetrahedra": 1000,
            "field_dofs": {"field_order": 4, "estimated_field_dofs": None},
        },
    }
    save(run / "metadata.json", metadata)
    save(
        run / "output/palace/palace.json",
        {"Problem": {"MeshElements": 1000, "DegreesOfFreedom": 1200}},
    )
    matrix = np.asarray([[1e-13, -3e-14], [-3e-14, 1e-13]])
    mutual = np.asarray([[7e-14, 3e-14], [3e-14, 7e-14]])
    for name, values in (
        ("C", matrix),
        ("Cm", mutual),
        ("Cinv", np.linalg.inv(matrix)),
    ):
        unit = "1/F" if name == "Cinv" else "F"
        np.savetxt(
            run / f"output/palace/terminal-{name}.csv",
            np.column_stack(([1, 2], values)),
            delimiter=",",
            header=f"i,value ({unit}),value ({unit})",
            comments="",
            encoding="utf-8",
        )
    (run / "output/palace/error-indicators.csv").write_text(
        "norm,min,max,mean\n0.1,0.01,0.2,0.05\n", encoding="utf-8"
    )
    (run / "solver.log").write_text(
        "".join(
            f"It {i}/2: Index = {i}\n0 KSP residual norm 1.0\n2 KSP residual norm 1.0e-10\nLinear solver converged in 2 iterations\n"
            for i in (1, 2)
        ),
        encoding="utf-8",
    )
    return {
        "run": run,
        "point": point,
        "settings": settings,
        "provenance": provenance,
        "source": {"driver_sha256": "fixture-driver", "recovery": False},
        "values": {
            "maxwell_capacitance": matrix.tolist(),
            "mutual_capacitance": mutual.tolist(),
            "fem_error_indicator_norm": [0.1],
            "solver_iterations": [2.0],
            "solver_relative_residual": [1e-10],
        },
    }


def check_packet(packet: dict) -> dict:
    """Exercise the real native proof collector in float64."""
    with jax.enable_x64():
        return curator._native_run(**packet, legacy=True)


def test_original_schema_and_genuine_dofs(packet: dict) -> None:
    proof = check_packet(packet)
    assert proof["field_dofs"] == 1200
    assert proof["metadata"]["mesh"]["field_dofs"]["estimated_field_dofs"] is None
    assert proof["matrices"]["C"][0][0] == pytest.approx(1e-13, rel=0, abs=0)
    assert not proof["native_order_override"]


def test_failed_batch_status_stays_failed(
    tmp_path: Path, table: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid replacement must not rewrite the original scheduler outcome."""
    frame, metadata = table
    baseline = write(tmp_path / "baseline", metadata, frame)
    root = tmp_path / "controls"
    name = "numerical"
    task = {
        "recipe": "double_pad_transmon",
        "mode": "control",
        "label": "bare-default-p5",
        "point": dict(zip(curator.GRID, [250.0, 400.0, 15.0, 200.0], strict=True)),
        "settings": curator.SETTINGS | {"order": 5},
        "anchor": "default",
        "control": "p5",
    }
    save(
        root / "source/plan.json",
        {"jobs": [{"name": name, "expected_points": 1, "tasks": [task]}]},
    )
    save(
        root / "submission-0.json",
        {"name": name, "returncode": 0, "stdout": "21244724\n"},
    )
    save(
        root / f"results/{name}/summary.json",
        [task | {"status": "failed", "error": "less_than_equal input_value=5"}],
    )
    (root / f"results/{name}/exit-status.txt").write_text("1\n", encoding="utf-8")
    (root / "scheduler.txt").write_text(
        "21244724|FAILED|04:33:56|1:0\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        curator, "_source", lambda *_a, **_k: {"fixture": "source checked separately"}
    )
    report = {"campaigns": [], "references": [], "failed": [], "pending": []}
    assert curator._campaign(root, report, baseline) == []
    job = report["campaigns"][0]["jobs"][0]
    assert job["native_exit_status"] == "1"
    assert job["scheduler_records"] == ["21244724|FAILED|04:33:56|1:0"]
    assert report["failed"][0]["raw_summary"]["status"] == "failed"


def test_control_schema_without_an_archived_log(packet: dict) -> None:
    """Control diagnostics remain explicitly attributed to native Parquet."""
    old = packet["run"]
    inputs = json.loads((old / "inputs.json").read_text(encoding="utf-8"))
    inputs["provenance"].pop("experiment_sha256")
    inputs["mesh_inputs"] = {
        "driver_sha256": "fixture-driver",
        "settings": packet["settings"],
        "minimum_feature_elements": 4,
    }
    inputs["terminal_order"] = list(curator.TERMINALS)
    run = old.with_name(
        hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:20]
    )
    old.rename(run)
    packet["run"] = run
    save(run / "inputs.json", inputs)
    (run / "solver.log").unlink()
    with jax.enable_x64():
        proof = curator._native_run(**packet)
        assert proof["diagnostics_scope"] == "native Parquet, solver log not retained"
        assert "solver.log" not in proof["files"]
        packet["source"]["driver_sha256"] = "different-driver"
        with pytest.raises(ValueError, match="meshing inputs"):
            curator._native_run(**packet)


@pytest.mark.parametrize("synthetic", [False, True])
def test_task_reads_native_files_again(
    packet: dict, table: tuple, tmp_path: Path, synthetic: bool
) -> None:
    """A task is checked from files even after an earlier successful preview."""
    frame, metadata = table
    point = packet["point"]
    frame = frame.filter(pl.all_horizontal([pl.col(k) == v for k, v in point.items()]))
    frame = frame.sample(fraction=1, shuffle=True, seed=23)
    folder = tmp_path / "task"
    data = write(
        folder / "data",
        metadata.model_copy(
            update={"synthetic": synthetic, "provenance": packet["provenance"]}
        ),
        frame,
    )
    frame.write_parquet(
        folder / "data/part-0000.parquet",
        metadata={METADATA_KEY: data.metadata.to_json()},
    )
    run = folder / "runs" / packet["run"].name
    run.parent.mkdir(parents=True)
    packet["run"].rename(run)
    source = packet["source"] | {
        "fingerprints": {
            k: packet["provenance"][k]
            for k in (
                "generator_sha256",
                "experiment_sha256",
                "production_driver_sha256",
                "helper_sha256",
            )
        }
    }
    with jax.enable_x64():
        proof = {}
        if synthetic:
            with pytest.raises(ValueError, match="convention"):
                curator._task(
                    folder,
                    {k: [v] for k, v in point.items()},
                    packet["settings"],
                    source,
                    proof,
                    legacy=True,
                )
            assert proof["raw_native_rows"] == frame.to_dicts()
            return
        curator._task(
            folder,
            {k: [v] for k, v in point.items()},
            packet["settings"],
            source,
            proof,
            legacy=True,
        )
        assert len(proof["native"]) == 1
        path = run / "output/palace/terminal-C.csv"
        path.write_text(
            path.read_text(encoding="utf-8").replace("(F)", "(pF)"), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="units"):
            curator._task(
                folder,
                {k: [v] for k, v in point.items()},
                packet["settings"],
                source,
                {},
                legacy=True,
            )


def test_retained_sources_are_not_an_original_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bind original native fingerprints without inventing a historical manifest."""
    source = tmp_path / "source"
    for name, text in {
        "batch.py": "driver",
        (curator.KERNEL / "_palace.py").as_posix(): "helper",
        (curator.KERNEL / "_experiment.py").as_posix(): "experiment",
        (
            curator.KERNEL / "double_pad_transmon.py"
        ).as_posix(): "GRID = {}\nSETTINGS = 4\n",
    }.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(
        curator, "PRODUCTION_DRIVER", curator._sha256(source / "batch.py")
    )
    monkeypatch.setattr(
        curator,
        "PRODUCTION_HELPER",
        curator._sha256(source / curator.KERNEL / "_palace.py"),
    )
    proof = curator._source(source, legacy=True)
    assert proof["inventory_scope"].startswith("retrospective")
    assert (
        proof["fingerprints"]["production_driver_sha256"] == curator.PRODUCTION_DRIVER
    )
    assert not (source / "source-sha256.json").exists()
    (source / "batch.py").write_text("changed driver", encoding="utf-8")
    with pytest.raises(ValueError, match="driver"):
        curator._source(source, legacy=True)


def test_contemporaneous_manifest_checked_before_using_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A saved passing status cannot hide changed frozen source bytes."""
    source = tmp_path / "source"
    for name, text in {
        "batch.py": "driver",
        (curator.KERNEL / "_palace.py").as_posix(): "helper",
        (curator.KERNEL / "_experiment.py").as_posix(): "experiment",
        (
            curator.KERNEL / "double_pad_transmon.py"
        ).as_posix(): "GRID = {}\nSETTINGS = 4\n",
        "baseline-data/part-0000.parquet": "fixture bytes",
        "gsim-fixture.whl": "fixture wheel",
    }.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(
        curator,
        "CONTROL_HELPER",
        curator._sha256(source / curator.KERNEL / "_palace.py"),
    )
    monkeypatch.setattr(
        curator,
        "BASELINE_SHA256",
        curator._sha256(source / "baseline-data/part-0000.parquet"),
    )
    plan = {
        "baseline_part_sha256": curator.BASELINE_SHA256,
        "grids": {"double_pad_transmon": curator.GRID},
        "numerical_target": {"relative": 0.02, "absolute_F": 1e-17},
        "source_revision": "fixture",
        "gsim_wheel_sha256": curator._sha256(source / "gsim-fixture.whl"),
        "validation_complete": True,
    }
    save(source / "plan.json", plan)
    manifest = {
        p.relative_to(source).as_posix(): curator._sha256(p)
        for p in source.rglob("*")
        if p.is_file()
    }
    save(source / "source-sha256.json", manifest)
    assert curator._source(source)["inventory_scope"].startswith(
        "verified contemporaneous"
    )
    (source / curator.KERNEL / "_experiment.py").write_text(
        "changed kernel", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Frozen source differs"):
        curator._source(source)


@pytest.mark.parametrize(
    "damage",
    [
        "C",
        "Cm",
        "Cinv",
        "units",
        "indicator",
        "residual",
        "dofs",
        "order",
        "completion",
        "cache",
    ],
)
def test_raw_proof_corruption(packet: dict, damage: str) -> None:
    run = packet["run"]
    match damage:
        case "C" | "Cm" | "Cinv":
            p = run / f"output/palace/terminal-{damage}.csv"
            raw = np.loadtxt(p, delimiter=",", skiprows=1, encoding="utf-8")
            raw[0, 1] *= 1.01
            np.savetxt(
                p,
                raw,
                delimiter=",",
                header=p.read_text(encoding="utf-8").splitlines()[0],
                comments="",
                encoding="utf-8",
            )
        case "units":
            p = run / "output/palace/terminal-C.csv"
            p.write_text(
                p.read_text(encoding="utf-8").replace("(F)", "(pF)"), encoding="utf-8"
            )
        case "indicator":
            (run / "output/palace/error-indicators.csv").write_text(
                "n,min,max,mean\n0.2,0.01,0.3,0.05\n", encoding="utf-8"
            )
        case "residual":
            p = run / "solver.log"
            p.write_text(
                p.read_text(encoding="utf-8").replace("1.0e-10", "1.0e-4"),
                encoding="utf-8",
            )
        case "dofs":
            save(
                run / "output/palace/palace.json",
                {"Problem": {"MeshElements": 1000, "DegreesOfFreedom": None}},
            )
        case "order":
            p = run / "metadata.json"
            m = json.loads(p.read_text(encoding="utf-8"))
            m["simulation"]["field_order"] = 3
            save(p, m)
        case "completion":
            save(
                run / "complete.json", {"terminals": list(reversed(curator.TERMINALS))}
            )
        case "cache":
            p = run / "inputs.json"
            m = json.loads(p.read_text(encoding="utf-8"))
            m["point"]["pad_gap"] = 20
            save(p, m)
    with pytest.raises((ValueError, RuntimeError)):
        check_packet(packet)


@pytest.mark.parametrize(
    "diagnostic", ["solver_iterations", "solver_relative_residual"]
)
def test_native_log_must_agree_with_retained_diagnostics(
    packet: dict, diagnostic: str
) -> None:
    """A converged log cannot silently replace a different retained table value."""
    packet["values"][diagnostic] = [3.0 if diagnostic == "solver_iterations" else 2e-10]
    with pytest.raises(
        ValueError, match=f"Native {diagnostic} diagnostic differs from Parquet"
    ):
        check_packet(packet)


def test_only_documented_recovery_can_override_setup_order(packet: dict) -> None:
    packet["settings"]["order"] = 5
    packet["provenance"]["helper_sha256"] = curator.RECOVERY_HELPER
    old = packet["run"]
    inputs = json.loads((old / "inputs.json").read_text(encoding="utf-8"))
    inputs["provenance"] = {
        k: v for k, v in packet["provenance"].items() if k != "generator_sha256"
    }
    run = old.with_name(
        hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:20]
    )
    old.rename(run)
    packet["run"] = run
    save(run / "inputs.json", inputs)
    for file in ("config.json", "output/palace/config_resolved.json"):
        p = run / file
        c = json.loads(p.read_text(encoding="utf-8"))
        c["Solver"]["Order"] = 5
        save(p, c)
    with pytest.raises(ValueError, match="metadata"):
        check_packet(packet)
    packet["source"]["recovery"] = True
    assert check_packet(packet)["native_order_override"]


@pytest.mark.parametrize(
    "damage", ["quantity", "coordinate", "mutual", "negative", "fractional", "residual"]
)
def test_task_quantity_gates(table: tuple, damage: str) -> None:
    frame, meta = table
    q = pl.col("quantity")
    match damage:
        case "quantity":
            frame = frame.filter(q != "solver_iterations")
        case "coordinate":
            frame = frame.filter(
                ~((q == "fem_error_indicator_norm") & (pl.col("pad_width") == 350))
            )
        case _:
            name, value = {
                "mutual": ("mutual_capacitance", 1e-9),
                "negative": ("fem_error_indicator_norm", -1),
                "fractional": ("solver_iterations", 1.5),
                "residual": ("solver_relative_residual", 1e-4),
            }[damage]
            frame = frame.with_columns(
                pl.when(q == name).then(value).otherwise(pl.col("value")).alias("value")
            )
    with jax.enable_x64(), pytest.raises(ValueError, match=r"Native|Dataset"):
        validate_native_capacitances(frame, meta, curator.GRID, {}, tolerance=1e-9)


@pytest.mark.parametrize(
    "damage", ["picofarads", "missing_diagnostic", "complex_matrix"]
)
def test_native_quantity_metadata_corruption_is_rejected_after_storage(
    table: tuple, tmp_path: Path, damage: str
) -> None:
    """Retained static evidence needs real SI quantities and every diagnostic."""
    frame, meta = table
    if damage == "missing_diagnostic":
        meta = meta.model_copy(
            update={
                "quantities": tuple(
                    q for q in meta.quantities if q.name != "solver_iterations"
                )
            }
        )
        frame = frame.filter(pl.col("quantity") != "solver_iterations")
        message = "quantities or diagnostics are missing"
    else:
        meta = meta.model_copy(
            update={
                "quantities": tuple(
                    q.model_copy(
                        update={"unit": "pF"}
                        if damage == "picofarads"
                        else {"complex": True}
                    )
                    if q.name == "maxwell_capacitance"
                    else q
                    for q in meta.quantities
                )
            }
        )
        selected = pl.col("quantity") == "maxwell_capacitance"
        if damage == "picofarads":
            frame = frame.with_columns(
                pl
                .when(selected)
                .then(pl.lit("pF"))
                .otherwise(pl.col("unit"))
                .alias("unit"),
                pl
                .when(selected)
                .then(pl.col("value") * 1e12)
                .otherwise(pl.col("value"))
                .alias("value"),
            )
        else:
            frame = frame.with_columns(
                pl
                .when(selected)
                .then(0.0)
                .otherwise(pl.col("value_imag"))
                .alias("value_imag")
            )
        message = (
            "must be stored in 'F'"
            if damage == "picofarads"
            else "quantity kinds or units differ"
        )
    frame.write_parquet(
        tmp_path / "part-0000.parquet", metadata={METADATA_KEY: meta.to_json()}
    )
    if damage == "picofarads":
        with pytest.raises(ValueError, match=message):
            Dataset(tmp_path)
        return
    retained = Dataset(tmp_path)
    with jax.enable_x64(), pytest.raises(ValueError, match=message):
        validate_native_capacitances(
            retained.table, retained.metadata, curator.GRID, {}, tolerance=1e-9
        )


@pytest.fixture
def qualification(
    tmp_path: Path, table: tuple, monkeypatch: pytest.MonkeyPatch
) -> tuple:
    """Isolate final admission policy from artifact IO, tested separately above."""
    frame, meta = table
    baseline = write(tmp_path / "baseline", meta, frame)
    roots = tuple(tmp_path / n for n in ("production", "controls", "recovery", "faces"))
    matrix = [[1e-13, -3e-14], [-3e-14, 1e-13]]

    def row(label: str, anchor: str, control: str, point: dict) -> dict:
        settings = curator.SETTINGS.copy()
        if control == "fine-h":
            settings |= {"near_mesh": 1.4, "far_mesh": 60.48}
        if control == "p5":
            settings |= {"order": 5}
        if control == "domain":
            settings |= {"domain_pad": 2400.0}
        return {
            "label": label,
            "anchor": anchor,
            "control": control,
            "recipe": "double_pad_transmon",
            "mode": "control",
            "point": point,
            "settings": settings,
            "maxwell_F": matrix,
            "native": {
                "source": f"procedural/{label}",
                "tetrahedra": 2000 if control == "fine-h" else 1000,
                "field_dofs": 2400 if control == "p5" else 1200,
                "field_order": settings["order"],
            },
        }

    centres = product(*[
        [(a + b) / 2 for a, b in pairwise(v)] for v in curator.GRID.values()
    ])
    controls = [
        row(
            f"held-{i}",
            f"cell-{i}",
            "held-out",
            dict(zip(curator.GRID, p, strict=True)),
        )
        for i, p in enumerate(centres)
    ]
    anchors = [
        [250.0, 400.0, 15.0, 200.0],
        [350.0, 550.0, 10.0, 300.0],
        [150.0, 250.0, 10.0, 100.0],
        [150.0, 550.0, 40.0, 100.0],
    ]
    recovery = []
    for i, coords in enumerate(anchors):
        p = dict(zip(curator.GRID, coords, strict=True))
        controls.extend(
            row(f"a{i}-{c}", str(i), c, p) for c in ("baseline", "fine-h", "domain")
        )
        recovery.extend(row(f"a{i}-{c}", str(i), c, p) for c in ("baseline", "p5"))
    face_rows = []
    for name, values in curator.GRID.items():
        for value in (values[0], values[-1]):
            p = {k: (v[0] + v[1]) / 2 for k, v in curator.GRID.items()} | {name: value}
            face_rows.append(
                row(f"face-{name}-{value}", f"face-{name}-{value}", "held-out", p)
            )

    def production(_root: Path, report: dict) -> Dataset:
        report["production"] = [{"procedural_fixture": True}] * 9
        return baseline

    def campaign(root: Path, report: dict, _baseline: Dataset, **_kwargs: Any) -> list:
        manifest = {
            "plan.json": root.name,
            (curator.KERNEL / "_palace.py").as_posix(): "recovery"
            if root == roots[2]
            else "original",
        }
        report["campaigns"].append({"source": {"files": manifest}})
        if root == roots[1]:
            report["failed"].extend(
                {
                    "campaign": str(root),
                    "task": {
                        k: v for k, v in r.items() if k not in {"native", "maxwell_F"}
                    },
                    "error": "less_than_equal input_value=5",
                }
                for r in recovery
                if r["control"] == "p5"
            )
            return deepcopy(controls)
        return deepcopy(recovery if root == roots[2] else face_rows)

    monkeypatch.setattr(curator, "_production", production)
    monkeypatch.setattr(curator, "_campaign", campaign)
    return roots, controls, recovery, face_rows, baseline


def test_qualified_export_embeds_proof_without_changing_rows(
    qualification: tuple, tmp_path: Path
) -> None:
    roots, _, _, _, baseline = qualification
    output = tmp_path / "qualified"
    report = curator.verify(*roots, output=output)
    assert report["passed"]
    assert report["coverage"]["passed"]
    assert len(report["superseded_failures"]) == 4
    exported = Dataset(output)
    assert exported.table.equals(baseline.table)
    assert exported.metadata.provenance["validation"] == report
    assert all(
        a.validated == (min(curator.GRID[a.name]), max(curator.GRID[a.name]))
        for a in exported.metadata.axes
    )


@pytest.mark.parametrize(
    "damage",
    [
        "pending_face",
        "corner_face",
        "duplicate_face",
        "cell",
        "h_tets",
        "p_dofs",
        "p_order",
        "wrong_setting",
        "accuracy",
    ],
)
def test_gate_failures_block_export(
    qualification: tuple, tmp_path: Path, damage: str
) -> None:
    roots, controls, recovered, faces, _ = qualification
    match damage:
        case "pending_face":
            faces.pop()
        case "corner_face":
            faces[0]["point"] = {k: v[0] for k, v in curator.GRID.items()}
        case "duplicate_face":
            faces[-1] = faces[0]
        case "cell":
            controls.pop(0)
        case "h_tets":
            next(r for r in controls if r["control"] == "fine-h")["native"][
                "tetrahedra"
            ] = 1000
        case "p_dofs":
            next(r for r in recovered if r["control"] == "p5")["native"][
                "field_dofs"
            ] = 1200
        case "p_order":
            next(r for r in recovered if r["control"] == "p5")["native"][
                "field_order"
            ] = 4
        case "wrong_setting":
            next(r for r in controls if r["control"] == "domain")["settings"][
                "permittivity"
            ] = 10
        case "accuracy":
            faces[0]["maxwell_F"] = [[2e-13, -3e-14], [-3e-14, 2e-13]]
    output = tmp_path / "rejected"
    with pytest.raises(
        ValueError, match=r"Numerical|native|Native|qualification|refine|DOFs"
    ):
        curator.verify(*roots, output=output)
    assert not output.exists()


def test_preview_is_fresh_after_a_good_result(qualification: tuple) -> None:
    roots, _, _, faces, _ = qualification
    assert curator.verify(*roots)["passed"]
    faces.pop()
    assert not curator.verify(*roots)["passed"]


@pytest.fixture
def complete_campaign(
    tmp_path: Path, table: tuple, monkeypatch: pytest.MonkeyPatch
) -> dict:
    """Build a complete on-disk procedural campaign, stubbing only raw backend proof.

    Raw proof is exercised separately with native-format files. Source binding,
    task/table validation, refinement, interpolation, recovery and export use
    their actual implementations here. No real FEM results are manufactured.

    Returns:
        Disposable campaign roots, baseline and planned procedural tasks.
    """
    frame, template = table
    production, controls, recovery, faces = (
        tmp_path / n for n in ("production", "controls", "recovery", "faces")
    )
    helpers = {
        "production": "procedural original helper",
        "controls": "procedural control helper",
        "recovery": "procedural recovery helper",
    }
    driver = "procedural retained driver"
    monkeypatch.setattr(
        curator, "PRODUCTION_DRIVER", hashlib.sha256(driver.encode()).hexdigest()
    )
    for key, constant in (
        ("production", "PRODUCTION_HELPER"),
        ("controls", "CONTROL_HELPER"),
        ("recovery", "RECOVERY_HELPER"),
    ):
        monkeypatch.setattr(
            curator, constant, hashlib.sha256(helpers[key].encode()).hexdigest()
        )

    def snapshot(source: Path, helper: str) -> None:
        files = {
            "batch.py": driver,
            (curator.KERNEL / "_palace.py").as_posix(): helpers[helper],
            (
                curator.KERNEL / "_experiment.py"
            ).as_posix(): "procedural experiment kernel",
            (
                curator.KERNEL / "double_pad_transmon.py"
            ).as_posix(): "GRID = {}\nSETTINGS = 4\n",
        }
        for name, text in files.items():
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

    snapshot(production / "source-transmon-production-00", "production")
    fingerprints = curator._source(
        production / "source-transmon-production-00", legacy=True
    )["fingerprints"]
    physics = dict.fromkeys(curator.PHYSICS, "procedural fixture, not FEM")
    physics |= {"solver": "Palace", "linear_initial_guess": False, "recipe_version": 1}
    versions = dict.fromkeys(("gsim", "gdsfactory", "meshwell", "gmsh"), "procedural")
    provenance = (
        physics
        | fingerprints
        | {
            "processes": 1,
            "settings": curator.SETTINGS,
            "versions": versions,
            "evidence_kind": "procedural_fixture_not_FEM",
        }
    )
    metadata = template.model_copy(
        update={"synthetic": False, "provenance": provenance}
    )
    baseline = write(
        production / "candidates/double_pad_transmon_refined_palace", metadata, frame
    )
    baseline_path = (
        production / "candidates/double_pad_transmon_refined_palace/part-0000.parquet"
    )
    monkeypatch.setattr(curator, "BASELINE_SHA256", curator._sha256(baseline_path))

    def solve(**_point: float) -> dict:
        matrix = jnp.asarray([[1e-13, -3e-14], [-3e-14, 1e-13]])
        return {
            "maxwell_capacitance": matrix,
            "mutual_capacitance": maxwell_to_mutual(matrix),
            "fem_error_indicator_norm": 0.1,
            "solver_iterations": 2.0,
            "solver_relative_residual": 1e-10,
        }

    def task_files(folder: Path, grid: dict, settings: dict, source: dict) -> None:
        meta = metadata.model_copy(
            update={
                "provenance": provenance
                | source["fingerprints"]
                | {"settings": settings}
            }
        )
        with jax.enable_x64():
            write(folder / "data", meta, sweep(meta, solve, grid))
        for values in product(*grid.values()):
            inputs = {
                "point": dict(zip(grid, values, strict=True)),
                "evidence_kind": "procedural_backend_stub",
            }
            name = hashlib.sha256(
                json.dumps(inputs, sort_keys=True).encode()
            ).hexdigest()[:20]
            save(folder / "runs" / name / "inputs.json", inputs)

    def backend(
        run: Path,
        point: dict,
        settings: dict,
        provenance: dict,
        source: dict,
        values: dict,
        *,
        legacy: bool = False,
    ) -> dict:
        return {
            "source": str(run),
            "point": point,
            "provenance": provenance,
            "input_schema": "original production" if legacy else "control",
            "inputs": curator._json(run / "inputs.json"),
            "files": {"inputs.json": curator._sha256(run / "inputs.json")},
            "evidence_kind": "procedural_backend_stub_not_FEM",
            "source_fingerprints": source["fingerprints"],
            "tetrahedra": 2000 if settings["near_mesh"] < 2 else 1000,
            "field_dofs": 2400 if settings["order"] == 5 else 1200,
            "field_order": settings["order"],
            "matrices": {"C": values["maxwell_capacitance"]},
        }

    monkeypatch.setattr(curator, "_native_run", backend)
    for index, (width, height) in enumerate(
        product(curator.GRID["pad_width"], curator.GRID["pad_height"])
    ):
        name = f"transmon-production-{index:02}"
        source = production / f"source-{name}"
        snapshot(source, "production")
        grid = curator.GRID | {"pad_width": [width], "pad_height": [height]}
        save(
            source / "production-plan.json",
            {
                "name": name,
                "recipe": "double_pad_transmon",
                "grid": grid,
                "settings": {
                    k: curator.SETTINGS[k]
                    for k in ("near_mesh", "far_mesh", "domain_pad", "order")
                },
            },
        )
        folder = production / "results" / name
        task_files(folder, grid, curator.SETTINGS, curator._source(source, legacy=True))
        save(
            production / f"submission-{name}.json",
            {"job": {"name": name}, "returncode": 0, "stdout": f"{1000 + index}\n"},
        )
        save(
            folder / "generation-complete.json",
            {"grid": grid, "settings": curator.SETTINGS},
        )
        (folder / "exit-status.txt").write_text("0\n", encoding="utf-8")

    def task(label: str, anchor: str, control: str, point: dict) -> dict:
        settings = curator.SETTINGS.copy()
        if control in {"fine-h", "held-out"}:
            settings |= {"near_mesh": 1.4, "far_mesh": 60.48}
        if control == "p5":
            settings |= {"order": 5}
        if control == "domain":
            settings |= {"domain_pad": 2400.0}
        return {
            "recipe": "double_pad_transmon",
            "label": label,
            "mode": "control",
            "point": point,
            "anchor": anchor,
            "control": control,
            "settings": settings,
        }

    centres = product(*[
        [(a + b) / 2 for a, b in pairwise(v)] for v in curator.GRID.values()
    ])
    r23 = [
        task(
            f"held-{i}",
            f"cell-{i}",
            "held-out",
            dict(zip(curator.GRID, p, strict=True)),
        )
        for i, p in enumerate(centres)
    ]
    r27 = []
    for anchor, coords in zip(
        ("default", "large", "small", "aspect"),
        (
            [250.0, 400.0, 15.0, 200.0],
            [350.0, 550.0, 10.0, 300.0],
            [150.0, 250.0, 10.0, 100.0],
            [150.0, 550.0, 40.0, 100.0],
        ),
        strict=True,
    ):
        point = dict(zip(curator.GRID, coords, strict=True))
        r23.extend(
            task(f"{anchor}-{c}", anchor, c, point)
            for c in ("baseline", "fine-h", "p5", "domain")
        )
        r27.extend((
            task(f"{anchor}-p5", anchor, "p5", point),
            task(f"repeat-{anchor}-p4", anchor, "baseline", point),
        ))
    r36 = [
        task(
            f"face-{name}-{value}",
            f"face-{name}-{value}",
            "held-out",
            {k: (v[0] + v[1]) / 2 for k, v in curator.GRID.items()} | {name: value},
        )
        for name, coords in curator.GRID.items()
        for value in (coords[0], coords[-1])
    ]
    for root, tasks, job_id in (
        (controls, r23, 2000),
        (recovery, r27, 3000),
        (faces, r36, 4000),
    ):
        helper = "recovery" if root == recovery else "controls"
        source = root / "source"
        snapshot(source, helper)
        baseline_copy = source / "baseline-data/part-0000.parquet"
        baseline_copy.parent.mkdir(parents=True)
        baseline_copy.write_bytes(baseline_path.read_bytes())
        wheel = source / "gsim-procedural.whl"
        wheel.write_text("procedural solver wheel, not executable", encoding="utf-8")
        job = {"name": "references", "expected_points": len(tasks), "tasks": tasks}
        plan = {
            "grids": {"double_pad_transmon": curator.GRID},
            "baseline_part_sha256": curator.BASELINE_SHA256,
            "numerical_target": {"relative": 0.02, "absolute_F": 1e-17},
            "source_revision": "procedural_fixture",
            "gsim_wheel_sha256": curator._sha256(wheel),
            "jobs": [job],
        }
        save(source / "plan.json", plan)
        save(
            source / "source-sha256.json",
            {
                p.relative_to(source).as_posix(): curator._sha256(p)
                for p in source.rglob("*")
                if p.is_file()
            },
        )
        proof = curator._source(source, recovery=root == recovery)
        folder = root / "results/references"
        folder.mkdir(parents=True)
        summary = []
        for row in tasks:
            if root == controls and row["control"] == "p5":
                summary.append(
                    {k: row[k] for k in ("recipe", "label", "mode", "settings")}
                    | {"status": "failed", "error": "less_than_equal input_value=5"}
                )
                save(
                    folder / row["label"] / "runs/failed-setup/inputs.json",
                    {"point": row["point"], "partial": True},
                )
            else:
                grid = {k: [v] for k, v in row["point"].items()}
                task_files(folder / row["label"], grid, row["settings"], proof)
                summary.append(
                    row
                    | {"status": "ok", "maxwell_F": [[1e-13, -3e-14], [-3e-14, 1e-13]]}
                )
        save(folder / "summary.json", summary)
        save(
            root / "submission-0.json",
            {"name": "references", "returncode": 0, "stdout": f"{job_id}\n"},
        )
        (folder / "exit-status.txt").write_text(
            "1\n" if root == controls else "0\n", encoding="utf-8"
        )
        (root / "scheduler.txt").write_text(
            f"{job_id}|FAILED|00:01:00|1:0\n"
            if root == controls
            else f"{job_id}|COMPLETED|00:01:00|0:0\n",
            encoding="utf-8",
        )
    return {
        "roots": (production, controls, recovery, faces),
        "baseline": baseline,
        "tasks": (r23, r27, r36),
    }


def test_complete_collectors_and_export_roundtrip(
    complete_campaign: dict, tmp_path: Path
) -> None:
    """Exercise real collectors, all qualification gates and the CLI export tail."""
    output, report_path = tmp_path / "qualified-procedural", tmp_path / "report.json"
    curator.main(*complete_campaign["roots"], output=output, report=report_path)
    report = curator._json(report_path)
    assert report["passed"]
    assert len(report["production"]) == 9
    assert sum(len(p["native"]) for p in report["production"]) == 135
    assert len(report["numerical"]) == 12
    assert len(report["interpolation"]) == 60
    assert len(report["superseded_failures"]) == 4
    assert len(report["repeats"]) == 4
    assert report["coverage"]["interior_cells_tested"] == 32
    assert len(report["coverage"]["strict_faces"]) == 8
    assert report["campaigns"][0]["jobs"][0]["native_exit_status"] == "1"
    assert "FAILED" in report["campaigns"][0]["jobs"][0]["scheduler_records"][0]
    exported = Dataset(output)
    assert exported.table.equals(complete_campaign["baseline"].table)
    assert exported.metadata.provenance["validation"] == report
    assert report["production"][0]["source"]["inventory_scope"].startswith(
        "retrospective"
    )
    assert (
        report["production"][0]["native"][0]["evidence_kind"]
        == "procedural_backend_stub_not_FEM"
    )


def test_successful_reference_summary_cannot_replace_missing_native_artifact(
    complete_campaign: dict, tmp_path: Path
) -> None:
    """A retained success summary cannot qualify a face whose Parquet is missing."""
    faces = complete_campaign["roots"][3]
    task = complete_campaign["tasks"][2][-1]
    part = faces / "results/references" / task["label"] / "data/part-0000.parquet"
    part.unlink()
    preview = curator.verify(*complete_campaign["roots"])
    assert not preview["passed"]
    assert any(f"{faces.name}/{task['label']}" in item for item in preview["pending"])
    evidence = next(
        row
        for row in preview["references"]
        if row["campaign"] == str(faces) and row["task"]["label"] == task["label"]
    )
    assert evidence["raw_summary"]["status"] == "ok"
    assert str(part.parent) in evidence["error"]
    assert not evidence.get("passed", False)
    output = tmp_path / "missing-native-rejected"
    with pytest.raises(ValueError, match="qualification is incomplete or failed"):
        curator.verify(*complete_campaign["roots"], output=output)
    assert not output.exists()


@pytest.mark.parametrize(
    "damage",
    [
        "missing_face",
        "late_point",
        "unrelated_p5_failure",
        "p5_accuracy",
        "missing_native_point",
        "duplicate_native_point",
        "production_drift",
        "plan_defaults",
        "recovery_matrix_lie",
        "reference_physics",
        "target_tolerance",
        "wheel_mismatch",
    ],
)
def test_complete_campaign_rejects_incomplete_or_deceptive_evidence(
    complete_campaign: dict, tmp_path: Path, damage: str
) -> None:
    """Backend proof alone cannot bypass complete grids or recovery policy."""
    production, controls, recovery, faces = complete_campaign["roots"]
    match damage:
        case "missing_face":
            path = faces / "results/references/summary.json"
            rows = curator._json(path)
            rows.pop()
            save(path, rows)
        case "late_point":
            path = production / "results/transmon-production-08/data/part-0000.parquet"
            data = Dataset(path.parent)
            frame = data.table.filter(
                ~(
                    (pl.col("pad_gap") == 40)
                    & (pl.col("ground_clearance") == 300)
                    & (pl.col("quantity") == "solver_relative_residual")
                )
            )
            frame.write_parquet(path, metadata={METADATA_KEY: data.metadata.to_json()})
        case "missing_native_point" | "duplicate_native_point":
            runs = production / "results/transmon-production-08/runs"
            path = next(runs.glob("*/inputs.json"))
            if damage == "missing_native_point":
                path.unlink()
            else:
                save(runs / "duplicate/inputs.json", curator._json(path))
        case "plan_defaults":
            path = production / "source-transmon-production-08/production-plan.json"
            plan = curator._json(path)
            plan["settings"]["order"] = 3
            save(path, plan)
        case "target_tolerance" | "wheel_mismatch":
            source = controls / "source"
            path = source / "plan.json"
            plan = curator._json(path)
            if damage == "target_tolerance":
                plan["numerical_target"]["relative"] = 0.2
            else:
                plan["gsim_wheel_sha256"] = "different solver wheel"
            save(path, plan)
            manifest_path = source / "source-sha256.json"
            manifest = curator._json(manifest_path)
            manifest["plan.json"] = curator._sha256(path)
            save(manifest_path, manifest)
        case "recovery_matrix_lie":
            path = recovery / "results/references/summary.json"
            rows = curator._json(path)
            next(r for r in rows if r["label"] == "default-p5")["maxwell_F"] = [
                [2e-13, -3e-14],
                [-3e-14, 2e-13],
            ]
            save(path, rows)
        case "reference_physics":
            path = recovery / "results/references/default-p5/data/part-0000.parquet"
            data = Dataset(path.parent)
            meta = data.metadata.model_copy(
                update={
                    "provenance": data.metadata.provenance
                    | {"ground": "different physical reference ground"}
                }
            )
            data.table.write_parquet(path, metadata={METADATA_KEY: meta.to_json()})
        case "unrelated_p5_failure":
            path = controls / "results/references/summary.json"
            rows = curator._json(path)
            next(r for r in rows if r["label"] == "default-p5")["error"] = (
                "unrelated solver failure"
            )
            save(path, rows)
        case "p5_accuracy" | "production_drift":
            path = (
                recovery / "results/references/default-p5/data/part-0000.parquet"
                if damage == "p5_accuracy"
                else production
                / "results/transmon-production-08/data/part-0000.parquet"
            )
            data = Dataset(path.parent)
            frame = data.table.with_columns(
                pl
                .when(
                    pl.col("quantity").is_in([
                        "maxwell_capacitance",
                        "mutual_capacitance",
                    ])
                )
                .then(pl.col("value") * 1.2)
                .otherwise(pl.col("value"))
                .alias("value")
            )
            frame.write_parquet(path, metadata={METADATA_KEY: data.metadata.to_json()})
            if damage == "p5_accuracy":
                matrix = frame.filter(pl.col("quantity") == "maxwell_capacitance")[
                    "value"
                ].to_list()
                path = recovery / "results/references/summary.json"
                rows = curator._json(path)
                next(r for r in rows if r["label"] == "default-p5")["maxwell_F"] = [
                    matrix[:2],
                    matrix[2:],
                ]
                save(path, rows)
                preview = curator.verify(*complete_campaign["roots"])
                assert not preview["passed"]
                assert not preview["failed"]
                assert len(preview["numerical"]) == 12
                assert any(not row["passed"] for row in preview["numerical"])
    output = tmp_path / "rejected-procedural"
    with pytest.raises(
        ValueError,
        match=r"qualification|Native|native|Dataset|p5 replacement|Production|Frozen",
    ):
        curator.verify(*complete_campaign["roots"], output=output)
    assert not output.exists()
