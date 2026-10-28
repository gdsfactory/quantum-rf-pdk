# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "qpdk[models] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git",
#   "typer>=0.24,<1",
#   "gsim @ git+https://github.com/nikosavola/gsim.git@946c3973b6311f0a4a3cf3bd601336ad7a186dfb",
# ]
# ///
"""Qualify retained bare double-pad production and independent native references.

Every preview reads native evidence again. Original production has no complete
source manifest or mesh-group export. Retained-source hashes are retrospective,
while driver and kernel fingerprints are checked against native provenance.
Native capacitance CSV values are in F. The FEM indicator is not an error bound.
"""

# Preserve rejected artifacts in the preview instead of aborting collection.
# ruff: file-ignore[raise-within-try,too-many-statements-in-try-clause]

import ast
import hashlib
import json
from itertools import pairwise, product
from math import isfinite
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import typer

from qpdk import PDK, logger
from qpdk.models.datasets import Dataset, GridInterpolator
from qpdk.models.datasets.generate import write
from qpdk.models.datasets.recipes._double_pad_geometry import (
    TERMINALS,
    check_config,
    check_geometry,
)
from qpdk.models.datasets.recipes._palace import solver_diagnostics
from qpdk.models.datasets.validation import (
    compare_capacitances,
    validate_native_capacitances,
)

GRID = {
    "pad_width": [150.0, 250.0, 350.0],
    "pad_height": [250.0, 400.0, 550.0],
    "pad_gap": [10.0, 15.0, 20.0, 30.0, 40.0],
    "ground_clearance": [100.0, 200.0, 300.0],
}
BASELINE_SHA256 = "ab352f361f3481ccf51156b176d0ab289c079603b02096145fee7da782027a63"
PRODUCTION_DRIVER = "b37b640a9e8bdf2cc39f689818eb53bf535ae9049a786144e6853c34c3c84428"
PRODUCTION_HELPER = "93ce673fb961bce7d6a3485bd4e24a9ea914cb3e09287f0ed4aafa2b39251c4e"
CONTROL_HELPER = "f555018ce97db2fc01f860e755d9decc58e24f867cbdfcaa9a902736693fc7b0"
RECOVERY_HELPER = "ffc8dc4baa7ba04c50d07ef61f93eeaf53d902f36ed603b92f42f2793d7eb55c"
SETTINGS = dict(
    near_mesh=2.0,
    far_mesh=75.6,
    domain_pad=1600.0,
    permittivity=11.45,
    order=4,
    tolerance=1e-9,
)
PHYSICS = (
    "metal",
    "ground",
    "junction",
    "excluded_environment",
    "runtime_sha256",
    "solver",
    "linear_initial_guess",
    "container_binary",
    "recipe_version",
)
KERNEL = Path("qpdk/models/datasets/data")


def _sha256(path: Path) -> str:
    """Read a content fingerprint without modifying the source."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> Any:
    """Read one retained artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    """Keep nonfinite rejected evidence representable in strict JSON."""
    if isinstance(value, float) and not isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    return value


def _receipt(path: Path, name: str, *, legacy: bool = False) -> dict:
    """Bind a task to its successful scheduler submission receipt."""
    record = _json(path)
    identity = record["job"]["name"] if legacy else record["name"]
    if (
        record["returncode"] != 0
        or not record["stdout"].strip().isdigit()
        or identity != name
    ):
        raise ValueError("Scheduler receipt differs from the frozen task")
    return {"record": record, "sha256": _sha256(path)}


def _source(source: Path, *, legacy: bool = False, recovery: bool = False) -> dict:
    """Check retained kernels and distinguish inventories from original manifests."""
    if legacy:
        manifest = {
            p.relative_to(source).as_posix(): _sha256(p)
            for p in source.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        }
        scope = "retrospective retained-source inventory, no original full manifest"
    else:
        manifest = _json(source / "source-sha256.json")
        for name, expected in manifest.items():
            path = (source / name).resolve()
            if not path.is_relative_to(source.resolve()) or _sha256(path) != expected:
                raise ValueError(f"Frozen source differs: {name}")
        scope = "verified contemporaneous campaign source manifest"
        plan = _json(source / "plan.json")
        if (
            plan["baseline_part_sha256"] != BASELINE_SHA256
            or plan["grids"]["double_pad_transmon"] != GRID
            or plan["numerical_target"] != {"relative": 0.02, "absolute_F": 1e-17}
            or manifest["baseline-data/part-0000.parquet"] != BASELINE_SHA256
        ):
            raise ValueError("Frozen validation target or tolerance differs")
        wheels = [n for n in manifest if n.startswith("gsim-") and n.endswith(".whl")]
        if len(wheels) != 1 or manifest[wheels[0]] != plan["gsim_wheel_sha256"]:
            raise ValueError("Frozen solver wheel differs")
    helper = manifest[(KERNEL / "_palace.py").as_posix()]
    expected_helper = (
        PRODUCTION_HELPER if legacy else RECOVERY_HELPER if recovery else CONTROL_HELPER
    )
    if helper != expected_helper or (
        legacy and manifest["batch.py"] != PRODUCTION_DRIVER
    ):
        raise ValueError("Retained native driver or extraction helper differs")
    recipe = source / KERNEL / "double_pad_transmon.py"
    nodes = [
        node
        for node in ast.parse(recipe.read_text(encoding="utf-8")).body
        if not (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and (t.id == "GRID" or t.id.endswith("_GRID"))
                for t in node.targets
            )
        )
    ]
    fingerprints = {
        "generator_sha256": _sha256(recipe),
        "extraction_sha256": hashlib.sha256(
            "".join(ast.dump(n) for n in nodes).encode()
        ).hexdigest(),
        "experiment_sha256": manifest[(KERNEL / "_experiment.py").as_posix()],
        "helper_sha256": helper,
    }
    if legacy:
        fingerprints["production_driver_sha256"] = manifest["batch.py"]
    else:
        fingerprints |= {
            "study_source_sha256": _sha256(source / "source-sha256.json"),
            "validation_target_part_sha256": BASELINE_SHA256,
            "gsim_wheel_sha256": plan["gsim_wheel_sha256"],
            "qpdk_geometry_revision": plan["source_revision"],
        }
    return {
        "source": str(source),
        "inventory_scope": scope,
        "files": manifest,
        "fingerprints": fingerprints,
        "driver_sha256": manifest["batch.py"],
        "recovery": recovery,
    }


def _native_run(
    run: Path,
    point: dict,
    settings: dict,
    provenance: dict,
    source: dict,
    values: dict,
    *,
    legacy: bool = False,
) -> dict:
    """Bind raw native matrices, solver proof and current geometry to one row."""
    inputs = _json(run / "inputs.json")
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:20]
    omitted = (
        {"generator_sha256"} if legacy else {"generator_sha256", "experiment_sha256"}
    )
    if (
        run.name != key
        or inputs["point"] != point
        or inputs["provenance"]
        != {k: v for k, v in provenance.items() if k not in omitted}
        or inputs["minimum_feature_elements"] != 4
        or inputs["mesh_threads"] != provenance["processes"]
        or _json(run / "complete.json") != {"terminals": list(TERMINALS)}
    ):
        raise ValueError("Native cache key, point, provenance or completion differs")
    if not legacy and (
        inputs["mesh_inputs"]
        != {
            "driver_sha256": source["driver_sha256"],
            "settings": settings,
            "minimum_feature_elements": 4,
        }
        or inputs["terminal_order"] != list(TERMINALS)
    ):
        raise ValueError("Native meshing inputs or terminal order differs")
    geometry = check_geometry(inputs, settings)
    config = _json(run / "config.json")
    resolved = _json(run / "output/palace/config_resolved.json")
    check_config(config, settings)
    check_config(resolved, settings)
    metadata = _json(run / "metadata.json")
    simulation, mesh = metadata["simulation"], metadata["mesh"]
    # R27 overrides the native config after the gsim p4 setup wrote metadata.
    override = (
        source["recovery"]
        and settings["order"] == 5
        and provenance["helper_sha256"] == RECOVERY_HELPER
    )
    recorded_order = 4 if override else settings["order"]
    problem = _json(run / "output/palace/palace.json")["Problem"]
    if (
        metadata["solver"] != "palace"
        or simulation["problem_type"] != "Electrostatic"
        or simulation["field_order"] != recorded_order
        or mesh["field_dofs"]["field_order"] != recorded_order
        or simulation["linear_solver"] != config["Solver"]["Linear"]
        or simulation["refinement"] != config["Model"]["Refinement"]
        or any(
            type(problem[k]) is not int or problem[k] <= 0
            for k in ("MeshElements", "DegreesOfFreedom")
        )
        or problem["MeshElements"] != mesh["tetrahedra"]
    ):
        raise ValueError("Native mesh, DOFs or recorded setup metadata differs")
    matrices = {}
    for name in ("C", "Cm", "Cinv"):
        path = run / f"output/palace/terminal-{name}.csv"
        header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
        unit = "(1/F)" if name == "Cinv" else "(F)"
        raw = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2, encoding="utf-8")
        if (
            len(header) != 3
            or any(not c.strip().endswith(unit) for c in header[1:])
            or raw.shape != (2, 3)
            or not np.array_equal(raw[:, 0], [1, 2])
            or not bool(np.isfinite(raw).all())
        ):
            raise ValueError(f"Native {name} units, indices or matrix shape differs")
        matrices[name] = raw[:, 1:].tolist()
    for quantity, name in (("maxwell_capacitance", "C"), ("mutual_capacitance", "Cm")):
        actual = np.asarray(values[quantity]).reshape(2, 2)
        # Cm is derived from C in Parquet, while Palace prints it independently.
        matches = (
            np.array_equal(actual, matrices[name])
            if name == "C"
            else np.allclose(actual, matrices[name], rtol=1e-10, atol=1e-27)
        )
        if not matches:
            raise ValueError(f"Native {name} matrix differs from Parquet")
    if not bool(
        jnp.allclose(
            jnp.asarray(matrices["C"]) @ jnp.asarray(matrices["Cinv"]),
            jnp.eye(2),
            rtol=1e-8,
            atol=1e-10,
        )
    ):
        raise ValueError("Native inverse capacitance differs from Maxwell matrix")
    indicator = np.loadtxt(
        run / "output/palace/error-indicators.csv",
        delimiter=",",
        skiprows=1,
        ndmin=2,
        encoding="utf-8",
    )
    if (
        indicator.shape != (1, 4)
        or not bool(np.isfinite(indicator).all() & (indicator >= 0).all())
        or not np.isclose(
            indicator[0, 0],
            np.asarray(values["fem_error_indicator_norm"]).item(),
            rtol=1e-12,
            atol=0,
        )
    ):
        raise ValueError("Native FEM indicator differs from Parquet")
    log = run / "solver.log"
    quality = {
        name: np.asarray(values[name]).item()
        for name in ("solver_relative_residual", "solver_iterations")
    }
    if legacy or log.exists():
        quality = solver_diagnostics(
            log.read_text(encoding="utf-8"),
            terminals=2,
            tolerance=settings["tolerance"],
        )
    for name, value in quality.items():
        if not np.isclose(value, np.asarray(values[name]).item(), rtol=1e-12, atol=0):
            raise ValueError(f"Native {name} diagnostic differs from Parquet")
    files = [
        "inputs.json",
        "config.json",
        "complete.json",
        "metadata.json",
        "output/palace/config_resolved.json",
        "output/palace/palace.json",
        "output/palace/error-indicators.csv",
        *(f"output/palace/terminal-{n}.csv" for n in matrices),
    ]
    if log.exists():
        files.append("solver.log")
    return {
        "source": str(run),
        "files": {f: _sha256(run / f) for f in files},
        "inputs": inputs,
        "geometry": geometry,
        "config": config,
        "resolved_config": resolved,
        "metadata": metadata,
        "matrices": matrices,
        "problem": problem,
        "tetrahedra": problem["MeshElements"],
        "field_dofs": problem["DegreesOfFreedom"],
        "field_order": settings["order"],
        "setup_order": recorded_order,
        "native_order_override": override,
        "diagnostics": quality,
        "diagnostics_scope": "native solver log"
        if log.exists()
        else "native Parquet, solver log not retained",
        "indicator_statistics": indicator[0].tolist(),
    }


def _task(
    folder: Path,
    grid: dict,
    settings: dict,
    source: dict,
    evidence: dict,
    *,
    legacy: bool = False,
) -> Dataset:
    """Validate a complete task grid before deriving any native proof."""
    data = Dataset(folder / "data")
    meta, frame = data.metadata, data.table
    evidence |= {
        "metadata": meta.model_dump(mode="json"),
        "raw_native_rows": frame.to_dicts(),
        "parts": {
            p.name: _sha256(p) for p in sorted((folder / "data").glob("*.parquet"))
        },
    }
    if (
        meta.synthetic
        or meta.terminals != TERMINALS
        or meta.variants
        or meta.reference_ground != "coplanar_chip_ground_frame"
        or tuple(a.name for a in meta.axes) != tuple(GRID)
        or any(a.unit != "um" for a in meta.axes)
        or meta.provenance["settings"] != settings
        or any(meta.provenance[k] != v for k, v in source["fingerprints"].items())
    ):
        raise ValueError(
            "Native dataset convention, settings or source provenance differs"
        )
    all_values = validate_native_capacitances(
        frame, meta, grid, {}, tolerance=settings["tolerance"]
    )
    runs = {}
    for path in sorted((folder / "runs").glob("*/inputs.json")):
        point = _json(path)["point"]
        key = tuple(point[k] for k in grid)
        if key in runs:
            raise ValueError("Duplicate native runs for one point")
        runs[key] = path.parent
    expected = set(product(*grid.values()))
    if set(runs) != expected:
        raise ValueError("Native runs differ from the complete task grid")
    evidence["native"] = []
    for key in sorted(expected):
        point = dict(zip(grid, key, strict=True))
        indices = tuple(grid[a.name].index(point[a.name]) for a in meta.axes)
        values = {}
        for name, samples in all_values.items():
            value = samples
            for index in indices:
                value = value[index]
            values[name] = value
        evidence["native"].append(
            _native_run(
                runs[key],
                point,
                settings,
                meta.provenance,
                source,
                values,
                legacy=legacy,
            )
        )
    return data


def _production(root: Path, report: dict) -> Dataset:
    """Recover the original nine shards without relabelling their source snapshots."""
    baseline_path = (
        root / "candidates/double_pad_transmon_refined_palace/part-0000.parquet"
    )
    if _sha256(baseline_path) != BASELINE_SHA256:
        raise ValueError("Production baseline SHA256 differs")
    baseline = Dataset(baseline_path.parent)
    parts = []
    for index, (width, height) in enumerate(
        product(GRID["pad_width"], GRID["pad_height"])
    ):
        name = f"transmon-production-{index:02}"
        source = _source(root / f"source-{name}", legacy=True)
        plan = _json(root / f"source-{name}/production-plan.json")
        grid = GRID | {"pad_width": [width], "pad_height": [height]}
        if (
            plan["name"] != name
            or plan["recipe"] != "double_pad_transmon"
            or plan["grid"] != grid
            or (SETTINGS | plan["settings"]) != SETTINGS
        ):
            raise ValueError("Production shard plan or partial settings differ")
        folder = root / "results" / name
        evidence = {
            "job": name,
            "source": source,
            "plan": plan,
            "receipt": _receipt(root / f"submission-{name}.json", name, legacy=True),
        }
        report["production"].append(evidence)
        if (folder / "exit-status.txt").read_text(encoding="utf-8").strip() != "0":
            raise ValueError("Production task did not finish successfully")
        completed = _json(folder / "generation-complete.json")
        if completed["grid"] != grid or completed["settings"] != SETTINGS:
            raise ValueError("Production completion differs from its plan")
        data = _task(folder, grid, SETTINGS, source, evidence, legacy=True)
        if any(
            data.metadata.provenance[k] != baseline.metadata.provenance[k]
            for k in PHYSICS
        ):
            raise ValueError("Production physics differs from baseline")
        parts.append(data.table)
    frame = pl.concat(parts)
    order = [*GRID, "quantity", "row", "col"]
    if not frame.sort(order).equals(baseline.table.sort(order)):
        raise ValueError(
            "Production rows differ from original baseline values or coordinates"
        )
    validate_native_capacitances(
        baseline.table, baseline.metadata, GRID, {}, tolerance=1e-9
    )
    report["baseline_part_sha256"] = BASELINE_SHA256
    return baseline


def _campaign(
    root: Path, report: dict, baseline: Dataset, *, recovery: bool = False
) -> list[dict]:
    """Read fresh controls, keeping failed and pending tasks visible in previews."""
    source = _source(root / "source", recovery=recovery)
    plan = _json(root / "source/plan.json")
    scheduler = root / "scheduler.txt"
    scheduler_text = scheduler.read_text(encoding="utf-8") if scheduler.exists() else ""
    campaign = {
        "source": source,
        "plan": plan,
        "jobs": [],
        "scheduler_snapshot": {"text": scheduler_text, "sha256": _sha256(scheduler)}
        if scheduler.exists()
        else None,
    }
    report["campaigns"].append(campaign)
    records = []
    labels = set()
    for index, job in enumerate(plan["jobs"]):
        folder = root / "results" / job["name"]
        summary_path = folder / "summary.json"
        summary = _json(summary_path) if summary_path.exists() else []
        tasks = {t["label"]: t for t in job["tasks"]}
        if (
            len(tasks) != len(job["tasks"])
            or labels.intersection(tasks)
            or len({r["label"] for r in summary}) != len(summary)
            or any(r["label"] not in tasks for r in summary)
            or job["expected_points"] != len(tasks)
            or any(
                t["recipe"] != "double_pad_transmon"
                or t["mode"] != "control"
                or set(t["point"]) != set(GRID)
                or set(t["settings"]) != set(SETTINGS)
                for t in tasks.values()
            )
        ):
            raise ValueError("Duplicate, unknown or inconsistent campaign tasks")
        labels.update(tasks)
        saved = {r["label"]: r for r in summary}
        receipt = _receipt(root / f"submission-{index}.json", job["name"])
        job_id = receipt["record"]["stdout"].strip()
        exit_status = folder / "exit-status.txt"
        campaign["jobs"].append({
            "name": job["name"],
            "job_id": job_id,
            "receipt": receipt,
            "native_exit_status": exit_status.read_text(encoding="utf-8").strip()
            if exit_status.exists()
            else None,
            "exit_status_sha256": _sha256(exit_status)
            if exit_status.exists()
            else None,
            "scheduler_records": [
                line
                for line in scheduler_text.splitlines()
                if line.startswith(f"{job_id}|")
            ],
        })
        for task in job["tasks"]:
            evidence = {
                "campaign": str(root),
                "task": task,
                "source": source,
                "receipt": receipt,
            }
            report["references"].append(evidence)
            row = saved.get(task["label"])
            if row is None:
                report["pending"].append(f"{root.name}/{task['label']}")
                continue
            evidence["raw_summary"] = row
            if any(
                row.get(k) != task.get(k)
                for k in ("recipe", "label", "mode", "settings")
            ):
                raise ValueError("Native summary differs from frozen task")
            if row["status"] != "ok":
                evidence["error"] = row.get("error", "Native task failed")
                evidence["partial_native"] = [
                    {
                        "source": str(path.parent),
                        "inputs": _json(path),
                        "files": {
                            p.relative_to(path.parent).as_posix(): _sha256(p)
                            for p in path.parent.rglob("*")
                            if p.is_file() and p.suffix in {".json", ".csv", ".log"}
                        },
                    }
                    for path in sorted(
                        (folder / task["label"] / "runs").glob("*/inputs.json")
                    )
                ]
                report["failed"].append(evidence)
                continue
            if any(row.get(k) != task.get(k) for k in ("point", "anchor", "control")):
                raise ValueError(
                    "Successful summary differs from frozen reference point"
                )
            try:
                data = _task(
                    folder / task["label"],
                    {k: [v] for k, v in task["point"].items()},
                    task["settings"],
                    source,
                    evidence,
                )
                native = evidence["native"][0]
                if native["matrices"]["C"] != row["maxwell_F"]:
                    raise ValueError("Summary matrix differs from native CSV")
                if any(
                    data.metadata.provenance[k] != baseline.metadata.provenance[k]
                    for k in PHYSICS
                ) or any(
                    data.metadata.provenance["versions"][k]
                    != baseline.metadata.provenance["versions"][k]
                    for k in ("gsim", "gdsfactory", "meshwell", "gmsh")
                ):
                    raise ValueError(
                        "Reference physics or solver versions differ from production"
                    )
                records.append(
                    task | {"maxwell_F": native["matrices"]["C"], "native": native}
                )
                evidence["passed"] = True
            except FileNotFoundError as error:
                evidence["error"] = str(error)
                report["pending"].append(f"{root.name}/{task['label']}: {error}")
            except (ValueError, KeyError, TypeError, RuntimeError) as error:
                evidence["error"] = str(error)
                report["failed"].append(evidence)
    return records


def _coverage(rows: list[dict], faces: list[dict]) -> dict:
    """Require all cell centres and fresh off-grid strict boundary faces."""
    centres = set(
        product(*[[(lo + hi) / 2 for lo, hi in pairwise(v)] for v in GRID.values()])
    )
    held = [r for r in rows if r["control"] == "held-out"]
    actual = [tuple(r["point"][k] for k in GRID) for r in held]
    expected_faces = {
        (k, v) for k, coords in GRID.items() for v in (coords[0], coords[-1])
    }
    tested = set()
    face_points = [tuple(r["point"][k] for k in GRID) for r in faces]
    old_points = {tuple(r["point"][k] for k in GRID) for r in rows}
    training = set(product(*GRID.values()))
    for row in faces:
        p = row["point"]
        for name, coords in GRID.items():
            if p[name] in {coords[0], coords[-1]} and all(
                v[0] < p[k] < v[-1] for k, v in GRID.items() if k != name
            ):
                tested.add((name, p[name]))
    return {
        "interior_cells_tested": len(set(actual)),
        "interior_cells_required": 32,
        "tested_cell_centres": sorted(actual),
        "strict_faces": sorted(tested),
        "fresh_face_points": sorted(face_points),
        "passed": len(actual) == 32
        and set(actual) == centres
        and len(faces) == len(set(face_points)) == 8
        and all(r["control"] == "held-out" for r in faces)
        and tested == expected_faces
        and not set(face_points).intersection(old_points | training),
    }


def _numerical(candidate: dict, reference: dict) -> dict:
    """Check independent h, p or domain refinement at identical geometry."""
    control = reference["control"]
    allowed = {
        "fine-h": {"near_mesh", "far_mesh"},
        "p5": {"order"},
        "domain": {"domain_pad"},
    }[control]
    a, b = candidate["settings"], reference["settings"]
    changed = {k for k in a if a[k] != b[k]}
    if (
        candidate["point"] != reference["point"]
        or a != SETTINGS
        or not changed
        or not changed <= allowed
        or any((b[k] >= a[k] if control == "fine-h" else b[k] <= a[k]) for k in changed)
    ):
        raise ValueError(
            "Numerical control changes unrelated settings or does not refine"
        )
    native, refined = candidate["native"], reference["native"]
    if control == "fine-h" and refined["tetrahedra"] <= native["tetrahedra"]:
        raise ValueError("Fine-h native tetrahedra did not increase")
    if control == "p5" and (
        refined["field_order"] <= native["field_order"]
        or refined["field_dofs"] <= native["field_dofs"]
    ):
        raise ValueError("p5 native order and genuine Palace DOFs must increase")
    return {
        "anchor": candidate["anchor"],
        "control": control,
        "candidate_native": native["source"],
        "reference_native": refined["source"],
        **compare_capacitances(
            candidate["maxwell_F"],
            reference["maxwell_F"],
            terminals=TERMINALS,
            junction=TERMINALS,
        ),
    }


def verify(
    production: Path,
    controls: Path,
    recovery: Path,
    faces: Path,
    *,
    output: Path | None = None,
) -> dict:
    """Recompute native qualification and optionally export unchanged production rows.

    Returns:
        Portable evidence and fresh numerical, coverage and geometry gate results.

    Raises:
        ValueError: If frozen sources differ or an unqualified export is requested.
    """
    PDK.activate()
    report = {
        "passed": False,
        "admission_status": "candidate_pending_native_qualification",
        "scope": "Bare M1 pads, finite 300 um coplanar ground frame, lossless silicon. No junction metal, readout, package, nonlinear or distributed-mode qualification. Measured references are not a continuous-domain error bound. Mesh-group membership is unavailable. DOFs come only from native Palace JSON.",
        "production": [],
        "campaigns": [],
        "references": [],
        "failed": [],
        "pending": [],
        "superseded_failures": [],
        "numerical": [],
        "interpolation": [],
        "repeats": [],
        "verification_sources": {
            p.name: _sha256(p)
            for p in [
                Path(__file__),
                Path(__file__).with_name("_double_pad_geometry.py"),
                Path(__file__).with_name("_palace.py"),
                Path(__file__).parents[1] / "validation.py",
                Path(__file__).parents[3] / "cells/transmon.py",
                Path(__file__).parents[3] / "tech.py",
            ]
        },
    }
    with jax.enable_x64():
        baseline = _production(production, report)
        rows = _campaign(controls, report, baseline)
        recovered = _campaign(recovery, report, baseline, recovery=True)
        face_rows = _campaign(faces, report, baseline)
        r23, r27, r36 = (c["source"]["files"] for c in report["campaigns"])
        if (
            set(r23) != set(r27)
            or set(r23) != set(r36)
            or {k for k in r23 if r23[k] != r27[k]}
            != {"plan.json", (KERNEL / "_palace.py").as_posix()}
            or {k for k in r23 if r23[k] != r36[k]} != {"plan.json"}
        ):
            raise ValueError(
                "Recovery or strict-face campaign changed unrelated frozen sources"
            )
        for replacement in [r for r in recovered if r["control"] == "p5"]:
            failed = [
                f
                for f in report["failed"]
                if f["campaign"] == str(controls)
                and f["task"]
                == {
                    k: v
                    for k, v in replacement.items()
                    if k not in {"maxwell_F", "native"}
                }
            ]
            if (
                len(failed) != 1
                or "less_than_equal" not in failed[0]["error"]
                or "input_value=5" not in failed[0]["error"]
            ):
                raise ValueError(
                    "p5 replacement lacks its original documented setup failure"
                )
            report["failed"].remove(failed[0])
            report["superseded_failures"].append({
                "original": failed[0],
                "replacement": replacement,
            })
            rows.append(replacement)
        anchors = {r["anchor"]: r for r in rows if r["control"] == "baseline"}
        for candidate in anchors.values():
            for control in ("fine-h", "p5", "domain"):
                references = [
                    r
                    for r in rows
                    if r["anchor"] == candidate["anchor"] and r["control"] == control
                ]
                if len(references) == 1:
                    report["numerical"].append(_numerical(candidate, references[0]))
            repeats = [
                r
                for r in recovered
                if r["anchor"] == candidate["anchor"] and r["control"] == "baseline"
            ]
            if len(repeats) == 1:
                repeat = repeats[0]
                if (
                    repeat["point"] != candidate["point"]
                    or repeat["settings"] != candidate["settings"]
                ):
                    raise ValueError("Recovery p4 repeat differs from original anchor")
                report["repeats"].append({
                    "anchor": candidate["anchor"],
                    **compare_capacitances(
                        candidate["maxwell_F"],
                        repeat["maxwell_F"],
                        terminals=TERMINALS,
                        junction=TERMINALS,
                    ),
                })
        lookup = GridInterpolator(
            baseline.grid("maxwell_capacitance", allow_missing=False)
        )
        for row in (
            rows + face_rows + [r for r in recovered if r["control"] == "baseline"]
        ):
            report["interpolation"].append({
                "label": row["label"],
                "point": row["point"],
                "reference_maxwell_F": row["maxwell_F"],
                **compare_capacitances(
                    lookup(**row["point"]),
                    row["maxwell_F"],
                    terminals=TERMINALS,
                    junction=TERMINALS,
                ),
            })
        report["coverage"] = _coverage(rows, face_rows)
        report["passed"] = (
            not report["failed"]
            and not report["pending"]
            and len(rows) == 48
            and len(recovered) == 8
            and len(anchors) == 4
            and len(report["superseded_failures"]) == 4
            and len(report["repeats"]) == 4
            and len(report["numerical"]) == 12
            and len(report["interpolation"]) == 60
            and report["coverage"]["passed"]
            and all(
                c["passed"]
                for c in report["numerical"]
                + report["interpolation"]
                + report["repeats"]
            )
        )
    if report["passed"]:
        report["admission_status"] = "validated_bare_electrostatic_model"
    report = _json_safe(report)
    if output is not None:
        if not report["passed"]:
            raise ValueError(
                "Bare double-pad native qualification is incomplete or failed"
            )
        meta = baseline.metadata
        axes = tuple(
            a.model_copy(update={"validated": (min(GRID[a.name]), max(GRID[a.name]))})
            for a in meta.axes
        )
        write(
            output,
            meta.model_copy(
                update={
                    "axes": axes,
                    "provenance": meta.provenance
                    | {
                        "admission_status": report["admission_status"],
                        "validation": report,
                        "previous_validation": meta.provenance.get("validation"),
                    },
                }
            ),
            baseline.table,
        )
    return report


def main(
    production: Path,
    controls: Path,
    recovery: Path,
    faces: Path,
    output: Path | None = None,
    report: Path | None = None,
) -> None:
    """Preview native evidence and optionally write a qualified dataset."""
    result = verify(production, controls, recovery, faces, output=output)
    if report is not None:
        report.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    logger.info(
        "Bare double-pad passed={}, pending={}, failed={}",
        result["passed"],
        len(result["pending"]),
        len(result["failed"]),
    )


if __name__ == "__main__":
    typer.run(main)
