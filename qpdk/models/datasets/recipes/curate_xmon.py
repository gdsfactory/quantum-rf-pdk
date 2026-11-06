# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "qpdk[models] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git",
#   "typer>=0.24,<1",
# ]
# ///
"""Curate a bare Xmon grid from frozen tasks and independently checked native runs.

Recorded polygons are rebuilt with current cells. Native terminal and material
attributes are bound to the frozen sheet driver and verified solver wheel.
R30 did not record mesh groups or usable field DOF counts, so neither is claimed.
"""

# Keep raw evidence when a task fails instead of aborting the preview.
# ruff: file-ignore[raise-within-try,too-many-statements-in-try-clause]

import ast
import hashlib
import json
from itertools import product
from math import isfinite, prod
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import polars as pl
import typer

from qpdk import PDK, logger
from qpdk.models.datasets import Dataset, GridInterpolator
from qpdk.models.datasets.generate import write
from qpdk.models.datasets.recipes._experiment import Settings
from qpdk.models.datasets.recipes._xmon_geometry import check_config, check_geometry
from qpdk.models.datasets.recipes.curate_qubit_readout import (
    _coverage,
    _numerical_comparison,
)
from qpdk.models.datasets.validation import (
    compare_capacitances,
    validate_native_capacitances,
)

STABLE = (
    "runtime_sha256",
    "versions",
    "gsim_revision",
    "qpdk_geometry_revision",
    "metal",
    "ground",
    "junction",
    "scope",
    "solver",
    "linear_initial_guess",
    "container_binary",
    "extraction_sha256",
    "helper_sha256",
    "gsim_source_verification",
)
EVIDENCE_SCOPE = "Recorded input polygons, pins, frame and domain rebuilt from current cells. Terminal 1, ground 2, air 3 and silicon 4 follow the frozen canonical sheet driver and verified gsim wheel. Mesh-group membership and field DOF counts are unavailable. Residuals and iterations come from native Parquet; the FEM indicator is also checked against raw CSV and is not an accuracy bound."


def _sha256(path: Path) -> str:
    """Hash a source or native evidence file without changing it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> Any:
    """Read a recorded JSON artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    """Preserve nonfinite failure values as strings in strict JSON."""
    if isinstance(value, float) and not isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _task_grid(task: dict) -> dict:
    """Select the frozen production grid or reference point."""
    return task.get("grid") or {key: [value] for key, value in task["point"].items()}


def _faces(grid: dict, rows: list[dict]) -> set[tuple[str, float]]:
    """Find strict boundary faces covered by held references."""
    return {
        (name, row["point"][name])
        for row in rows
        for name, coords in grid.items()
        if row["point"][name] in {coords[0], coords[-1]}
        and all(
            values[0] < row["point"][other] < values[-1]
            for other, values in grid.items()
            if other != name
        )
    }


def _structure(plan: dict) -> dict:
    """Require the frozen bare Xmon production and reference structure."""
    grid = plan["grids"]["xmon"]
    if set(grid) != {"arm_scale", "gap_width"} or any(
        len(coords) != 5
        or coords != sorted(set(coords))
        or not all(isfinite(x) and x > 0 for x in coords)
        for coords in grid.values()
    ):
        raise ValueError("Bare Xmon requires a finite positive five-by-five task grid")
    tasks = [task for job in plan["jobs"] for task in job["tasks"]]
    if len({job["name"] for job in plan["jobs"]}) != len(plan["jobs"]):
        raise ValueError("Frozen Xmon job names must be unique")
    production = [t for t in tasks if t["mode"] == "production"]
    controls = [t for t in tasks if t["mode"] == "control"]
    held = [t for t in controls if t["anchor"].startswith("dense-held-")]
    anchors = [t for t in controls if t["control"] == "baseline"]
    if (
        len(production) != 1
        or production[0]["grid"] != grid
        or len(anchors) != 4
        or len({t["anchor"] for t in anchors}) != 4
        or len(held) != 20
        or len(controls) != 36
        or len(tasks) != 37
    ):
        raise ValueError(
            "Bare Xmon requires one production task, four anchors and twenty held references"
        )
    if len({tuple(t["point"][key] for key in grid) for t in held}) != len(held) or len({
        tuple(t["point"][key] for key in grid) for t in anchors
    }) != len(anchors):
        raise ValueError(
            "Xmon reference geometries must be distinct within their roles"
        )
    for anchor in anchors:
        pairs = [t for t in controls if t["anchor"] == anchor["anchor"]]
        if (
            sorted(t["control"] for t in pairs)
            != ["baseline", "domain", "fine-h", "hp4"]
            or any(t["point"] != anchor["point"] for t in pairs)
            or any(anchor["point"][key] not in coords for key, coords in grid.items())
        ):
            raise ValueError(
                "Every Xmon anchor requires baseline, h, p and domain controls at one grid point"
            )
    if any(
        _task_grid(task) != {key: [value] for key, value in task["point"].items()}
        for task in controls
    ):
        raise ValueError("Each Xmon reference task must select one point")
    coverage = _coverage(grid, held, require_all=True)
    faces = {
        (name, x) for name, coords in grid.items() for x in {coords[0], coords[-1]}
    }
    if (
        not coverage["passed"]
        or coverage["interior_cells_tested"] != 16
        or _faces(grid, held) != faces
    ):
        raise ValueError(
            "Frozen Xmon holdouts must cover sixteen cells and four strict faces"
        )
    for job in plan["jobs"]:
        if (
            len({t["label"] for t in job["tasks"]}) != len(job["tasks"])
            or any(t["recipe"] != "xmon" for t in job["tasks"])
            or job["expected_points"]
            != sum(prod(map(len, _task_grid(t).values())) for t in job["tasks"])
        ):
            raise ValueError("Frozen Xmon task identities or point counts differ")
    return {
        "native_points": sum(j["expected_points"] for j in plan["jobs"]),
        "production_points": prod(map(len, grid.values())),
        "anchors": len(anchors),
        "numerical": 3 * len(anchors),
        "held": len(held),
        "references": len(controls),
        "interior_cells": coverage["total_interior_cells"],
        "strict_faces": sorted(faces),
    }


def _source_provenance(source: Path, plan: dict, manifest: dict) -> dict:
    """Bind native provenance to the verified frozen source bundle."""
    recipe = source / "qpdk/models/datasets/data/xmon.py"
    nodes = [
        node
        for node in ast.parse(recipe.read_text(encoding="utf-8")).body
        if not (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and (target.id == "GRID" or target.id.endswith("_GRID"))
                for target in node.targets
            )
        )
    ]
    wheels = [
        name for name in manifest if name.startswith("gsim-") and name.endswith(".whl")
    ]
    if (
        len(wheels) != 1
        or manifest[wheels[0]] != plan["gsim_wheel_sha256"]
        or plan["gsim_source_verification"]["all_package_files_match_source"]
        is not True
        or _sha256(recipe) != plan["recipe_sha256"]
    ):
        raise ValueError("Frozen Xmon recipe or verified solver wheel differs")
    return {
        "solver": "Palace",
        "linear_initial_guess": False,
        "generator_sha256": _sha256(recipe),
        "extraction_sha256": hashlib.sha256(
            "".join(ast.dump(node) for node in nodes).encode()
        ).hexdigest(),
        "helper_sha256": manifest["qpdk/models/datasets/data/_palace.py"],
        "experiment_sha256": manifest["qpdk/models/datasets/data/_experiment.py"],
        "study_source_sha256": _sha256(source / "source-sha256.json"),
        "qpdk_geometry_revision": plan["source_revision"],
        "gsim_revision": plan["gsim_revision"],
        "gsim_wheel_sha256": plan["gsim_wheel_sha256"],
        "gsim_source_verification": plan["gsim_source_verification"],
    }


def _native_run(
    run: Path,
    point: dict,
    settings: dict,
    provenance: dict,
    driver_sha256: str,
    values: dict,
) -> dict:
    """Check one recorded native solve against its table and current geometry."""
    inputs = _json(run / "inputs.json")
    if (
        run.name
        != hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:20]
    ):
        raise ValueError("Native input run hash differs")
    if inputs["point"] != point or inputs["mesh_inputs"] != {
        "driver_sha256": driver_sha256,
        "settings": settings,
        "minimum_feature_elements": 4,
    }:
        raise ValueError("Native point, settings or frozen mesh driver differs")
    if (
        inputs["provenance"]
        != {
            k: v
            for k, v in provenance.items()
            if k not in {"generator_sha256", "experiment_sha256"}
        }
        or inputs["mesh_threads"] != provenance["processes"]
        or inputs["minimum_feature_elements"] != 4
    ):
        raise ValueError(
            "Native input provenance or meshing controls differ from Parquet"
        )
    if _json(run / "complete.json") != {"terminals": ["island"]}:
        raise ValueError("Native solve terminal completion differs")
    geometry = check_geometry(inputs, settings)
    config = _json(run / "config.json")
    resolved = _json(run / "output/palace/config_resolved.json")
    check_config(config, settings)
    check_config(resolved, settings)
    metadata = _json(run / "metadata.json")
    simulation, mesh = metadata["simulation"], metadata["mesh"]
    if (
        metadata["solver"] != "palace"
        or simulation["problem_type"] != "Electrostatic"
        or simulation["field_order"] != settings["order"]
        or mesh["field_dofs"]["field_order"] != settings["order"]
        or simulation["linear_solver"] != config["Solver"]["Linear"]
        or simulation["refinement"] != config["Model"]["Refinement"]
    ):
        raise ValueError("Native metadata field order or solver configuration differs")
    if type(mesh["tetrahedra"]) is not int or mesh["tetrahedra"] <= 0:
        raise ValueError("Native tetrahedron count must be a positive integer")
    matrices = {}
    for name in ("C", "Cm", "Cinv"):
        path = run / f"output/palace/terminal-{name}.csv"
        unit = "1/F" if name == "Cinv" else "F"
        if f"({unit})" not in path.read_text(encoding="utf-8").splitlines()[0]:
            raise ValueError(f"Native {name} matrix units differ")
        raw = np.loadtxt(
            path,
            delimiter=",",
            skiprows=1,
            ndmin=2,
            encoding="utf-8",
        )
        if raw.shape != (1, 2) or raw[0, 0] != 1 or not bool(jnp.isfinite(raw).all()):
            raise ValueError(f"Native {name} matrix shape, indices or values differ")
        matrices[name] = raw[:, 1:].tolist()
    for key, name in (("maxwell_capacitance", "C"), ("mutual_capacitance", "Cm")):
        if not bool(
            jnp.array_equal(
                jnp.asarray(values[key]).reshape(1, 1), jnp.asarray(matrices[name])
            )
        ):
            raise ValueError(f"Native {name} matrix differs from Parquet")
    if not bool(
        jnp.allclose(
            jnp.asarray(matrices["C"]) @ jnp.asarray(matrices["Cinv"]),
            jnp.eye(1),
            rtol=1e-8,
            atol=1e-12,
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
        or not bool(jnp.isfinite(indicator).all() & (indicator >= 0).all())
        or not bool(
            jnp.isclose(
                indicator[0, 0],
                jnp.asarray(values["fem_error_indicator_norm"]).item(),
                rtol=1e-12,
                atol=0,
            )
        )
    ):
        raise ValueError("Native FEM indicator differs from Parquet")
    files = [
        run / "inputs.json",
        run / "complete.json",
        run / "config.json",
        run / "metadata.json",
        run / "output/palace/config_resolved.json",
        *(run / f"output/palace/terminal-{name}.csv" for name in matrices),
        run / "output/palace/error-indicators.csv",
    ]
    return {
        "source": str(run),
        "files": {str(p.relative_to(run)): _sha256(p) for p in files},
        "inputs": inputs,
        "geometry": geometry,
        "config": config,
        "resolved_config": resolved,
        "metadata": metadata,
        "matrices": matrices,
        "indicator_statistics": dict(
            zip(
                ("norm", "minimum", "maximum", "mean"),
                indicator[0].tolist(),
                strict=True,
            )
        ),
        "tetrahedra": mesh["tetrahedra"],
        "field_order": simulation["field_order"],
        "field_dofs": mesh["field_dofs"],
    }


def _task(
    folder: Path, task: dict, saved: dict, source: dict, driver: str, evidence: dict
) -> Dataset:
    """Validate one complete task and collect its native evidence."""
    data = Dataset(folder / task["label"] / "data")
    meta, frame = data.metadata, data.table
    evidence["metadata"] = meta.model_dump(mode="json")
    if (
        meta.synthetic
        or meta.terminals != ("island",)
        or meta.reference_ground != "finite_coplanar_chip_ground"
    ):
        raise ValueError(
            "Synthetic data or a different Xmon terminal convention cannot validate FEM"
        )
    if (
        saved["status"] != "ok"
        or any(saved.get(key) != task.get(key) for key in ("recipe", "mode"))
        or (
            task["mode"] == "control"
            and any(
                saved.get(key) != task.get(key)
                for key in ("point", "anchor", "control")
            )
        )
    ):
        raise ValueError("Native summary failed or differs from its frozen task")
    provenance = meta.provenance
    if any(provenance[key] != value for key, value in source.items()):
        raise ValueError("Native source provenance differs from frozen sources")
    defaults = Settings()
    settings = {
        "permittivity": defaults.permittivity,
        "tolerance": defaults.tolerance,
        **task["settings"],
    }
    if settings != provenance["settings"] or settings != saved["settings"]:
        raise ValueError("Native solver settings differ from the frozen task")
    grid = _task_grid(task)
    if saved["grid"] != grid or saved["points"] != prod(map(len, grid.values())):
        raise ValueError("Native summary grid differs from frozen task")
    evidence["values"] = validate_native_capacitances(
        frame, meta, grid, {}, tolerance=settings["tolerance"]
    )
    if (
        task["mode"] == "control"
        and jnp
        .asarray(evidence["values"]["maxwell_capacitance"])
        .reshape(1, 1)
        .tolist()
        != saved["maxwell_F"]
    ):
        raise ValueError("Reference summary matrix differs from native Parquet")
    runs = folder / task["label"] / "runs"
    by_point = {}
    for path in sorted(runs.glob("*/inputs.json")):
        point = _json(path)["point"]
        key = tuple(point[name] for name in grid)
        if key in by_point:
            raise ValueError("Duplicate native runs for one task point")
        by_point[key] = path.parent
    expected = set(product(*grid.values()))
    if set(by_point) != expected:
        raise ValueError("Native runs do not cover the complete frozen task grid")
    evidence["native"] = []
    for coords in sorted(expected):
        point = dict(zip(grid, coords, strict=True))
        selected = frame.filter(
            pl.all_horizontal([pl.col(name) == value for name, value in point.items()])
        )
        values = {
            q.name: selected.filter(pl.col("quantity") == q.name)["value"].to_list()
            for q in meta.quantities
        }
        evidence["native"].append(
            _native_run(by_point[coords], point, settings, provenance, driver, values)
        )
    return data


def _numerical(candidate: dict, reference: dict) -> dict:
    """Require numerical accuracy and recorded mesh or order refinement."""
    comparison = _numerical_comparison(
        candidate, reference, ("island",), ("island", None), STABLE
    )
    baseline, refined = candidate["native"][0], reference["native"][0]
    if (
        reference["control"] == "fine-h"
        and refined["tetrahedra"] <= baseline["tetrahedra"]
    ):
        raise ValueError("Fine-h native tetrahedra did not increase")
    if (
        reference["control"] == "hp4"
        and refined["field_order"] <= baseline["field_order"]
    ):
        raise ValueError("p-refinement native field order did not increase")
    return comparison | {
        "candidate_mesh": baseline["metadata"],
        "reference_mesh": refined["metadata"],
    }


def verify(study: Path, output: Path | None = None) -> dict[str, Any]:
    """Recompute geometry, native data and independent accuracy gates before export.

    Missing tasks remain pending. Rejected native rows and raw proofs remain in
    the preview. An output is written only after every frozen task passes.

    Returns:
        Native evidence, coverage and the combined qualification result.

    Raises:
        FileNotFoundError: If frozen source files are unavailable.
        ValueError: If frozen sources differ or an export fails qualification.
    """
    PDK.activate()
    source = study / "source"
    manifest = _json(source / "source-sha256.json")
    for name, expected in manifest.items():
        if _sha256(source / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    plan = _json(source / "plan.json")
    expected = _structure(plan)
    provenance = _source_provenance(source, plan, manifest)
    grid = plan["grids"]["xmon"]
    report = {
        "passed": False,
        "admission_status": "candidate_pending_independent_numerical_validation",
        "scope": plan["scope"],
        "evidence_scope": EVIDENCE_SCOPE,
        "source_manifest_sha256": _sha256(source / "source-sha256.json"),
        "source_manifest": manifest,
        "plan": plan,
        "verification_sources": {
            Path(__file__).name: _sha256(Path(__file__)),
            "_xmon_geometry.py": _sha256(Path(__file__).with_name("_xmon_geometry.py")),
            "validation.py": _sha256(Path(__file__).parents[1] / "validation.py"),
            "cells/transmon.py": _sha256(
                Path(__file__).parents[3] / "cells/transmon.py"
            ),
            "tech.py": _sha256(Path(__file__).parents[3] / "tech.py"),
        },
        "expected": expected,
        "pending": [],
        "failed": [],
        "sources": [],
        "raw_references": [],
        "numerical": [],
        "interpolation": [],
    }
    parts, rows = [], []
    for index, job in enumerate(plan["jobs"]):
        folder = study / "results" / job["name"]
        summary = folder / "summary.json"
        saved_rows = _json(summary) if summary.exists() else []
        labels = [row["label"] for row in saved_rows]
        tasks = {task["label"]: task for task in job["tasks"]}
        for saved in saved_rows:
            if labels.count(saved["label"]) != 1 or saved["label"] not in tasks:
                report["failed"].append({
                    "job": job["name"],
                    "raw_summary": saved,
                    "error": "Duplicate or unknown native summary task",
                })
        saved_by_label = {
            r["label"]: r
            for r in saved_rows
            if labels.count(r["label"]) == 1 and r["label"] in tasks
        }
        for task in job["tasks"]:
            identity = f"{job['name']}/{task['label']}"
            if task["label"] not in saved_by_label:
                report["pending"].append(identity)
                continue
            saved = saved_by_label[task["label"]]
            evidence = {"job": job["name"], "task": task, "raw_summary": saved}
            report["sources"].append(evidence)
            try:
                receipt = study / f"submission-{index}.json"
                evidence["receipt"] = _json(receipt)
                evidence["receipt_sha256"] = _sha256(receipt)
                if (
                    evidence["receipt"]["returncode"] != 0
                    or not evidence["receipt"]["stdout"].strip().isdigit()
                    or evidence["receipt"]["name"] != job["name"]
                ):
                    raise ValueError("Native scheduler receipt differs")
                files = sorted((folder / task["label"] / "data").glob("*.parquet"))
                if not files:
                    if saved["status"] != "ok":
                        raise ValueError("Failed native task has no completed Parquet")
                    raise FileNotFoundError(identity + "/native-Parquet")
                evidence["parts"] = {
                    str(p.relative_to(study)): _sha256(p) for p in files
                }
                evidence["raw_native_rows"] = pl.read_parquet(files).to_dicts()
                data = _task(
                    folder, task, saved, provenance, manifest["batch.py"], evidence
                )
                evidence["passed"] = True
                if task["mode"] == "production":
                    parts.append(data)
                else:
                    row = (
                        task
                        | saved
                        | {
                            "source": {
                                "job": job["name"],
                                "job_id": evidence["receipt"]["stdout"].strip(),
                                "parts": evidence["parts"],
                                "provenance": data.metadata.provenance,
                            },
                            "native": evidence["native"],
                        }
                    )
                    rows.append(row)
                    report["raw_references"].append(row)
            except FileNotFoundError as error:
                evidence |= {"passed": False, "error": str(error)}
                report["pending"].append(identity + ": " + str(error))
            except (ValueError, KeyError, TypeError) as error:
                evidence |= {"passed": False, "error": str(error)}
                report["failed"].append(evidence)
                logger.error(f"Rejected {identity}: {error}")
    anchors = {r["anchor"]: r for r in rows if r["control"] == "baseline"}
    for anchor, candidate in anchors.items():
        for control in ("fine-h", "hp4", "domain"):
            reference = next(
                (r for r in rows if r["anchor"] == anchor and r["control"] == control),
                None,
            )
            if reference is None:
                continue
            try:
                report["numerical"].append({
                    "anchor": anchor,
                    **_numerical(candidate, reference),
                })
            except (ValueError, KeyError) as error:
                report["failed"].append({
                    "candidate": candidate,
                    "reference": reference,
                    "error": str(error),
                })
    held = [r for r in rows if r["anchor"].startswith("dense-held-")]
    report["coverage"] = _coverage(grid, held, require_all=True)
    report["strict_faces"] = sorted(_faces(grid, held))
    if len(parts) == 1:
        data = parts[0]
        lookup = GridInterpolator(data.grid("maxwell_capacitance", allow_missing=False))
        for row in rows:
            try:
                if any(
                    row["source"]["provenance"][key] != data.metadata.provenance[key]
                    for key in STABLE
                ) or (
                    row["control"] == "baseline"
                    and row["settings"] != data.metadata.provenance["settings"]
                ):
                    raise ValueError(
                        "Reference physics or calibrated settings differ from production"
                    )
                prediction = lookup(**row["point"])
                comparison = {
                    "label": row["label"],
                    "point": row["point"],
                    "candidate_F": prediction.tolist(),
                    "reference_F": row["maxwell_F"],
                    **compare_capacitances(
                        prediction,
                        row["maxwell_F"],
                        terminals=("island",),
                        junction=("island", None),
                    ),
                }
                report["interpolation"].append(comparison)
                if not comparison["passed"]:
                    logger.error(f"Xmon interpolation failed: {comparison}")
            except (ValueError, KeyError) as error:
                report["failed"].append({"reference": row, "error": str(error)})
    report["completed_native_points"] = sum(
        len(s.get("native", [])) for s in report["sources"] if s.get("passed")
    )
    report["passed"] = (
        not report["pending"]
        and not report["failed"]
        and len(parts) == 1
        and report["completed_native_points"] == expected["native_points"]
        and len(anchors) == expected["anchors"]
        and len(held) == expected["held"]
        and len(report["numerical"]) == expected["numerical"]
        and len(report["interpolation"]) == expected["references"]
        and report["coverage"]["passed"]
        and report["strict_faces"] == expected["strict_faces"]
        and all(c["passed"] for c in report["numerical"] + report["interpolation"])
    )
    if report["passed"]:
        report["admission_status"] = "validated_bare_electrostatic_model"
    report = _json_safe(report)
    if output is not None:
        if not report["passed"]:
            raise ValueError("Xmon dataset is incomplete or failed native validation")
        meta = parts[0].metadata
        axes = tuple(
            a.model_copy(update={"validated": (min(grid[a.name]), max(grid[a.name]))})
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
                    },
                }
            ),
            parts[0].table,
        )
    return report


def main(study: Path, output: Path | None = None, report: Path | None = None) -> None:
    """Preview the native study, optionally writing qualified data and a report."""
    result = verify(study, output=output)
    if report is not None:
        report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    logger.info(
        f"Bare Xmon passed={result['passed']}, pending={len(result['pending'])}, failed={len(result['failed'])}"
    )


if __name__ == "__main__":
    typer.run(main)
