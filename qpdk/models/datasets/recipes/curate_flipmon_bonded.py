# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "qpdk[models] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git",
#   "typer>=0.24,<1",
# ]
# ///
"""Qualify the final bare bonded-flipmon grid using frozen native evidence.

The final study supplies new production and fresh held references. Calibration
also supplies exactly the declared reused production and older retuning points.
Retuning comparisons still gate accuracy, but never supply fresh coverage.
"""

import ast
import hashlib
import inspect
import json
from itertools import pairwise, product
from math import isfinite, prod
from pathlib import Path
from typing import Any

import polars as pl
import typer

from qpdk import logger
from qpdk.models.datasets import (
    Dataset,
    DatasetMetadata,
    GridInterpolator,
    QuantityKind,
)
from qpdk.models.datasets.generate import write
from qpdk.models.datasets.recipes._bare_flipmon_geometry import (
    AXES,
    TERMINALS,
    verify_geometry,
)
from qpdk.models.datasets.recipes._experiment import Settings
from qpdk.models.datasets.table import to_grid
from qpdk.models.datasets.validation import (
    compare_capacitances,
    validate_native_capacitances,
)

RECIPE = "flipmon_bonded"
JUNCTION = ("island", "ring")
QUANTITIES = {
    "maxwell_capacitance": (QuantityKind.MAXWELL_CAPACITANCE, "F"),
    "mutual_capacitance": (QuantityKind.MUTUAL_CAPACITANCE, "F"),
    "fem_error_indicator_norm": (QuantityKind.CIRCUIT_PARAMETER, "1"),
    "solver_relative_residual": (QuantityKind.CIRCUIT_PARAMETER, "1"),
    "solver_iterations": (QuantityKind.CIRCUIT_PARAMETER, "1"),
}


CONTROLS = ("baseline", "fine-h", "hp4", "domain", "ideal-wire", "ideal-wire-fine-h")


STABLE = (
    "runtime_sha256",
    "versions",
    "gsim_geometry_revision",
    "gsim_wheel_sha256",
    "qpdk_geometry_revision",
    "reference_planes",
    "junction",
    "ground",
    "container_binary",
    "solver",
    "linear_initial_guess",
    "extraction_sha256",
    "helper_sha256",
    "experiment_sha256",
    "diagnostics",
    "metal_thickness_um",
    "substrate_thickness_um",
    "outer_ring",
)


def _sha256(path: Path) -> str:
    """Hash retained evidence without loading large parts into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _child(root: Path, name: str) -> Path:
    """Keep manifest entries inside their frozen source directory."""
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError("Frozen evidence paths must stay inside their directory")
    return path


def _plan(campaign: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify every frozen source before reading the planned experiment."""
    source = campaign / "source"
    manifest_path = source / "source-sha256.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "plan.json" not in manifest:
        raise ValueError("Frozen manifest must include plan.json")
    for name, expected in manifest.items():
        if _sha256(_child(source, name)) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    plan = json.loads((source / "plan.json").read_text(encoding="utf-8"))
    return plan, {
        "campaign": campaign.name,
        "source_manifest_sha256": _sha256(manifest_path),
        "manifest": manifest,
        "plan": plan,
    }


def _mesh_refinement(candidate: dict, reference: dict, control: str) -> dict:
    """Confirm that numerical controls changed the actual mesh or field order."""
    baseline, refined = (row["mesh_evidence"] for row in (candidate, reference))
    if control == "fine-h" and any(
        refined[key] <= baseline[key] for key in ("nodes", "tetrahedra")
    ):
        raise ValueError("Native h refinement did not increase mesh resolution")
    if control == "hp4" and refined["field_order"] <= baseline["field_order"]:
        raise ValueError("Native p refinement did not increase field order")
    return {
        role: {key: mesh[key] for key in ("nodes", "tetrahedra", "field_order")}
        for role, mesh in (("baseline", baseline), ("reference", refined))
    }


def _held_coverage(
    grid: dict[str, list[float]], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Require distinct cell centres and faces with no training-node coordinates."""
    for row in rows:
        point = row["point"]
        if set(point) != set(grid):
            raise ValueError("Fresh held-out point must name every target axis")
        faces = 0
        for name, coords in grid.items():
            value = point[name]
            if value in {coords[0], coords[-1]}:
                faces += 1
            elif value not in {(lo + hi) / 2 for lo, hi in pairwise(coords)}:
                raise ValueError(
                    "Fresh held-out coordinates must be strict cell centres or axis faces"
                )
        if faces > 1:
            raise ValueError(
                "Fresh held-out faces must have interior coordinates on the other axes"
            )
    return _coverage(
        grid, [row | {"anchor": "dense-held-fresh"} for row in rows], require_all=True
    )


def _coverage(
    grid: dict[str, list[float]],
    rows: list[dict[str, Any]],
    *,
    require_all: bool = False,
) -> dict[str, Any]:
    """Count distinct interior cells and both boundaries of every grid axis."""
    cells = set()
    boundaries = set()
    for row in rows:
        if not row["anchor"].startswith("dense-held-"):
            continue
        point = row["point"]
        cell = []
        for name, coords in grid.items():
            if point[name] in {coords[0], coords[-1]}:
                boundaries.add((name, point[name]))
            interval = next(
                (
                    i
                    for i, (lo, hi) in enumerate(pairwise(coords))
                    if lo < point[name] < hi
                ),
                None,
            )
            if interval is not None:
                cell.append(interval)
        if len(cell) == len(grid):
            cells.add(tuple(cell))
    total = prod(len(coords) - 1 for coords in grid.values())
    all_intervals = all(
        set(range(len(coords) - 1)) <= {cell[axis] for cell in cells}
        for axis, coords in enumerate(grid.values())
    )
    return {
        "interior_cells_tested": len(cells),
        "total_interior_cells": total,
        "tested_cell_indices": [list(cell) for cell in sorted(cells)],
        "boundaries_tested": [list(boundary) for boundary in sorted(boundaries)],
        "passed": len(cells) >= (total if require_all else min(64, total))
        and len(boundaries) == 2 * len(grid)
        and all_intervals,
    }


def _numerical_comparison(
    candidate: dict[str, Any],
    row: dict[str, Any],
    terminals: tuple[str, ...],
    junction: tuple[str, str | None],
    stable: tuple[str, ...],
) -> dict[str, Any]:
    """Compare identical geometry with one independently refined solver setting."""
    control = row["control"]
    if row["point"] != candidate["point"]:
        raise ValueError("Numerical controls must use the identical geometry")
    changed = {
        key
        for key in candidate["settings"]
        if candidate["settings"][key] != row["settings"][key]
    }
    allowed = {
        "fine-h": {"near_mesh", "far_mesh"},
        "hp4": {"order"},
        "domain": {"domain_pad"},
    }[control]
    if not changed or not changed <= allowed:
        raise ValueError("Numerical control changes unrelated solver settings")
    if (
        control == "fine-h"
        and any(row["settings"][key] >= candidate["settings"][key] for key in changed)
    ) or (
        control != "fine-h"
        and any(row["settings"][key] <= candidate["settings"][key] for key in changed)
    ):
        raise ValueError("Numerical control must refine the candidate")
    if any(
        candidate["source"]["provenance"][key] != row["source"]["provenance"][key]
        for key in stable
        if key != "settings"
    ):
        raise ValueError("Numerical control physics or solver provenance differs")
    return {
        "control": control,
        "point": row["point"],
        "candidate_source": candidate["source"],
        "reference_source": row["source"],
        "candidate_settings": candidate["settings"],
        "reference_settings": row["settings"],
        "candidate_maxwell_F": candidate["maxwell_F"],
        "reference_maxwell_F": row["maxwell_F"],
        **compare_capacitances(
            candidate["maxwell_F"],
            row["maxwell_F"],
            terminals=terminals,
            junction=junction,
        ),
    }


def _grid(task: dict) -> dict:
    """Resolve the externally frozen production grid or reference point."""
    return (
        task["grid"]
        if task["mode"] == "production"
        else {k: [v] for k, v in task["point"].items()}
    )


def _key(point: dict) -> tuple:
    """Keep point identities independent of serialized dictionary order."""
    return tuple(point[name] for name in AXES)


def _fingerprints(campaign: Path, manifest: dict) -> dict:
    """Bind native provenance to the frozen bare recipe and extraction helpers."""
    prefix = "qpdk/models/datasets/recipes/"
    names = {
        "generator_sha256": "flipmon_bonded.py",
        "helper_sha256": "_palace.py",
        "experiment_sha256": "_experiment.py",
    }
    source = _child(campaign / "source", prefix + names["generator_sha256"]).read_text(
        encoding="utf-8"
    )
    nodes = [
        node
        for node in ast.parse(source).body
        if not (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and (target.id == "GRID" or target.id.endswith("_GRID"))
                for target in node.targets
            )
        )
    ]
    return {
        **{key: manifest[prefix + name] for key, name in names.items()},
        "extraction_sha256": hashlib.sha256(
            "".join(ast.dump(node) for node in nodes).encode()
        ).hexdigest(),
    }


def _selection(
    study: Path, calibration: Path, plans: dict[Path, dict]
) -> tuple[list[tuple], dict]:
    """Select disjoint training shards and separate calibration, fresh and old roles."""
    final = plans[study]
    target = final["grids"][RECIPE]
    if (
        set(target) != set(AXES)
        or tuple(len(target[a]) for a in AXES) != (7, 3, 3, 3)
        or any(
            values != sorted(set(values)) or not all(isfinite(v) for v in values)
            for values in target.values()
        )
    ):
        raise ValueError(
            "Bare final grid requires seven chip gaps and three nodes on each other axis"
        )
    if (
        final["calibration_campaign"] != calibration.name
        or final["retuning_references"] != calibration.name
    ):
        raise ValueError(
            "Frozen calibration and retuning campaign differ from the supplied directory"
        )
    entries = final["reused_production"]
    declared = {(r["campaign"], r["job"], r["label"]): r for r in entries}
    if (
        len(declared) != 3
        or len(entries) != len(declared)
        or any(k[0] != calibration.name for k in declared)
        or any(r.get("recipe", RECIPE) != RECIPE for r in entries)
    ):
        raise ValueError(
            "Final bare grid requires three uniquely declared original production shards"
        )
    selected, found, training = [], set(), set()
    counts = dict.fromkeys(
        ("production", "new_production_points", "calibration", "fresh", "retuning"), 0
    )
    controls, fresh, retuning = [], [], []
    for campaign in (calibration, study):
        jobs = plans[campaign]["jobs"]
        if len({j["name"] for j in jobs}) != len(jobs):
            raise ValueError("Frozen job names must be unique")
    jobs_with_campaign = (
        (p, i, j) for p in (calibration, study) for i, j in enumerate(plans[p]["jobs"])
    )
    for campaign, index, job in jobs_with_campaign:
        if len({t["label"] for t in job["tasks"]}) != len(job["tasks"]):
            raise ValueError("Frozen task labels must be unique")
        if job["expected_points"] != sum(
            prod(map(len, _grid(t).values())) for t in job["tasks"]
        ):
            raise ValueError("Frozen job point count differs from its tasks")
        for task in job["tasks"]:
            if task["recipe"] != RECIPE or task["mode"] not in {
                "production",
                "control",
            }:
                raise ValueError("Unexpected bare flipmon recipe or task mode")
            if task["mode"] == "control" and set(task["point"]) != set(AXES):
                raise ValueError(
                    "Reference points must select every bare geometry axis"
                )
            if task["mode"] == "production":
                identity = (campaign.name, job["name"], task["label"])
                if campaign == calibration:
                    if identity not in declared:
                        continue
                    if declared[identity]["grid"] != task["grid"]:
                        raise ValueError(
                            "Reused grid differs from its original frozen task"
                        )
                    found.add(identity)
                grid = task["grid"]
                if set(grid) != set(AXES) or any(
                    not values
                    or values != sorted(set(values))
                    or not set(values) <= set(target[a])
                    for a, values in grid.items()
                ):
                    raise ValueError("Production shard leaves the target grid")
                points = set(product(*(grid[a] for a in AXES)))
                if training & points or task["bond"] != "physical_bump":
                    raise ValueError(
                        "Production shards overlap or use a nonphysical bond"
                    )
                training |= points
                role = "production"
                if campaign == study:
                    counts["new_production_points"] += len(points)
            elif campaign == study:
                role = "fresh"
                fresh.append(task)
                if task["control"] != "held-out" or task["bond"] != "physical_bump":
                    raise ValueError(
                        "Fresh holdouts must use independent physical-bump references"
                    )
            elif task["anchor"].startswith("dense-held-"):
                role = "retuning"
                retuning.append(task)
                if task["control"] != "held-out" or task["bond"] != "physical_bump":
                    raise ValueError(
                        "Old retuning references differ from the original held role"
                    )
            else:
                role = "calibration"
                controls.append(task)
            counts[role] += 1
            selected.append((campaign, index, job, task, role))
    if found != set(declared) or training != set(product(*(target[a] for a in AXES))):
        raise ValueError(
            "Declared production does not cover the exact final target grid"
        )
    if counts != {
        "production": 7,
        "new_production_points": 108,
        "calibration": 24,
        "fresh": 56,
        "retuning": 24,
    }:
        raise ValueError(
            "Bare final study requires seven shards, 24 calibration, 56 fresh and 24 retuning tasks"
        )
    if (
        final["new_production_geometries"] != 108
        or final["fresh_held_out_points"] != 56
        or final["datasets"]["flipmon_bonded_palace"] != 189
    ):
        raise ValueError("Frozen final study point totals differ")
    anchors = {t["anchor"] for t in controls}
    if (
        len(anchors) != 4
        or len({_key(t["point"]) for t in controls if t["control"] == "baseline"}) != 4
    ):
        raise ValueError("Bare calibration requires four anchors")
    for anchor in anchors:
        rows = [t for t in controls if t["anchor"] == anchor]
        if (
            sorted(t["control"] for t in rows) != sorted(CONTROLS)
            or len({_key(t["point"]) for t in rows}) != 1
            or any(
                t["bond"]
                != (
                    "ideal_wire"
                    if t["control"].startswith("ideal-wire")
                    else "physical_bump"
                )
                for t in rows
            )
        ):
            raise ValueError(
                "Each anchor requires unique physical and ideal-wire numerical controls"
            )
    fresh_points, old_points = (
        {_key(t["point"]) for t in fresh},
        {_key(t["point"]) for t in retuning},
    )
    if (
        len(fresh_points) != 56
        or len(old_points) != 24
        or fresh_points & (training | old_points)
    ):
        raise ValueError(
            "Fresh points overlap training, old retuning points or each other"
        )
    coverage = _held_coverage(target, fresh)
    if (
        not coverage["passed"]
        or coverage["interior_cells_tested"] != 48
        or len(coverage["boundaries_tested"]) != 8
    ):
        raise ValueError(
            "Fresh holdouts must cover 48 strict cell centres and eight faces"
        )
    return selected, {
        "tasks": len(selected),
        "native_points": len(training) + len(controls) + len(fresh) + len(retuning),
        "production_shards": counts["production"],
        "production_points": len(training),
        "calibration": 24,
        "fresh": 56,
        "retuning": 24,
        "numerical": 12,
        "ideal_wire_refinement": 4,
        "interpolation": 72,
        "retuning_at_training_nodes": len(old_points & training),
    }


def _validate_table(
    frame: pl.DataFrame,
    metadata: DatasetMetadata,
    grid: dict,
    *,
    bond: str,
    tolerance: float,
) -> dict:
    """Check every declared quantity on the frozen two-terminal task grid.

    Returns:
        Native values without an admission claim.

    Raises:
        ValueError: If quantities, terminal conventions or diagnostics differ.
    """
    if (
        metadata.terminals != TERMINALS
        or metadata.reference_ground != "both_chip_ground_frames"
        or metadata.variants != ("bond",)
        or set(grid) != set(AXES)
    ):
        raise ValueError(
            "Native bare flipmon terminal, ground or axis convention differs"
        )
    return validate_native_capacitances(
        frame, metadata, grid, {"bond": bond}, tolerance=tolerance
    )


def _load_task(
    campaign: Path, index: int, job: dict, task: dict, frozen: dict, evidence: dict
) -> Dataset:
    """Read a completed native task, preserving its provenance and raw failure rows."""
    folder = _child(campaign / "results", job["name"])
    task_dir = _child(folder, task["label"])
    evidence["path"] = str(task_dir)
    receipt = campaign / f"submission-{index}.json"
    evidence["receipt"] = json.loads(receipt.read_text(encoding="utf-8"))
    evidence["receipt_sha256"] = _sha256(receipt)
    if (
        evidence["receipt"]["returncode"] != 0
        or evidence["receipt"]["name"] != job["name"]
        or not evidence["receipt"]["stdout"].strip().isdigit()
    ):
        raise ValueError("Native scheduler receipt differs")
    files = sorted((task_dir / "data").glob("*.parquet"))
    saved = evidence["raw_summary"]
    if not files:
        if saved["status"] != "ok":
            raise ValueError("Native task failed without a completed table")
        raise FileNotFoundError(f"Missing native Parquet for {task_dir}")
    evidence["parts"] = {str(p.relative_to(campaign)): _sha256(p) for p in files}
    evidence["raw_native_rows"] = pl.read_parquet(files).to_dicts()
    data = Dataset(task_dir / "data")
    meta = data.metadata
    evidence["metadata"] = meta.model_dump(mode="json")
    if meta.synthetic:
        raise ValueError("Synthetic data cannot supply native admission evidence")
    combined = task | saved
    if saved["status"] != "ok" or any(
        combined.get(k) != task.get(k)
        for k in ("recipe", "mode", "point", "anchor", "control", "bond")
    ):
        raise ValueError("Native summary failed or differs from its frozen task")
    defaults = Settings()
    settings = {
        "permittivity": defaults.permittivity,
        "tolerance": defaults.tolerance,
        **task["settings"],
    }
    provenance = meta.provenance
    plan, manifest = frozen["plan"], frozen["manifest"]
    fingerprints = _fingerprints(campaign, manifest)
    if (
        any(provenance[k] != v for k, v in fingerprints.items())
        or fingerprints["generator_sha256"] != plan["recipe_sha256"]
    ):
        raise ValueError("Native fingerprints differ from the frozen recipe or helpers")
    if (
        provenance["settings"] != settings
        or saved["settings"] != settings
        or provenance["study_source_sha256"] != frozen["source_manifest_sha256"]
        or provenance["qpdk_geometry_revision"] != plan["source_revision"]
        or provenance["gsim_geometry_revision"] != plan["gsim_revision"]
        or provenance["gsim_wheel_sha256"] != plan["gsim_wheel_sha256"]
    ):
        raise ValueError("Native settings or campaign provenance differs")
    wheels = [
        value
        for name, value in manifest.items()
        if name.startswith("gsim-") and name.endswith(".whl")
    ]
    if wheels != [plan["gsim_wheel_sha256"]]:
        raise ValueError("Frozen solver wheel differs from the campaign provenance")
    grid = _grid(task)
    if saved["grid"] != grid or saved["points"] != prod(map(len, grid.values())):
        raise ValueError("Native summary grid differs from the frozen task")
    evidence["values"] = _validate_table(
        data.table, meta, grid, bond=task["bond"], tolerance=settings["tolerance"]
    )
    if (
        task["mode"] == "control"
        and data
        .grid("maxwell_capacitance", bond=task["bond"], allow_missing=False)
        .values.reshape(2, 2)
        .tolist()
        != saved["maxwell_F"]
    ):
        raise ValueError("Native summary matrix differs from its Parquet matrix")
    evidence["geometry"] = verify_geometry(task_dir, data, grid, task["bond"])
    return data


def _safe(value: Any) -> Any:
    """Keep nonfinite rejected values explicit in strict JSON reports."""
    if isinstance(value, float) and not isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return value


def _compare(candidate: dict, reference: dict, control: str) -> dict:
    """Require independent numerical accuracy and actual h or p refinement."""
    return {
        "mesh_refinement": _mesh_refinement(candidate, reference, control),
        **_numerical_comparison(
            candidate, reference | {"control": control}, TERMINALS, JUNCTION, STABLE
        ),
    }


def _interpolate(data: Dataset, lookup: GridInterpolator, row: dict) -> dict:
    """Compare a physical reference only after its physics matches production."""
    if any(
        row["source"]["provenance"][k] != data.metadata.provenance[k]
        for k in (*STABLE, "bond")
    ):
        raise ValueError("Reference physics differs from production")
    if (
        row["role"] == "calibration"
        and row["control"] == "baseline"
        and row["settings"] != data.metadata.provenance["settings"]
    ):
        raise ValueError("Calibrated settings differ from production")
    if row["role"] == "fresh":
        _numerical_comparison(
            row | {"settings": data.metadata.provenance["settings"]},
            row | {"control": "fine-h"},
            TERMINALS,
            JUNCTION,
            (*STABLE, "bond"),
        )
    prediction = lookup(**row["point"])
    return {
        "reference": row,
        "candidate_F": prediction.tolist(),
        **compare_capacitances(
            prediction, row["maxwell_F"], terminals=TERMINALS, junction=JUNCTION
        ),
    }


def verify(
    study: Path, calibration: Path, *, output: Path | None = None
) -> dict[str, Any]:
    """Qualify new production and fresh holdouts alongside original calibration.

    Calibration also supplies explicitly declared reused production and old held
    references. Every selected native point must pass raw geometry verification.

    Returns:
        Portable raw evidence and separate numerical and geometry gate results.

    Raises:
        ValueError: If frozen sources or selections conflict, or export is blocked.
    """
    study, calibration = study.resolve(), calibration.resolve()
    if study == calibration or study.name == calibration.name:
        raise ValueError("Final and original studies must have distinct identities")
    frozen = {campaign: _plan(campaign) for campaign in (study, calibration)}
    selected, expected = _selection(
        study, calibration, {p: value[0] for p, value in frozen.items()}
    )
    target = frozen[study][0]["grids"][RECIPE]
    report: dict[str, Any] = {
        "passed": False,
        "numerical_passed": False,
        "admission_status": "candidate_pending_independent_numerical_validation",
        "scope": frozen[study][0]["scope"],
        "expected": expected,
        "campaigns": [value[1] for value in frozen.values()],
        "pending": [],
        "failed": [],
        "sources": [],
        "raw_references": [],
        "numerical": [],
        "ideal_wire_refinement": [],
        "finite_bump_difference": [],
        "interpolation": [],
        "retuning_diagnostics": [],
        "geometry_parity": {"passed": False, "tasks": []},
        "verification_source_sha256": {
            path.name: _sha256(path)
            for path in (
                Path(__file__),
                Path(inspect.getfile(verify_geometry)),
                Path(inspect.getfile(Settings)),
                Path(inspect.getfile(validate_native_capacitances)),
            )
        },
    }
    parts, records = [], []
    for campaign, index, job, task, role in selected:
        identity = f"{campaign.name}/{job['name']}/{task['label']}"
        summary = _child(campaign / "results", job["name"]) / "summary.json"
        saved = (
            json.loads(summary.read_text(encoding="utf-8")) if summary.exists() else []
        )
        labels = [r["label"] for r in saved]
        planned = {t["label"] for t in job["tasks"]}
        invalid = [
            r
            for r in saved
            if labels.count(r["label"]) != 1 or r["label"] not in planned
        ]
        if invalid:
            report["failed"].append({
                "identity": identity,
                "raw_summary": invalid,
                "error": "Duplicate or unknown native summary task",
            })
        row = next(
            (
                r
                for r in saved
                if r["label"] == task["label"] and labels.count(r["label"]) == 1
            ),
            None,
        )
        if row is None:
            report["pending"].append(identity)
            continue
        evidence: dict[str, Any] = {
            "identity": identity,
            "role": role,
            "task": task,
            "raw_summary": row,
        }
        report["sources"].append(evidence)
        try:
            data = _load_task(campaign, index, job, task, frozen[campaign][1], evidence)
        except FileNotFoundError as error:
            evidence |= {"passed": False, "error": str(error)}
            report["pending"].append(identity + ": " + str(error))
        except (ValueError, KeyError, TypeError) as error:
            evidence |= {"passed": False, "error": str(error)}
            report["failed"].append(evidence)
            logger.error(f"Rejected {identity}: {error}")
        else:
            evidence["passed"] = True
            report["geometry_parity"]["tasks"].append({
                "identity": identity,
                **evidence["geometry"],
            })
            if role == "production":
                parts.append(data)
            else:
                proof = evidence["geometry"]["points"][0]
                records.append(
                    task
                    | row
                    | {
                        "role": role,
                        "source": {
                            "campaign": campaign.name,
                            "provenance": data.metadata.provenance,
                            "receipt": evidence["receipt"],
                            "parts": evidence["parts"],
                        },
                        "mesh_evidence": proof["mesh"]
                        | {"field_order": proof["field_order"]},
                    }
                )
    report["raw_references"] = records
    calibration_rows = {
        (r["anchor"], r["control"]): r for r in records if r["role"] == "calibration"
    }
    for (anchor, baseline_control), baseline in calibration_rows.items():
        if baseline_control != "baseline":
            continue
        for control in ("fine-h", "hp4", "domain"):
            reference = calibration_rows.get((anchor, control))
            if reference is not None:
                try:
                    comparison = _compare(baseline, reference, control)
                except (ValueError, KeyError) as error:
                    report["failed"].append({
                        "candidate": baseline,
                        "reference": reference,
                        "error": str(error),
                    })
                else:
                    report["numerical"].append({"anchor": anchor, **comparison})
        ideal, fine = (
            calibration_rows.get((anchor, c))
            for c in ("ideal-wire", "ideal-wire-fine-h")
        )
        if ideal is not None and fine is not None:
            if (
                ideal["point"] != baseline["point"]
                or ideal["settings"] != baseline["settings"]
                or any(
                    ideal["source"]["provenance"][k]
                    != baseline["source"]["provenance"][k]
                    for k in STABLE
                )
            ):
                raise ValueError(
                    "Ideal-wire control changes unrelated physics or settings"
                )
            report["ideal_wire_refinement"].append({
                "anchor": anchor,
                **_compare(ideal, fine, "fine-h"),
            })
            report["finite_bump_difference"].append({
                "anchor": anchor,
                "used_for_admission": False,
                **compare_capacitances(
                    baseline["maxwell_F"],
                    ideal["maxwell_F"],
                    terminals=TERMINALS,
                    junction=JUNCTION,
                ),
            })
    held = [r for r in records if r["role"] == "fresh"]
    report["held_out_coverage"] = _held_coverage(target, held)
    frame: pl.DataFrame | None = None
    if len(parts) == expected["production_shards"]:
        meta = parts[0].metadata
        if any(
            (d.metadata.axes, d.metadata.quantities) != (meta.axes, meta.quantities)
            or any(
                d.metadata.provenance[k] != meta.provenance[k]
                for k in (*STABLE, "settings", "bond")
            )
            for d in parts
        ):
            raise ValueError("Production shard physics or solver profiles differ")
        frame = pl.concat([d.table for d in parts])
        report["production_values"] = _validate_table(
            frame,
            meta,
            target,
            bond="physical_bump",
            tolerance=meta.provenance["settings"]["tolerance"],
        )
        data = parts[0]
        lookup = GridInterpolator(
            to_grid(
                frame,
                meta,
                "maxwell_capacitance",
                bond="physical_bump",
                allow_missing=False,
            )
        )
        for row in records:
            if row["bond"] != "physical_bump":
                continue
            comparison = _interpolate(data, lookup, row)
            key = (
                "retuning_diagnostics" if row["role"] == "retuning" else "interpolation"
            )
            if key == "retuning_diagnostics":
                comparison |= {
                    "used_for_accuracy_gate": True,
                    "used_for_fresh_coverage": False,
                }
            report[key].append(comparison)
    report["numerical_passed"] = (
        not report["pending"]
        and not report["failed"]
        and len(report["numerical"]) == 12
        and len(report["ideal_wire_refinement"]) == 4
        and len(report["interpolation"]) == 72
        and len(report["retuning_diagnostics"]) == 24
        and report["held_out_coverage"]["passed"]
        and all(
            r["passed"]
            for key in (
                "numerical",
                "ideal_wire_refinement",
                "interpolation",
                "retuning_diagnostics",
            )
            for r in report[key]
        )
    )
    geometry = report["geometry_parity"]
    geometry["verified_points"] = sum(len(t["points"]) for t in geometry["tasks"])
    geometry["passed"] = (
        not report["pending"]
        and not report["failed"]
        and len(geometry["tasks"]) == expected["tasks"]
        and geometry["verified_points"] == expected["native_points"]
        and all(t["passed"] for t in geometry["tasks"])
    )
    report["passed"] = report["numerical_passed"] and geometry["passed"]
    if report["passed"]:
        report["admission_status"] = "validated_bare_electrostatic_model"
    report = _safe(report)
    if output is not None:
        if not report["passed"] or frame is None:
            raise ValueError(
                "Dataset export blocked: incomplete numerical or raw geometry qualification"
            )
        meta = parts[0].metadata
        axes = tuple(
            a.model_copy(
                update={"validated": (min(target[a.name]), max(target[a.name]))}
            )
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
            frame,
        )
    return report


def main(
    study: Path,
    calibration: Path,
    output: Path | None = None,
    report: Path | None = None,
) -> None:
    """Preview frozen bare campaigns or export a fully qualified grid."""
    result = verify(study, calibration, output=output)
    if report is not None:
        report.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    logger.info(
        f"Bare flipmon passed={result['passed']}, pending={len(result['pending'])}, failed={len(result['failed'])}, geometry_points={result['geometry_parity']['verified_points']}"
    )


if __name__ == "__main__":
    typer.run(main)
