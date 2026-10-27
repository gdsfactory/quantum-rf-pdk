# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "qpdk[models] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git@a93c7641c5c234ab87c0497c91a02bbf9baf47dd",
#   "typer>=0.24,<1",
# ]
# ///
"""Curate local qubit grids only after independent numerical and held-out checks.

The frozen campaign plan identifies every production shard and reference solve.
Source hashes, per-shard metadata and the raw reference matrices travel inside
Parquet metadata. A missing or failed task cannot publish a dataset. Reusing a
production shard preserves its original provenance rather than relabelling it
as a new solve. Pass earlier refinement directories with repeated ``--references``
options so their independent checks remain part of admission. Preview reports
remain operational JSON, outside the dataset.

Lower-order probes are retained as negative controls. They need a same-mesh
comparison with the calibrated production profile and cannot replace refinement
references or held-out coverage.
"""

import hashlib
import json
from itertools import pairwise, product
from math import prod
from pathlib import Path
from typing import Any

import polars as pl
import typer

from qpdk import logger
from qpdk.models.datasets import Dataset, GridInterpolator, check_maxwell
from qpdk.models.datasets.generate import write
from qpdk.models.datasets.table import to_grid
from qpdk.models.datasets.validation import compare_capacitances


def _records(campaign: Path, recipe: str) -> list[dict[str, Any]]:
    """Collect completed control rows with their scheduler and source identities."""
    plan = json.loads((campaign / "source/plan.json").read_text())
    records = []
    manifest = campaign / "source/source-sha256.json"
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    for index, job in enumerate(plan["jobs"]):
        path = campaign / "results" / job["name"] / "summary.json"
        if path.exists():
            tasks = {t["label"]: t for t in job["tasks"]}
            receipt = json.loads((campaign / f"submission-{index}.json").read_text())
            origin = {
                "campaign": campaign.name,
                "job": job["name"],
                "job_id": receipt["stdout"].strip(),
                "source_manifest_sha256": manifest_hash,
            }
            for saved in json.loads(path.read_text()):
                task = tasks[saved["label"]]
                row = {**task, **saved}
                if row["recipe"] != recipe:
                    continue
                if {task.get("control"), row.get("control")} & {
                    "lower-order",
                    "same-mesh",
                } and (
                    any(
                        row.get(key) != task.get(key)
                        for key in ("mode", "control", "anchor", "point")
                    )
                    or any(
                        row["settings"].get(key) != value
                        for key, value in task["settings"].items()
                    )
                ):
                    raise ValueError("Diagnostic summary differs from its frozen task")
                row["source"] = dict(origin)
                if row["status"] == "ok" and row["mode"] == "control":
                    data = Dataset(path.parent / row["label"] / "data")
                    if data.metadata.synthetic:
                        raise ValueError(
                            "Synthetic references cannot validate FEM data"
                        )
                    if (
                        row["control"] in {"lower-order", "same-mesh"}
                        and data.metadata.provenance["settings"] != row["settings"]
                    ):
                        raise ValueError(
                            "Diagnostic settings differ from its Parquet provenance"
                        )
                    row["source"]["provenance"] = data.metadata.provenance
                    residual = data.grid(
                        "solver_relative_residual", allow_missing=False
                    ).values.max()
                    if residual > row["settings"]["tolerance"]:
                        raise ValueError("A reference exceeds its residual tolerance")
                    matrix = data.grid(
                        "maxwell_capacitance", allow_missing=False
                    ).values.reshape(len(data.metadata.terminals), -1)
                    if matrix.tolist() != row["maxwell_F"]:
                        raise ValueError(
                            "Reference summary differs from its Parquet matrix"
                        )
                records.append(row)
    return records


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


def verify(
    study: Path,
    calibration: Path,
    recipe: str,
    *,
    references: list[Path] | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    """Check a complete frozen study and optionally publish its validated grid.

    Args:
        study: Final campaign containing the target grid and reused shard list.
        calibration: Campaign with independent mesh, order and domain controls.
        recipe: Local qubit extraction recipe.
        references: Earlier refinement campaigns whose raw evidence must be retained.
        output: Destination for the admitted Parquet dataset, or preview only.

    Returns:
        Progress and accuracy evidence, including all raw reference matrices.

    Raises:
        ValueError: If completed shards disagree or publication gates fail.
    """
    plan = json.loads((study / "source/plan.json").read_text())
    target = plan["grids"][recipe]
    stable = (
        "settings",
        "runtime_sha256",
        "versions",
        "gsim_revision",
        "qpdk_geometry_revision",
        "reference_planes",
        "junction",
        "scope",
    )
    report: dict[str, Any] = {
        "recipe": recipe,
        "rtol": 0.02,
        "atol_F": 1e-17,
        "pending": [],
        "failed": [],
        "sources": [],
        "numerical": [],
        "interpolation": [],
        "passed": False,
    }
    campaigns = list(dict.fromkeys((study, calibration, *(references or []))))
    if len({c.name for c in campaigns}) != len(campaigns):
        raise ValueError("Campaign names must identify a unique source directory")
    reuse = [r for r in plan.get("reused_production", []) if r["recipe"] == recipe]
    reuse_keys = {(r["campaign"], r["job"], r["label"]) for r in reuse}
    if len(reuse_keys) != len(reuse):
        raise ValueError("A reused production shard is listed more than once")
    found_reuse = set()
    parts = []
    for campaign in campaigns:
        campaign_plan = json.loads((campaign / "source/plan.json").read_text())
        manifest = campaign / "source/source-sha256.json"
        for filename, expected in json.loads(manifest.read_text()).items():
            if (
                hashlib.sha256(
                    (campaign / "source" / filename).read_bytes()
                ).hexdigest()
                != expected
            ):
                raise ValueError(f"Frozen source changed: {filename}")
        for index, job in enumerate(campaign_plan["jobs"]):
            folder = campaign / "results" / job["name"]
            path = folder / "summary.json"
            rows = (
                {r["label"]: r for r in json.loads(path.read_text())}
                if path.exists()
                else {}
            )
            for task in job["tasks"]:
                if task["recipe"] != recipe:
                    continue
                identity = (campaign.name, job["name"], task["label"])
                if identity in reuse_keys:
                    found_reuse.add(identity)
                relevant = (
                    campaign == study
                    or task["mode"] == "control"
                    or identity in reuse_keys
                )
                if not relevant:
                    continue
                label = f"{campaign.name}/{job['name']}/{task['label']}"
                row = rows.get(task["label"])
                if row is None:
                    report["pending"].append(label)
                elif row["status"] != "ok":
                    if task["mode"] == "production":
                        report["failed"].append({
                            "label": label,
                            "error": row.get("error"),
                        })
                elif task["mode"] == "production":
                    dataset = Dataset(folder / task["label"] / "data")
                    receipt = campaign / f"submission-{index}.json"
                    report["sources"].append({
                        "campaign": campaign.name,
                        "job": job["name"],
                        "job_id": json.loads(receipt.read_text())["stdout"].strip(),
                        "source_manifest_sha256": hashlib.sha256(
                            manifest.read_bytes()
                        ).hexdigest(),
                        "metadata": dataset.metadata.model_dump(mode="json"),
                        "parts": {
                            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(
                                (folder / task["label"] / "data").glob("*.parquet")
                            )
                        },
                    })
                    parts.append(dataset)
    for identity in sorted(reuse_keys - found_reuse):
        report["pending"].append("/".join(identity))
    expected_parts = sum(
        t["recipe"] == recipe and t["mode"] == "production"
        for j in plan["jobs"]
        for t in j["tasks"]
    ) + sum(r["recipe"] == recipe for r in plan.get("reused_production", []))
    calibration_records = _records(calibration, recipe)
    study_records = [
        row
        for campaign in campaigns
        if campaign != calibration
        for row in _records(campaign, recipe)
    ]
    calibration_rows = [
        r
        for r in calibration_records
        if r["recipe"] == recipe and r["mode"] == "control" and r["status"] == "ok"
    ]
    study_rows = [
        r
        for r in study_records
        if r["recipe"] == recipe and r["mode"] == "control" and r["status"] == "ok"
    ]
    candidates = {
        r["anchor"]: r for r in calibration_rows if r["control"] == "candidate"
    }
    report["superseded_failures"] = []
    for row in calibration_records + study_records:
        if row["recipe"] != recipe or row["status"] == "ok":
            continue
        replacement = next(
            (
                r
                for r in study_rows
                if r["point"] == row["point"] and r["settings"] == row["settings"]
            ),
            None,
        )
        if replacement is None:
            report["failed"].append(row)
        else:
            report["superseded_failures"].append({
                "original": row,
                "replacement_source": replacement["source"],
                "replacement_label": replacement["label"],
            })
    terminals = (
        ("left_pad", "right_pad", "readout")
        if recipe == "double_pad_readout"
        else ("island", "readout")
    )
    junction = (
        ("left_pad", "right_pad")
        if recipe == "double_pad_readout"
        else ("island", None)
    )
    report["diagnostic_controls"] = []
    for row in study_rows:
        if row["control"] != "lower-order":
            continue
        if row["anchor"].startswith("dense-held-"):
            raise ValueError("Diagnostic pairs cannot claim held-out coverage")
        candidate = next(
            (
                r
                for r in study_rows
                if r["control"] == "same-mesh"
                and r["anchor"] == row["anchor"]
                and r["point"] == row["point"]
            ),
            None,
        )
        if candidate is None:
            raise ValueError("Lower-order diagnostic needs its same-mesh candidate")
        if not any(candidate["settings"] == r["settings"] for r in candidates.values()):
            raise ValueError(
                "Diagnostic candidate must use the calibrated production settings"
            )
        if any(
            candidate["source"]["provenance"][key] != row["source"]["provenance"][key]
            for key in stable
            if key != "settings"
        ):
            raise ValueError(
                "Diagnostic physics or solver provenance differs from candidate"
            )
        changed = {
            key
            for key in candidate["settings"]
            if candidate["settings"][key] != row["settings"][key]
        }
        if (
            changed != {"order"}
            or row["settings"]["order"] >= candidate["settings"]["order"]
        ):
            raise ValueError(
                "Lower-order diagnostic must only decrease polynomial order"
            )
        report["diagnostic_controls"].append({
            "label": row["label"],
            "point": row["point"],
            "candidate_source": candidate["source"],
            "diagnostic_source": row["source"],
            "candidate_settings": candidate["settings"],
            "diagnostic_settings": row["settings"],
            "candidate_maxwell_F": candidate["maxwell_F"],
            "diagnostic_maxwell_F": row["maxwell_F"],
            "used_for_admission": False,
            "reason": "A lower polynomial order is a negative control, not a refinement reference.",
            **compare_capacitances(
                candidate["maxwell_F"],
                row["maxwell_F"],
                terminals=terminals,
                junction=junction,
            ),
        })
    study_rows = [r for r in study_rows if r["control"] != "lower-order"]
    report["held_out_coverage"] = _coverage(
        target,
        study_rows,
        require_all=plan.get("reference_coverage_target", {}).get(
            "all_cells_required_for_publication", False
        ),
    )
    controls = {
        (r["anchor"], r["control"]): r
        for r in calibration_rows + study_rows
        if r["anchor"] in candidates and r["control"] in {"fine-h", "hp4", "domain"}
    }
    for (anchor, control), row in controls.items():
        candidate = candidates[anchor]
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
            and any(
                row["settings"][key] >= candidate["settings"][key] for key in changed
            )
        ) or (
            control != "fine-h"
            and any(
                row["settings"][key] <= candidate["settings"][key] for key in changed
            )
        ):
            raise ValueError("Numerical control must refine the candidate")
        report["numerical"].append({
            "anchor": anchor,
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
        })
    numerical_complete = len(candidates) == 5 and set(controls) == set(
        product(candidates, ("fine-h", "hp4", "domain"))
    )
    if len(parts) == expected_parts:
        metadata = parts[0].metadata
        for dataset in parts:
            a, b = metadata, dataset.metadata
            if (a.axes, a.terminals, a.quantities, a.reference_ground) != (
                b.axes,
                b.terminals,
                b.quantities,
                b.reference_ground,
            ):
                raise ValueError("Production shard conventions differ")
            if (
                a.synthetic
                or b.synthetic
                or any(a.provenance[k] != b.provenance[k] for k in stable)
            ):
                raise ValueError(
                    "Production shard physics or solver provenance differs"
                )
        if any(
            c["settings"] != metadata.provenance["settings"]
            for c in candidates.values()
        ):
            raise ValueError("Calibration mesh does not match production settings")
        frame = pl.concat([d.scan().collect() for d in parts])
        if frame.select(list(target)).unique().height != prod(
            map(len, target.values())
        ) or any(
            frame[k].unique().sort().to_list() != values for k, values in target.items()
        ):
            raise ValueError(
                "Production shards do not cover exactly the requested grid"
            )
        for quantity in metadata.quantities:
            to_grid(frame, metadata, quantity.name, allow_missing=False)
        grid = to_grid(frame, metadata, "maxwell_capacitance", allow_missing=False)
        check_maxwell(grid.values)
        if (
            frame.filter(pl.col("quantity") == "solver_relative_residual")[
                "value"
            ].max()
            > metadata.provenance["settings"]["tolerance"]
        ):
            raise ValueError("A production solve exceeds its residual tolerance")
        lookup = GridInterpolator(grid)
        references = [
            r for r in calibration_rows if r["control"] != "candidate"
        ] + study_rows
        for row in references:
            if any(
                row["source"]["provenance"][key] != metadata.provenance[key]
                for key in stable
                if key != "settings"
            ):
                raise ValueError(
                    "Reference physics or solver provenance differs from production"
                )
            report["interpolation"].append({
                "label": row["label"],
                "source": row["source"],
                "point": row["point"],
                "reference_settings": row["settings"],
                "reference_maxwell_F": row["maxwell_F"],
                **compare_capacitances(
                    lookup(**row["point"]),
                    row["maxwell_F"],
                    terminals=terminals,
                    junction=junction,
                ),
            })
        report["passed"] = (
            numerical_complete
            and report["held_out_coverage"]["passed"]
            and not report["pending"]
            and not report["failed"]
            and all(r["passed"] for r in report["numerical"] + report["interpolation"])
        )
        if output is not None and report["passed"]:
            provenance = {
                k: v
                for k, v in metadata.provenance.items()
                if k
                not in {
                    "study_source_sha256",
                    "generator_sha256",
                    "extraction_sha256",
                    "experiment_sha256",
                }
            }
            provenance |= {
                "admission_status": "validated_local_electrostatic_model",
                "validation": report,
            }
            axes = tuple(
                a.model_copy(
                    update={"validated": (min(target[a.name]), max(target[a.name]))}
                )
                for a in metadata.axes
            )
            write(
                output,
                metadata.model_copy(update={"axes": axes, "provenance": provenance}),
                frame,
            )
            logger.info(f"Published validated local dataset: {output}")
    if output is not None and not report["passed"]:
        raise ValueError("Dataset is incomplete or failed independent accuracy checks")
    return report


def main(
    study: Path,
    calibration: Path,
    recipe: str,
    output: Path | None = None,
    report: Path | None = None,
    references: list[Path] | None = None,
) -> None:
    """Preview frozen campaign verification or publish a validated dataset."""
    result = verify(study, calibration, recipe, references=references, output=output)
    if report is not None:
        report.write_text(json.dumps(result, indent=2) + "\n")
    logger.info(
        f"{recipe}: passed={result['passed']}, pending={len(result['pending'])}, failed={len(result['failed'])}"
    )


if __name__ == "__main__":
    typer.run(main)
