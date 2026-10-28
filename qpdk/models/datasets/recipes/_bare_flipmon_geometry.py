"""Rebuild bare bonded-flipmon masks and verify retained native mesh evidence."""

import hashlib
import importlib
import json
import re
from collections.abc import Mapping, Sequence
from itertools import chain, product, starmap
from math import isclose
from pathlib import Path
from typing import Any

import gdsfactory as gf
import numpy as np

from qpdk import PDK, cells
from qpdk.models.datasets import Dataset
from qpdk.models.datasets.table import to_grid
from qpdk.tech import LAYER

AXES = ("chip_gap", "alignment", "inner_radius", "top_radius")
TERMINALS = ("island", "ring")


def _require(condition: bool, message: str) -> None:
    """Reject inconsistent native geometry with its physical context."""
    if not condition:
        raise ValueError(f"Flipmon geometry: {message}")


def _sha256(path: Path) -> str:
    """Bind compact verification records to retained native files."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _same(actual: Any, expected: Any) -> bool:
    """Compare serialized stack settings within floating point roundoff."""
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and all(_same(actual[k], v) for k, v in expected.items())
        )
    if isinstance(expected, (list, tuple)):
        return (
            isinstance(actual, (list, tuple))
            and len(actual) == len(expected)
            and all(starmap(_same, zip(actual, expected, strict=True)))
        )
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and bool(np.isclose(actual, expected, rtol=0, atol=1e-10))
        )
    return actual == expected


def _stack(point: dict, settings: dict) -> dict:
    """Describe the two-chip process used by the local capacitance model."""
    top = point["chip_gap"] + 0.2
    return {
        "layers": {
            name: {
                "name": name,
                "gds_layer": list(gf.get_layer_tuple(layer)),
                "zmin": lo,
                "zmax": hi,
                "thickness": hi - lo,
                "material": "pec",
                "layer_type": kind,
            }
            for name, layer, lo, hi, kind in (
                ("bottom", LAYER.M1_DRAW, 0, 0.2, "conductor"),
                ("top", LAYER.M2_DRAW, top, top + 0.2, "conductor"),
                ("bump", LAYER.IND, 0.2, top, "via"),
            )
        },
        "materials": {
            "pec": {"conductivity": 1e30},
            "silicon": {"permittivity": settings["permittivity"]},
            "air": {"permittivity": 1.0},
        },
        "dielectrics": [
            {"name": "bottom_chip", "zmin": -300, "zmax": 0, "material": "silicon"},
            {"name": "gap", "zmin": 0, "zmax": top + 0.2, "material": "air"},
            {
                "name": "top_chip",
                "zmin": top + 0.2,
                "zmax": top + 300.2,
                "material": "silicon",
            },
        ],
    }


def _groups(metadata: dict, kind: str) -> dict:
    """Resolve unique named mesh groups with nonempty element sets."""
    groups = metadata["mesh"]["groups"][kind]
    _require(
        len({g["tag"] for g in groups}) == len(groups)
        and len({g["name"] for g in groups}) == len(groups),
        "duplicate mesh groups",
    )
    _require(
        all(
            type(g["tag"]) is int and g["tag"] > 0 and g["elements"] > 0 for g in groups
        ),
        "empty or invalid mesh groups",
    )
    return {g["name"]: g["tag"] for g in groups}


def _geometry(inputs: dict, bond: str) -> dict:
    """Reconstruct the bare island, ring, shifted top pad and both ground frames."""
    point = inputs["point"]
    device = cells.flipmon_with_bbox(
        inner_circle_radius=point["inner_radius"], top_circle_radius=point["top_radius"]
    )
    dbu = device.kcl.dbu
    frame = gf.kdb.Region(gf.kdb.DBox(-400, -400, 400, 400).to_itype(dbu))
    shifted = gf.Component()
    shifted.add_ref(device.extract([LAYER.M2_DRAW, LAYER.M2_ETCH])).dmovex(
        point["alignment"]
    )
    expected = [
        (
            LAYER.M1_DRAW,
            (
                (frame - device.get_region(LAYER.M1_ETCH))
                | device.get_region(LAYER.M1_DRAW)
            ).merged(),
        ),
        (
            LAYER.M2_DRAW,
            (
                (frame - shifted.get_region(LAYER.M2_ETCH))
                | shifted.get_region(LAYER.M2_DRAW)
            ).merged(),
        ),
    ]
    if bond == "physical_bump":
        expected.append((LAYER.IND, device.get_region(LAYER.IND)))
    _require(
        set(inputs["polygons_um"])
        == {str(gf.get_layer_tuple(layer)) for layer, _ in expected},
        "native mask layers differ",
    )
    _require(
        inputs["terminals"] == {"island": [0.0, 0.0], "ring": [100.0, 0.0]},
        "terminal pins differ",
    )
    native = []
    for layer, target in expected:
        region = gf.kdb.Region()
        for vertices in inputs["polygons_um"][str(gf.get_layer_tuple(layer))]:
            _require(bool(np.isfinite(vertices).all()), "nonfinite polygon")
            region.insert(
                gf.kdb.DPolygon(list(starmap(gf.kdb.DPoint, vertices))).to_itype(dbu)
            )
        region.merge()
        _require(
            (target - region.sized(1)).is_empty()
            and (region - target.sized(1)).is_empty(),
            "native footprint differs",
        )
        native.append(region)
    bottom, top = native[:2]
    _require(
        bottom.count() == 3 and top.count() == 2, "native conductor components differ"
    )
    selected = []
    for name in TERMINALS:
        x, y = inputs["terminals"][name]
        probe = gf.kdb.Region(
            gf.kdb.DBox(x - dbu, y - dbu, x + dbu, y + dbu).to_itype(dbu)
        )
        region = bottom.interacting(probe).merged()
        _require(
            region.count() == 1
            and all((region & other).is_empty() for other in selected),
            "native terminals shorted or missing",
        )
        selected.append(region)
    _require(
        (bottom - (selected[0] | selected[1])).merged().count() == 1,
        "bottom ground disconnected",
    )
    center = point["alignment"]
    pad = top.interacting(
        gf.kdb.Region(gf.kdb.DBox(center - dbu, -dbu, center + dbu, dbu).to_itype(dbu))
    ).merged()
    _require(
        pad.count() == 1 and (top - pad).merged().count() == 1,
        "top pad or ground disconnected",
    )
    if bond == "physical_bump":
        via = native[2]
        landing = top.interacting(via).merged()
        _require(
            via.count() == 1 and (via - selected[0]).is_empty(),
            "bump misses inner island",
        )
        _require(
            landing.count() == 1
            and (via - landing).is_empty()
            and (landing ^ pad).is_empty(),
            "bump misses top pad",
        )
    return {
        "dbu_um": dbu,
        "boundary_tolerance_dbu": 1,
        "boundary_tolerance_um": dbu,
        "bottom_conductors": bottom.count(),
        "top_conductors": top.count(),
    }


def _topology(
    metadata: dict, terminals: list, config: dict, bond: str, settings: dict
) -> dict:
    """Check electrode, ground and material assignments in the resolved mesh."""
    _require(
        [(t["index"], t["name"], t["floating"]) for t in terminals]
        == [(i, name, False) for i, name in enumerate(TERMINALS, 1)],
        "terminal identities differ",
    )
    nets = [t["nets"] for t in terminals]
    names = [t["net_names"] for t in terminals]
    _require(
        [len(n) for n in nets] == ([1, 1] if bond == "physical_bump" else [2, 1])
        and len(set(chain.from_iterable(nets))) == sum(map(len, nets)),
        "terminal nets overlap or differ",
    )
    terminal_names = {name for group in names for name in group}
    _require(
        names[1:] == [["ring"]]
        and names[0][0] == "island"
        and [len(n) for n in names] == list(map(len, nets))
        and len(terminal_names) == sum(map(len, names)),
        "terminal net names differ",
    )
    surfaces = _groups(metadata, "surfaces")
    assigned = {}
    for name in TERMINALS:
        assigned[name] = {f"bottom_{face}@{name}" for face in ("xy", "z")}
    top_name = "island" if bond == "physical_bump" else names[0][1]
    assigned["island"] |= {f"top_{face}@{top_name}" for face in ("xy", "z")}
    if bond == "physical_bump":
        assigned["island"].add("air__bump@island")
    metal = {name for name in surfaces if "@" in name}
    remaining = metal - set.union(*assigned.values())
    ground_names = {name.split("@")[1] for name in remaining}
    _require(
        len(ground_names) == 2 and ground_names.isdisjoint(terminal_names),
        "ground nets overlap terminals or each other",
    )
    for layer in ("bottom", "top"):
        ground = {name for name in remaining if name.startswith(layer + "_")}
        _require(
            len(ground) == 2
            and {name.split("@")[0] for name in ground} == {f"{layer}_xy", f"{layer}_z"}
            and len({name.split("@")[1] for name in ground}) == 1,
            "ground frame surfaces differ",
        )
    _require(len(remaining) == 4, "unexpected conductor surfaces")
    assigned["ground"] = remaining
    boundaries = config["Boundaries"]
    _require(
        set(boundaries) == {"Terminal", "Ground"}
        and [t["Index"] for t in boundaries["Terminal"]] == [1, 2],
        "resolved boundary identities differ",
    )
    for name, entry in [
        *zip(TERMINALS, boundaries["Terminal"], strict=True),
        ("ground", boundaries["Ground"]),
    ]:
        attrs = entry["Attributes"]
        _require(
            assigned[name] <= surfaces.keys()
            and len(attrs) == len(set(attrs))
            and set(attrs) == {surfaces[n] for n in assigned[name]},
            "resolved surface attributes differ",
        )
    volumes = _groups(metadata, "volumes")
    expected_volumes = {"silicon", "air"} | (
        {"bump"} if bond == "physical_bump" else set()
    )
    _require(set(volumes) == expected_volumes, "material volumes differ")
    materials = config["Domains"]["Materials"]
    attrs = [tag for m in materials for tag in m["Attributes"]]
    _require(
        len(materials) == len(volumes)
        and len(attrs) == len(set(attrs))
        and set(attrs) == set(volumes.values()),
        "resolved material attributes differ",
    )
    for name, tag in volumes.items():
        material = next(m for m in materials if tag in m["Attributes"])
        expected = {
            "Permittivity": settings["permittivity"] if name == "silicon" else 1.0,
            "Permeability": 1.0,
            "Conductivity": 1e30 if name == "bump" else 0.0,
            "LossTan": 0.0,
            "LondonDepth": 0.0,
        }
        _require(
            all(material.get(k) == v for k, v in expected.items()),
            "resolved material properties differ",
        )
    return {
        name: {n: surfaces[n] for n in sorted(group)}
        for name, group in assigned.items()
    }


def _matrix(path: Path) -> np.ndarray:
    """Read native capacitance or inverse units and return an indexed SI matrix."""
    with path.open(encoding="utf-8") as stream:
        columns = stream.readline().strip().split(",")
    units = [re.search(r"\(([^()]+)\)\s*$", column) for column in columns[1:]]
    _require(
        len(units) == 2 and all(unit is not None for unit in units),
        "raw matrix units are missing",
    )
    names = [unit.group(1) for unit in units if unit is not None]
    scales = {"F": 1.0, "pF": 1e-12, "fF": 1e-15}
    if path.name == "terminal-Cinv.csv":
        scales = {"1/" + unit: 1 / scale for unit, scale in scales.items()}
    _require(
        len(set(names)) == 1 and names[0] in scales,
        "raw matrix units are unsupported or inconsistent",
    )
    values = np.loadtxt(path, delimiter=",", skiprows=1, encoding="utf-8")
    _require(
        values.shape == (2, 3)
        and bool(np.isfinite(values).all())
        and np.array_equal(values[:, 0], [1, 2]),
        "raw matrix indices or values differ",
    )
    return values[:, 1:] * scales[names[0]]


def verify_geometry(
    task_dir: Path, data: Dataset, grid: Mapping[str, Sequence[float]], bond: str
) -> dict:
    """Bind a completed native task to the current local layout and raw matrices.

    The caller validates frozen sources and the task table first. This verifies
    retained mesh-group assignments, not a reconstruction of mesh connectivity.
    Procedural test fixtures exercise this geometry contract only.

    Returns:
        Compact per-point evidence suitable for embedding in dataset metadata.

    Raises:
        ValueError: If synthetic data or inconsistent native evidence is found.
    """
    PDK.activate()
    _require(
        not data.metadata.synthetic, "synthetic data cannot supply admission evidence"
    )
    _require(
        bond in {"physical_bump", "ideal_wire"}
        and set(grid) == set(AXES)
        and all(len(values) > 0 for values in grid.values()),
        "bond or point selection differs",
    )
    _require(data.metadata.terminals == TERMINALS, "dataset terminal order differs")
    points = list(product(*(grid[a] for a in AXES)))
    _require(len(points) == len(set(points)), "duplicate selected points")
    provenance = {
        k: v
        for k, v in data.metadata.provenance.items()
        if k not in {"generator_sha256", "experiment_sha256"}
    }
    settings = provenance["settings"]
    runs = {}
    for path in sorted((task_dir / "runs").glob("*/inputs.json")):
        inputs = json.loads(path.read_text())
        point = inputs["point"]
        key = tuple(point[a] for a in AXES)
        _require(
            set(point) == {*AXES, "bond"}
            and point["bond"] == bond
            and key in points
            and key not in runs,
            "unexpected or duplicate native point",
        )
        _require(
            path.parent.name
            == hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[
                :20
            ],
            "native run key differs",
        )
        _require(
            inputs["provenance"] == provenance
            and inputs["terminal_order"] == list(TERMINALS),
            "native provenance or terminal order differs",
        )
        runs[key] = (path.parent, inputs)
    _require(set(runs) == set(points), "missing native runs for selected points")
    matrices = {
        q: to_grid(data.table, data.metadata, q, bond=bond)
        for q in (
            "maxwell_capacitance",
            "mutual_capacitance",
            "fem_error_indicator_norm",
        )
    }
    for matrix in matrices.values():
        _require(
            matrix.axis_names == AXES
            and all(
                list(coords) == list(grid[a])
                for a, coords in zip(AXES, matrix.coords, strict=True)
            ),
            "table point selection differs",
        )
    records = []
    for index, point in enumerate(points):
        run, inputs = runs[point]
        files = {
            name: run / name
            for name in (
                "inputs.json",
                "metadata.json",
                "terminals.json",
                "complete.json",
                "output/palace/config_resolved.json",
                "output/palace/terminal-C.csv",
                "output/palace/terminal-Cm.csv",
                "output/palace/terminal-Cinv.csv",
                "output/palace/error-indicators.csv",
            )
        }
        raw = {
            name: json.loads(files[name].read_text())
            for name in (
                "metadata.json",
                "terminals.json",
                "complete.json",
                "output/palace/config_resolved.json",
            )
        }
        metadata, terminals, complete, config = raw.values()
        _require(
            complete == {"terminals": list(TERMINALS)},
            "native completion marker differs",
        )
        processes = provenance["processes"]
        _require(type(processes) is int and processes > 0, "invalid mesh process count")
        pad = settings["domain_pad"]
        expected_mesh = {
            "stack": _stack(inputs["point"], settings),
            "airbox": dict(margin_x=pad, margin_y=pad, z_above=pad, z_below=pad),
            "mesh": dict(
                refined_mesh_size=settings["near_mesh"],
                max_mesh_size=settings["far_mesh"],
                planar_conductors=False,
                merge_via_distance=0,
                decimate_tolerance=0,
                threads=processes,
                surface_threads=1,
                verbose=True,
            ),
            "bond": bond,
        }
        _require(
            _same(inputs["mesh_inputs"], expected_mesh)
            and inputs["mesh_threads"] == processes
            and inputs["minimum_feature_elements"] == 4,
            "native stack or mesh settings differ",
        )
        _require(
            _same(inputs["bounds_um"], [-400 - pad, -400 - pad, 400 + pad, 400 + pad]),
            "native domain bounds differ",
        )
        expected_bbox = dict(
            xmin=-400 - pad,
            ymin=-400 - pad,
            zmin=-300 - pad,
            xmax=400 + pad,
            ymax=400 + pad,
            zmax=inputs["point"]["chip_gap"] + 300.4 + pad,
        )
        _require(
            set(metadata["mesh"]["bbox"]) == set(expected_bbox)
            and all(
                np.isclose(metadata["mesh"]["bbox"][k], v, rtol=0, atol=1e-6)
                for k, v in expected_bbox.items()
            ),
            "meshed domain bounds differ",
        )
        _require(
            metadata["solver"] == "palace"
            and metadata["mesh"]["dimension"] == 3
            and metadata["simulation"]["problem_type"] == "Electrostatic"
            and metadata["simulation"]["field_order"] == settings["order"]
            and metadata["mesh"]["field_dofs"]["field_order"] == settings["order"],
            "mesh simulation differs",
        )
        _require(
            all(
                type(metadata["mesh"][key]) is int and metadata["mesh"][key] > 0
                for key in ("nodes", "elements", "tetrahedra")
            )
            and metadata["mesh"]["sicn"]["invalid"] == 0
            and metadata["mesh"]["kappa"]["singular_elements"] == 0,
            "invalid native mesh diagnostics",
        )
        _require(
            config["Problem"]["Type"] == "Electrostatic"
            and isclose(config["Model"]["L0"], 1e-6, rel_tol=1e-12)
            and config["Solver"]["Order"] == settings["order"]
            and config["Solver"]["Linear"]["InitialGuess"] is False
            and config["Solver"]["Linear"]["Tol"] == settings["tolerance"],
            "resolved solver settings differ",
        )
        _require(
            all(
                config["Solver"]["Linear"].get(k) == v
                for k, v in metadata["simulation"]["linear_solver"].items()
            )
            and all(
                config["Model"]["Refinement"].get(k) == v
                for k, v in metadata["simulation"]["refinement"].items()
            )
            and all(
                config["Model"]["Refinement"].get(k, 0) == 0
                for k in ("MaxIts", "UniformLevels", "SerialUniformLevels")
            ),
            "recorded solver or static mesh refinement differs",
        )
        geometry = _geometry(inputs, bond)
        topology = _topology(metadata, terminals, config, bond, settings)
        c, cm, inverse = (
            _matrix(files[f"output/palace/terminal-{name}.csv"])
            for name in ("C", "Cm", "Cinv")
        )
        for name, values in (("maxwell_capacitance", c), ("mutual_capacitance", cm)):
            expected = matrices[name].values.reshape(-1, 2, 2)[index]
            _require(
                np.allclose(values, expected, rtol=1e-12, atol=0),
                "raw matrix differs from Parquet",
            )
        mutual = -c.copy()
        np.fill_diagonal(mutual, c.sum(axis=1))
        _require(
            np.allclose(cm, mutual, rtol=1e-10, atol=1e-26)
            and np.allclose(c @ inverse, np.eye(2), rtol=0, atol=1e-9),
            "raw matrices inconsistent",
        )
        indicator = np.loadtxt(
            files["output/palace/error-indicators.csv"],
            delimiter=",",
            skiprows=1,
            ndmin=2,
            encoding="utf-8",
        )
        expected_indicator = matrices["fem_error_indicator_norm"].values.reshape(-1)[
            index
        ]
        _require(
            indicator.shape == (1, 4)
            and bool(np.isfinite(indicator).all())
            and bool((indicator >= 0).all())
            and bool(
                np.isclose(indicator[0, 0], expected_indicator, rtol=1e-12, atol=0)
            ),
            "raw FEM indicator differs from Parquet",
        )
        records.append({
            "point": inputs["point"],
            "run": run.name,
            "raw_sha256": {name: _sha256(path) for name, path in files.items()},
            "mesh_hash": metadata["mesh"]["mesh_hash"],
            "raw": {"inputs": inputs, **raw},
            "matrices": {"C": c.tolist(), "Cm": cm.tolist(), "Cinv": inverse.tolist()},
            "matrix_units": {"C": "F", "Cm": "F", "Cinv": "1/F"},
            "matrix_headers": {
                name: files[f"output/palace/terminal-{name}.csv"]
                .read_text(encoding="utf-8")
                .splitlines()[0]
                for name in ("C", "Cm", "Cinv")
            },
            "indicator_statistics": dict(
                zip(
                    ("norm", "minimum", "maximum", "mean"),
                    indicator[0].tolist(),
                    strict=True,
                )
            ),
            "mesh": {
                key: metadata["mesh"][key]
                for key in (
                    "nodes",
                    "elements",
                    "tetrahedra",
                    "geometry_orders",
                    "edge_length",
                    "quality",
                    "sicn",
                    "kappa",
                    "field_dofs",
                )
            },
            "field_order": metadata["simulation"]["field_order"],
            "topology": topology,
            **geometry,
        })
    modules = {
        fn.__module__
        for fn in (
            cells.flipmon_with_bbox,
            cells.flipmon,
            cells.indium_bump,
            cells.squid_junction_long,
        )
    } | {"qpdk.tech", __name__}
    sources = {}
    for name in sorted(modules):
        source = importlib.import_module(name).__file__
        if source is None:
            raise ValueError(f"Missing layout module source: {name}")
        sources[name] = _sha256(Path(source))
    return {
        "passed": True,
        "points": records,
        "factory": "qpdk.cells.flipmon_with_bbox",
        "factory_source_sha256": sources,
        "scope": "Local electrostatic inputs and retained mesh-group assignments. No distributed line, bump inductance or nonlinear spectrum.",
    }
