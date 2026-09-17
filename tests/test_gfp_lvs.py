"""LVS checks for the checked-in schematic and layout fallback."""

from __future__ import annotations

import ast
import json
import xml.etree.ElementTree as ET  # ruff: ignore[suspicious-xml-etree-import]
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import gdsfactory as gf
import pytest
from conftest import import_gfp_module

from qpdk import PDK

if TYPE_CHECKING:
    from types import ModuleType

pytestmark = pytest.mark.gfp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_STEM = "resonator_test_chip_yaml"
SAMPLE_DIR = PROJECT_ROOT / "qpdk/samples"
GSCH_PATH = SAMPLE_DIR / f"{SAMPLE_STEM}.gsch"
PIC_YML_PATH = SAMPLE_DIR / f"{SAMPLE_STEM}.pic.yml"


@dataclass(frozen=True, slots=True, kw_only=True)
class SchematicProject:
    """Temporary project containing the checked-in schematic."""

    root: Path
    gsch: Path


@pytest.fixture(scope="module")
def gfp_check() -> ModuleType:
    """Provide the GDSFactory+ verification API."""
    return import_gfp_module("gdsfactoryplus.check")


@pytest.fixture(scope="module")
def schematic_project(tmp_path_factory: pytest.TempPathFactory) -> SchematicProject:
    """Copy the real schematic with the model index needed by LVS."""
    root = tmp_path_factory.mktemp("gfp_lvs")
    document = json.loads(GSCH_PATH.read_text())
    model_ids = sorted({
        item["model"]
        for item in document.values()
        if isinstance(item, dict) and item.get("model")
    })
    gsch = root / GSCH_PATH.name
    gsch.write_text(GSCH_PATH.read_text())
    (root / "models.nyanlib").write_text(
        json.dumps({
            f"models:{model_id}": {
                "name": model_id.rsplit(".", maxsplit=1)[-1],
                "type": "ckt",
                "tags": ["qpdk"],
            }
            for model_id in model_ids
        })
    )
    return SchematicProject(root=root, gsch=gsch)


def _write_layout(pic_yml: Path, output_dir: Path) -> Path:
    """Materialize a fallback using the schematic's top-cell name."""
    PDK.activate()
    gds_path = output_dir / f"{SAMPLE_STEM}.gds"
    gf.read.from_yaml(pic_yml, name=SAMPLE_STEM).write_gds(gds_path)
    return gds_path


def _item_description(item: ET.Element) -> str:
    descriptions = []
    for value in item.findall("./values/value"):
        text = value.text or ""
        if not text.startswith("text: "):
            continue
        description = text.removeprefix("text: ").strip()
        with suppress(SyntaxError, ValueError):
            description = ast.literal_eval(description)
        descriptions.append(description)
    return "; ".join(descriptions)


def test_lvs_matches_checked_in_sources(
    gfp_check: ModuleType,
    schematic_project: SchematicProject,
    tmp_path: Path,
) -> None:
    """Match the real schematic against its generated layout fallback."""
    xml = gfp_check.check_lvs(
        str(_write_layout(PIC_YML_PATH, tmp_path)),
        str(schematic_project.gsch),
        "qpdk",
        project_root=str(schematic_project.root),
    )
    root = ET.fromstring(xml)  # ruff: ignore[suspicious-xml-element-tree-usage]

    assert root.tag == "report-database"
    assert not list(root.iter("item"))


def test_lvs_rejects_broken_layout_fallback(
    gfp_check: ModuleType,
    schematic_project: SchematicProject,
    tmp_path: Path,
) -> None:
    """Catch a top-level connection dropped from the layout fallback."""
    broken_pic_yml = tmp_path / PIC_YML_PATH.name
    broken_pic_yml.write_text(
        PIC_YML_PATH.read_text().replace("  o4: probe_east_bot,waveport\n", "")
    )
    xml = gfp_check.check_lvs(
        str(_write_layout(broken_pic_yml, tmp_path)),
        str(schematic_project.gsch),
        "qpdk",
        project_root=str(schematic_project.root),
    )
    root = ET.fromstring(xml)  # ruff: ignore[suspicious-xml-element-tree-usage]
    items = list(root.iter("item"))
    descriptions = " | ".join(_item_description(item) for item in items)

    assert items
    assert "o4" in descriptions
