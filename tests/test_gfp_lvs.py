"""Verification checks of gdsfactoryplus v2 (LVS, connectivity, DRC smoke).

Covers the gdsfactoryplus 2.0.0 verification API (``gdsfactoryplus.check``)
against this PDK:

- ``check_lvs``: the ``resonator_test_chip_yaml`` sample layout matches its
  ``resonator_test_chip_yaml.pic.yml`` schematic with zero violations
  (elvis engine).
- ``check_lvs`` detects real violations (negative control, so the passing
  assertion above cannot become vacuous).
- ``check_connectivity``: the report is parseable and contains no
  short/overlap/mismatch violations. ``DanglingPort`` items are expected
  noise for hierarchical GDS: library leaf-cell ports connect in their
  parent, and ``resonator_o1`` of ``quarter_wave_resonator_coupled`` is
  intentionally unterminated (capacitive coupling through the gap).
- ``check_drc``: remote submission smoke test against the
  ``quantum_rf`` deck, skipped unless ``GFP_API_KEY`` is set.

Behavioral notes these tests encode:

- ``check_lvs`` accepts ``.gsch`` (via nyancir) and ``.pic.yml`` schematics;
  ``.pic.yml`` files are resolved recursively from ``project_root`` by
  component name, so only components with their own ``*.pic.yml`` expand.
- The SDK LVS flow does not forward qpdk's ``[tool.elvis.equivalent-ports]``
  config (``launcher``: ``o1``/``waveport``). Empirically this does not
  break the resonator test chip: elvis derives equivalent-port groups from
  GDS pin metadata and the launcher nets produce neither opens nor port
  mismatches.

The schematic/layout pairing mirrors the gfp app: the ``.pic.yml`` sample is
a wrapper around ``resonator_test_chip_python``, so the compared layout is a
wrapper cell containing that chip as a single instance — the hierarchy the
gfp app builds from the schematic. A flattened ``resonator_test_chip_python``
GDS does NOT match this schematic (its top cell directly contains the
resonator/probeline instances) and is intentionally not used here.

Reports are parsed with stdlib ``xml.etree`` (bandit XML rules are ignored
at the use sites): the LYRDB output is generated locally by elvis/klayout,
never received from untrusted sources.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET  # ruff: ignore[suspicious-xml-etree-import]
from pathlib import Path
from typing import TYPE_CHECKING

import gdsfactory as gf
import pytest
from conftest import import_gfp_module

from qpdk import PDK
from qpdk.cells import double_pad_transmon
from qpdk.samples.resonator_test_chip import resonator_test_chip_python

if TYPE_CHECKING:
    from types import ModuleType

pytestmark = pytest.mark.gfp

#: qpdk repository root (parent of ``tests/``); the pic.yml search root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Schematic describing the resonator test chip (wrapper sample).
SCHEMATIC_PATH = PROJECT_ROOT / "qpdk/samples/resonator_test_chip_yaml.pic.yml"

#: Environment variable holding the gdsfactoryplus DRC API key.
GFP_API_KEY_ENV = "GFP_API_KEY"

#: ``[tool.gdsfactoryplus.drc] pdk`` in pyproject.toml; also the portal
#: project slug for this PDK's DRC submissions (qpdk configures no
#: ``[tool.gdsfactoryplus.project].identifier``).
QUANTUM_RF_SLUG = "quantum_rf"

#: Connectivity-check categories that indicate real defects. ``DanglingPort``
#: is excluded on purpose (see module docstring).
CONNECTIVITY_VIOLATION_CATEGORIES = frozenset({
    "PortOverlap",
    "InstanceOverlap",
    "CellShapeInstanceOverlap",
    "PortMismatch",
})


@pytest.fixture
def gfp_check() -> ModuleType:
    """Provide the gdsfactoryplus.check module (see import_gfp_module)."""
    return import_gfp_module("gdsfactoryplus.check")


def _wrapper_component(name: str, *, with_o4: bool = True) -> gf.Component:
    """Build and return the layout the ``resonator_test_chip_yaml.pic.yml`` describes.

    A single instance ``resonator_test_chip`` of ``resonator_test_chip_python``
    with all four probeline ports exposed on the wrapper, matching the
    ``instances``/``ports`` sections of the schematic.

    Returns:
        Wrapper component whose top-cell name matches the schematic.
    """
    chip = resonator_test_chip_python()
    wrapper = gf.Component(name)
    ref = wrapper.add_ref(chip, name="resonator_test_chip")
    for port_name in ("o1", "o2", "o3", "o4"):
        if port_name == "o4" and not with_o4:
            continue
        wrapper.add_port(port_name, port=ref.ports[port_name])
    return wrapper


@pytest.fixture(scope="module")
def chip_yaml_gds(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write the schematic-matching test-chip layout GDS to a temp directory."""
    PDK.activate()  # not applied to module-scoped fixtures by the autouse hook
    gds_path = tmp_path_factory.mktemp("gfp_verify") / "resonator_test_chip_yaml.gds"
    _wrapper_component("resonator_test_chip_yaml").write_gds(gds_path)
    return gds_path


def _leaf_category(item: ET.Element) -> str:
    """Return the leaf category path of a LYRDB item, e.g. ``LVS.open``."""
    return item.findtext("category") or ""


def _item_description(item: ET.Element) -> str:
    """Return the human-readable text values of a LYRDB item."""
    return "; ".join(
        (value.text or "").removeprefix("text: ").strip()
        for value in item.findall("./values/value")
        if (value.text or "").startswith("text: ")
    )


def _describe_violations(root: ET.Element) -> str:
    """Format all LYRDB items for assertion failure messages."""
    return "\n".join(
        f"[{_leaf_category(item)}] {_item_description(item)}"
        for item in root.iter("item")
    )


def test_lvs_resonator_test_chip_matches_schematic(
    gfp_check: ModuleType, chip_yaml_gds: Path
) -> None:
    """Elvis LVS reports zero violations for the sample chip vs its pic.yml."""
    xml = gfp_check.check_lvs(
        str(chip_yaml_gds),
        str(SCHEMATIC_PATH),
        "qpdk",
        project_root=str(PROJECT_ROOT),
    )
    root = ET.fromstring(xml)  # ruff: ignore[suspicious-xml-element-tree-usage]

    assert root.tag == "report-database", xml[:500]
    violations = _describe_violations(root)
    assert not list(root.iter("item")), f"LVS violations:\n{violations}"


def test_lvs_reports_a_broken_layout(gfp_check: ModuleType, tmp_path: Path) -> None:
    """Negative control: dropping the o4 port must produce LVS violations."""
    gds_path = tmp_path / "resonator_test_chip_yaml_broken.gds"
    _wrapper_component("resonator_test_chip_yaml_broken", with_o4=False).write_gds(
        gds_path
    )

    xml = gfp_check.check_lvs(
        str(gds_path),
        str(SCHEMATIC_PATH),
        "qpdk",
        project_root=str(PROJECT_ROOT),
    )
    root = ET.fromstring(xml)  # ruff: ignore[suspicious-xml-element-tree-usage]
    items = list(root.iter("item"))
    descriptions = " | ".join(_item_description(item) for item in items)

    assert items, "LVS accepted a layout that lost the o4 port connection"
    assert "o4" in descriptions, f"Violations do not mention o4:\n{descriptions}"


def test_connectivity_no_shorts_or_overlaps(
    gfp_check: ModuleType, chip_yaml_gds: Path
) -> None:
    """Connectivity check finds no shorts/overlaps/mismatches on the sample.

    ``check_connectivity`` aggregates kfactory's port mismatch, dangling
    port, instance overlap, and shape/instance overlap checks into one
    LYRDB report. ``DanglingPort`` items are expected (see module
    docstring); every other category indicates a real defect.
    """
    xml = gfp_check.check_connectivity(str(chip_yaml_gds), verbose=False)
    root = ET.fromstring(xml)  # ruff: ignore[suspicious-xml-element-tree-usage]

    assert root.tag == "report-database", xml[:500]
    unexpected = [
        f"[{_leaf_category(item)}] {_item_description(item)}"
        for item in root.iter("item")
        if _leaf_category(item).rsplit(".", maxsplit=1)[-1]
        in CONNECTIVITY_VIOLATION_CATEGORIES
    ]
    assert not unexpected, "Connectivity violations:\n" + "\n".join(unexpected)


@pytest.mark.skipif(
    not os.environ.get(GFP_API_KEY_ENV),
    reason="GFP_API_KEY not set — remote DRC submission would fail",
)
def test_drc_submission_smoke(gfp_check: ModuleType, tmp_path: Path) -> None:
    """Submit a small qpdk cell to the remote DRC service without error."""
    gds_path = tmp_path / "double_pad_transmon.gds"
    double_pad_transmon().write_gds(gds_path)

    submission = gfp_check.check_drc(
        str(gds_path),
        project_slug=QUANTUM_RF_SLUG,
        api_key=os.environ[GFP_API_KEY_ENV],
    )

    assert submission["sub_id"], f"Missing sub_id in submission: {submission}"
    assert submission["status"], f"Missing status in submission: {submission}"
