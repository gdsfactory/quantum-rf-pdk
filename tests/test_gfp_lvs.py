"""Verification checks of gdsfactoryplus v2 (LVS, connectivity, DRC smoke).

Covers the gdsfactoryplus 2.0.0 verification API (``gdsfactoryplus.check``)
against this PDK:

- ``check_lvs``: a hand-authored ``.gsch`` nyancir wrapper schematic matches
  its wrapper layout with zero violations (elvis engine), plus a negative
  control so the passing assertion cannot become vacuous. Both checked-in
  ``.pic.yml`` samples are covered against their directly materialized
  layouts, so a gfp release dropping ``.pic.yml`` schematic resolution does
  not go unnoticed: the resonator test chip matches with zero violations,
  the qubit test chip keeps two by-design opens.
- ``check_connectivity``: the report is parseable and contains no
  short/overlap/mismatch violations. ``DanglingPort`` items are expected
  noise for hierarchical GDS: library leaf-cell ports connect in their
  parent, and ``resonator_o1`` of ``quarter_wave_resonator_coupled`` is
  intentionally unterminated (capacitive coupling through the gap).
- ``check_drc``: remote submission smoke test, skipped unless
  ``GFP_API_KEY`` is set and ``[tool.gdsfactoryplus.project].identifier``
  (the portal project slug submissions are filed under) is configured.

Behavioral notes these tests encode:

- ``check_lvs`` resolves a ``.gsch`` schematic and its ``models.nyanlib``
  through nyancad's ``FileAPI`` relative to ``project_root``.
- ``check_lvs`` resolves a ``.pic.yml`` schematic purely through YAML: it
  rglobs ``*.pic.yml`` under ``project_root`` and recursively expands only
  components that have their own ``.pic.yml`` file. No ``models.nyanlib``
  is involved, so ``project_root`` can be the repository root directly.
- The SDK LVS flow does not forward qpdk's ``[tool.elvis.equivalent-ports]``
  config (``launcher``: ``o1``/``waveport``). Empirically this does not
  break the resonator test chip: elvis derives equivalent-port groups from
  GDS pin metadata and the launcher nets produce neither opens nor port
  mismatches.

The schematic/layout pairing mirrors the gfp app: the nyancir is a wrapper
around ``resonator_test_chip_python``, so the compared layout is a wrapper
cell containing that chip as a single instance. Both ``.pic.yml`` samples,
in contrast, are fully detailed netlists and LVS-match their directly
materialized layouts, because gdsfactory derives their nets from the very
placement and routes it materializes. The qubit sample keeps two by-design
opens: its tees end on the launchers' ``o1`` pins, leaving the
``launcher_bot``/``launcher_top`` waveports unterminated
(``_EXPECTED_OPEN_DESCRIPTIONS``).

Reports are parsed with stdlib ``xml.etree`` (bandit XML rules are ignored
at the use sites): the LYRDB output is generated locally by elvis/klayout,
never received from untrusted sources.
"""

from __future__ import annotations

import ast
import json
import os
import tomllib
import xml.etree.ElementTree as ET  # ruff: ignore[suspicious-xml-etree-import]
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gdsfactory as gf
import pytest
from conftest import import_gfp_module

from qpdk import PDK
from qpdk.cells import double_pad_transmon
from qpdk.samples.resonator_test_chip import resonator_test_chip_python

if TYPE_CHECKING:
    from types import ModuleType

pytestmark = pytest.mark.gfp

#: qpdk repository root (parent of ``tests/``); where pyproject.toml lives.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Fully qualified factory id of the resonator test chip sample.
_CHIP_QUALNAME = "qpdk.samples.resonator_test_chip.resonator_test_chip_python"

#: Chip ports the nyancir exposes; must match the wrapper GDS ports.
_NYANCIR_PORTS = ("o1", "o2", "o3", "o4")

#: Checked-in ``.pic.yml`` samples covered by the YAML-path LVS tests.
RESONATOR_YML_SAMPLE = PROJECT_ROOT / "qpdk/samples/resonator_test_chip_yaml.pic.yml"
PIC_YML_SAMPLE = PROJECT_ROOT / "qpdk/samples/qubit_test_chip.pic.yml"

#: Launcher waveports the ``.pic.yml`` sample intentionally leaves unterminated
#: (its tees end on the launchers' ``o1`` pins); elvis still flags them as opens.
_EXPECTED_OPEN_DESCRIPTIONS = frozenset({
    "Open port: launcher_bot['waveport'] is not connected",
    "Open port: launcher_top['waveport'] is not connected",
})

#: Environment variable holding the gdsfactoryplus DRC API key.
GFP_API_KEY_ENV = "GFP_API_KEY"


def _portal_project_slug() -> str | None:
    """Read the GDSFactory+ portal project slug from ``pyproject.toml``.

    ``[tool.gdsfactoryplus.project].identifier`` is the cloud portal project
    that DRC submissions are filed under (orthogonal to
    ``[tool.gdsfactoryplus.drc] pdk``, which names the DRC process deck).
    qpdk leaves it unset; the GF+ extension writes it on first interactive
    DRC use.

    Returns:
        The configured slug, or ``None`` when not configured.
    """
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        data = tomllib.load(stream)
    slug = (
        data
        .get("tool", {})
        .get("gdsfactoryplus", {})
        .get("project", {})
        .get("identifier")
    )
    return slug or None


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
    """Build the wrapper layout the nyancir schematic describes.

    A single instance ``resonator_test_chip`` of ``resonator_test_chip_python``
    with all four probeline ports exposed on the wrapper, matching the
    ``nets`` of the schematic's chip instance and port markers.

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
    """Write the resonator ``.pic.yml`` sample's materialized layout GDS."""
    PDK.activate()  # not applied to module-scoped fixtures by the autouse hook
    gds_path = tmp_path_factory.mktemp("gfp_verify") / "resonator_test_chip_yaml.gds"
    _materialize_pic_yml(RESONATOR_YML_SAMPLE, gds_path.stem).write_gds(gds_path)
    return gds_path


def _nyancir_document() -> dict[str, Any]:
    """Build a ``.gsch`` nyancir mirroring the wrapper layout.

    One ``type: "ckt"`` instance of the sample factory named like the
    wrapper GDS instance (``resonator_test_chip``), with its four ports
    exposed through ``type: "port"`` markers.

    Returns:
        Nyancir document ready to be serialized to a ``.gsch`` file.
    """
    document: dict[str, Any] = {
        "chip:resonator_test_chip": {
            "type": "ckt",
            "model": _CHIP_QUALNAME,
            "name": "resonator_test_chip",
            "transform": [1, 0, 0, 1, 0, 0],
            "x": 0,
            "y": 0,
            "props": {},
            "nets": {port: f"net_{port}" for port in _NYANCIR_PORTS},
        }
    }
    for port in _NYANCIR_PORTS:
        document[f"chip:{port}"] = {
            "type": "port",
            "name": port,
            "x": 0,
            "y": 0,
            "nets": {"P": f"net_{port}"},
        }
    return document


def _materialize_pic_yml(yml_path: Path, name: str) -> gf.Component:
    """Materialize a ``.pic.yml`` sample as a component named like its GDS.

    Elvis reads the top cell by the GDS file stem, so the component name and
    the file name must agree; without ``name`` the component is ``Unnamed_0``.

    Returns:
        The materialized sample component.
    """
    return gf.read.from_yaml(str(yml_path), name=name)


@pytest.fixture(scope="module")
def pic_yml_chip_gds(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write the ``.pic.yml`` sample's materialized layout GDS to a temp directory."""
    PDK.activate()  # not applied to module-scoped fixtures by the autouse hook
    gds_path = tmp_path_factory.mktemp("gfp_pic_yml") / "qubit_test_chip_lvs.gds"
    _materialize_pic_yml(PIC_YML_SAMPLE, gds_path.stem).write_gds(gds_path)
    return gds_path


@pytest.fixture(scope="module")
def nyancir_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a self-contained project root holding a ``.gsch`` schematic.

    ``check_lvs`` resolves a ``.gsch`` schematic and its
    ``models.nyanlib`` through nyancad's ``FileAPI`` relative to
    ``project_root``; the hand-authored nyanlib entry mirrors what the
    ``gfp`` binary generates for the sample factory (only the component
    ``name`` is read for a Python-factory leaf).

    Returns:
        Path to the temporary project root.
    """
    root = tmp_path_factory.mktemp("gfp_nyancir")
    (root / "resonator_test_chip_gfp.gsch").write_text(
        json.dumps(_nyancir_document(), indent=2)
    )
    (root / "models.nyanlib").write_text(
        json.dumps(
            {
                f"models:{_CHIP_QUALNAME}": {
                    "name": "resonator_test_chip_python",
                    "type": "ckt",
                    "tags": ["qpdk"],
                }
            },
            indent=2,
        )
    )
    return root


@pytest.fixture(scope="module")
def nyancir_chip_gds(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write the nyancir-matching wrapper layout GDS to a temp directory."""
    PDK.activate()  # not applied to module-scoped fixtures by the autouse hook
    gds_path = tmp_path_factory.mktemp("gfp_verify") / "resonator_test_chip_gfp.gds"
    _wrapper_component("resonator_test_chip_gfp").write_gds(gds_path)
    return gds_path


def _leaf_category(item: ET.Element) -> str:
    """Return the leaf category path of a LYRDB item, e.g. ``LVS.open``."""
    return item.findtext("category") or ""


def _item_description(item: ET.Element) -> str:
    """Return the human-readable text values of a LYRDB item."""
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


def _describe_violations(root: ET.Element) -> str:
    """Format all LYRDB items for assertion failure messages."""
    return "\n".join(
        f"[{_leaf_category(item)}] {_item_description(item)}"
        for item in root.iter("item")
    )


def test_lvs_resonator_test_chip_matches_nyancir(
    gfp_check: ModuleType, nyancir_root: Path, nyancir_chip_gds: Path
) -> None:
    """Elvis LVS reports zero violations for the sample chip vs a ``.gsch``."""
    xml = gfp_check.check_lvs(
        str(nyancir_chip_gds),
        str(nyancir_root / "resonator_test_chip_gfp.gsch"),
        "qpdk",
        project_root=str(nyancir_root),
    )
    root = ET.fromstring(xml)  # ruff: ignore[suspicious-xml-element-tree-usage]

    assert root.tag == "report-database", xml[:500]
    violations = _describe_violations(root)
    assert not list(root.iter("item")), f"LVS violations:\n{violations}"


def test_lvs_nyancir_reports_a_broken_layout(
    gfp_check: ModuleType, nyancir_root: Path, tmp_path: Path
) -> None:
    """Negative control: a dropped o4 port must violate the ``.gsch`` LVS."""
    gds_path = tmp_path / "resonator_test_chip_gfp_broken.gds"
    _wrapper_component("resonator_test_chip_gfp_broken", with_o4=False).write_gds(
        gds_path
    )

    xml = gfp_check.check_lvs(
        str(gds_path),
        str(nyancir_root / "resonator_test_chip_gfp.gsch"),
        "qpdk",
        project_root=str(nyancir_root),
    )
    root = ET.fromstring(xml)  # ruff: ignore[suspicious-xml-element-tree-usage]
    items = list(root.iter("item"))
    descriptions = " | ".join(_item_description(item) for item in items)

    assert items, "LVS accepted a layout that lost the o4 port connection"
    assert "o4" in descriptions, f"Violations do not mention o4:\n{descriptions}"


def test_lvs_resonator_test_chip_matches_schematic(
    gfp_check: ModuleType, chip_yaml_gds: Path
) -> None:
    """Elvis LVS reports zero violations for the resonator chip vs its pic.yml."""
    xml = gfp_check.check_lvs(
        str(chip_yaml_gds),
        str(RESONATOR_YML_SAMPLE),
        "qpdk",
        project_root=str(PROJECT_ROOT),
    )
    root = ET.fromstring(xml)  # ruff: ignore[suspicious-xml-element-tree-usage]

    assert root.tag == "report-database", xml[:500]
    violations = _describe_violations(root)
    assert not list(root.iter("item")), f"LVS violations:\n{violations}"


def test_lvs_resonator_test_chip_yaml_reports_a_broken_layout(
    gfp_check: ModuleType, tmp_path: Path
) -> None:
    """Negative control: a dropped o4 port must violate the resonator LVS."""
    broken_yml = tmp_path / "resonator_test_chip_yaml_broken.pic.yml"
    broken_yml.write_text(
        RESONATOR_YML_SAMPLE.read_text().replace("  o4: probe_east_bot,waveport\n", "")
    )
    gds_path = tmp_path / "resonator_test_chip_yaml_broken.gds"
    _materialize_pic_yml(broken_yml, gds_path.stem).write_gds(gds_path)

    xml = gfp_check.check_lvs(
        str(gds_path),
        str(RESONATOR_YML_SAMPLE),
        "qpdk",
        project_root=str(PROJECT_ROOT),
    )
    root = ET.fromstring(xml)  # ruff: ignore[suspicious-xml-element-tree-usage]
    items = list(root.iter("item"))
    descriptions = " | ".join(_item_description(item) for item in items)

    assert items, "LVS accepted a layout that lost the o4 port connection"
    assert "o4" in descriptions, f"Violations do not mention o4:\n{descriptions}"


def test_lvs_pic_yml_sample_matches_layout(
    gfp_check: ModuleType, pic_yml_chip_gds: Path
) -> None:
    """Elvis LVS matches the ``.pic.yml`` sample against its materialized layout.

    The sample's tees end on the launchers' ``o1`` pins, so the
    ``launcher_bot``/``launcher_top`` waveports dangle by design and elvis
    flags them as opens (``_EXPECTED_OPEN_DESCRIPTIONS``); every other violation
    category means a real mismatch.
    """
    xml = gfp_check.check_lvs(
        str(pic_yml_chip_gds),
        str(PIC_YML_SAMPLE),
        "qpdk",
        project_root=str(PROJECT_ROOT),
    )
    root = ET.fromstring(xml)  # ruff: ignore[suspicious-xml-element-tree-usage]

    assert root.tag == "report-database", xml[:500]
    items = list(root.iter("item"))
    assert len(items) == 2, _describe_violations(root)
    assert {_leaf_category(item) for item in items} == {"LVS.open"}
    assert {_item_description(item) for item in items} == _EXPECTED_OPEN_DESCRIPTIONS


def test_lvs_pic_yml_reports_a_broken_layout(
    gfp_check: ModuleType, tmp_path: Path
) -> None:
    """Negative control: a dropped o2 port must violate the ``.pic.yml`` LVS."""
    broken_yml = tmp_path / "qubit_test_chip_broken.pic.yml"
    broken_yml.write_text(
        PIC_YML_SAMPLE.read_text().replace("  o2: launcher_out,waveport\n", "")
    )
    gds_path = tmp_path / "qubit_test_chip_broken.gds"
    _materialize_pic_yml(broken_yml, gds_path.stem).write_gds(gds_path)

    xml = gfp_check.check_lvs(
        str(gds_path),
        str(PIC_YML_SAMPLE),
        "qpdk",
        project_root=str(PROJECT_ROOT),
    )
    root = ET.fromstring(xml)  # ruff: ignore[suspicious-xml-element-tree-usage]
    items = list(root.iter("item"))
    descriptions = " | ".join(_item_description(item) for item in items)

    assert items, "LVS accepted a layout that lost the o2 port connection"
    assert "o2" in descriptions, f"Violations do not mention o2:\n{descriptions}"


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
    project_slug = _portal_project_slug()
    if project_slug is None:
        pytest.skip(
            "[tool.gdsfactoryplus.project].identifier not set in pyproject.toml "
            "— no portal project configured for DRC submissions",
        )
    gds_path = tmp_path / "double_pad_transmon.gds"
    double_pad_transmon().write_gds(gds_path)

    submission = gfp_check.check_drc(
        str(gds_path),
        project_slug=project_slug,
        api_key=os.environ[GFP_API_KEY_ENV],
    )

    assert submission["sub_id"], f"Missing sub_id in submission: {submission}"
    assert submission["status"], f"Missing status in submission: {submission}"
