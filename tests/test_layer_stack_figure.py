"""Keep docs/figures/layer-stack.yaml, the data behind the layer-stack figure, in step with qpdk.tech."""

from pathlib import Path

import pytest
import yaml
from gdsfactory.technology import LayerLevel, LogicalLayer
from gdsfactory.technology.layer_views import LayerViews

from qpdk import tech
from qpdk.tech import LAYER

_DOCS = Path(__file__).parents[1] / "docs" / "figures" / "layer-stack.yaml"
_LAYER_VIEWS = Path(tech.__file__).parent / "layers.yaml"
_STACKS: dict[str, dict] = yaml.safe_load(_DOCS.read_text())["stacks"]
_LOOKS = {"substrate", "vacuum", "oxide"}


def _figure_levels(name: str) -> dict[str, dict]:
    """Every level the figure attributes to stack ``name``, inherited ones first."""
    entry = _STACKS[name]
    levels: dict[str, dict] = {}
    for parent in entry.get("extends", []):
        levels |= _figure_levels(parent)
    return levels | entry.get("reuses", {}) | entry["levels"]


@pytest.mark.parametrize("name", _STACKS)
def test_figure_lists_every_level(name: str) -> None:
    """The figure covers each stack exactly, through its own and inherited levels."""
    stack = getattr(tech, name)
    assert set(_figure_levels(name)) == set(stack.layers)


@pytest.mark.parametrize(
    ("name", "level"),
    [(name, level) for name in _STACKS for level in _figure_levels(name)],
)
def test_figure_level_matches_tech(name: str, level: str) -> None:
    """Material, thickness, z and drawn layer match the stack in qpdk.tech."""
    want = _figure_levels(name)[level]
    got: LayerLevel = getattr(tech, name).layers[level]
    assert got.material == want["material"]
    assert got.thickness == pytest.approx(want["thickness"])
    assert got.zmin == pytest.approx(want["zmin"])
    if want.get("derived", False):
        assert not isinstance(got.layer, LogicalLayer)
    else:
        assert isinstance(got.layer, LogicalLayer)
        assert got.layer.layer == getattr(LAYER, want["layer"])


def test_reused_levels_point_at_listed_levels() -> None:
    """A reused level borrows the legend number of a level listed elsewhere."""
    listed = {level for entry in _STACKS.values() for level in entry["levels"]}
    for entry in _STACKS.values():
        assert set(entry.get("reuses", {})) <= listed


def test_listed_level_names_are_unique() -> None:
    """Callouts find their legend number by level name."""
    names = [level for entry in _STACKS.values() for level in entry["levels"]]
    assert len(names) == len(set(names))


def test_figure_colours_exist_in_layer_views() -> None:
    """Every layer and colour the figure names is a qpdk/layers.yaml entry."""
    views = LayerViews.from_yaml(_LAYER_VIEWS).get_layer_views()
    for entry in _STACKS.values():
        for level in (entry["levels"] | entry.get("reuses", {})).values():
            assert level["layer"] in views
            color = level.get("color", level["layer"])
            assert color in _LOOKS or color in views
