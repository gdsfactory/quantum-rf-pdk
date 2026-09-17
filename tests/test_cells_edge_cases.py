"""Tests for cell validation edge cases — resonator, waveguides, snspd, unimon."""

from functools import partial

import gdsfactory as gf
import pytest
from kfactory import kdb

from qpdk.cells.resonator import resonator
from qpdk.cells.snspd import snspd
from qpdk.cells.unimon import unimon, unimon_arm, unimon_coupled
from qpdk.cells.waveguides import bend_circular, tee
from qpdk.logger import logger
from qpdk.tech import LAYER


class TestResonatorValidation:
    """Tests for resonator validation branches."""

    @staticmethod
    def test_negative_straights_raises() -> None:
        """Test that start_with_bend + end_with_bend + 0 meanders raises."""
        with pytest.raises(ValueError, match="fewer than 0 straight sections"):
            resonator(
                length=4000,
                meanders=0,
                start_with_bend=True,
                end_with_bend=True,
            )

    @staticmethod
    def test_too_short_for_meanders_raises() -> None:
        """Test that a very short length with many meanders raises ValueError."""
        with pytest.raises(ValueError, match="too short"):
            resonator(length=10, meanders=6)

    @staticmethod
    def test_zero_meanders_succeeds() -> None:
        """Test that 0 meanders creates a straight resonator."""
        c = resonator(length=1000, meanders=0)
        assert c is not None

    @staticmethod
    def test_start_with_bend() -> None:
        """Test resonator starting with a bend."""
        c = resonator(length=4000, meanders=4, start_with_bend=True)
        assert c is not None

    @staticmethod
    def test_end_with_bend() -> None:
        """Test resonator ending with a bend."""
        c = resonator(length=4000, meanders=4, end_with_bend=True)
        assert c is not None

    @staticmethod
    def test_both_bends() -> None:
        """Test resonator with both start and end bends."""
        c = resonator(length=4000, meanders=4, start_with_bend=True, end_with_bend=True)
        assert c is not None

    @staticmethod
    def test_open_end() -> None:
        """Test resonator with open end (half-wave)."""
        c = resonator(length=4000, meanders=4, open_end=True)
        assert c is not None

    @staticmethod
    def test_closed_start() -> None:
        """Test resonator with closed start."""
        c = resonator(length=4000, meanders=4, open_start=False)
        assert c is not None


class TestTee:
    """Tests for tee junction consistency with the attached straights."""

    @staticmethod
    @pytest.mark.parametrize(
        "cross_section_name",
        ["cpw", "meander_inductor_cross_section", "superinductor_cross_section"],
    )
    def test_junction_widths_match_cross_section(cross_section_name: str) -> None:
        """Test that junction ports match the cross-section width and layer.

        The junction must not be built with the default CPW width (10 µm) or
        layer (M1_DRAW) when ``tee`` is given a different cross-section.
        """
        cross_section = gf.get_cross_section(cross_section_name)
        c = tee(cross_section=cross_section_name)
        junction = next(inst for inst in c.insts if inst.cell.name.startswith("nxn"))
        for port in junction.ports:
            assert port.width == pytest.approx(cross_section.width)
            assert port.layer == gf.get_layer(cross_section.layer)


class TestBendCircularEdgeCases:
    """Tests for bend_circular radius correction."""

    @staticmethod
    def test_very_small_radius_gets_corrected() -> None:
        """Test that a radius smaller than min is corrected."""
        # Use a very small radius — the function should correct it
        c = bend_circular(radius=0.1)
        assert c is not None

    @staticmethod
    def test_normal_radius_succeeds() -> None:
        """Test that a normal radius works without issues."""
        c = bend_circular(radius=100.0)
        assert c is not None

    @staticmethod
    def test_various_angles() -> None:
        """Test bend with different angles."""
        for angle in [45.0, 90.0, 180.0]:
            c = bend_circular(angle=angle, radius=100.0)
            assert c is not None


class TestSNSPDEdgeCases:
    """Tests for snspd port_type handling and small-size validation."""

    @staticmethod
    def test_default_port_type_electrical() -> None:
        """Test that the default port_type produces electrical ports."""
        c = snspd()
        assert {p.name: str(p.port_type) for p in c.ports} == {
            "e1": "electrical",
            "e2": "electrical",
        }

    @staticmethod
    def test_port_type_optical() -> None:
        """Test that port_type="optical" produces optical ports."""
        c = snspd(port_type="optical")
        assert {p.name: str(p.port_type) for p in c.ports} == {
            "o1": "optical",
            "o2": "optical",
        }

    @staticmethod
    def test_port_type_optical_netlist_round_trip() -> None:
        """Test that the optical-port snspd survives a component->netlist->component round trip.

        The flat snspd leaf drops its ports in `get_netlist()`, so wrap it in a
        hierarchical component whose netlist records the ports and the
        `port_type` setting of the instance.
        """
        c = gf.Component()
        sref = c << snspd(port_type="optical")
        c.add_port(name="o1", port=sref.ports["o1"])
        c.add_port(name="o2", port=sref.ports["o2"])
        netlist = c.get_netlist()

        yaml_str = c.write_netlist(netlist)
        c2 = gf.read.from_yaml(yaml_str)
        assert {p.name: str(p.port_type) for p in c2.ports} == {
            "o1": "optical",
            "o2": "optical",
        }
        instances = c2.get_netlist()["instances"]
        settings = instances[next(iter(instances))]["settings"]
        assert settings["port_type"] == "optical"

    @staticmethod
    def test_valid_snspd_builds() -> None:
        """Test that a representative valid snspd builds connected geometry."""
        c = snspd(size=(10, 8), wire_pitch=0.6)
        assert c is not None
        assert c.info["xsize"] > 0
        # The original defect class was a silently unconnected pad: the wire
        # on the NbTiN layer must merge into a single connected region.
        layer_index = gf.get_layer(LAYER.NbTiN)
        region = gf.kdb.Region(c.begin_shapes_rec(layer_index)).merged()
        assert len(region) == 1

    @staticmethod
    def test_num_squares_overrides_size() -> None:
        """Test that num_squares builds a square SNSPD sized from the squares."""
        c = snspd(num_squares=1000)
        assert c is not None
        assert c.info["xsize"] == c.info["ysize"]
        # The meander count is quantized to whole wire pitches, so the achieved
        # square count is close to but not exactly the requested one.
        assert abs(c.info["num_squares"] - 1000) / 1000 < 0.05

    @staticmethod
    @pytest.mark.parametrize(
        ("ysize", "terminals_same_side", "num_squares_expected"),
        [(1.3, False, 150.0), (2.0, True, 200.0)],
    )
    def test_minimal_meanders_connected(
        ysize: float, terminals_same_side: bool, num_squares_expected: float
    ) -> None:
        """Test that the smallest accepted sizes build one connected wire region.

        These sit on the boundary the validation moved: 3 meanders
        (`num_meanders == 3`) and 4 with same-side terminals, the shapes
        closest to the old silently-unconnected output.
        """
        c = snspd(
            size=(10, ysize), wire_pitch=0.6, terminals_same_side=terminals_same_side
        )
        assert c.info["num_squares"] == num_squares_expected
        layer_index = gf.get_layer(LAYER.NbTiN)
        region = gf.kdb.Region(c.begin_shapes_rec(layer_index)).merged()
        assert len(region) == 1

    @staticmethod
    def test_nonpositive_num_squares_raises() -> None:
        """Test that num_squares <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="num_squares"):
            snspd(num_squares=0)

    @staticmethod
    def test_nonpositive_size_raises() -> None:
        """Test that nonpositive size dimensions raise ValueError."""
        with pytest.raises(ValueError, match="size"):
            snspd(size=(0, 8))

    @staticmethod
    def test_wire_pitch_le_wire_width_raises() -> None:
        """Test that wire_pitch <= wire_width raises ValueError."""
        with pytest.raises(ValueError, match="wire_pitch"):
            snspd(wire_width=0.6, wire_pitch=0.2)

    @staticmethod
    def test_too_small_size_raises() -> None:
        """Test that a size fitting fewer than 3 meanders raises ValueError."""
        with pytest.raises(ValueError, match="num_meanders"):
            snspd(size=(10, 0.5), wire_pitch=0.6)

    @staticmethod
    def test_too_small_size_same_side_raises() -> None:
        """Test that same-side terminals with fewer than 3 meanders raise."""
        with pytest.raises(ValueError, match="num_meanders"):
            snspd(size=(10, 1), wire_pitch=0.6, terminals_same_side=True)


class TestUnimonMeanderRadius:
    """Tests that unimon metadata records the actual bend radius."""

    @staticmethod
    def test_default_radius_matches_bend_default() -> None:
        """Test that the default meander radius equals the default bend radius."""
        assert unimon_arm().info["radius"] == pytest.approx(100.0)
        c = unimon()
        assert c.info["meander_radius"] == pytest.approx(100.0)

    @staticmethod
    def test_non_default_radius_recorded() -> None:
        """Test that a non-default bend radius is recorded, not the 100 µm fallback."""
        radius = 50.0
        c = unimon(bend_spec=partial(bend_circular, radius=radius))
        assert c.info["meander_radius"] == radius

    @staticmethod
    def test_coupled_follows_meander_radius() -> None:
        """Test that unimon_coupled computes coupling_radius from the true meander radius.

        coupling_radius should be
        ``meander_radius + coupling_gap + width_resonator/2 + width_coupler/2``.
        """
        radius = 50.0
        coupling_gap = 30.0
        c = unimon_coupled(
            bend_spec=partial(bend_circular, radius=radius), coupling_gap=coupling_gap
        )
        assert c.info["meander_radius"] == radius
        xs = gf.get_cross_section("cpw")
        assert c.info["coupling_radius"] == radius + coupling_gap + xs.width

    @staticmethod
    def test_coupled_drawn_at_computed_radius() -> None:
        """Test that the drawn coupler bend uses the computed coupling_radius."""
        c = unimon_coupled()
        coupler_inst = next(i for i in c.insts if "half_circle_coupler" in i.cell.name)
        bend_inst = next(i for i in coupler_inst.cell.insts if "bend" in i.cell.name)
        assert bend_inst.cell.info["radius"] == c.info["coupling_radius"]

    @staticmethod
    def test_coupled_keeps_coupling_gap() -> None:
        """Test that the coupler keeps an edge-to-edge coupling_gap M1_DRAW spacing.

        The spacing is measured between the coupler M1_DRAW metal and the
        unimon meander M1_DRAW metal.
        """
        c = unimon_coupled()
        unimon_inst = next(i for i in c.insts if "unimon" in i.cell.name)
        coupler_inst = next(i for i in c.insts if "half_circle_coupler" in i.cell.name)
        reg_u = _m1_draw_region(unimon_inst)
        reg_c = _m1_draw_region(coupler_inst)
        assert (reg_u & reg_c).is_empty()
        gap = _min_spacing(reg_u, reg_c)
        assert gap == pytest.approx(30.0, abs=0.5)

    @staticmethod
    def test_coupled_small_radius_warns() -> None:
        """Test that a bend radius bringing the coupler onto a meander straight warns."""
        messages = []
        handler_id = logger.add(messages.append, level="WARNING")
        try:
            c = unimon_coupled(bend_spec=partial(bend_circular, radius=22.0))
        finally:
            logger.remove(handler_id)
        assert any("overlap the resonator" in str(m) for m in messages)
        assert c.info["coupling_radius"] == pytest.approx(62.0)

        messages.clear()
        handler_id = logger.add(messages.append, level="WARNING")
        try:
            unimon_coupled()
        finally:
            logger.remove(handler_id)
        assert not any("overlap the resonator" in str(m) for m in messages)


def _m1_draw_region(inst) -> kdb.Region:
    """Return the M1_DRAW polygons of an instance transformed to top-level coordinates."""
    reg = kdb.Region()
    layer_index = inst.cell.kdb_cell.layout().layer(
        kdb.LayerInfo(LAYER.M1_DRAW.layer, LAYER.M1_DRAW.datatype)
    )
    it = inst.cell.kdb_cell.begin_shapes_rec(layer_index)
    while not it.at_end():
        reg.insert(it.shape().polygon.transformed(it.trans()))
        it.next()
    return reg.transformed(inst.trans)


def _min_spacing(reg_a: kdb.Region, reg_b: kdb.Region, hi_um: float = 200.0) -> float:
    """Return the edge-to-edge spacing in µm between two disjoint regions.

    Binary-searches the sizing amount at which the grown regions first touch.
    """
    lo, hi = 0.0, hi_um
    for _ in range(40):
        mid = (lo + hi) / 2
        if (reg_a & reg_b.sized(round(mid * 1000))).is_empty():
            lo = mid
        else:
            hi = mid
    return lo
