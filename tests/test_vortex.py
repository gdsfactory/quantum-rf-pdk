"""Tests for the vortex-trapping cells in :mod:`qpdk.cells.vortex`."""

import gdsfactory as gf
import pytest

from qpdk.cells.vortex import (
    cpw_bulge,
    neck_termination,
    resonator_half_wave_bulge,
    resonator_quarter_wave_neck,
    vortex_trapping_threshold,
)
from qpdk.tech import LAYER

_FLUX_QUANTUM = 2.067833848e-15


def _region(component: gf.Component, layer: tuple[int, int]) -> gf.kdb.Region:
    return gf.kdb.Region(component.begin_shapes_rec(gf.get_layer(layer))).merged()


class TestTrappingThreshold:
    """Tests for :func:`vortex_trapping_threshold`."""

    @staticmethod
    def test_kuit_value() -> None:
        """A 2 µm strip traps above about 0.85 mT for the Kuit prefactor."""
        assert vortex_trapping_threshold(2) == pytest.approx(
            1.65 * _FLUX_QUANTUM / 4e-12
        )
        assert vortex_trapping_threshold(2) == pytest.approx(0.853e-3, rel=1e-3)

    @staticmethod
    def test_scales_as_inverse_width_squared() -> None:
        """Doubling the width quarters the threshold."""
        assert vortex_trapping_threshold(4) == pytest.approx(
            vortex_trapping_threshold(2) / 4
        )

    @staticmethod
    @pytest.mark.parametrize(("width", "prefactor"), [(0, 1.65), (-1, 1.65), (2, 0)])
    def test_invalid_arguments_raise(width: float, prefactor: float) -> None:
        """Non-positive arguments are rejected."""
        with pytest.raises(ValueError, match="must be positive"):
            vortex_trapping_threshold(width, prefactor)


class TestNeckTermination:
    """Tests for :func:`neck_termination`."""

    @staticmethod
    def test_ports_and_info() -> None:
        """One port at the origin facing the resonator, info records geometry."""
        c = neck_termination()
        assert [p.name for p in c.ports] == ["o1"]
        assert c.ports["o1"].orientation == 180
        assert tuple(c.ports["o1"].center) == (0, 0)
        assert c.info["neck_width"] == 2
        assert c.info["neck_length"] == 325
        assert c.info["window_length"] == pytest.approx(425)
        assert c.info["neck_trapping_threshold"] == pytest.approx(
            vortex_trapping_threshold(2)
        )

    @staticmethod
    def test_neck_shorts_to_ground() -> None:
        """The conductor reaches the far edge of the etched window."""
        c = neck_termination(trap_hole_rows=0)
        etch = _region(c, LAYER.M1_ETCH)
        draw = _region(c, LAYER.M1_DRAW)
        assert draw.bbox().right == etch.bbox().right

    @staticmethod
    def test_neck_is_narrowest_conductor() -> None:
        """The conductor at the far end of the window has the neck width."""
        c = neck_termination(neck_width=3, trap_hole_rows=0)
        draw = _region(c, LAYER.M1_DRAW)
        x = round((c.info["window_length"] - 1) * 1000)
        probe = draw & gf.kdb.Region(gf.kdb.Box(x, -100_000, x + 10, 100_000))
        assert probe.bbox().height() == pytest.approx(3000, abs=2)

    @staticmethod
    def test_hole_band_web_narrower_than_neck() -> None:
        """Default hole web is narrower than the default neck width."""
        c = neck_termination()
        holes = _region(c, LAYER.M1_ETCH) - _region(
            neck_termination(trap_hole_rows=0), LAYER.M1_ETCH
        )
        assert holes.count() > 0
        assert c.info["neck_width"] > 5.5 - 4.0

    @staticmethod
    def test_smooth_taper_differs_from_linear() -> None:
        """The raised-cosine profile changes the conductor outline."""
        linear = _region(neck_termination(trap_hole_rows=0), LAYER.M1_DRAW)
        smooth = _region(
            neck_termination(taper_shape="smooth", trap_hole_rows=0), LAYER.M1_DRAW
        )
        assert not (linear ^ smooth).is_empty()

    @staticmethod
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"neck_width": 0}, "neck_width must be positive"),
            ({"neck_length": -1}, "neck_length must be positive"),
            ({"taper_length": 0}, "taper_length must be positive"),
            ({"neck_width": 20}, "must not exceed the CPW width"),
            ({"window_width": 10}, "at least the CPW footprint"),
            ({"taper_shape": "cubic"}, "taper_shape"),
            ({"trap_hole_rows": -1}, "trap_hole_rows"),
            ({"trap_hole_size": 6.0}, "trap_hole_size"),
        ],
    )
    def test_invalid_arguments_raise(kwargs: dict, match: str) -> None:
        """Out-of-range dimensions are rejected."""
        with pytest.raises(ValueError, match=match):
            neck_termination(**kwargs)


class TestCpwBulge:
    """Tests for :func:`cpw_bulge`."""

    @staticmethod
    def test_ports_and_flux_step() -> None:
        """Two ports, and the flux step is one flux quantum over the bulge area."""
        c = cpw_bulge(bulge_width=8 + 10, bulge_length=50)
        assert {p.name for p in c.ports} == {"o1", "o2"}
        assert c.info["bulge_flux_step"] == pytest.approx(
            _FLUX_QUANTUM / (18 * 50 * 1e-12)
        )

    @staticmethod
    def test_paper_flux_step() -> None:
        """An 8 µm x 50 µm bulge gives the ~5 µT step of Nsanzineza and Plourde."""
        assert pytest.approx(5.17e-6, rel=1e-2) == _FLUX_QUANTUM / (8 * 50e-12)

    @staticmethod
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"bulge_width": 10}, "must exceed the CPW width"),
            ({"bulge_length": 0}, "bulge_length must be positive"),
            ({"taper_length": -5}, "taper_length must be positive"),
        ],
    )
    def test_invalid_arguments_raise(kwargs: dict, match: str) -> None:
        """Out-of-range dimensions are rejected."""
        with pytest.raises(ValueError, match=match):
            cpw_bulge(**kwargs)


class TestResonatorQuarterWaveNeck:
    """Tests for :func:`resonator_quarter_wave_neck`."""

    @staticmethod
    def test_builds_with_one_port_and_neck_info() -> None:
        """The resonator exposes the coupling port and the neck parameters."""
        c = resonator_quarter_wave_neck()
        assert [p.name for p in c.ports] == ["o1"]
        assert c.info["neck_width"] == 2
        assert "neck_trapping_threshold" in c.info

    @staticmethod
    def test_window_clear_of_meander() -> None:
        """The etched window does not overlap the meander's conductor."""
        plain = resonator_quarter_wave_neck(trap_hole_rows=0)
        neck = neck_termination(trap_hole_rows=0)
        draw = _region(plain, LAYER.M1_DRAW)
        # Conductor inside the window can only be the taper and neck: its
        # area must equal the neck termination's own conductor area.
        window = gf.kdb.Region()
        for inst in plain.insts:
            if inst.cell.name.startswith("neck_termination"):
                window = gf.kdb.Region(inst.dbbox().to_itype(plain.kcl.dbu))
        assert not window.is_empty()
        inside = draw & window
        assert inside.area() == pytest.approx(
            _region(neck, LAYER.M1_DRAW).area(), rel=1e-3
        )


class TestResonatorHalfWaveBulge:
    """Tests for :func:`resonator_half_wave_bulge`."""

    @staticmethod
    def test_builds_with_two_ports() -> None:
        """The half-wave resonator has two open-end ports and the bulge info."""
        c = resonator_half_wave_bulge()
        assert {p.name for p in c.ports} == {"o1", "o2"}
        assert "bulge_flux_step" in c.info

    @staticmethod
    def test_odd_meanders_raise() -> None:
        """The bulge can only be centred with an even number of meanders."""
        with pytest.raises(ValueError, match="meanders must be even"):
            resonator_half_wave_bulge(meanders=5)

    @staticmethod
    def test_too_short_raises() -> None:
        """A length too short for the meanders and bulge is rejected."""
        with pytest.raises(ValueError, match="too short"):
            resonator_half_wave_bulge(length=500)
