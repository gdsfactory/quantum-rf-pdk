"""Tests for the VACASK subprocess wrapper and raw-file reader."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from qpdk.simulation.vacask import (
    RawPlot,
    VacaskError,
    find_vacask,
    read_raw,
    run_vacask,
    vacask_model_path,
)

DATA = Path(__file__).parent / "data" / "vacask"

try:
    find_vacask()
    HAS_VACASK = True
except VacaskError:
    HAS_VACASK = False

needs_vacask = pytest.mark.skipif(not HAS_VACASK, reason="VACASK binary not found")

JJ_FILES = {"josephson_junction.va": vacask_model_path("josephson_junction")}


def test_read_raw_real() -> None:
    (plot,) = read_raw(DATA / "tran1.raw")
    assert plot.plotname == "Transient Analysis"
    assert plot.names == ("time", "1", "j1:ph")
    assert plot.data.dtype == np.float64
    assert plot.data.shape == (180, 3)
    time = plot["time"]
    assert time[0] == 0
    assert np.all(np.diff(time) > 0)
    assert time[-1] == pytest.approx(20e-12)


def test_read_raw_complex() -> None:
    (plot,) = read_raw(DATA / "ac1.raw")
    assert plot.data.dtype == np.complex128
    np.testing.assert_allclose(plot["frequency"].real, np.linspace(5e9, 7e9, 5))
    np.testing.assert_array_equal(plot["frequency"].imag, 0)
    assert np.all(np.abs(plot["1"]) > 0)


@pytest.mark.parametrize("name", ["tran1", "ac1", "acsw"])
def test_read_raw_ascii_matches_binary(name: str) -> None:
    (binary,) = read_raw(DATA / f"{name}.raw")
    (ascii_,) = read_raw(DATA / f"{name}a.raw")
    assert ascii_.names == binary.names
    np.testing.assert_allclose(ascii_.data, binary.data, rtol=1e-14)


def test_read_raw_sweep_split() -> None:
    (plot,) = read_raw(DATA / "acsw.raw")
    assert plot.names[0] == "bias"
    groups = plot.split("bias")
    assert [g["bias"][0].real for g in groups] == [0, 0.5e-6]
    for group in groups:
        assert isinstance(group, RawPlot)
        np.testing.assert_allclose(group["frequency"].real, [5e9, 6e9, 7e9])
    # The bias point of ``acsw`` repeats the single analysis ``ac1``
    (single,) = read_raw(DATA / "ac1.raw")
    np.testing.assert_allclose(groups[1]["1"], single["1"][[0, 2, 4]], rtol=1e-12)


def test_read_raw_multiple_plots() -> None:
    plots = read_raw(DATA / "multi.raw")
    assert [p.names[0] for p in plots] == ["frequency", "bias"]
    np.testing.assert_array_equal(plots[0].data, read_raw(DATA / "ac1.raw")[0].data)
    np.testing.assert_array_equal(plots[1].data, read_raw(DATA / "acsw.raw")[0].data)


def test_read_raw_truncated(tmp_path: Path) -> None:
    raw = (DATA / "ac1.raw").read_bytes()
    (tmp_path / "bad.raw").write_bytes(raw[:-8])
    with pytest.raises(VacaskError, match="truncated"):
        read_raw(tmp_path / "bad.raw")


def test_plot_missing_variable() -> None:
    (plot,) = read_raw(DATA / "ac1.raw")
    assert "1" in plot
    with pytest.raises(KeyError, match="nope"):
        plot["nope"]


def test_find_vacask_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VACASK", str(tmp_path / "vacask"))
    assert find_vacask() == tmp_path / "vacask"


def test_find_vacask_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VACASK", raising=False)
    monkeypatch.delenv("QPDK_VACASK", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(VacaskError, match=r"codeberg\.org"):
        find_vacask()


def test_netlist_needs_control_block() -> None:
    with pytest.raises(VacaskError, match="control"):
        run_vacask("Title\nr1 (1 0) r r=1\n")


def test_models_ship_as_package_data() -> None:
    for name in ("josephson_junction", "squid"):
        assert "module" in vacask_model_path(name).read_text()


B1_NETLIST = """JJ LC resonance vs DC bias
load "josephson_junction.va"
load "capacitor.osdi"
model isource isource
model capacitor capacitor
model jj jj
iin (0 1) isource dc=0 mag=1n
j1 (1 0) jj ic=1u r=1e6
c1 (1 0) capacitor c=2p
control
  options tolscale=1e-6 reltol=1e-6
  sweep bias instance="iin" parameter="dc" values=[0, 0.3u, 0.5u, 0.8u]
  analysis ac1 ac from=1G to=10G mode="lin" points=90000
endc
"""


@needs_vacask
def test_jj_resonance_vs_bias() -> None:
    """Junction plus capacitor resonates at 1/(2 pi sqrt(L_J C)), L_J = Phi_0/(2 pi Ic cos phi)."""
    plot = run_vacask(B1_NETLIST, JJ_FILES)["ac1"]
    peaks = [
        group["frequency"].real[np.argmax(np.abs(group["1"]))]
        for group in plot.split("bias")
    ]
    np.testing.assert_allclose(
        peaks, [6.20350e9, 6.05900e9, 5.77300e9, 4.80520e9], rtol=1e-5
    )


KERR_JPA_NETLIST = """Kerr JPA
load "josephson_junction.va"
load "capacitor.osdi"
load "resistor.osdi"
model isource isource
model capacitor capacitor
model resistor resistor
model jj jj
ipump (0 in) isource type="sine" sinedc=0 ampl=40n freq=5.84G spur={[1]} smag=[1n]
rp (in 0) resistor r=50
cc (in 1) capacitor c=0.185p
j1 (1 0) jj ic=1u
c1 (1 0) capacitor c=2p
control
  options tolscale=1e-6 reltol=1e-6
  analysis hbac1 hbac freq=[5.84G] nharm=7 outspur={[1],[-1]} from=0.5M to=5M mode="lin" points=9
endc
"""


@needs_vacask
def test_kerr_jpa_gain() -> None:
    plot = run_vacask(KERR_JPA_NETLIST, JJ_FILES)["hbac1"]
    source_current, resistance = 1e-9, 50
    gain = 20 * np.log10(np.abs(2 * plot["in;1"] / (source_current * resistance) - 1))
    idler = 20 * np.log10(np.abs(2 * plot["in;-1"] / (source_current * resistance)))
    assert gain[0] == pytest.approx(11.81, abs=0.1)
    assert idler[0] == pytest.approx(11.51, abs=0.1)
    assert gain[-1] == pytest.approx(11.50, abs=0.1)
    # Manley-Rowe: |S_ss|^2 - |S_is|^2 = 1 for a lossless degenerate amplifier
    assert 10 ** (gain[0] / 10) - 10 ** (idler[0] / 10) == pytest.approx(1, abs=0.01)


@needs_vacask
def test_singular_dc_loop_raises() -> None:
    """An inductor in parallel with a junction has no unique DC phase."""
    netlist = """Inductor loop
load "josephson_junction.va"
load "capacitor.osdi"
load "inductor.osdi"
model isource isource
model capacitor capacitor
model inductor inductor
model jj jj
iin (0 1) isource dc=0.5u
l1 (1 0) inductor l=100p
j1 (1 0) jj ic=1u
c1 (1 0) capacitor c=1p
control
  analysis op1 op
endc
"""
    with pytest.raises(VacaskError, match="zero pivot"):
        run_vacask(netlist, JJ_FILES)
