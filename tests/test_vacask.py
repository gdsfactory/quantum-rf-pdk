"""Tests for the VACASK netlist builder, runner, raw-file reader and rational fit.

Tests that run the VACASK binary carry the ``vacask`` marker and are deselected
by default; run them with ``just test-vacask`` (or ``pytest -m vacask``).
"""

import shutil
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from qpdk import PDK
from qpdk.models.waveguides import straight
from qpdk.simulation.vacask import (
    VACASK_MODELS_DIR,
    Analysis,
    Instance,
    Netlist,
    RationalModel,
    RawFile,
    RawPlot,
    Sweep,
    Vacask,
    VacaskError,
    cpw_tline_params,
    format_value,
)

DATA = Path(__file__).parent / "data" / "vacask"
PHI_0 = 2.067833848e-15  # Wb

# --------------------------------------------------------------------------- #
# Raw files
# --------------------------------------------------------------------------- #


def test_raw_real() -> None:
    raw = RawFile(DATA / "tran1.raw")
    assert len(raw) == 1
    plot = raw[0]
    assert plot.plotname == "Transient Analysis"
    assert plot.names == ("time", "1", "j1:ph")
    assert plot.data.dtype == np.float64
    assert plot.data.shape == (180, 3)
    time = plot["time"]
    assert time[0] == 0
    assert np.all(np.diff(time) > 0)
    assert time[-1] == pytest.approx(20e-12)


def test_raw_complex() -> None:
    plot = RawFile(DATA / "ac1.raw")[0]
    assert plot.data.dtype == np.complex128
    np.testing.assert_allclose(plot["frequency"].real, np.linspace(5e9, 7e9, 5))
    np.testing.assert_array_equal(plot["frequency"].imag, 0)
    assert np.all(np.abs(plot["1"]) > 0)


@pytest.mark.parametrize("name", ["tran1", "ac1", "acsw"])
def test_raw_ascii_matches_binary(name: str) -> None:
    binary = RawFile(DATA / f"{name}.raw")[0]
    ascii_ = RawFile(DATA / f"{name}a.raw")[0]
    assert ascii_.names == binary.names
    np.testing.assert_allclose(ascii_.data, binary.data, rtol=1e-14)


def test_raw_sweep_split() -> None:
    plot = RawFile(DATA / "acsw.raw")[0]
    assert plot.names[0] == "bias"
    groups = plot.split("bias")
    assert [g["bias"][0].real for g in groups] == [0, 0.5e-6]
    for group in groups:
        assert isinstance(group, RawPlot)
        np.testing.assert_allclose(group["frequency"].real, [5e9, 6e9, 7e9])
    # The bias point of ``acsw`` repeats the single analysis ``ac1``
    single = RawFile(DATA / "ac1.raw")[0]
    np.testing.assert_allclose(groups[1]["1"], single["1"][[0, 2, 4]], rtol=1e-12)


def test_raw_multiple_plots() -> None:
    raw = RawFile(DATA / "multi.raw")
    assert [p.names[0] for p in raw.plots] == ["frequency", "bias"]
    np.testing.assert_array_equal(raw[0].data, RawFile(DATA / "ac1.raw")[0].data)
    np.testing.assert_array_equal(raw[1].data, RawFile(DATA / "acsw.raw")[0].data)


def test_raw_binary_truncated(tmp_path: Path) -> None:
    raw = (DATA / "ac1.raw").read_bytes()
    (tmp_path / "bad.raw").write_bytes(raw[:-8])
    with pytest.raises(VacaskError, match="truncated"):
        RawFile(tmp_path / "bad.raw")


def test_raw_ascii_truncated(tmp_path: Path) -> None:
    text = (DATA / "tran1a.raw").read_text()
    (tmp_path / "bad.raw").write_text(text[: len(text) // 2])
    with pytest.raises(VacaskError, match="truncated"):
        RawFile(tmp_path / "bad.raw")


def test_raw_ascii_malformed(tmp_path: Path) -> None:
    lines = (DATA / "tran1a.raw").read_text().splitlines()
    index = lines.index("Values:") + 2
    lines[index] = "\tnot-a-number"
    (tmp_path / "bad.raw").write_text("\n".join(lines) + "\n")
    with pytest.raises(VacaskError, match="malformed"):
        RawFile(tmp_path / "bad.raw")


def test_raw_bad_header(tmp_path: Path) -> None:
    text = (DATA / "tran1a.raw").read_text()
    (tmp_path / "bad.raw").write_text(text.replace("No. Points:", "No. Pts:", 1))
    with pytest.raises(VacaskError, match="header"):
        RawFile(tmp_path / "bad.raw")


def test_plot_missing_variable() -> None:
    plot = RawFile(DATA / "ac1.raw")[0]
    assert "1" in plot
    with pytest.raises(KeyError, match="nope"):
        plot["nope"]


# --------------------------------------------------------------------------- #
# Netlists
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("value", "text"),
    [
        ("sine", '"sine"'),
        (True, "1"),
        (3, "3"),
        (1e9, "1e+09"),
        (0.185e-12, "1.85e-13"),
        (1 / 3, repr(1 / 3)),
        ([1, 2.5], "[1, 2.5]"),
        (np.array([0.0, 1e-6]), "[0, 1e-06]"),
        (["vp1", "rp1"], '["vp1", "rp1"]'),
        ([[1], [-1]], "{[1],[-1]}"),
    ],
)
def test_format_value(value: object, text: str) -> None:
    assert format_value(value) == text


def test_format_value_exact() -> None:
    """Every float survives the round trip through the netlist exactly."""
    for value in np.random.default_rng(0).uniform(-1e12, 1e12, 100):
        assert float(format_value(value)) == value  # ruff: ignore[float-equality-comparison]


def test_format_value_rejects_unknown() -> None:
    with pytest.raises(TypeError):
        format_value(object())


def test_netlist_render() -> None:
    netlist = Netlist(
        "JJ resonance",
        loads=["josephson_junction.va", "capacitor.osdi"],
        models={"isource": "isource", "jj": "jj"},
        instances=[
            Instance("iin", ("0", "1"), "isource", {"dc": 0, "mag": 1e-9}),
            Instance("j1", ("1", "0"), "jj", {"ic": 1e-6}),
        ],
        options={"reltol": 1e-6},
        sweeps=[Sweep("bias", "iin", "dc", {"values": [0, 5e-7]})],
        analyses=[Analysis("ac1", "ac", {"from": 1e9, "to": 1e10, "mode": "lin"})],
    )
    assert netlist.render() == (
        "JJ resonance\n"
        'load "josephson_junction.va"\n'
        'load "capacitor.osdi"\n'
        "model isource isource\n"
        "model jj jj\n"
        "iin (0 1) isource dc=0 mag=1e-09\n"
        "j1 (1 0) jj ic=1e-06\n"
        "control\n"
        "  abort always\n"
        "  options reltol=1e-06\n"
        '  sweep bias instance="iin" parameter="dc" values=[0, 5e-07]\n'
        '  analysis ac1 ac from=1e+09 to=1e+10 mode="lin"\n'
        "endc\n"
    )


def test_netlist_copies_qpdk_models_only(tmp_path: Path) -> None:
    extra = tmp_path / "extra.va"
    netlist = Netlist(
        "t",
        loads=["josephson_junction.va", "squid.va", "capacitor.osdi"],
        files={"extra.va": extra},
    )
    assert netlist.all_files() == {
        "josephson_junction.va": VACASK_MODELS_DIR / "josephson_junction.va",
        "squid.va": VACASK_MODELS_DIR / "squid.va",
        "extra.va": extra,
    }


@pytest.mark.parametrize("name", ["josephson_junction.va", "squid.va"])
def test_models_ship_as_package_data(name: str) -> None:
    assert "module" in (VACASK_MODELS_DIR / name).read_text()


def test_find_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VACASK", str(tmp_path / "vacask"))
    monkeypatch.setenv("OPENVAF", str(tmp_path / "openvaf-r"))
    vacask = Vacask.find()
    assert vacask.binary == tmp_path / "vacask"
    assert vacask.openvaf == tmp_path / "openvaf-r"


def test_find_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VACASK", raising=False)
    monkeypatch.delenv("QPDK_VACASK", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(VacaskError, match=r"codeberg\.org"):
        Vacask.find()


# --------------------------------------------------------------------------- #
# CPW conversion and rational fit (need the ``models`` extra)
# --------------------------------------------------------------------------- #


def _tline_s(f: np.ndarray, z0: float, td: float, z_ref: float) -> np.ndarray:
    """S-matrix of a lossless line, shape ``(len(f), 2, 2)``."""
    gl = 2j * np.pi * f * td
    denominator = 2 * z0 * z_ref * np.cosh(gl) + (z0**2 + z_ref**2) * np.sinh(gl)
    s11 = (z0**2 - z_ref**2) * np.sinh(gl) / denominator
    s21 = 2 * z0 * z_ref / denominator
    return np.stack([np.stack([s11, s21], -1), np.stack([s21, s11], -1)], -2)


def test_cpw_tline_params_matches_cpw_model() -> None:
    """``tline_ideal`` with these parameters reproduces the qpdk CPW model."""
    PDK.activate()
    length = 4000.0
    z0, td = cpw_tline_params(length)
    assert 40 < z0 < 60
    assert td == pytest.approx(length * 1e-6 / 3e8 * np.sqrt(6.2), rel=0.05)

    f = np.linspace(1e9, 20e9, 50)
    sdict = straight(f=f, length=length)  # referenced to the line's own Z0
    expected = _tline_s(f, z0, td, z_ref=z0)
    # The CPW model includes the dielectric loss tangent; the ideal line does not.
    np.testing.assert_allclose(sdict["o2", "o1"], expected[:, 1, 0], atol=1e-4)
    np.testing.assert_allclose(sdict["o1", "o1"], expected[:, 0, 0], atol=1e-4)


@cache
def _coupled_line() -> tuple[np.ndarray, dict[tuple[str, str], np.ndarray]]:
    """A 50 Ω-referenced 44 Ω line behind a 20 fF series capacitor."""
    f = np.linspace(10e6, 70e9, 1500)
    z0, td, z_ref, c = 44.0, 33e-12, 50.0, 20e-15
    gl = 2j * np.pi * f * td
    zc = 1 / (2j * np.pi * f * c)
    # ABCD of the series capacitor times ABCD of the line
    a = np.cosh(gl) + zc * np.sinh(gl) / z0
    b = z0 * np.sinh(gl) + zc * np.cosh(gl)
    cc = np.sinh(gl) / z0
    d = np.cosh(gl)
    denominator = a + b / z_ref + cc * z_ref + d
    s = {
        ("o1", "o1"): (a + b / z_ref - cc * z_ref - d) / denominator,
        ("o2", "o2"): (-a + b / z_ref - cc * z_ref + d) / denominator,
        ("o1", "o2"): 2 * (a * d - b * cc) / denominator,
        ("o2", "o1"): 2 / denominator,
    }
    return f, s


def _dense(sdict: dict, ports: tuple[str, ...]) -> np.ndarray:
    return np.stack(
        [np.stack([np.asarray(sdict[i, j]) for j in ports], -1) for i in ports], -2
    )


@cache
def _coupled_fit() -> RationalModel:
    pytest.importorskip("skrf")
    pytest.importorskip("sax")
    f, sdict = _coupled_line()
    return RationalModel.fit(sdict, f)


def test_fit_accuracy() -> None:
    model = _coupled_fit()
    f, sdict = _coupled_line()
    np.testing.assert_allclose(
        model.s_matrix(f), _dense(sdict, ("o1", "o2")), atol=1e-2
    )
    assert model.ports == ("o1", "o2")
    assert model.residues.shape == (2, 2, len(model.poles))
    assert model.constant.shape == (2, 2)


def test_fit_is_stable_and_passive() -> None:
    model = _coupled_fit()
    assert model.is_stable()
    assert np.all(model.poles.real < 0)
    # Passive: no singular value of S exceeds 1, also far outside the fit band.
    f = np.geomspace(1e3, 1e13, 4000)
    sigma = np.linalg.svd(model.s_matrix(f), compute_uv=False)
    assert sigma.max() <= 1 + 1e-6


def test_fit_sdict_roundtrip() -> None:
    model = _coupled_fit()
    f = np.linspace(1e9, 2e9, 3)
    sdict = model.sdict(f)
    assert set(sdict) == {(i, j) for i in ("o1", "o2") for j in ("o1", "o2")}
    np.testing.assert_array_equal(sdict["o2", "o1"], model.s_matrix(f)[:, 1, 0])


def test_fit_port_order() -> None:
    pytest.importorskip("skrf")
    f, sdict = _coupled_line()
    model = RationalModel.fit(sdict, f, ports=("o2", "o1"))
    assert model.ports == ("o2", "o1")
    np.testing.assert_allclose(
        model.s_matrix(f), _dense(sdict, ("o2", "o1")), atol=1e-2
    )


def test_fit_missing_port() -> None:
    pytest.importorskip("skrf")
    f, sdict = _coupled_line()
    with pytest.raises(ValueError, match="o3"):
        RationalModel.fit(sdict, f, ports=("o1", "o3"))


def test_fit_sparse_sdict() -> None:
    """Entries missing from a sparse SDict are zero; scalars broadcast."""
    pytest.importorskip("skrf")
    f, sdict = _coupled_line()
    sparse = {**sdict, ("o3", "o3"): 0.0}
    model = RationalModel.fit(sparse, f, ports=("o1", "o2", "o3"))
    s = model.s_matrix(f)
    np.testing.assert_allclose(s[:, :2, :2], _dense(sdict, ("o1", "o2")), atol=1e-2)
    np.testing.assert_allclose(s[:, 2, :], 0, atol=1e-2)
    np.testing.assert_allclose(s[:, :, 2], 0, atol=1e-2)


def test_fit_rejects_non_passive_result() -> None:
    """A model that stays active after enforcement is an error, not a warning."""
    pytest.importorskip("skrf")
    f = np.linspace(10e6, 20e9, 500)
    # S21 = 1.2: an amplifier, which no passive model can represent.
    sdict = {("o1", "o2"): 1.2, ("o2", "o1"): 1.2}
    with pytest.raises(ValueError, match="not passive"):
        RationalModel.fit(sdict, f, ports=("o1", "o2"))


# --------------------------------------------------------------------------- #
# Simulations (need the VACASK binary)
# --------------------------------------------------------------------------- #

JJ_MODELS = {"isource": "isource", "capacitor": "capacitor", "jj": "jj"}


@pytest.mark.vacask
def test_jj_resonance_vs_bias() -> None:
    """Junction plus capacitor resonates at 1/(2 pi sqrt(L_J C)), L_J = Phi_0/(2 pi Ic cos phi)."""
    netlist = Netlist(
        "JJ LC resonance vs DC bias",
        loads=["josephson_junction.va", "capacitor.osdi"],
        models=JJ_MODELS,
        instances=[
            Instance("iin", ("0", "1"), "isource", {"dc": 0, "mag": 1e-9}),
            Instance("j1", ("1", "0"), "jj", {"ic": 1e-6, "r": 1e6}),
            Instance("c1", ("1", "0"), "capacitor", {"c": 2e-12}),
        ],
        options={"tolscale": 1e-6, "reltol": 1e-6},
        sweeps=[Sweep("bias", "iin", "dc", {"values": [0, 0.3e-6, 0.5e-6, 0.8e-6]})],
        analyses=[
            Analysis(
                "ac1", "ac", {"from": 1e9, "to": 10e9, "mode": "lin", "points": 90000}
            )
        ],
    )
    plot = Vacask.find().run(netlist)["ac1"]
    peaks = [
        group["frequency"].real[np.argmax(np.abs(group["1"]))]
        for group in plot.split("bias")
    ]
    np.testing.assert_allclose(
        peaks, [6.20350e9, 6.05900e9, 5.77300e9, 4.80520e9], rtol=1e-5
    )


@pytest.mark.vacask
def test_kerr_jpa_gain() -> None:
    netlist = Netlist(
        "Kerr JPA",
        loads=["josephson_junction.va", "capacitor.osdi", "resistor.osdi"],
        models={**JJ_MODELS, "resistor": "resistor"},
        instances=[
            Instance(
                "ipump",
                ("0", "in"),
                "isource",
                {
                    "type": "sine",
                    "sinedc": 0,
                    "ampl": 40e-9,
                    "freq": 5.84e9,
                    "spur": [[1]],
                    "smag": [1e-9],
                },
            ),
            Instance("rp", ("in", "0"), "resistor", {"r": 50}),
            Instance("cc", ("in", "1"), "capacitor", {"c": 0.185e-12}),
            Instance("j1", ("1", "0"), "jj", {"ic": 1e-6}),
            Instance("c1", ("1", "0"), "capacitor", {"c": 2e-12}),
        ],
        options={"tolscale": 1e-6, "reltol": 1e-6},
        analyses=[
            Analysis(
                "hbac1",
                "hbac",
                {
                    "freq": [5.84e9],
                    "nharm": 7,
                    "outspur": [[1], [-1]],
                    "from": 0.5e6,
                    "to": 5e6,
                    "mode": "lin",
                    "points": 9,
                },
            )
        ],
    )
    plot = Vacask.find().run(netlist)["hbac1"]
    source_current, resistance = 1e-9, 50
    gain = 20 * np.log10(np.abs(2 * plot["in;1"] / (source_current * resistance) - 1))
    idler = 20 * np.log10(np.abs(2 * plot["in;-1"] / (source_current * resistance)))
    assert gain[0] == pytest.approx(11.81, abs=0.1)
    assert idler[0] == pytest.approx(11.51, abs=0.1)
    assert gain[-1] == pytest.approx(11.50, abs=0.1)
    # Manley-Rowe: |S_ss|^2 - |S_is|^2 = 1 for a lossless degenerate amplifier
    assert 10 ** (gain[0] / 10) - 10 ** (idler[0] / 10) == pytest.approx(1, abs=0.01)


@pytest.mark.vacask
def test_squid_inductance_vs_flux() -> None:
    """Small-signal SQUID inductance is Phi_0 / (2 pi Ic cos(pi Phi/Phi_0)).

    The model keeps the sign of the cosine, so at zero bias the inductance
    changes sign across half a flux quantum instead of folding back.
    """
    ic, flux = 2e-6, [0, 0.25, 0.4, 0.6, 0.75]
    netlist = Netlist(
        "SQUID inductance vs flux",
        loads=["squid.va"],
        models={"isource": "isource", "vsource": "vsource", "squid": "squid"},
        instances=[
            Instance("iin", ("0", "1"), "isource", {"dc": 0, "mag": 1}),
            Instance("vfl", ("fl", "0"), "vsource", {"dc": 0}),
            Instance("s1", ("1", "0", "fl"), "squid", {"ic_tot": ic}),
        ],
        sweeps=[Sweep("flux", "vfl", "dc", {"values": flux})],
        analyses=[Analysis("ac1", "ac", {"values": [1e9, 2e9]})],
    )
    plot = Vacask.find().run(netlist)["ac1"]
    groups = plot.split("flux")
    np.testing.assert_allclose([g["flux"][0].real for g in groups], flux)
    for group, phi in zip(groups, flux, strict=True):
        omega = 2 * np.pi * group["frequency"].real
        inductance = group["1"] / (1j * omega)
        expected = PHI_0 / (2 * np.pi * ic * np.cos(np.pi * phi))
        np.testing.assert_allclose(inductance.real, expected, rtol=1e-6)
        np.testing.assert_allclose(inductance.imag, 0, atol=1e-6 * abs(expected))


@pytest.mark.vacask
def test_rational_subckt_matches_model() -> None:
    """VACASK's S-parameters of the emitted subcircuit equal the fit's own."""
    model = _coupled_fit()
    f = np.linspace(1e9, 20e9, 191)
    netlist = Netlist(
        "Rational model",
        loads=["resistor.osdi", "capacitor.osdi"],
        models={"resistor": "resistor", "capacitor": "capacitor", "vsource": "vsource"},
        subckts=[model.subckt("fit")],
        instances=[
            Instance("x1", ("1", "2"), "fit"),
            Instance("vp1", ("a1", "0"), "vsource", {"dc": 0}),
            Instance("rp1", ("a1", "1"), "resistor", {"r": 50}),
            Instance("vp2", ("a2", "0"), "vsource", {"dc": 0}),
            Instance("rp2", ("a2", "2"), "resistor", {"r": 50}),
        ],
        analyses=[
            Analysis("sp", "acsp", {"ports": ["vp1", "rp1", "vp2", "rp2"], "values": f})
        ],
    )
    plot = Vacask.find().run(netlist)["sp"]
    s = model.s_matrix(f)
    for i in range(2):
        for j in range(2):
            np.testing.assert_allclose(
                plot[f"s({i + 1},{j + 1})"], s[:, i, j], atol=1e-9
            )


@pytest.mark.vacask
def test_singular_dc_loop_raises() -> None:
    """An inductor in parallel with a junction has no unique DC phase."""
    netlist = Netlist(
        "Inductor loop",
        loads=["josephson_junction.va", "capacitor.osdi", "inductor.osdi"],
        models={**JJ_MODELS, "inductor": "inductor"},
        instances=[
            Instance("iin", ("0", "1"), "isource", {"dc": 0.5e-6}),
            Instance("l1", ("1", "0"), "inductor", {"l": 100e-12}),
            Instance("j1", ("1", "0"), "jj", {"ic": 1e-6}),
            Instance("c1", ("1", "0"), "capacitor", {"c": 1e-12}),
        ],
        analyses=[Analysis("op1", "op")],
    )
    with pytest.raises(VacaskError, match="zero pivot"):
        Vacask.find().run(netlist)
