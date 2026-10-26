# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "qpdk[models] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git",
#   "matplotlib",
# ]
# ///
"""Simulate the gatemon YAML schematic or extracted layout with gdsfactory and SAX.

Run with ``python -m qpdk.samples.simulate_gatemon_test_chip --help`` after
installing ``qpdk[models]`` and matplotlib. Frequencies are sampled uniformly.
Gate ports are matched to 50 ohm unless explicitly terminated with an RF short
at the launchers. The qubit is a linearized Josephson oscillator, not a
quantum transition or a two-tone spectroscopy model.
"""

import argparse
from pathlib import Path
from typing import Literal

import gdsfactory as gf
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import sax
import yaml

from qpdk import PDK, logger
from qpdk.colormaps import QPDK
from qpdk.config import PATH
from qpdk.models.gatemon import gatemon_island
from qpdk.models.generic import short


def _netlist(source: Literal["schematic", "layout"]) -> dict:
    """Read logical YAML connections or extract every physical CPW route."""
    PDK.activate()
    if source == "layout":
        # Preserve capacitances in F; the default rounding would erase them.
        netlist = gf.get_component("gatemon_test_chip").get_netlist(
            serialization_max_digits=18, on_dangling_port="ignore"
        )
        return {
            "instances": netlist["instances"],
            "connections": {net["p1"]: net["p2"] for net in netlist["nets"]},
            "ports": netlist["ports"],
        }
    document = yaml.safe_load((PATH.samples / "gatemon_test_chip.pic.yml").read_text())
    connections = dict(document["connections"])
    for route in document["routes"].values():
        connections.update(route["links"])
    return {
        "instances": document["instances"],
        "connections": connections,
        "ports": document["ports"],
    }


def _circuit(
    source: Literal["schematic", "layout"],
    gate_load: Literal["matched", "short"] = "matched",
) -> tuple[sax.SDictModel, dict]:
    """Bind the layout's island adapter and optional external RF gate loads."""
    netlist = _netlist(source)
    if gate_load == "short":
        for port in ("gate_a", "gate_b"):
            name = f"load_{port}"
            netlist["instances"][name] = {"component": "short"}
            netlist["connections"][netlist["ports"].pop(port)] = f"{name},o1"
    # The legacy gatemon model has a ground port; the layout uses readout/gate.
    models = dict(PDK.models) | {"gatemon": gatemon_island, "short": short}
    circuit, _ = sax.circuit(
        netlist, models=models, ignore_impossible_connections=False
    )
    return circuit, netlist


def _main() -> None:
    """Save the circuit response as CSV and a vector and raster spectrum plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("schematic", "layout"), default="layout")
    parser.add_argument("--gate-load", choices=("matched", "short"), default="matched")
    parser.add_argument("--f-min-ghz", type=float, default=1.0)
    parser.add_argument("--f-max-ghz", type=float, default=15.0)
    parser.add_argument("--points", type=int, default=4001)
    parser.add_argument("--transmission", type=float, default=0.8)
    parser.add_argument("--output", type=Path, default=Path("build/gatemon-sax"))
    args = parser.parse_args()
    if not 0 < args.f_min_ghz < args.f_max_ghz or args.points < 2:
        parser.error("Use positive ordered frequencies and at least two points")
    if not 0 <= args.transmission <= 1:
        parser.error("Channel transmission must be between zero and one")
    circuit, netlist = _circuit(args.source, args.gate_load)
    settings = {
        name: {"transmissions": (args.transmission,)}
        for name, instance in netlist["instances"].items()
        if instance["component"] == "gatemon"
    }
    f = jnp.linspace(args.f_min_ghz * 1e9, args.f_max_ghz * 1e9, args.points)
    response = circuit(f=f, **settings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = [np.asarray(f)]
    labels = ["frequency_hz"]
    for (a, b), values in sorted(response.items()):
        columns.extend((np.asarray(values.real), np.asarray(values.imag)))
        labels.extend((f"{a}__{b}_real", f"{a}__{b}_imag"))
    np.savetxt(
        args.output.with_suffix(".csv"),
        np.column_stack(columns),
        delimiter=",",
        header=",".join(labels),
        comments="",
    )
    try:
        plt.style.use(PATH.docs / "qpdk.mplstyle")
    except OSError:
        plt.style.use({"axes.grid": True, "grid.alpha": 0.3, "image.cmap": QPDK.name})
    fig, ax = plt.subplots(figsize=(9, 4))
    pairs = [("o2", "o1"), ("o4", "o3")]
    if args.gate_load == "matched":
        pairs.extend((("gate_a", "o1"), ("gate_b", "o3")))
    for a, b in pairs:
        magnitude = 20 * jnp.log10(jnp.maximum(jnp.abs(response[a, b]), 1e-8))
        ax.plot(f / 1e9, magnitude, label=f"{a} from {b}")
    ax.set(xlabel=r"$f\;[\text{GHz}]$", ylabel="Magnitude [dB]", ylim=(-100, 1))
    ax.set_title(
        f"{args.source}, {args.gate_load} gate, transmission {args.transmission:g}"
    )
    ax.legend()
    fig.tight_layout()
    for suffix in (".png", ".svg"):
        fig.savefig(args.output.with_suffix(suffix), dpi=160)
    plt.close(fig)
    logger.info(f"Saved {args.source} SAX response to {args.output}.csv and .png/.svg")


if __name__ == "__main__":
    _main()
