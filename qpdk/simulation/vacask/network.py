"""Turn S-parameters into VACASK subcircuits. Needs the ``models`` extra."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from qpdk.logger import logger

from .netlist import _templates

if TYPE_CHECKING:
    from gdsfactory.typings import CrossSectionSpec
    from skrf.vectorFitting import VectorFitting

MODELS_EXTRA_HINT = (
    "Install it with `uv sync --extra models` or `pip install 'qpdk[models]'`."
)


def cpw_tline_params(
    length: float, cross_section: CrossSectionSpec = "cpw"
) -> tuple[float, float]:
    """Characteristic impedance and delay of a lossless CPW for ``tline_ideal``.

    Uses :func:`~qpdk.models.cpw.cpw_parameters` with zero loss tangent, so the
    result matches VACASK's lossless ``tline_ideal`` device. Needs the
    ``models`` extra.

    Args:
        length: Line length in µm.
        cross_section: CPW cross-section.

    Returns:
        ``(z0, td)``: characteristic impedance in Ω and propagation delay in s.

    Raises:
        ImportError: If the ``models`` extra is not installed.
    """
    try:
        from qpdk.models.constants import c_0  # ruff: ignore[import-outside-top-level]
        from qpdk.models.cpw import (  # ruff: ignore[import-outside-top-level]
            cpw_parameters,
            get_cpw_dimensions,
        )
    except ImportError as error:
        raise ImportError(
            f"cpw_tline_params needs the qpdk CPW models. {MODELS_EXTRA_HINT}"
        ) from error

    width, gap = get_cpw_dimensions(cross_section)
    ep_eff, z0 = cpw_parameters(width, gap, tand=0.0)
    td = length * 1e-6 * np.sqrt(float(np.real(ep_eff))) / c_0
    return float(np.real(z0)), float(td)


def vector_fit(
    sdict: Mapping[tuple[str, str], ArrayLike],
    f: ArrayLike,
    *,
    ports: Sequence[str] | None = None,
    z0: float = 50.0,
    n_poles_real: int = 1,
    n_poles_cmplx: int = 10,
    enforce_passivity: bool = True,
    n_samples: int = 1000,
) -> VectorFitting:
    """Fit S-parameters with :class:`skrf.vectorFitting.VectorFitting`.

    A thin wrapper that builds the :class:`skrf.Network` from an SDict and
    checks that the result can be used as a circuit: every pole stable and the
    model passive. Evaluate the fit with
    :meth:`~skrf.vectorFitting.VectorFitting.get_model_response` and turn it
    into a VACASK subcircuit with :func:`vector_fit_subckt`.

    Args:
        sdict: S-parameters on ``f``, referenced to ``z0``. Missing entries
            of a sparse SDict are zero.
        f: Frequencies in Hz. Cover every frequency the circuit will see,
            e.g. up to ``nharm`` times the pump frequency for harmonic balance.
        ports: Port order. Defaults to the order in which ``sdict`` names them.
            Stored as the network's ``port_names``.
        z0: Reference impedance in Ω.
        n_poles_real: Number of initial real poles.
        n_poles_cmplx: Number of initial complex-conjugate pole pairs.
        enforce_passivity: Make the model passive if the fit is not.
        n_samples: Frequency samples for the passivity enforcement.

    Returns:
        The fitted :class:`~skrf.vectorFitting.VectorFitting` object.

    Raises:
        ImportError: If the ``models`` extra is not installed.
        ValueError: If a port is missing from ``sdict``, the fit has an
            unstable pole, or it is still not passive after enforcement.
    """
    try:
        import jax.numpy as jnp  # ruff: ignore[import-outside-top-level]
        import sax  # ruff: ignore[import-outside-top-level]
        import skrf  # ruff: ignore[import-outside-top-level]
        from skrf.vectorFitting import (  # ruff: ignore[import-outside-top-level]
            VectorFitting,
        )
    except ImportError as error:
        raise ImportError(
            f"vector_fit needs sax and scikit-rf. {MODELS_EXTRA_HINT}"
        ) from error

    f = np.asarray(f, dtype=float)
    dense, port_map = sax.sdense({
        k: jnp.broadcast_to(jnp.asarray(v), f.shape) for k, v in sdict.items()
    })
    ports = tuple(ports) if ports is not None else tuple(port_map)
    missing = set(ports) - set(port_map)
    if missing:
        raise ValueError(f"Ports {sorted(missing)} are not in the SDict")
    order = [port_map[p] for p in ports]
    s = np.asarray(dense)[:, order][:, :, order]

    network = skrf.Network(frequency=skrf.Frequency.from_f(f, unit="Hz"), s=s, z0=z0)
    network.port_names = list(ports)
    active = np.flatnonzero(np.any(s != 0, axis=(0, 1)) | np.any(s != 0, axis=(0, 2)))
    if not len(active):
        # An exactly matched network has no dynamics for scikit-rf to identify.
        vf = VectorFitting(network)
        vf.poles = np.empty(0, dtype=complex)
        vf.residues = np.empty((len(ports) ** 2, 0), dtype=complex)
        vf.constant_coeff = np.zeros(len(ports) ** 2)
        vf.proportional_coeff = np.zeros(len(ports) ** 2)
        return vf
    if 0 < len(active) < len(ports):
        # Exactly matched, isolated ports destabilize scikit-rf's residue enforcement.
        fitted = vector_fit(
            {(ports[i], ports[j]): s[:, i, j] for i in active for j in active},
            f,
            ports=[ports[i] for i in active],
            z0=z0,
            n_poles_real=n_poles_real,
            n_poles_cmplx=n_poles_cmplx,
            enforce_passivity=enforce_passivity,
            n_samples=n_samples,
        )
        vf = VectorFitting(network)
        vf.poles = fitted.poles
        indices = np.array([i * len(ports) + j for i in active for j in active])
        for name in ("residues", "constant_coeff", "proportional_coeff"):
            values = getattr(fitted, name)
            padded = np.zeros((len(ports) ** 2, *values.shape[1:]), dtype=values.dtype)
            padded[indices] = values
            setattr(vf, name, padded)
        return vf
    vf = VectorFitting(network)
    vf.vector_fit(n_poles_real=n_poles_real, n_poles_cmplx=n_poles_cmplx)
    if enforce_passivity and not vf.is_passive():
        vf.passivity_enforce(n_samples=n_samples, f_max=float(f.max()))
        if not vf.is_passive():
            raise ValueError(
                "Vector fit is still not passive after passivity enforcement; "
                "try fewer poles or a wider frequency range"
            )
    poles = vf.poles
    if poles is None:
        raise ValueError("Vector fit produced no poles")
    if np.any(poles.real >= 0):
        raise ValueError("Vector fit has unstable poles")
    logger.info(
        "Vector fit: {} poles, RMS error {:.2e}", len(poles), vf.get_rms_error()
    )
    return vf


def vector_fit_subckt(vf: VectorFitting, name: str) -> str:
    r"""Emit a vector fit as a VACASK subcircuit.

    The fitted S-matrix is

    .. math::

        S(s) = D + \sum_k \frac{R_k}{s - p_k},

    where each complex pole stands for itself and its conjugate.

    Each port is a resistor :math:`Z_0` to ground in parallel with a current
    source :math:`2 b / \sqrt{Z_0}`, which gives the port voltage
    :math:`\sqrt{Z_0}(a + b)`. The incident waves :math:`a` drive one
    first-order state per real pole and two per complex pair, and the
    reflected waves :math:`b` are weighted sums of the states. Waves and
    states are node voltages scaled by :math:`\sqrt{Z_0}`; each state
    capacitor is :math:`1/|p_k|`, so all conductances and gains are of order
    one.

    The subcircuit uses the builtin ``vccs`` and the ``resistor`` and
    ``capacitor`` models, so the netlist must load ``resistor.osdi`` and
    ``capacitor.osdi`` and define models of those names.

    Args:
        vf: A fit from :func:`vector_fit`, with a constant term and no
            proportional term, referenced to one real impedance.
        name: Subcircuit name.

    Returns:
        The subcircuit text, with terminals ``p1``, ``p2``, ... in port order.

    Raises:
        ValueError: If the fit has a proportional term, an unstable pole, or
            port-dependent or complex reference impedances.
    """
    network = vf.network
    n = network.nports
    z0_all = np.asarray(network.z0)
    z0 = float(z0_all.flat[0].real)
    if not np.allclose(z0_all, z0):
        raise ValueError("vector_fit_subckt needs one real reference impedance")
    if np.any(np.asarray(vf.proportional_coeff) != 0):
        raise ValueError("vector_fit_subckt does not support a proportional term")
    poles = np.asarray(vf.poles)
    if np.any(poles.real >= 0):
        raise ValueError("Vector fit has unstable poles")
    residues = np.asarray(vf.residues).reshape(n, n, -1)
    constant = np.asarray(vf.constant_coeff).reshape(n, n)

    terminals = [f"p{i + 1}" for i in range(n)]
    elements: list[tuple[str, tuple[str, ...], float]] = []
    for i in range(n):
        # Port: V = Z0 I + 2 sqrt(Z0) b
        elements += [
            ("r", (terminals[i],), z0),
            ("g", (terminals[i], f"b{i}"), 2 / z0),
            # Incident wave: sqrt(Z0) a = V - sqrt(Z0) b
            ("r", (f"a{i}",), 1.0),
            ("g", (f"a{i}", terminals[i]), 1.0),
            ("g", (f"a{i}", f"b{i}"), -1.0),
            ("r", (f"b{i}",), 1.0),
        ]
    for j in range(n):
        for k, pole in enumerate(poles):
            mag = abs(pole)
            u, v = f"x{j}_{k}", f"y{j}_{k}"
            # Scaled state |p| a / (s - p), split into real and imaginary parts
            elements += [
                ("c", (u,), 1 / mag),
                ("r", (u,), mag / -pole.real),
                ("g", (u, f"a{j}"), 1.0),
            ]
            if pole.imag != 0:
                elements += [
                    ("g", (u, v), -pole.imag / mag),
                    ("c", (v,), 1 / mag),
                    ("r", (v,), mag / -pole.real),
                    ("g", (v, u), pole.imag / mag),
                ]
            scale = 2 / mag if pole.imag != 0 else 1 / mag
            for i in range(n):
                residue = residues[i, j, k]
                elements.append(("g", (f"b{i}", u), scale * residue.real))
                if pole.imag != 0:
                    elements.append(("g", (f"b{i}", v), -scale * residue.imag))
        for i in range(n):
            elements.append(("g", (f"b{i}", f"a{j}"), float(constant[i, j].real)))

    port_names = network.port_names or terminals
    return (
        _templates()
        .get_template("subckt.sim.j2")
        .render(
            name=name,
            ports=port_names,
            terminals=terminals,
            n_poles=len(poles),
            elements=elements,
        )
    )
