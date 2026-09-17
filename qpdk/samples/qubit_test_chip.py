"""Qubit test chip sample, materialized from its ``.gsch`` schematic."""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import gdsfactory as gf

GSCH_SAMPLE = Path(__file__).parent / "qubit_test_chip.gsch"


@gf.cell(tags=["samples", "qubits"])
def qubit_test_chip() -> gf.Component:
    """Layout of the qubit test chip, built from its ``.gsch`` schematic.

    The full chip layout is defined in the schematic; this factory only runs
    the gdsfactoryplus build pipeline (``.gsch`` to GDS) and reads the
    result back, so editing the schematic edits the layout.

    Returns:
        The materialized sample component.

    Raises:
        ImportError: The gdsfactoryplus 2.0 SDK is not importable. It is not
            on PyPI; run ``just fetch-gfp`` and source
            ``build/gfp-vsix/env.sh``.
        ValueError: The build dropped unrouted connections, so the layout
            would silently miss nets.
    """
    try:
        from gdsfactoryplus.compile import (  # ruff: ignore[import-outside-top-level]
            build_nyancir_gds,
        )
    except ImportError as e:
        raise ImportError(
            "Building qubit_test_chip from its .gsch needs the "
            "gdsfactoryplus 2.0 SDK, which is not on PyPI. "
            "Run `just fetch-gfp` and source build/gfp-vsix/env.sh."
        ) from e

    with tempfile.TemporaryDirectory() as tmp:
        materialized_name = f"_qpdk_qubit_test_chip_{uuid.uuid4().hex}"
        schematic_path = Path(tmp) / f"{materialized_name}.gsch"
        schematic_path.write_bytes(GSCH_SAMPLE.read_bytes())
        gds_path = Path(tmp) / f"{materialized_name}.gds"
        result = build_nyancir_gds(
            str(schematic_path),
            str(gds_path),
            "qpdk.PDK",
            str(Path(tmp) / f"{materialized_name}.dschematic"),
        )
        warnings = result["warnings"]
        if warnings:
            raise ValueError(
                "qubit_test_chip.gsch dropped unrouted connections: "
                + "; ".join(warnings)
            )
        return gf.kcl[materialized_name]
