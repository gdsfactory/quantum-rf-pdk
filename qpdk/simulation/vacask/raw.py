"""Reader for the SPICE raw files written by VACASK."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import VacaskError


@dataclass(frozen=True)
class RawPlot:
    """One plot of a SPICE raw file.

    Attributes:
        title: Circuit title.
        plotname: Analysis description written by the simulator.
        names: Variable names, in column order.
        data: Array of shape ``(points, variables)``; complex for complex plots.
    """

    title: str
    plotname: str
    names: tuple[str, ...]
    data: np.ndarray

    def __getitem__(self, name: str) -> np.ndarray:
        """Return the column of variable ``name``.

        Raises:
            KeyError: If the plot has no variable called ``name``.
        """
        try:
            return self.data[:, self.names.index(name)]
        except ValueError:
            raise KeyError(f"{name!r} not in {self.names}") from None

    def __contains__(self, name: object) -> bool:
        """Whether the plot has a variable called ``name``."""
        return name in self.names

    def split(self, name: str) -> list[RawPlot]:
        """Split a swept plot into one plot per value of the sweep variable ``name``.

        A new group starts wherever the sweep column changes value.

        Returns:
            One plot per sweep point, in order.
        """
        column = self[name]
        starts = [0, *(np.flatnonzero(column[1:] != column[:-1]) + 1), len(column)]
        return [
            RawPlot(self.title, self.plotname, self.names, self.data[a:b])
            for a, b in itertools.pairwise(starts)
        ]


class RawFile:
    """A SPICE raw file (binary or ASCII) as written by VACASK.

    The file may hold several plots back to back. Binary data is little-endian
    ``float64`` (``complex128`` for complex plots), stored point-major.

    Attributes:
        path: Path of the file.
        plots: The plots in file order.

    Raises:
        VacaskError: If the file is truncated or malformed.
    """

    def __init__(self, path: str | Path) -> None:
        """Read and parse ``path``."""
        self.path = Path(path)
        self.plots = self._parse(self.path.read_bytes())

    def __len__(self) -> int:
        """Number of plots."""
        return len(self.plots)

    def __getitem__(self, index: int) -> RawPlot:
        """Plot number ``index``."""
        return self.plots[index]

    def _error(self, message: str) -> VacaskError:
        """Return an error that names this raw file."""
        return VacaskError(f"{self.path}: {message}")

    @staticmethod
    def _parse_header(lines: list[str]) -> tuple[dict[str, str], list[str]]:
        """Parse the text header of one plot into its fields and variable names."""
        header: dict[str, str] = {}
        names: list[str] = []
        in_variables = False
        for line in lines:
            if in_variables and line[:1] in {"\t", " "}:
                names.append(line.split()[1])
                continue
            key, _, value = line.partition(":")
            in_variables = key == "Variables"
            header[key] = value.strip()
        return header, names

    def _parse_ascii(
        self, text: str, npoints: int, nvars: int, dtype: type
    ) -> np.ndarray:
        """Parse the ``Values:`` section of an ASCII raw file."""
        tokens = text.split()
        if len(tokens) < npoints * (nvars + 1):
            raise self._error("ASCII data is truncated")
        data = np.empty((npoints, nvars), dtype=dtype)
        pos = 0
        try:  # ruff: ignore[too-many-statements-in-try-clause]
            for point in range(npoints):
                pos += 1  # point index
                for var in range(nvars):
                    token = tokens[pos]
                    pos += 1
                    if dtype is complex:
                        real, _, imag = token.partition(",")
                        data[point, var] = complex(float(real), float(imag or 0.0))
                    else:
                        data[point, var] = float(token)
        except ValueError as error:
            raise self._error(f"malformed ASCII value {tokens[pos - 1]!r}") from error
        return data

    def _parse(self, raw: bytes) -> list[RawPlot]:
        """Split a raw file into its plots and parse each one."""
        plots: list[RawPlot] = []
        pos = 0
        while pos < len(raw) and raw[pos:].strip():
            markers = [
                m
                for m in (raw.find(b"Binary:\n", pos), raw.find(b"Values:\n", pos))
                if m >= 0
            ]
            if not markers:
                raise self._error("no 'Binary:' or 'Values:' section")
            marker = min(markers)
            header, names = self._parse_header(raw[pos:marker].decode().splitlines())
            try:
                npoints = int(header["No. Points"])
                nvars = int(header["No. Variables"])
            except (KeyError, ValueError) as error:
                raise self._error(f"bad header field {error}") from error
            if len(names) != nvars:
                raise self._error(f"header lists {len(names)} of {nvars} variables")
            is_complex = "complex" in header.get("Flags", "")
            start = marker + len(b"Binary:\n")
            if raw.startswith(b"Binary:\n", marker):
                dtype = np.dtype("<c16" if is_complex else "<f8")
                end = start + npoints * nvars * dtype.itemsize
                if end > len(raw):
                    raise self._error("binary data is truncated")
                data = np.frombuffer(raw[start:end], dtype=dtype).reshape(
                    npoints, nvars
                )
            else:
                next_plot = raw.find(b"Title:", start)
                end = len(raw) if next_plot < 0 else next_plot
                data = self._parse_ascii(
                    raw[start:end].decode(),
                    npoints,
                    nvars,
                    complex if is_complex else float,
                )
            plots.append(
                RawPlot(
                    header.get("Title", ""),
                    header.get("Plotname", ""),
                    tuple(names),
                    data,
                )
            )
            pos = end
        return plots
