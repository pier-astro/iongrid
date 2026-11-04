"""Interactive inspection of tabulated iongrid and OGIP spectra."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import SCHEMA_VERSION


class TableInspector:
    """Browse stored grid nodes without interpolating their spectra.

    Accept an iongrid FITS cube or a regular OGIP/XSPEC table model. Energy
    is displayed at bin centres in keV; spectral values retain the file's
    native convention (for example, optical depth in a cube or transmission
    in a multiplicative table). Use :meth:`show` in Jupyter for sliders.
    """

    def __init__(self, path: str | Path):
        from astropy.io import fits

        self.path = Path(path)
        with fits.open(self.path, memmap=True) as hdul:
            if "GRID" in hdul and "SPECTRA" in hdul:
                self._read_cube(hdul)
            elif all(name in hdul for name in ("PARAMETERS", "ENERGIES", "SPECTRA")):
                self._read_ogip(hdul)
            else:
                raise ValueError("Expected an iongrid FITS cube or an OGIP table model.")
        if self.values.shape != tuple(map(len, self.axes)) + (len(self.energy),):
            raise ValueError("Spectrum shape does not match parameter and energy axes.")
        if len(self.energy) < 2 or np.any(np.diff(self.energy) <= 0):
            raise ValueError("Energy bin centres must be strictly increasing.")

    def _read_cube(self, hdul) -> None:
        """Read the iongrid cube schema."""
        header = hdul[0].header
        if "IGFMTVER" in header and int(header["IGFMTVER"]) != SCHEMA_VERSION:
            raise ValueError("Unsupported iongrid cube schema version.")
        grid = hdul["GRID"].data
        self.names = ("logxi", "vturb", "log_nh")
        self.axes = tuple(np.asarray(grid[name][0], dtype=float) for name in
                          ("LOGXI", "VTURB", "LOG_NH"))
        self.energy = np.asarray(grid["ENERGY"][0], dtype=float)
        self.values = np.array(hdul["SPECTRA"].data, copy=True)
        self.quantity = str(header.get("QUANTITY", "Spectrum"))
        self.unit = str(hdul["SPECTRA"].header.get("BUNIT", ""))
        self.format = "iongrid cube"

    def _read_ogip(self, hdul) -> None:
        """Read a regular OGIP/XSPEC table without evaluating interpolation."""
        header = hdul[0].header
        if "IGFMTVER" in header and int(header["IGFMTVER"]) != SCHEMA_VERSION:
            raise ValueError("Unsupported iongrid OGIP schema version.")
        parameters = hdul["PARAMETERS"].data
        self.names = tuple(_text(value) for value in parameters["NAME"])
        self.axes = tuple(np.asarray(row["VALUE"][: int(row["NUMBVALS"])], dtype=float)
                          for row in parameters)
        energies = hdul["ENERGIES"].data
        self.energy = 0.5 * (np.asarray(energies["ENERG_LO"], dtype=float)
                             + np.asarray(energies["ENERG_HI"], dtype=float))
        spectra = hdul["SPECTRA"].data
        expected = np.stack(np.meshgrid(*self.axes, indexing="ij"), axis=-1).reshape(-1, len(self.axes))
        stored = np.asarray(spectra["PARAMVAL"], dtype=float).reshape(expected.shape)
        if not np.allclose(stored, expected):
            raise ValueError("OGIP rows do not follow the regular parameter-grid order.")
        self.values = np.array(spectra["INTPSPEC"], copy=True).reshape(
            *(len(axis) for axis in self.axes), len(self.energy)
        )
        model_type = _text(header.get("XFXP0001", "additive" if header.get("ADDMODEL") else "multiplicative"))
        self.quantity = {"additive": "Bin-integrated photon flux",
                         "multiplicative": "Transmission",
                         "exponential": "Optical depth"}.get(model_type, "Table value")
        self.unit = str(header.get("MODLUNIT", ""))
        self.format = "OGIP table"

    def node(self, **parameters: float) -> np.ndarray:
        """Return one stored spectrum selected by its exact grid coordinates."""
        if set(parameters) != set(self.names):
            raise ValueError(f"Specify every grid coordinate: {', '.join(self.names)}")
        indices = []
        for name, axis in zip(self.names, self.axes):
            matches = np.flatnonzero(np.isclose(axis, parameters[name], rtol=1e-6, atol=0))
            if len(matches) != 1:
                raise ValueError(f"{name}={parameters[name]} is not a stored grid coordinate.")
            indices.append(int(matches[0]))
        return self.values[tuple(indices)]

    def show(self, *, emin: float | None = None, emax: float | None = None,
             yscale: str = "linear", ylabel: str | None = None):
        """Display a matplotlib plot with one selection slider per grid axis.

        Requires Jupyter, ipywidgets and an interactive matplotlib backend
        such as ``%matplotlib widget``. Returns the figure and slider widgets.
        """
        import matplotlib.pyplot as plt
        import ipywidgets as widgets
        from IPython.display import display

        mask = np.ones(len(self.energy), dtype=bool)
        if emin is not None:
            mask &= self.energy >= emin
        if emax is not None:
            mask &= self.energy <= emax
        if not mask.any():
            raise ValueError("The requested energy range contains no table bins.")
        sliders = [widgets.SelectionSlider(
            options=[(f"{value:.6g}", index) for index, value in enumerate(axis)],
            value=0, description=name, continuous_update=False,
            style={"description_width": "initial"},
        ) for name, axis in zip(self.names, self.axes)]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        line, = ax.plot(self.energy[mask], self.values[(0,) * len(self.axes)][mask])
        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel(ylabel or f"{self.quantity}{f' ({self.unit})' if self.unit else ''}")
        ax.set_xscale("log")
        ax.set_yscale(yscale)

        def update(_=None):
            index = tuple(slider.value for slider in sliders)
            line.set_ydata(self.values[index][mask])
            ax.set_title(", ".join(f"{name}={axis[i]:.6g}"
                                   for name, axis, i in zip(self.names, self.axes, index)))
            ax.relim()
            ax.autoscale_view(scalex=False)
            fig.canvas.draw_idle()

        for slider in sliders:
            slider.observe(update, names="value")
        update()
        display(widgets.VBox(sliders))
        display(fig.canvas)
        return fig, sliders


def _text(value) -> str:
    return value.decode().strip() if isinstance(value, bytes) else str(value).strip()
