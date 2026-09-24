#!/usr/bin/env python3
"""Pack a complete XSTAR grid into the common FITS cube format."""

from iongrid.xstar.cube import write_fits_cube
from settings import GRID_DIR

manifest = GRID_DIR / "manifest.csv"
results = GRID_DIR / "results"
for name, key, quantity, unit in (
    ("optdepth.fits", "tau_total", "TOTAL OPTICAL DEPTH", "dimensionless"),
    ("linem.fits", "emt_lines", "OUTWARD LINE EMISSION", "ph cm-2 s-1"),
    ("emission.fits", "emt_total", "OUTWARD TOTAL EMISSION", "ph cm-2 s-1"),
):
    write_fits_cube(manifest, results / name, npz_key=key,
                    quantity=quantity, unit=unit, overwrite=True)
