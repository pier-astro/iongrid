#!/usr/bin/env python3
from pathlib import Path

from iongrid.cloudy.cube import write_fits_cube

GRID_DIR = Path(__file__).resolve().parent
MANIFEST = GRID_DIR / "manifest.csv"

# .npz keys : tau_absorption, tau_scattering, tau_total,
#             emt_continuum, emt_lines, emt_total,
#             incident, transmitted

# Write a unique absorption FITS cube
write_fits_cube(
    MANIFEST,
    GRID_DIR / 'results' / 'optdepth.fits',
    npz_key='tau_absorption',
    quantity='ABSORPTION OPTICAL DEPTH',
    unit='dimensionless',
    overwrite=True,
)

# Write a unique emission FITS cube
write_fits_cube(
    MANIFEST,
    GRID_DIR / 'results' / 'linem.fits',
    npz_key='emt_lines',
    quantity='EMISSION LINES',
    unit='ph cm-2 s-1',
    overwrite=True,
)