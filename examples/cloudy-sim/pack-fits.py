#!/usr/bin/env python3
import sys

sys.dont_write_bytecode = True

from iongrid import Grid
from settings import ROOT

grid = Grid.open(ROOT)
for point, error in grid.collect().items():
    print(f"{point}: {error}")

# .npz keys : tau_absorption, tau_scattering, tau_total,
#             emt_continuum, emt_lines, emt_total,
#             incident, transmitted

# Write a unique absorption FITS table
grid.pack(
    ROOT / 'results' / 'optdepth.fits',
    key='tau_absorption',
    quantity='ABSORPTION OPTICAL DEPTH',
    unit='dimensionless',
    overwrite=True,
)

# Write a unique emission FITS table
grid.pack(
    ROOT / 'results' / 'linem.fits',
    key='emt_lines',
    quantity='EMISSION LINES',
    unit='ph cm-2 s-1',
    overwrite=True,
)
