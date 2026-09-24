#!/usr/bin/env python3
"""Convert XSTAR cubes to OGIP/XSPEC tables."""

from iongrid.ogip import write_ogip_table
from settings import GRID_DIR

results = GRID_DIR / "results"
interpolation = {"logxi": "linear", "vturb": "log", "lognH": "linear"}
for cube, table, model_name, model_type in (
    ("optdepth.fits", "optdepth.emod", "xstar_abs", "exponential"),
    ("optdepth.fits", "transmission.mmod", "xstar_trans", "multiplicative"),
    ("linem.fits", "linem.amod", "xstar_lines", "additive"),
    ("emission.fits", "emission.amod", "xstar_emission", "additive"),
):
    write_ogip_table(results / cube, results / table, model_name=model_name,
                     model_type=model_type, interpolation=interpolation, overwrite=True)
