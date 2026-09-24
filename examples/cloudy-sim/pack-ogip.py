#!/usr/bin/env python3
from pathlib import Path

from iongrid.ogip import write_ogip_table

GRID_DIR = Path(__file__).resolve().parent
RESULTS_DIR = GRID_DIR / "results"
INTERPOLATION = {"logxi": "linear", "vturb": "log", "lognH": "linear"}  # "linear", "log", or {"logxi": "linear", "vturb": "log", ...}

# Write the optical depth model
write_ogip_table(
    cube_path   = RESULTS_DIR / "optdepth.fits",
    output_path = RESULTS_DIR / "optdepth.emod",
    model_name  = "abs",
    model_type  = "exponential",
    redshift    = True,
    interpolation = INTERPOLATION,
    overwrite   = True,
)

# Write the transmission model (mtable)
write_ogip_table(
    cube_path   = RESULTS_DIR / "optdepth.fits",
    output_path = RESULTS_DIR / "transmission.mmod",
    model_name  = "trans",
    model_type  = "multiplicative",
    redshift    = True,
    interpolation = INTERPOLATION,
    overwrite   = True,
)

# Write the emission model
write_ogip_table(
    cube_path   = RESULTS_DIR / "linem.fits",
    output_path = RESULTS_DIR / "linem.amod",
    model_name  = "emis",
    model_type  = "additive",
    redshift    = True,
    interpolation = INTERPOLATION,
    overwrite   = True,
)
