#!/usr/bin/env python3
"""Convert XSTAR point tables to OGIP/XSPEC tables."""

import sys

sys.dont_write_bytecode = True

from iongrid.ogip import write_ogip_table
from settings import CLEAN, ROOT

results = ROOT / "results"
interpolation = {"logxi": "linear",
                 "vturb": "log",
                 "lognh": "linear",
                 "feabund": "linear",
                 "gamma": "linear"}

for cube, table, model_name, model_type in (
    ("optdepth.fits", "optdepth.emod", "xstar_abs", "exponential"),
    ("optdepth.fits", "transmission.mmod", "xstar_trans", "multiplicative"),
    ("linem.fits", "linem.amod", "xstar_lines", "additive"),
    ("emission.fits", "emission.amod", "xstar_emission", "additive"),
):
    versions = ("",) if cube == "emission.fits" else {
        "raw": ("",), "clean": ("_clean",), "both": ("", "_clean"),
    }[CLEAN]
    for suffix in versions:
        stem, extension = table.rsplit(".", 1)
        write_ogip_table(results / f"{cube[:-5]}{suffix}.fits", results / f"{stem}{suffix}.{extension}",
                         model_name=model_name, model_type=model_type, interpolation=interpolation, overwrite=True)
