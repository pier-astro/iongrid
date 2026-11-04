#!/usr/bin/env python3
"""Pack the XSTAR grid into FITS: a table of parameter rows.

CLEAN in settings.py selects the versions written:
  "raw"   optdepth.fits, linem.fits, emission.fits
  "clean" optdepth_clean.fits, linem_clean.fits, emission.fits
  "both"  all of the above
emission.fits (the total) is never modified. Cleaned files carry NOISCLN / EPS_*
header keywords.
"""

import sys

sys.dont_write_bytecode = True

from iongrid import Grid
from iongrid.xstar.clean import make_postprocess
from settings import CLEAN, EPS_E, EPS_T, ROOT

if CLEAN not in {"raw", "clean", "both"}:
    raise SystemExit("CLEAN must be 'raw', 'clean' or 'both'")

grid = Grid.open(ROOT)
for point, error in grid.collect().items():
    print(f"{point}: {error}")
results = ROOT / "results"

# (file stem, npz key, quantity, unit, cleaning kind or None)
PRODUCTS = (
    ("optdepth", "tau_total", "TOTAL OPTICAL DEPTH", "dimensionless", "tau"),
    ("linem", "emt_lines", "OUTWARD LINE EMISSION", "ph cm-2 s-1", "lines"),
    ("emission", "emt_total", "OUTWARD TOTAL EMISSION", "ph cm-2 s-1", None),
)

for stem, key, quantity, unit, kind in PRODUCTS:
    if CLEAN in {"raw", "both"} or kind is None:
        grid.pack(results / f"{stem}.fits", key=key, quantity=quantity,
                  unit=unit, overwrite=True)
    if CLEAN in {"clean", "both"} and kind is not None:
        post, header = make_postprocess(kind, eps_T=EPS_T, eps_E=EPS_E)
        grid.pack(results / f"{stem}_clean.fits", key=key, quantity=quantity,
                  unit=unit, overwrite=True, postprocess=post, header=header)
