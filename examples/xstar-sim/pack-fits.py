#!/usr/bin/env python3
"""Pack the XSTAR grid into FITS: a cube (Cartesian) or a scattered table (LHS).

CLEAN in settings.py selects the versions written:
  "raw"   optdepth.fits, linem.fits, emission.fits            
  "clean" optdepth_clean.fits, linem_clean.fits, emission.fits
  "both"  all of the above
emission.fits (the total) is never modified. Cleaned files carry NOISCLN / EPS_*
header keywords.
"""

from iongrid.xstar.clean import make_postprocess
from iongrid.xstar.cube import write_fits_cube
from settings import CLEAN, EPS_E, EPS_T, GRID_DIR, GRID_MODE

if CLEAN not in {"raw", "clean", "both"}:
    raise SystemExit("CLEAN must be 'raw', 'clean' or 'both'")

layout = "cube" if GRID_MODE == "cartesian" else "table"
manifest = GRID_DIR / "manifest.csv"
results = GRID_DIR / "results"

# (file stem, npz key, quantity, unit, cleaning kind or None)
PRODUCTS = (
    ("optdepth", "tau_total", "TOTAL OPTICAL DEPTH", "dimensionless", "tau"),
    ("linem", "emt_lines", "OUTWARD LINE EMISSION", "ph cm-2 s-1", "lines"),
    ("emission", "emt_total", "OUTWARD TOTAL EMISSION", "ph cm-2 s-1", None),
)

for stem, key, quantity, unit, kind in PRODUCTS:
    if CLEAN in {"raw", "both"} or kind is None:
        write_fits_cube(manifest, results / f"{stem}.fits", npz_key=key, quantity=quantity,
                        unit=unit, overwrite=True, layout=layout)
    if CLEAN in {"clean", "both"} and kind is not None:
        post, header = make_postprocess(kind, eps_T=EPS_T, eps_E=EPS_E)
        write_fits_cube(manifest, results / f"{stem}_clean.fits", npz_key=key, quantity=quantity,
                        unit=unit, overwrite=True, layout=layout, postprocess=post, extra_header=header)
