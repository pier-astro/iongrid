"""Edit these values in the copied campaign directory."""

from pathlib import Path

from iongrid.sampling import cartesian, lhs
from iongrid.sed import native

ROOT = Path(__file__).resolve().parent
ENGINE = "xstar"
EXECUTABLE = "xstar"
TEMPLATE = ROOT / "xstar.par"
SED = native()
SAMPLING = "cartesian"

if SAMPLING == "cartesian":
    PARAMS = cartesian(
        logxi=[3.0, 4.0],
        vturb=[100.0, 1000.0],
        lognh=[22.0, 23.0],
        feabund=[0.5, 1.5],
        gamma=[1.5, 2.5],
        logdensity=12.0,
        logluminosity=47.0,
    )
elif SAMPLING == "lhs":
    PARAMS = lhs(
        bounds={
            "logxi": (3.0, 4.0),
            "vturb": (100.0, 1000.0),
            "lognh": (22.0, 23.0),
            "feabund": (0.5, 1.5),
            "gamma": (1.5, 2.5)},
        samples=500,
        seed=12345,
        logarithmic=("vturb",),
        logdensity=12.0,
        logluminosity=47.0,
    )
else:
    raise ValueError("SAMPLING must be 'cartesian' or 'lhs'")

CLEAN = "raw"  # "raw", "clean", or "both".
EPS_T = 1e-5
EPS_E = 1e-5
