"""Edit these values in the copied campaign directory."""

from pathlib import Path

from iongrid.sampling import cartesian, lhs
from iongrid.sed import native

ROOT = Path(__file__).resolve().parent
ENGINE = "xstar"
EXECUTABLE = "/data-fast/dragon/software/heasoft-6.36/x86_64-pc-linux-gnu-libc2.39/bin/xstar"
TEMPLATE = ROOT / "xstar.par"
SED = native()
SAMPLING = "lhs"

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
    PARAMS = (
        lhs(
            bounds={
                "logxi": (2.0, 7.0),
                "vturb": (50.0, 10000.0),
                "lognh": (20.0, 25.0),
                "feabund": (0.5, 2.0),
                "gamma": (1.4, 2.6)},
            samples=20,
            seed=12345,
            anchors=True,
            logarithmic=("vturb",)
            )
        + 
        lhs(
            bounds={
                "logxi": (2.0, 7.0), 
                "vturb": (50.0, 10000.0), 
                "lognh": (22.5, 25.0),
                "feabund": (0.5, 2.0), 
                "gamma": (1.4, 2.6)},
            samples=10,
            seed=777, 
            anchors=False, 
            logarithmic=("vturb",),
            )
        + 
        lhs(
            bounds={
                "logxi": (2.0, 5.0), 
                "vturb": (50.0, 10000.0), 
                "lognh": (20.0, 25.0),
                "feabund": (0.5, 2.0), 
                "gamma": (1.4, 2.6)},
            samples=20,
            seed=34562, 
            anchors=False, 
            logarithmic=("vturb",),
            )
    )
else:
    raise ValueError("SAMPLING must be 'cartesian' or 'lhs'")

CLEAN = "both"  # "raw", "clean", or "both".
EPS_T = 1e-5
EPS_E = 1e-5
