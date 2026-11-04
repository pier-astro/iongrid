"""Edit these values in the copied campaign directory."""

from pathlib import Path

from iongrid.sampling import cartesian
from iongrid.sed import native

ROOT = Path(__file__).resolve().parent
ENGINE = "cloudy"
EXECUTABLE = "cloudy"
TEMPLATE = None  # Or ROOT / "cloudy.in".
SED = native()

PARAMS = cartesian(
    logxi=[3.0, 4.75, 6.5],
    vturb=[100.0, 550.0, 1000.0],
    lognh=[21.0, 22.5, 24.0],
    logdensity=10.0,
    logluminosity=48.0,
    slope=-0.8,
)
