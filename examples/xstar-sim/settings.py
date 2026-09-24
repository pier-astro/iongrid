"""Edit this file in each copied run directory; it records the XSTAR setup."""

from pathlib import Path

GRID_DIR = Path(__file__).resolve().parent
LOGXI_VALUES = [3.0, 4.0]
VTURB_VALUES = [100.0, 1000.0]
LOG_NH_VALUES = [22.0, 23.0]
LOG_DENSITY = 10.0
LUMINOSITY_38 = 1.0  # XSTAR rlrad38: input luminosity in 1e38 erg/s
SED_FILE = None  # Or GRID_DIR / "sed.dat": XSTAR text SED copied into each point.
XSTAR_OPTIONS = {"emult": 0.5, "taumax": 5.0, "xeemin": 0.1, "critf": 1e-5}
XSTAR_PATH = "xstar"

SETUP = dict(
    logxi_values=LOGXI_VALUES, vturb_values=VTURB_VALUES,
    log_nh_values=LOG_NH_VALUES, log_density=LOG_DENSITY,
    luminosity_38=LUMINOSITY_38, xstar_options=XSTAR_OPTIONS,
    sed_file=SED_FILE,
)
