#!/usr/bin/env python3
import sys
from pathlib import Path
from iongrid.cloudy.grid import run_grid

# ---------------------------------------------------------
# Grid Configuration
# ---------------------------------------------------------
LOGXI_VALUES = [3.0, 4.75, 6.5]
VTURB_VALUES = [100.0, 550.0, 1000.0]
LOG_NH_VALUES = [21.0, 22.5, 24.0]

# Fixed parameters
LOG_DENSITY = 10.0  # log(cm^-3)
SED_SLOPE = -0.8    # Power law slope (defaults to -0.8)
SED_FILE = None     # Path to a custom SED file (overrides SED_SLOPE if provided)

# ---------------------------------------------------------
# Execution Configuration
# ---------------------------------------------------------
CLOUDY_PATH = "path/to/cloudy.exe"
WORKERS = 5
LIMIT = None
TEMPLATE_PATH = None
GRID_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "run"
    
    # Execute setup, run, refresh, or packnpz action
    run_grid(
        grid_dir=GRID_DIR,
        action=action,
        logxi_values=LOGXI_VALUES,
        vturb_values=VTURB_VALUES,
        log_nh_values=LOG_NH_VALUES,
        log_density=LOG_DENSITY,
        template_path=TEMPLATE_PATH,
        cloudy_path=CLOUDY_PATH,
        workers=WORKERS,
        limit=LIMIT,
        make_binaries=True,
        sed_slope=SED_SLOPE,
        sed_file=SED_FILE,
    )
