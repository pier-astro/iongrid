#!/usr/bin/env python3
import sys
from pathlib import Path

from iongrid.cloudy.grid import pack_grid_npz, refresh_grid, run_grid
from iongrid.cloudy.htcondor import write_condor_submission

# Full workflow:
# 1. Edit this file for the grid, Cloudy path, batch size, and cluster needs.
# 2. Run: python sim-condor.py condor
# 3. Submit the generated htc_submit/grid_submission.sub with condor_submit.
# 4. After all jobs finish, run: python sim-condor.py refresh (updates manifest.csv)
# 5. Compile binary .npz files: python sim-condor.py packnpz
#
# Cloudy input/output products are written under points/ exactly like local runs.
# HTCondor submission files, logs, batch lists, and per-batch status summaries
# are written under htc_submit/ in this same folder.

# ---------------------------------------------------------
# Grid Configuration
# ---------------------------------------------------------
LOGXI_VALUES = [3.0, 4.75, 6.5]
VTURB_VALUES = [100.0, 550.0, 1000.0]
LOG_NH_VALUES = [21.0, 22.5, 24.0]

# Fixed parameters
LOG_DENSITY = 10.0
SED_SLOPE = -0.8
SED_FILE = None

# ---------------------------------------------------------
# Execution Configuration
# ---------------------------------------------------------
CLOUDY_PATH = "path/to/cloudy.exe"
TEMPLATE_PATH = None
GRID_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------
# HTCondor Configuration
# ---------------------------------------------------------
BATCH_SIZE = 4 # Number of Cloudy runs per HTCondor job
SHUFFLE_SEED = 12345
REQUEST_CPUS = 1
REQUEST_MEMORY = "3G"
REQUIREMENTS = None  # Add site-specific HTCondor requirements if needed.
PYTHON_PATH = None  # None means: use $PYTHON on the worker, else python3


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "condor"

    if action == "setup":
        run_grid(
            grid_dir=GRID_DIR,
            action="setup",
            logxi_values=LOGXI_VALUES,
            vturb_values=VTURB_VALUES,
            log_nh_values=LOG_NH_VALUES,
            log_density=LOG_DENSITY,
            template_path=TEMPLATE_PATH,
            sed_slope=SED_SLOPE,
            sed_file=SED_FILE,
        )

    elif action == "condor":
        run_grid(
            grid_dir=GRID_DIR,
            action="setup",
            logxi_values=LOGXI_VALUES,
            vturb_values=VTURB_VALUES,
            log_nh_values=LOG_NH_VALUES,
            log_density=LOG_DENSITY,
            template_path=TEMPLATE_PATH,
            sed_slope=SED_SLOPE,
            sed_file=SED_FILE,
        )
        submit_file = write_condor_submission(
            grid_dir=GRID_DIR,
            cloudy_path=CLOUDY_PATH,
            batch_size=BATCH_SIZE,
            shuffle=True,
            seed=SHUFFLE_SEED,
            python_path=PYTHON_PATH,
            request_cpus=REQUEST_CPUS,
            request_memory=REQUEST_MEMORY,
            requirements=REQUIREMENTS,
        )
        print(f"Wrote HTCondor submission: {submit_file}")
        print(f"Submit with: condor_submit {submit_file}")

    elif action == "refresh":
        refresh_grid(GRID_DIR, make_binaries=False)

    elif action == "packnpz":
        pack_grid_npz(GRID_DIR)

    else:
        raise ValueError("action must be 'setup', 'condor', 'refresh', or 'packnpz'")
