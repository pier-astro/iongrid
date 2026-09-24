#!/usr/bin/env python3
"""Prepare HTCondor batches; submit explicitly after inspecting the files."""

import sys
from iongrid.xstar.grid import run_grid
from iongrid.xstar.htcondor import write_condor_submission
from settings import GRID_DIR, SETUP, XSTAR_PATH

BATCH_SIZE = 1
PYTHON_PATH = "python3"  # Must import iongrid on worker nodes.
HEADAS_INIT = None  # Set to the worker's headas-init.sh if needed.
REQUEST_CPUS = 1
REQUEST_MEMORY = "3G"
REQUIREMENTS = None

if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "condor"
    if action == "condor":
        run_grid(GRID_DIR, "setup", **SETUP)
        submit = write_condor_submission(
            GRID_DIR, xstar_path=XSTAR_PATH, batch_size=BATCH_SIZE,
            python_path=PYTHON_PATH, headas_init=HEADAS_INIT,
            request_cpus=REQUEST_CPUS, request_memory=REQUEST_MEMORY,
            requirements=REQUIREMENTS,
        )
        print(f"Inspect and submit with: condor_submit {submit}")
    elif action in {"setup", "refresh", "packnpz"}:
        run_grid(GRID_DIR, action, **SETUP)
    else:
        raise SystemExit("action must be setup, condor, refresh, or packnpz")
