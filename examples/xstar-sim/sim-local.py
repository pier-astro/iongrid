#!/usr/bin/env python3
"""Set up, execute or compile the XSTAR grid in this directory."""

import sys
from iongrid.xstar.grid import run_grid
from settings import GRID_DIR, SETUP, XSTAR_PATH

WORKERS = 4
LIMIT = None  # Set to 1 for a trial run.
TIMEOUT = None  # Seconds per point, or None to wait for convergence.

if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "setup"
    run_grid(GRID_DIR, action, workers=WORKERS, limit=LIMIT,
             xstar_path=XSTAR_PATH, timeout=TIMEOUT, **SETUP)
