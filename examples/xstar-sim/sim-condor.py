#!/usr/bin/env python3
"""Prepare HTCondor batches; submit explicitly after inspecting the files."""

import sys
from iongrid.xstar.grid import run_grid
from iongrid.xstar.htcondor import write_condor_submission
from settings import GRID_DIR, SETUP, XSTAR_PATH
import subprocess
from pathlib import Path

BATCH_SIZE = 1
PYTHON_PATH = "python3"  # Must import iongrid on worker nodes.
XSTAR_PATH = "/data-fast/dragon/software/heasoft-6.36/x86_64-pc-linux-gnu-libc2.39/bin/xstar"
HEADAS_INIT = "/data-fast/dragon/software/heasoft-6.36/x86_64-pc-linux-gnu-libc2.39/headas-init.sh"
REQUEST_CPUS = 1
REQUEST_MEMORY = "3G"
REQUIREMENTS = "(OpSysMajorVer == 24)"

if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "condor"
    if action == "condor":
        # Automatically generate the LHS grid before XSTAR setup (the LHS configuration is read from settings.py by make-lhs.py):
        lhs_file = SETUP.get("lhs_file")
        if lhs_file is not None:
            script_dir = Path(__file__).resolve().parent
            lhs_script = script_dir / "make-lhs.py"
            if not lhs_script.is_file():
                raise FileNotFoundError(f"LHS generator not found: {lhs_script}")
            print("=" * 72)
            print("LHS GRID GENERATION")
            print("=" * 72)
            print(f"Running: {lhs_script}")
            # Regenerate from settings.py so configuration changes are applied.
            subprocess.run([sys.executable, str(lhs_script)], cwd=script_dir, check=True)
            lhs_path = Path(lhs_file).expanduser().resolve()
            if not lhs_path.is_file():
                raise FileNotFoundError(f"LHS grid generation finished, but the CSV expected by SETUP['lhs_file'] was not found: {lhs_path}")
            print(f"LHS grid ready: {lhs_path}")
            print("=" * 72)
        
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
