# XSTAR campaign

Copy this directory into your run workspace (for runs inside the repository,
use `runs/<name>/`, which Git ignores). Everything that defines the campaign
lives in the copied scripts, so the setup stays next to the results.

## 1. Common setup

1. Activate the Python environment where `iongrid[fits]` is installed. The
   HTCondor workflow uses the interpreter that runs `sim-condor.py`, so
   activate the environment before running it.
2. Edit `settings.py`:
   - `EXECUTABLE`: path to the `xstar` binary.
   - `SAMPLING`: `"cartesian"` for a regular grid, or `"lhs"` for a scattered
     design. For `"lhs"`, several `lhs(...)` calls can be joined with `+`
     (use `anchors=False` on later calls to avoid repeating the boundary
     points).
   - `CLEAN`: `"raw"`, `"clean"` or `"both"` optical-depth and line-emission
     products.
3. Edit the supplied `xstar.par` as needed.

## 2a. Run locally

1. Run `sim-local.py`. It prepares a new campaign or
   resumes the existing manifest.
2. Set `SELECT = "failed"` in the script to retry failed points.

## 2b. Run on HTCondor

Workers need a shared filesystem containing the run directory, and HEASoft
and the Python environment must be reachable from them.

1. In `sim-condor.py`, set `ACTION = "write"`, then edit:
   - `INIT`: path to the HEASoft `headas-init.sh` (it is sourced on the worker
     before the job starts).
   - `DIRECTIVES`: native submit directives (CPUs, memory, extra
     `environment` entries, and so on).
   - `BATCH`: number of simulations per job.
   - `SELECT`: `"ready"` for new points, `"failed"` to retry failed ones.
2. Run `python sim-condor.py`. It generates a `condor-*/` directory with a
   submit file and prints the `condor_submit` command. It does not submit.
3. Submit the printed file with `condor_submit`.
4. When the jobs have finished, set `ACTION = "refresh"` and run
   `python sim-condor.py` to reconcile the manifest with the results. To
   resubmit unfinished points, go back to `ACTION = "write"` with the
   appropriate `SELECT`.

## 3. Pack the results (local and HTCondor)

1. Run `pack-fits.py` to collect and pack the available spectra.
2. Run `pack-ogip.py` after a complete Cartesian campaign to build XSPEC
   tables.
3. Use `calc-norm.py` to calculate the additive model normalization.

See the [campaign guide](../../docs/campaign.md) for SEDs, sampling, and
campaign extension.
