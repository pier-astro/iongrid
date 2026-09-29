# XSTAR grid recipe

Copy this directory into an ignored `runs/<name>/` directory. Edit `settings.py` there: the three grid axes, fixed density, input luminosity, optional spectral file and any extra XSTAR controls all remain beside the simulation results. `sim-local.py` and `sim-condor.py` share those settings. Install `iongrid[fits]` in the Python environment used for packing and make the XSTAR executable available in that environment.

For a custom SED, place an XSTAR-format text file in this run directory and set `SED_FILE = GRID_DIR / "sed.dat"` in `settings.py`. Setup copies that file to each point and selects `spectrum=file` and `spectun=0`; with `SED_FILE = None`, the default is XSTAR's power law. See [XSTAR products and conversion](../../docs/xstar-outputs.md) for the expected file structure and output choices.

## Local

```bash
python sim-local.py setup     # writes manifest.csv and points/*/xstar-params.json
python sim-local.py run       # executes pending points, with WORKERS and LIMIT from the script
python sim-local.py refresh   # recovers status from existing outputs
python sim-local.py packnpz   # compiles successful FITS products to point caches
python pack-fits.py
python pack-ogip.py
```

Each point runs in its own directory, so concurrent XSTAR processes cannot overwrite one another's `xout_*` files. Standard output, errors, exit code and elapsed time are retained per point. `pack-fits.py` requires a complete rectangular grid.

## HTCondor

```bash
python sim-condor.py condor
condor_submit htc_submit/grid_submission.sub
python sim-condor.py refresh
python sim-condor.py packnpz
python pack-fits.py
python pack-ogip.py
```

The `condor` action writes batch lists, a wrapper and a submit file; it does not submit jobs. Workers need a shared filesystem containing the run directory, an importable `iongrid` installation and XSTAR/HEASoft configured. Set `HEADAS_INIT` to the worker's `headas-init.sh` when the environment is not already initialized. Batch jobs write their own result CSVs and never edit the shared manifest; `refresh` reconstructs its status afterwards.

The supplied output recipe exports total optical depth and outward line/total emission. XSTAR's emission tables follow the predecessor xstar2xspec convention of a 1 kpc reference distance and a 10^38 erg/s reference luminosity. This differs from the supplied CLOUDY recipe's 10 kpc reference, so do not reuse a CLOUDY additive-table normalization without adjusting it. See `docs/schema.md` in the repository for the exact products.

To estimate an additive normalization, edit `L_ION_SIM` (`LUMINOSITY_38 * 1e38` erg/s), `L_ION_TARGET`, distance and covering fraction at the top of `calc-norm.py`, then run:

```bash
python calc-norm.py
```

These numbers are illustrative. See [emission normalization](../../docs/emission-normalization.md) for the cancellation of the XSTAR simulation luminosity and the assumptions behind fixing this value.
