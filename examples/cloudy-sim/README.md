# CLOUDY simulation recipe

Copy this directory to a separate run folder before changing its scripts. For a local run inside this repository, use `mkdir -p runs && cp -R examples/cloudy-sim runs/my-grid`; `runs/` is ignored by Git. Edit `LOGXI_VALUES`, `VTURB_VALUES`, `LOG_NH_VALUES`, `LOG_DENSITY`, the incident SED and `CLOUDY_PATH` in the script you use. Paths are relative to the script's directory, so the scripts themselves remain a readable record of the run. `TEMPLATE_PATH = None` copies the packaged `cloudy-template.in` to the run folder on first setup; edit that local copy to customise the CLOUDY deck.

## Local run

```bash
python sim-local.py setup
python sim-local.py run
python sim-local.py refresh
python sim-local.py packnpz
```

`run` executes pending points, refreshes `manifest.csv` and compiles successful points to `.npz`. `refresh` rescans CLOUDY `.out` files; `packnpz` recompiles successful nodes without rerunning CLOUDY. `WORKERS` sets local parallelism; `LIMIT` allows a short trial run. The manifest and all raw products remain in this folder.

## HTCondor run

```bash
python sim-condor.py condor
condor_submit htc_submit/grid_submission.sub
python sim-condor.py refresh
python sim-condor.py packnpz
```

Set the worker Python path and HTCondor requirements in `sim-condor.py`. The worker must be able to import `iongrid`. Job results go to `htc_submit/`; the ordered manifest remains in the run folder. `BATCH_SIZE` controls points per job.

## Export spectra

After every point has succeeded and been compiled:

```bash
python pack-fits.py
python pack-ogip.py
```

The supplied `pack-fits.py` writes absorption optical depth and line emission cubes. Edit its `npz_key`, `quantity` and `unit` arguments to export other products (`tau_scattering`, `tau_total`, `emt_continuum`, `emt_total`, `incident`, `transmitted`). The `pack-ogip.py` recipe creates an exponential optical-depth table (`.emod`), a multiplicative transmission table (`.mmod`) and an additive line-emission table (`.amod`). It records XSPEC axis interpolation choices with `METHOD`; WINE's Python spectral-value interpolation is configured separately.

Emission is stored as bin-integrated photon flux at 10 kpc. For a source at redshift `z`, the suggested initial XSPEC additive normalization is `(10 kpc / D_L(z))²`. For example:

```bash
python calc-norm.py 0.184
```

With no argument, `calc-norm.py` retains the original example redshift `0.184`. The calculation and cosmology are unchanged; set your own source redshift explicitly for a new analysis.

See [`../../docs/schema.md`](../../docs/schema.md) for units and [`../../docs/cloudy-outputs.md`](../../docs/cloudy-outputs.md) for the CLOUDY input/output rationale.
