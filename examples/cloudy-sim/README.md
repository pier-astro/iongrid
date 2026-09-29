# CLOUDY simulation recipe

Copy this directory to a separate run folder before changing its scripts. For a local run inside this repository, use `mkdir -p runs && cp -R examples/cloudy-sim runs/my-grid`; `runs/` is ignored by Git. Edit `LOGXI_VALUES`, `VTURB_VALUES`, `LOG_NH_VALUES`, `LOG_DENSITY`, `LOG_L_ION`, the incident SED and `CLOUDY_PATH` in the script you use. Paths are relative to the script's directory, so the scripts themselves remain a readable record of the run. `TEMPLATE_PATH = None` copies the packaged `cloudy-template.in` to the run folder on first setup; edit that local copy to customise the CLOUDY deck.

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

The supplied `pack-fits.py` writes absorption optical depth and line emission cubes. Edit its `npz_key`, `quantity` and `unit` arguments to export other products (`tau_scattering`, `tau_total`, `emt_continuum`, `emt_total`). Native luminosity-case `incident` and `transmitted` diagnostics remain in the `.npz`: their values can exceed the float32 FITS cube range. The `pack-ogip.py` recipe creates an exponential optical-depth table (`.emod`), a multiplicative transmission table (`.mmod`) and an additive line-emission table (`.amod`). It records XSPEC axis interpolation choices with `METHOD`; WINE's Python spectral-value interpolation is configured separately.

Emission is stored as bin-integrated photon flux at 10 kpc. The default template fixes the 1–1000 Ryd source luminosity at `LOG_L_ION` (initially 48) across the grid. For each `LOGXI_VALUES` node, setup computes `radius` from $r=\sqrt{L_{\rm ion}/(\xi n_H)}$; $N_H$ and turbulence remain independent axes. Edit the variables at the top of `calc-norm.py` to match the simulation and target source, then run:

```bash
python calc-norm.py
```

The bundled target values are illustrative. A common source luminosity, distance and global covering fraction now imply one physical normalization across the `logxi` axis. Check the smallest radius at the highest `logxi` and largest $N_H$ before using a grid where geometric dilution matters. See [emission normalization](../../docs/emission-normalization.md).

The compiler identifies the native emission units from each `.out` log and converts integrated luminosity to full-covering photon flux at 10 kpc. If the log lacks an emission-unit heading, it checks the retained `.in` deck. For other brightness commands, add an explicit `# iongrid emission input: intensity` or `# iongrid emission input: luminosity` comment to that deck after checking its output units. If you supply a custom `xi` intensity deck instead of `luminosity`, set `CLOUDY_INPUT = "xi"` in `calc-norm.py` and provide its ionization, density and radius; its implied luminosity can vary across nodes.

See [`../../docs/schema.md`](../../docs/schema.md) for units and [`../../docs/cloudy-outputs.md`](../../docs/cloudy-outputs.md) for the CLOUDY input/output rationale.
