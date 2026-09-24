# iongrid

`iongrid` builds spectral grids from photoionisation simulations and exports portable FITS cubes and OGIP/XSPEC table models. The current engine is CLOUDY. Simulation parameters and execution settings live in editable Python recipes beside each run, so the setup remains available when a costly grid is revisited.

The package generates rest-frame absorption optical depths and emission spectra.

To apply outflow geometries or relativistic effects see the [`wine`](https://github.com/pier-astro/wine) code.

## Install

Python 3.10 or newer is required. Install with FITS support in your chosen environment:

```bash
python -m pip install -e '.[fits]'
```

CLOUDY is installed separately. Set its executable path in the recipe used for a run. HTCondor is optional.

## Start a simulation

Copy [`examples/cloudy-sim/`](examples/cloudy-sim/) to a new simulation directory, edit the axis values, SED, CLOUDY executable and parallelism in `sim-local.py` or `sim-condor.py`, then run the selected script there. For runs inside this repository, use `runs/<name>/`; `runs/` is ignored by Git. The copied scripts and input template are the record of that simulation. See the [recipe guide](examples/cloudy-sim/README.md) for local, HTCondor and packaging steps.

The output path is `manifest.csv` → `points/<point_id>/*.npz` → `results/*.fits` → `results/*.amod|*.mmod|*.emod`. Generated products under the bundled example are ignored by Git; recipes and source remain tracked.

## Formats and code

- [`docs/schema.md`](docs/schema.md): common node, cube and OGIP output schema, units and version marker.
- [`docs/cloudy-outputs.md`](docs/cloudy-outputs.md): CLOUDY files read by the compiler and the resolution/runtime choices in the supplied template.
- `src/iongrid/cloudy/`: manifest, local and HTCondor runners, CLOUDY spectrum extraction and cube packing.
- `src/iongrid/ogip.py`: conversion of a spectral cube into an OGIP/XSPEC table model.

The project is distributed under [GPL-3.0-or-later](LICENSE).
