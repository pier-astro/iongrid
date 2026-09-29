# iongrid

Assembling spectral templates for ionised gas can be computationally expensive. `iongrid` runs grids with [Cloudy](https://gitlab.nublado.org/cloudy/cloudy/-/wikis/home) or [XSTAR](https://heasarc.gsfc.nasa.gov/docs/software/xstar/xstar.html), in parallel locally or on HTCondor, and exports portable FITS cubes and OGIP/XSPEC table models.

Simulation parameters and execution settings live in editable Python recipes beside each run, so the setup remains available when a costly grid is revisited.

The package generates rest-frame absorption optical depths and emission spectra as typical for photo-ionisation codes.

To apply outflow geometries or relativistic effects see the [`wine`](https://github.com/pier-astro/wine) code.

## Install

Python 3.10 or newer is required. Install with FITS support in your chosen environment:

```bash
python -m pip install -e '.[fits]'
```

CLOUDY and XSTAR are installed separately. Set the executable path in the selected recipe. HTCondor is optional.

## Start a simulation

Copy either the [CLOUDY recipe](examples/cloudy-sim/README.md) or the [XSTAR recipe](examples/xstar-sim/README.md) into a new simulation directory. Set the grid, source spectrum, executable and parallelism in the copied Python scripts, then run the local or HTCondor path. For runs inside this repository, use `runs/<name>/`; `runs/` is ignored by Git. The scripts and input spectrum kept there record the simulation setup.

The output path is `manifest.csv` → `points/<point_id>/*.npz` → `results/*.fits` → `results/*.amod|*.mmod|*.emod`. Generated products under the bundled example are ignored by Git; recipes and source remain tracked.

## Formats and code

- [`docs/schema.md`](docs/schema.md): common node, cube and OGIP output schema, units and version marker.
- [`docs/cloudy-outputs.md`](docs/cloudy-outputs.md): CLOUDY files read by the compiler and the resolution/runtime choices in the supplied template.
- [`docs/xstar-outputs.md`](docs/xstar-outputs.md): XSTAR files read by the compiler, conversions and reference normalization.
- [`docs/emission-normalization.md`](docs/emission-normalization.md): source luminosity, distance and covering factors for additive tables.
- `src/iongrid/cloudy/`: manifest, local and HTCondor runners, CLOUDY spectrum extraction and cube packing.
- `src/iongrid/xstar/`: the corresponding XSTAR workflow, including custom input SEDs.
- `src/iongrid/cube.py`: common FITS cube writer.
- `src/iongrid/ogip.py`: conversion of a spectral cube into an OGIP/XSPEC table model.

The project is distributed under [GPL-3.0-or-later](LICENSE).
