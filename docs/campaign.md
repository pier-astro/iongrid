# Simulation campaigns

Copy the [CLOUDY recipe](../examples/cloudy-sim/) or [XSTAR recipe](../examples/xstar-sim/)
to a run directory. Edit `settings.py` and run `sim-local.py` from VS Code or
Python. It prepares a new campaign or resumes an existing one. Set
`EXECUTABLE` to the simulator path. Use `TEMPLATE` to select a native deck or
parameter file.

`params` is a list of dictionaries. Each dictionary contains the numeric
coordinates for one simulation. In `cartesian()`, a scalar is fixed and an
array gives the exact axis values. In `lhs()`, pairs give bounds and
`samples` gives the total number of rows. `anchors=True` includes two
boundary points per sampled dimension in that count. Use `logarithmic` for
positive linear coordinates that should be sampled uniformly in their
logarithm. Already-logarithmic coordinates such as `lognh` use ordinary
sampling. Combine row lists with `+`.

Coordinates use base-10 logarithms for hydrogen column in cm⁻² (`lognh`),
hydrogen density in cm⁻³ (`logdensity`), ionization parameter in erg cm s⁻¹
(`logxi`), and 1–1000 Ryd ionizing luminosity in erg/s (`logluminosity`).
`vturb` is in km/s and `feabund` scales iron relative to the selected
engine's abundance mixture. The engine adapter handles native conversions.

## Incident spectrum

Use `native()` to pass SED controls to the engine. For native power laws the
photon index `gamma` maps to the energy index `1 - gamma`. CLOUDY also accepts
its native `slope` directly. Use `table()`
explicitly for file SEDs.

```python
from iongrid.sed import table

sed = table("reference.sed")
sed = table((energy, photons))
sed = table(files, axes=("gamma", "ktbb"))
sed = table(make_sed, axes=("gamma", "ktbb"))
```

A mapping associates exact coordinate keys with native files. A callable
receives the coordinates in the specified order and returns energy and photon
density arrays. Supply increasing energies in keV and relative photons per keV,
covering the ionizing band. The engine normalizes to the requested luminosity.
CLOUDY's log-interpolated table requires strictly positive photon densities.
An existing XSTAR file uses the `spectun` declared in the source parameter file.
The adapters write engine-specific tables during preparation. Workers consume the
staged inputs. Set the SED through `{sed_command}` in a templated CLOUDY deck.

## Resume or extend

```python
from iongrid import Grid

grid = Grid.open(".")
grid.refresh()
print(grid.status())
grid.extend(additional)
grid.run(workers=4)  # Select "failed" explicitly when retrying.
errors = grid.collect()
grid.pack("tau.fits", key="tau_total", quantity="Optical depth", unit="")
```

The saved campaign and per-point inputs are the execution record. The manifest
reports total, completed, and failed counts at its last update. `status()`
also returns pending and collected counts. The pending count includes work
that may still be queued or running.

`extend()` skips duplicate rows and retains completed results. A previously
fixed parameter can become an axis by supplying new values in additional
rows. Use the original parameter columns. For a new tabulated SED key,
pass `sed=table(...)` to `extend()`. Previously staged keys can be reused
without the original callable. Collection tracks failures separately from
simulation and can be retried on its own.

To export a cleaned XSTAR optical-depth table alongside a raw one, apply the
existing cleaning hook only when packing. The header records the thresholds.

```python
from iongrid.xstar.clean import make_postprocess

clean, header = make_postprocess("tau")
grid.pack("tau-clean.fits", key="tau_total", quantity="Optical depth", unit="",
          postprocess=clean, header=header)
```

## HTCondor

```python
from iongrid import condor

submit = condor.write(grid, batch=16, fragment="site.sub", init="cluster.sh")
print(submit)
```

Use native submit directives in the fragment, such as `requirements` and
`request_memory`. The generator owns the executable, arguments, initial
directory, and queue. It assumes a shared filesystem. Inspect and submit the
generated file, then call `grid.refresh()` to reconcile worker outcomes.

## Standalone points

Construct `CloudyPoint` or `XstarPoint`, then call `prepare(directory)`,
`run(directory)`, and `collect(directory)`. Pass `template=` for a native
file or `text=` for inline input. To collect CLOUDY output, use the standard
output filenames from the packaged template and supply the physical
coordinates on the point. XSTAR retains the source parameter file and
resolved overrides and uses a private writable `PFILES` directory.
