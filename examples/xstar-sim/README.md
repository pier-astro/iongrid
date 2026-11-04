# XSTAR campaign

Copy this directory into your run workspace. Edit `settings.py` and run
`sim-local.py` from VS Code or Python. It prepares a new campaign or resumes
the existing manifest. Set `SELECT = "failed"` in the driver to retry failed
points.

For HTCondor, edit the native submit directives in `sim-condor.py` and run
the script to generate a submit file. Submit that file, then set
`ACTION = "refresh"` to reconcile results. Set `INIT` to a worker
environment script if needed.

Run `pack-fits.py` to collect and pack the available spectra. Run
`pack-ogip.py` after a complete Cartesian campaign to build XSPEC tables.
Use `calc-norm.py` to calculate the additive model normalization.

See the [campaign guide](../../docs/campaign.md) for SEDs, sampling, and
campaign extension.

Edit the supplied `xstar.par` for native solver options. Set `SAMPLING = "lhs"`
for a scattered design, and use `CLEAN` to select raw, cleaned, or both
optical-depth and line-emission products.
