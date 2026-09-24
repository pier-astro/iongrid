# Spectral output schema

The engine-specific simulator produces one point cache. The portable interchange product is a FITS spectral cube; an OGIP table is a derived fitting product. These files are independent of the wind model and contain **rest-frame** spectra. The format version is an integer, currently `1`: `schema_version` in each new `.npz` and `IGFMTVER` in the primary header of each FITS cube and OGIP table. This is the *format* version, distinct from the Python package version and from OGIP `HDUVERS`. Readers should check it before interpreting a future version. Existing compatible caches/cubes without the marker can still be packed.

## Point cache (`.npz`)

Each `points/<point_id>/<point_id>.npz` contains one-dimensional arrays on the same increasing `energy` centres (keV):

| Key | Meaning | Unit |
| --- | --- | --- |
| `tau_absorption`, `tau_scattering`, `tau_total` | Absorption, scattering and summed optical depth | dimensionless |
| `emt_continuum`, `emt_lines`, `emt_total` | Diffuse continuum, lines and sum at the 10 kpc reference distance | ph cm⁻² s⁻¹ **per energy bin** |
| `incident`, `transmitted` | CLOUDY incident and transmitted continua retained for diagnostics | CLOUDY continuum units |
| `metadata` | Structured scalar with `point_id`, `logxi`, `vturb`, `log_nh`, `log_density`, `temperature` | as below |
| `schema_version` | Integer format version | — |

`logxi` is log₁₀ of ξ in erg cm s⁻¹; `vturb` is km s⁻¹; `log_nh` is log₁₀ of column density in cm⁻²; `log_density` is log₁₀ of hydrogen density in cm⁻³; temperature is K. The comma-separated `manifest.csv` indexes points and run status (`ready`, `ok`, `failed`) and stores execution time in seconds. It is a run ledger, not the spectral interchange format.

## FITS cube

A cube stores one selected `.npz` key over a complete rectangular `(logxi, vturb, log_nh)` grid. The primary header has `ORIGIN=iongrid`, `ENGINE=CLOUDY`, `IGFMTVER=1`, `NPZKEY` and `QUANTITY`. The `SPECTRA` image is float32 with NumPy shape `(n_logxi, n_vturb, n_log_nh, n_energy)` and `BUNIT` set by the packing recipe. The `GRID` table contains centre arrays `ENERGY` (keV), `LOGXI`, `VTURB` (km s⁻¹) and `LOG_NH`. A cube may contain optical depth or emission; the quantity and units determine how it is used. The writer checks that successful points fill the Cartesian grid and share the same energy axis.

## OGIP/XSPEC table

`write_ogip_table` converts a cube into a standard `PARAMETERS`, `ENERGIES`, `SPECTRA` table. Parameter axes with a single value are fixed and omitted from `PARAMETERS`. Parameter `METHOD=0` uses linear axis coordinates, `METHOD=1` logarithmic coordinates; these settings do **not** specify interpolation of spectral values. Spectral values in XSPEC tables are interpolated linearly by XSPEC.

- `model_type="exponential"`: stores optical depth for an XSPEC `etable` (`.emod`).
- `model_type="multiplicative"`: stores `exp(-optical depth)` for an XSPEC `mtable` (`.mmod`).
- `model_type="additive"`: stores bin-integrated photon flux for an XSPEC `atable` (`.amod`), normally `ph cm⁻² s⁻¹` at 10 kpc before table normalization.

The primary header also carries `IGFMTVER=1`, `ORIGIN=iongrid`, OGIP identifiers and `XFXP0001` for the explicit model type. Energy bins are reconstructed from the cube's energy centres. WINE's Python reader can consume these OGIP tables; its separate spectral-value interpolation default is logarithmic.
