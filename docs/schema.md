# Spectral output schema

Each simulator produces one cache per grid point. The portable interchange product is a FITS spectral cube; an OGIP table is a derived fitting product. All spectra are in the source rest frame, before WINE applies the wind geometry. The format version is currently `1`: `schema_version` in each new `.npz`, and `IGFMTVER` in each cube and OGIP table. It is independent of the Python package version and OGIP `HDUVERS`. Readers should check it before interpreting future formats. Older compatible caches and cubes without the marker can still be packed.

## Point cache (`.npz`)

Each `points/<point_id>/<point_id>.npz` contains an increasing one-dimensional `energy` axis of bin centres in keV. Spectral arrays have that same shape. The engine adapters publish the following keys where available:

| Key | Meaning | Unit |
| --- | --- | --- |
| `tau_total` | Total line-of-sight optical depth | dimensionless |
| `tau_absorption`, `tau_scattering` | CLOUDY absorption and scattering components | dimensionless |
| `emt_continuum`, `emt_lines`, `emt_total` | Diffuse emission components and total at the engine's stated reference | photons cm⁻² s⁻¹ per energy bin |
| `incident`, `transmitted` | Input and transmitted spectra retained for diagnostics | native simulator units |
| `metadata` | Scalar point parameters; fields vary by engine | see below |
| `schema_version` | Integer format version | — |

The grid coordinates in `metadata` are `point_id`, `logxi`, `vturb` and `log_nh`; adapters can add fields such as `log_density`, temperature or input luminosity. `logxi` is log₁₀ of ξ in erg cm s⁻¹; `vturb` is km s⁻¹; `log_nh` is log₁₀ of column density in cm⁻². The comma-separated `manifest.csv` indexes point status (`ready`, `ok`, `failed`) and execution time in seconds. It is a run ledger, not the spectral interchange format.

The supplied CLOUDY extraction expresses emission at 10 kpc. The XSTAR extraction follows the predecessor XSTAR-to-XSPEC convention: bin-integrated photon flux at 1 kpc for a reference luminosity of 10³⁸ erg s⁻¹. XSTAR caches also store `emission_reference_kpc` and `emission_reference_luminosity_erg_s`. These emission normalizations are different and must be accounted for when comparing additive spectra from the two engines. Optical depth is dimensionless.

## FITS cube

A cube stores one selected `.npz` key over a complete rectangular `(logxi, vturb, log_nh)` grid. Its primary header has `ORIGIN=iongrid`, `ENGINE=CLOUDY` or `XSTAR`, `IGFMTVER=1`, `NPZKEY` and `QUANTITY`. `REFDIST` (kpc) records the emission reference distance; XSTAR cubes also have `REFLUM` (erg s⁻¹). The `SPECTRA` image is float32 with NumPy shape `(n_logxi, n_vturb, n_log_nh, n_energy)` and `BUNIT` set by the packing recipe. The `GRID` table contains centre arrays `ENERGY` (keV), `LOGXI`, `VTURB` (km s⁻¹) and `LOG_NH`. The writer requires successful points to fill the Cartesian grid and share the energy axis.

## OGIP/XSPEC table

`write_ogip_table` converts a cube into a standard `PARAMETERS`, `ENERGIES`, `SPECTRA` table. It carries `ENGINE`, `REFDIST` and `REFLUM` through when present. Axes with a single value are fixed and omitted from `PARAMETERS`; at least one axis must vary. Parameter `METHOD=0` uses linear axis coordinates and `METHOD=1` logarithmic coordinates. These settings do **not** control interpolation of spectral values. XSPEC tables interpolate spectral values linearly; WINE's Python reader separately offers spectral-value interpolation, with a logarithmic default.

- `model_type="exponential"`: optical depth for an XSPEC `etable` (`.emod`).
- `model_type="multiplicative"`: `exp(-optical depth)` for an XSPEC `mtable` (`.mmod`).
- `model_type="additive"`: bin-integrated photon flux for an XSPEC `atable` (`.amod`), in photons cm⁻² s⁻¹ at the reference stated by the cube header, before table normalization.

The primary header also carries OGIP identifiers and `XFXP0001` for the explicit model type. Energy-bin edges are reconstructed from the cube's energy centres.
