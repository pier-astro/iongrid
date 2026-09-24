from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np

from . import SCHEMA_VERSION


def write_ogip_table(
    cube_path: str | Path,
    output_path: str | Path,
    *,
    model_name: str,
    model_type: str,
    model_unit: str = "",
    redshift: bool = True,
    interpolation: str | dict[str, str] = "linear",
    overwrite: bool = False,
) -> Path:
    """Convert an iongrid FITS cube into an OGIP/XSPEC table model.

    Parameters
    ----------
    cube_path
        FITS cube written by an iongrid engine adapter.
    output_path
        Output XSPEC table model path.
    model_name
        XSPEC model name stored in the FITS header.
    model_type
        One of ``"additive"``, ``"multiplicative"``, or ``"exponential"``.
        Use additive for emission, multiplicative for transmission, and
        exponential for raw optical depth tables used through XSPEC ``etable``.
    model_unit
        Units for additive models. Leave empty for multiplicative/exponential.
    redshift
        Whether XSPEC should add a redshift parameter.
    interpolation
        ``"linear"`` or ``"log"`` for all gridded parameters, or a dictionary
        keyed by parameter name. XSPEC stores these as METHOD 0 and 1.
    overwrite
        Replace an existing output file.
    """
    try:
        from astropy.io import fits
    except ImportError as e:
        raise ImportError("Writing XSPEC table models requires astropy.") from e

    model_type = model_type.lower()
    if model_type not in {"additive", "multiplicative", "exponential"}:
        raise ValueError("model_type must be 'additive', 'multiplicative', or 'exponential'")

    cube_path = Path(cube_path)
    output_path = Path(output_path)
    if cube_path.resolve() == output_path.resolve():
        raise ValueError("cube_path and output_path must be different files")

    try:
        hdul = fits.open(cube_path, memmap=True)
    except OSError as e:
        raise ValueError(f"Input is not a readable packed FITS cube: {cube_path}") from e

    with hdul:
        if "GRID" not in hdul or "SPECTRA" not in hdul:
            raise ValueError(f"Input FITS cube must contain GRID and SPECTRA HDUs: {cube_path}")
        if "IGFMTVER" in hdul[0].header and int(hdul[0].header["IGFMTVER"]) != SCHEMA_VERSION:
            raise ValueError(f"Unsupported iongrid schema version in {cube_path}")

        energy = np.atleast_1d(np.asarray(hdul["GRID"].data["ENERGY"][0], dtype=float))
        logxi = np.atleast_1d(np.asarray(hdul["GRID"].data["LOGXI"][0], dtype=float))
        vturb = np.atleast_1d(np.asarray(hdul["GRID"].data["VTURB"][0], dtype=float))
        log_nh = np.atleast_1d(np.asarray(hdul["GRID"].data["LOG_NH"][0], dtype=float))
        spectra = hdul["SPECTRA"].data

        expected_shape = (len(logxi), len(vturb), len(log_nh), len(energy))
        if spectra.shape != expected_shape:
            raise ValueError(f"SPECTRA shape {spectra.shape} does not match GRID axes {expected_shape}")

        primary = fits.PrimaryHDU()
        primary.header["HDUCLASS"] = "OGIP"
        primary.header["HDUCLAS1"] = "XSPEC TABLE MODEL"
        primary.header["HDUVERS"] = "1.1.0"
        primary.header["MODLNAME"] = model_name
        primary.header["MODLUNIT"] = model_unit
        primary.header["REDSHIFT"] = bool(redshift)
        primary.header["ADDMODEL"] = model_type == "additive"
        primary.header["LOELIMIT"] = 0.0
        primary.header["HIELIMIT"] = 0.0
        primary.header["XFXP0001"] = model_type
        primary.header["SOURCE"] = cube_path.name
        primary.header["ORIGIN"] = "iongrid"
        primary.header["IGFMTVER"] = SCHEMA_VERSION
        for key in ("ENGINE", "REFDIST", "REFLUM"):
            if key in hdul[0].header:
                primary.header[key] = hdul[0].header[key]

        all_names = ["logxi", "vturb", "lognH"]
        all_units = ["", "km/s", "cm-2"]
        all_axes = [logxi, vturb, log_nh]
        par_indices = [i for i, axis in enumerate(all_axes) if len(axis) > 1]
        if not par_indices:
            raise ValueError("XSPEC table needs at least one grid axis with more than one value")

        par_names = [all_names[i] for i in par_indices]
        par_units = [all_units[i] for i in par_indices]
        par_axes = [all_axes[i] for i in par_indices]
        parameters = _parameters_hdu(
            names=par_names,
            values=par_axes,
            units=par_units,
            methods=[_interpolation_method(interpolation, name) for name in par_names],
        )
        energies = _energies_hdu(_energy_edges(energy))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fits.HDUList([primary, parameters, energies]).writeto(output_path, overwrite=overwrite)
        _append_empty_spectra_table(output_path, nrows=int(np.prod([len(axis) for axis in all_axes])), npars=len(par_axes), nbins=len(energy))

        with fits.open(output_path, mode="update", memmap=True) as out_hdul:
            table = out_hdul["SPECTRA"].data
            row = 0
            total = int(np.prod([len(axis) for axis in all_axes]))
            for index in _progress(product(range(len(logxi)), range(len(vturb)), range(len(log_nh))), total):
                i, j, k = index
                values = [all_axes[n][index[n]] for n in par_indices]
                if len(values) == 1:
                    table["PARAMVAL"][row] = values[0]
                else:
                    table["PARAMVAL"][row] = values
                spec_val = np.asarray(spectra[i, j, k, :], dtype=np.float32)
                if model_type == "multiplicative":
                    spec_val = np.exp(-spec_val)
                table["INTPSPEC"][row] = spec_val
                row += 1
            out_hdul.flush()

    return output_path


def _parameters_hdu(names, values, units, methods):
    from astropy.io import fits

    max_values = max(len(v) for v in values)
    value_arrays = np.array([np.asarray(vals, dtype=np.float32) for vals in values], dtype=object)

    columns = [
        fits.Column(name="NAME", format="12A", array=np.asarray(names)),
        fits.Column(name="UNITS", format="12A", array=np.asarray(units)),
        fits.Column(name="METHOD", format="J", array=np.asarray(methods, dtype=np.int32)),
        fits.Column(name="INITIAL", format="E", array=np.array([v[0] for v in values], dtype=np.float32)),
        fits.Column(name="DELTA", format="E", array=np.array([_delta(v) for v in values], dtype=np.float32)),
        fits.Column(name="MINIMUM", format="E", array=np.array([v[0] for v in values], dtype=np.float32)),
        fits.Column(name="BOTTOM", format="E", array=np.array([v[0] for v in values], dtype=np.float32)),
        fits.Column(name="TOP", format="E", array=np.array([v[-1] for v in values], dtype=np.float32)),
        fits.Column(name="MAXIMUM", format="E", array=np.array([v[-1] for v in values], dtype=np.float32)),
        fits.Column(name="NUMBVALS", format="J", array=np.array([len(v) for v in values], dtype=np.int32)),
        fits.Column(name="VALUE", format=f"PE({max_values})", array=value_arrays),
    ]
    hdu = fits.BinTableHDU.from_columns(columns, name="PARAMETERS")
    hdu.header["HDUCLASS"] = "OGIP"
    hdu.header["HDUCLAS1"] = "XSPEC TABLE MODEL"
    hdu.header["HDUCLAS2"] = "PARAMETERS"
    hdu.header["HDUVERS"] = "1.0.0"
    hdu.header["NINTPARM"] = len(names)
    hdu.header["NADDPARM"] = 0
    return hdu


def _energies_hdu(edges):
    from astropy.io import fits

    hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="ENERG_LO", format="E", unit="keV", array=edges[:-1].astype(np.float32)),
            fits.Column(name="ENERG_HI", format="E", unit="keV", array=edges[1:].astype(np.float32)),
        ],
        name="ENERGIES",
    )
    hdu.header["HDUCLASS"] = "OGIP"
    hdu.header["HDUCLAS1"] = "XSPEC TABLE MODEL"
    hdu.header["HDUCLAS2"] = "ENERGIES"
    hdu.header["HDUVERS"] = "1.0.0"
    return hdu


def _append_empty_spectra_table(output_path, nrows, npars, nbins):
    from astropy.io import fits

    row_bytes = 4 * (npars + nbins)
    data_bytes = row_bytes * nrows
    padding = (-data_bytes) % 2880

    header = fits.Header()
    header["XTENSION"] = "BINTABLE"
    header["BITPIX"] = 8
    header["NAXIS"] = 2
    header["NAXIS1"] = row_bytes
    header["NAXIS2"] = nrows
    header["PCOUNT"] = 0
    header["GCOUNT"] = 1
    header["TFIELDS"] = 2
    header["TTYPE1"] = "PARAMVAL"
    header["TFORM1"] = f"{npars}E"
    header["TTYPE2"] = "INTPSPEC"
    header["TFORM2"] = f"{nbins}E"
    header["EXTNAME"] = "SPECTRA"
    header["HDUCLASS"] = "OGIP"
    header["HDUCLAS1"] = "XSPEC TABLE MODEL"
    header["HDUCLAS2"] = "MODEL SPECTRA"
    header["HDUVERS"] = "1.0.0"

    with Path(output_path).open("r+b") as f:
        f.seek(0, 2)
        f.write(header.tostring().encode("ascii"))
        f.seek(data_bytes + padding - 1, 1)
        f.write(b"\0")


def _energy_edges(centers):
    centers = np.asarray(centers, dtype=float)
    if len(centers) < 2:
        raise ValueError("energy grid must contain at least two bins")

    edges = np.empty(len(centers) + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    if edges[0] <= 0.0:
        edges[0] = centers[0] * np.sqrt(centers[0] / centers[1])
    return edges


def _delta(values):
    if len(values) < 2:
        return 0.1
    return 0.1 * float(values[1] - values[0])


def _interpolation_method(interpolation, name):
    value = interpolation.get(name, "linear") if isinstance(interpolation, dict) else interpolation
    value = value.lower()
    if value in {"linear", "lin"}:
        return 0
    if value in {"log", "logarithmic"}:
        return 1
    raise ValueError("interpolation must be 'linear', 'log', or a dict using those values")


def _progress(items, total):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return items
    return tqdm(items, total=total, desc="Writing XSPEC spectra", unit="spec")
