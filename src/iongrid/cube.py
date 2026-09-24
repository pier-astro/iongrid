from __future__ import annotations

from pathlib import Path

import numpy as np

from . import SCHEMA_VERSION


def read_fits_cube_grid(
    points: list,
    root: str | Path,
    npz_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """Read grid axes and validate that all compiled nodes are packable."""
    root = Path(root)
    points = [pt for pt in points if pt.status == "ok"]
    if not points:
        raise ValueError("manifest has no successful points to pack")

    logxi = np.array(sorted({pt.logxi for pt in points}), dtype=float)
    vturb = np.array(sorted({pt.vturb for pt in points}), dtype=float)
    log_nh = np.array(sorted({pt.log_nh for pt in points}), dtype=float)
    expected_count = len(logxi) * len(vturb) * len(log_nh)
    if len(points) != expected_count:
        raise ValueError(f"successful points do not form a complete rectangular grid: {len(points)} of {expected_count}")

    xi_idx = {value: i for i, value in enumerate(logxi)}
    vt_idx = {value: i for i, value in enumerate(vturb)}
    nh_idx = {value: i for i, value in enumerate(log_nh)}

    energy = None
    seen = set()
    for pt in points:
        npz_path = pt.dirpath(root) / f"{pt.point_id}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing compiled node cache: {npz_path}")

        with np.load(npz_path) as data:
            if npz_key not in data:
                raise KeyError(f"{npz_path} does not contain {npz_key!r}")
            if "schema_version" in data and int(data["schema_version"]) != SCHEMA_VERSION:
                raise ValueError(f"Unsupported iongrid schema version in {npz_path}")
            node_energy = np.asarray(data["energy"], dtype=float)
            if np.asarray(data[npz_key]).shape != node_energy.shape:
                raise ValueError(f"Spectrum and energy grid have different shapes in {npz_path}")

        if energy is None:
            energy = node_energy
        elif len(node_energy) != len(energy) or not np.allclose(node_energy, energy, rtol=1e-6, atol=0.0):
            raise ValueError(f"Energy grid mismatch in {npz_path}")

        index = (xi_idx[pt.logxi], vt_idx[pt.vturb], nh_idx[pt.log_nh])
        if index in seen:
            raise ValueError(f"Duplicate grid point in manifest: {pt.point_id}")
        seen.add(index)

    return logxi, vturb, log_nh, energy, points


def write_fits_cube(
    points: list,
    root: str | Path,
    output_path: str | Path,
    *,
    npz_key: str,
    quantity: str,
    unit: str,
    overwrite: bool = False,
    engine: str,
    reference_distance_kpc: float | None = None,
    reference_luminosity_erg_s: float | None = None,
) -> Path:
    """Write one FITS cube from a named spectrum stored in each node ``.npz``."""
    try:
        from astropy.io import fits
    except ImportError as e:
        raise ImportError("Writing FITS cubes requires astropy.") from e

    root = Path(root)
    logxi, vturb, log_nh, energy, points = read_fits_cube_grid(points, root, npz_key)

    primary = fits.PrimaryHDU()
    primary.header["ORIGIN"] = "iongrid"
    primary.header["ENGINE"] = engine.upper()
    primary.header["IGFMTVER"] = SCHEMA_VERSION
    primary.header["NPZKEY"] = npz_key
    primary.header["QUANTITY"] = quantity
    if reference_distance_kpc is not None:
        primary.header["REFDIST"] = (reference_distance_kpc, "Emission reference distance [kpc]")
    if reference_luminosity_erg_s is not None:
        primary.header["REFLUM"] = (reference_luminosity_erg_s, "Reference luminosity [erg/s]")

    shape = (len(logxi), len(vturb), len(log_nh), len(energy))
    spectra_header = fits.Header()
    spectra_header["XTENSION"] = "IMAGE"
    spectra_header["BITPIX"] = -32
    spectra_header["NAXIS"] = 4
    spectra_header["NAXIS1"] = shape[3]
    spectra_header["NAXIS2"] = shape[2]
    spectra_header["NAXIS3"] = shape[1]
    spectra_header["NAXIS4"] = shape[0]
    spectra_header["PCOUNT"] = 0
    spectra_header["GCOUNT"] = 1
    spectra_header["EXTNAME"] = "SPECTRA"
    spectra_header["BUNIT"] = unit
    spectra_header["ORDER"] = "(logxi, vturb, log_nh, energy)"

    grid = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="ENERGY", format=f"{len(energy)}D", unit="keV", array=[energy]),
            fits.Column(name="LOGXI", format=f"{len(logxi)}D", unit="log10(erg cm s-1)", array=[logxi]),
            fits.Column(name="VTURB", format=f"{len(vturb)}D", unit="km s-1", array=[vturb]),
            fits.Column(name="LOG_NH", format=f"{len(log_nh)}D", unit="log10(cm-2)", array=[log_nh]),
        ],
        name="GRID",
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_empty_cube_file(output_path, primary, spectra_header, grid, shape, overwrite)

    xi_idx = {value: i for i, value in enumerate(logxi)}
    vt_idx = {value: i for i, value in enumerate(vturb)}
    nh_idx = {value: i for i, value in enumerate(log_nh)}
    with fits.open(output_path, mode="update", memmap=True) as hdul:
        spectra = hdul["SPECTRA"].data
        for pt in _progress(points, f"Packing {npz_key}"):
            npz_path = pt.dirpath(root) / f"{pt.point_id}.npz"
            with np.load(npz_path) as data:
                spectra[xi_idx[pt.logxi], vt_idx[pt.vturb], nh_idx[pt.log_nh], :] = data[npz_key].astype(np.float32)
        hdul.flush()

    return output_path


def _write_empty_cube_file(output_path, primary, spectra_header, grid, shape, overwrite):
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"File already exists: {output_path}")
        output_path.unlink()

    data_bytes = int(np.prod(shape)) * 4
    padding = (-data_bytes) % 2880
    with output_path.open("wb") as f:
        f.write(primary.header.tostring().encode("ascii"))
        f.write(spectra_header.tostring().encode("ascii"))
        f.seek(data_bytes + padding - 1, 1)
        f.write(b"\0")

    from astropy.io import fits
    fits.append(output_path, grid.data, grid.header)


def _progress(items, label: str):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return items
    return tqdm(items, desc=label, unit="node")
