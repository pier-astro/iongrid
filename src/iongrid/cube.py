from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from . import SCHEMA_VERSION

# Cube axes in storage order. The first three are always present; the optional
# ones are added (in this order, before the energy axis) only when every point
# carries a value for them, so existing 3-axis cubes are written exactly as before.
CUBE_AXES = (
    ("logxi", "LOGXI", "log10(erg cm s-1)", True),
    ("vturb", "VTURB", "km s-1", True),
    ("log_nh", "LOG_NH", "log10(cm-2)", True),
    ("feabund", "FEABUND", "xstardef", False),
    ("gamma", "GAMMA", "", False),
)

# Optional hook: postprocess(npz_data, spectrum) -> spectrum. Called per node
# right after the spectrum is read, e.g. to apply XSTAR noise cleaning.
Postprocess = Callable[["np.lib.npyio.NpzFile", np.ndarray], np.ndarray]


def _validate_node(pt, root, npz_key, energy, postprocess):
    """Load one node, validate it and return (energy, spectrum)."""
    npz_path = pt.dirpath(root) / f"{pt.point_id}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing compiled node cache: {npz_path}")
    with np.load(npz_path) as data:
        if npz_key not in data:
            raise KeyError(f"{npz_path} does not contain {npz_key!r}")
        if "schema_version" in data and int(data["schema_version"]) != SCHEMA_VERSION:
            raise ValueError(f"Unsupported iongrid schema version in {npz_path}")
        node_energy = np.asarray(data["energy"], dtype=float)
        spectrum = np.asarray(data[npz_key])
        if spectrum.shape != node_energy.shape:
            raise ValueError(f"Spectrum and energy grid have different shapes in {npz_path}")
        if postprocess is not None:
            spectrum = np.asarray(postprocess(data, spectrum))
            if spectrum.shape != node_energy.shape:
                raise ValueError(f"postprocess changed the spectrum shape in {npz_path}")
        if not np.all(np.isfinite(spectrum)) or np.any(np.abs(spectrum) > np.finfo(np.float32).max):
            raise ValueError(f"{npz_key} in {npz_path} cannot be stored as finite float32 FITS values")
    if energy is not None and (len(node_energy) != len(energy) or not np.allclose(node_energy, energy, rtol=1e-6, atol=0.0)):
        raise ValueError(f"Energy grid mismatch in {npz_path}")
    return node_energy, spectrum


def _active_axes(points):
    """Return [(attr, FITS name, unit)] for the axes present in these points."""
    active = []
    for attr, fits_name, unit, required in CUBE_AXES:
        values = [getattr(pt, attr, None) for pt in points]
        n_none = sum(v is None for v in values)
        if required:
            active.append((attr, fits_name, unit))
        elif n_none == 0:
            active.append((attr, fits_name, unit))
        elif n_none != len(values):
            raise ValueError(f"axis {attr!r} is set for some points but not others")
    return active


def read_cube_grid_nd(points: list, root: str | Path, npz_key: str, postprocess: Postprocess | None = None):
    """Read the N-d grid axes and validate that all compiled nodes are packable.

    Returns ``(axes, energy, points)`` where ``axes`` is an ordered list of
    ``(attr, FITS name, unit, values)``.
    """
    root = Path(root)
    points = [pt for pt in points if pt.status == "ok"]
    if not points:
        raise ValueError("manifest has no successful points to pack")

    active = _active_axes(points)
    axes = [(a, n, u, np.array(sorted({getattr(pt, a) for pt in points}), dtype=float)) for a, n, u in active]
    expected_count = int(np.prod([len(v) for *_, v in axes]))
    if len(points) != expected_count:
        raise ValueError(
            "successful points do not form a complete rectangular grid: "
            f"{len(points)} of {expected_count} (use the scattered table layout for LHS grids)"
        )

    indexers = [{value: i for i, value in enumerate(v)} for *_, v in axes]
    energy, seen = None, set()
    for pt in points:
        node_energy, _ = _validate_node(pt, root, npz_key, energy, postprocess)
        if energy is None:
            energy = node_energy
        index = tuple(ix[getattr(pt, a)] for ix, (a, *_rest) in zip(indexers, axes))
        if index in seen:
            raise ValueError(f"Duplicate grid point in manifest: {pt.point_id}")
        seen.add(index)
    return axes, energy, points


def read_fits_cube_grid(points: list, root: str | Path, npz_key: str):
    """Backward-compatible 3-axis reader: (logxi, vturb, log_nh, energy, points)."""
    axes, energy, points = read_cube_grid_nd(points, root, npz_key)
    if len(axes) != 3:
        raise ValueError("grid has feabund/gamma axes; use read_cube_grid_nd")
    (_, _, _, logxi), (_, _, _, vturb), (_, _, _, log_nh) = axes
    return logxi, vturb, log_nh, energy, points


def _base_header(primary, *, engine, npz_key, quantity, reference_distance_kpc,
                 reference_luminosity_erg_s, extra_header):
    primary.header["ORIGIN"] = "iongrid"
    primary.header["ENGINE"] = engine.upper()
    primary.header["IGFMTVER"] = SCHEMA_VERSION
    primary.header["NPZKEY"] = npz_key
    primary.header["QUANTITY"] = quantity
    if reference_distance_kpc is not None:
        primary.header["REFDIST"] = (reference_distance_kpc, "Emission reference distance [kpc]")
    if reference_luminosity_erg_s is not None:
        primary.header["REFLUM"] = (reference_luminosity_erg_s, "Reference luminosity [erg/s]")
    for key, value in (extra_header or {}).items():
        primary.header[key] = value


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
    postprocess: Postprocess | None = None,
    extra_header: dict | None = None,
) -> Path:
    """Write one FITS cube from a named spectrum stored in each node ``.npz``."""
    try:
        from astropy.io import fits
    except ImportError as e:
        raise ImportError("Writing FITS cubes requires astropy.") from e

    root = Path(root)
    axes, energy, points = read_cube_grid_nd(points, root, npz_key, postprocess)

    primary = fits.PrimaryHDU()
    _base_header(primary, engine=engine, npz_key=npz_key, quantity=quantity,
                 reference_distance_kpc=reference_distance_kpc,
                 reference_luminosity_erg_s=reference_luminosity_erg_s, extra_header=extra_header)

    shape = tuple(len(v) for *_, v in axes) + (len(energy),)
    spectra_header = fits.Header()
    spectra_header["XTENSION"] = "IMAGE"
    spectra_header["BITPIX"] = -32
    spectra_header["NAXIS"] = len(shape)
    for k, n in enumerate(reversed(shape), start=1):
        spectra_header[f"NAXIS{k}"] = n
    spectra_header["PCOUNT"] = 0
    spectra_header["GCOUNT"] = 1
    spectra_header["EXTNAME"] = "SPECTRA"
    spectra_header["BUNIT"] = unit
    spectra_header["ORDER"] = "(" + ", ".join([a for a, *_ in axes] + ["energy"]) + ")"

    columns = [fits.Column(name="ENERGY", format=f"{len(energy)}D", unit="keV", array=[energy])]
    for _, fits_name, axis_unit, values in axes:
        columns.append(fits.Column(name=fits_name, format=f"{len(values)}D", unit=axis_unit, array=[values]))
    grid = fits.BinTableHDU.from_columns(columns, name="GRID")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_empty_cube_file(output_path, primary, spectra_header, grid, shape, overwrite)

    indexers = [{value: i for i, value in enumerate(v)} for *_, v in axes]
    with fits.open(output_path, mode="update", memmap=True) as hdul:
        spectra = hdul["SPECTRA"].data
        for pt in _progress(points, f"Packing {npz_key}"):
            _, spectrum = _validate_node(pt, root, npz_key, energy, postprocess)
            index = tuple(ix[getattr(pt, a)] for ix, (a, *_rest) in zip(indexers, axes))
            spectra[index + (slice(None),)] = spectrum.astype(np.float32)
        hdul.flush()

    return output_path


def write_fits_table(
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
    postprocess: Postprocess | None = None,
    extra_header: dict | None = None,
) -> Path:
    """Write scattered (e.g. Latin-hypercube) points as one FITS file.

    Layout (``LAYOUT = 'SCATTERED'``):

    * ``PARAMS``  binary table, one row per point (POINT_ID, LOGXI, VTURB,
      LOG_NH, and FEABUND / GAMMA when present).
    * ``ENERGY``  1-d image, keV.
    * ``SPECTRA`` 2-d image, shape ``(n_points, n_energy)``; row i matches
      row i of ``PARAMS``.

    The whole array is held in memory (float32, n_points x n_energy).
    This layout is not a rectilinear grid, so it cannot be turned into an
    OGIP/XSPEC table model.
    """
    try:
        from astropy.io import fits
    except ImportError as e:
        raise ImportError("Writing FITS tables requires astropy.") from e

    root = Path(root)
    points = sorted((pt for pt in points if pt.status == "ok"), key=lambda p: p.point_id)
    if not points:
        raise ValueError("manifest has no successful points to pack")
    active = _active_axes(points)

    energy, rows = None, []
    for pt in _progress(points, f"Packing {npz_key}"):
        node_energy, spectrum = _validate_node(pt, root, npz_key, energy, postprocess)
        if energy is None:
            energy = node_energy
        rows.append(spectrum.astype(np.float32))
    spectra = np.vstack(rows)

    columns = [fits.Column(name="POINT_ID", format="32A", array=np.array([p.point_id for p in points]))]
    for attr, fits_name, axis_unit in active:
        columns.append(fits.Column(name=fits_name, format="D", unit=axis_unit,
                                   array=np.array([getattr(p, attr) for p in points], dtype=float)))
    params = fits.BinTableHDU.from_columns(columns, name="PARAMS")

    primary = fits.PrimaryHDU()
    _base_header(primary, engine=engine, npz_key=npz_key, quantity=quantity,
                 reference_distance_kpc=reference_distance_kpc,
                 reference_luminosity_erg_s=reference_luminosity_erg_s, extra_header=extra_header)
    primary.header["LAYOUT"] = ("SCATTERED", "points are not on a rectilinear grid")

    energy_hdu = fits.ImageHDU(np.asarray(energy, dtype=np.float64), name="ENERGY")
    energy_hdu.header["BUNIT"] = "keV"
    spectra_hdu = fits.ImageHDU(spectra, name="SPECTRA")
    spectra_hdu.header["BUNIT"] = unit
    spectra_hdu.header["ORDER"] = "(point, energy)"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {output_path}")
    fits.HDUList([primary, params, energy_hdu, spectra_hdu]).writeto(output_path, overwrite=overwrite)
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
