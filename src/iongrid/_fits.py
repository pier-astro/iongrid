from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

import numpy as np

from . import POINTS_VERSION, SCHEMA_VERSION

# Optional hook: postprocess(npz_data, spectrum) -> spectrum. Called per node
# right after the spectrum is read, e.g. to apply XSTAR noise cleaning.
Postprocess = Callable[["np.lib.npyio.NpzFile", np.ndarray], np.ndarray]


def _validate_node(npz_path, npz_key, energy, postprocess):
    """Load one node, validate it and return (energy, spectrum)."""
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


def _base_header(primary, *, engine, npz_key, quantity, reference_distance_kpc,
                 reference_luminosity_erg_s, extra_header):
    primary.header["ORIGIN"] = "iongrid"
    primary.header["ENGINE"] = engine.upper()
    primary.header["IGFMTVER"] = POINTS_VERSION
    primary.header["NPZKEY"] = npz_key
    primary.header["QUANTITY"] = quantity
    if reference_distance_kpc is not None:
        primary.header["REFDIST"] = (reference_distance_kpc, "Emission reference distance [kpc]")
    if reference_luminosity_erg_s is not None:
        primary.header["REFLUM"] = (reference_luminosity_erg_s, "Reference luminosity [erg/s]")
    for key, value in (extra_header or {}).items():
        primary.header[key] = value


def write_fits_table(
    parameters: dict,
    directory: str | Path,
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
    """Stream one spectrum per parameter row to PARAMS, ENERGY and SPECTRA."""
    try:
        from astropy.io import fits
    except ImportError as e:
        raise ImportError("Writing FITS tables requires astropy.") from e

    directory = Path(directory)
    points = sorted(parameters)
    if not points:
        raise ValueError("manifest has no collected points to pack")

    first_path = directory / points[0] / f"{points[0]}.npz"
    energy, first = _validate_node(first_path, npz_key, None, postprocess)
    shape = (len(points), len(energy))
    units = {"vturb": "km/s", "ktbb": "keV"}
    columns = [fits.Column(name="POINT_ID", format="32A", array=points)]
    for name in parameters[points[0]]:
        columns.append(fits.Column(name=name.upper(), format="D", unit=units.get(name),
                                   array=[parameters[point][name] for point in points]))
    params = fits.BinTableHDU.from_columns(columns, name="PARAMS")

    primary = fits.PrimaryHDU()
    _base_header(primary, engine=engine, npz_key=npz_key, quantity=quantity,
                 reference_distance_kpc=reference_distance_kpc,
                 reference_luminosity_erg_s=reference_luminosity_erg_s, extra_header=extra_header)
    primary.header["LAYOUT"] = "POINTS"

    energy_hdu = fits.ImageHDU(np.asarray(energy, dtype=np.float64), name="ENERGY")
    energy_hdu.header["BUNIT"] = "keV"
    header = fits.ImageHDU(np.empty((1, 1), dtype=np.float32), name="SPECTRA").header
    header["NAXIS1"], header["NAXIS2"] = shape[1], shape[0]
    header["BUNIT"] = unit
    header["ORDER"] = "(point, energy)"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {output_path}")
    with TemporaryDirectory(dir=output_path.parent) as scratch:
        temporary = Path(scratch) / output_path.name
        fits.HDUList([primary, params, energy_hdu]).writeto(temporary)
        size = len(points) * len(energy) * 4
        with temporary.open("ab") as stream:
            stream.write(header.tostring().encode("ascii"))
            stream.truncate(stream.tell() + size + (-size) % 2880)
        with fits.open(temporary, mode="update", memmap=True) as hdul:
            spectra = hdul["SPECTRA"].data
            spectra[0] = first
            for index, point in enumerate(_progress(points[1:], f"Packing {npz_key}"), start=1):
                cache = directory / point / f"{point}.npz"
                _, spectra[index] = _validate_node(cache, npz_key, energy, postprocess)
        if overwrite:
            temporary.replace(output_path)
        else:
            output_path.hardlink_to(temporary)
    return output_path


def _progress(items, label: str):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return items
    return tqdm(items, desc=label, unit="node")
