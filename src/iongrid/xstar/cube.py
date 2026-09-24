"""Pack XSTAR point caches using the common iongrid FITS cube writer."""

from pathlib import Path

from iongrid.cube import write_fits_cube as _write_cube
from .grid import read_manifest


def write_fits_cube(
    manifest_path: str | Path, output_path: str | Path, *,
    npz_key: str, quantity: str, unit: str, overwrite: bool = False,
) -> Path:
    manifest_path = Path(manifest_path)
    return _write_cube(
        read_manifest(manifest_path), manifest_path.parent, output_path,
        npz_key=npz_key, quantity=quantity, unit=unit,
        overwrite=overwrite, engine="XSTAR", reference_distance_kpc=1.0,
        reference_luminosity_erg_s=1e38,
    )
