"""Pack XSTAR point caches using the common iongrid FITS writers."""

from pathlib import Path

from iongrid.cube import write_fits_cube as _write_cube
from iongrid.cube import write_fits_table as _write_table
from .grid import read_manifest


def write_fits_cube(
    manifest_path: str | Path, output_path: str | Path, *,
    npz_key: str, quantity: str, unit: str, overwrite: bool = False,
    layout: str = "cube", postprocess=None, extra_header: dict | None = None,
) -> Path:
    """Write a rectilinear ``cube`` (Cartesian grids) or a ``table`` of scattered points (LHS grids). 
    ``postprocess`` / ``extra_header`` are forwarded
    (see iongrid.xstar.clean.make_postprocess)."""
    if layout not in {"cube", "table"}:
        raise ValueError("layout must be 'cube' or 'table'")
    writer = _write_cube if layout == "cube" else _write_table
    manifest_path = Path(manifest_path)
    return writer(
        read_manifest(manifest_path), manifest_path.parent, output_path,
        npz_key=npz_key, quantity=quantity, unit=unit,
        overwrite=overwrite, engine="XSTAR", reference_distance_kpc=1.0,
        reference_luminosity_erg_s=1e38,
        postprocess=postprocess, extra_header=extra_header,
    )
