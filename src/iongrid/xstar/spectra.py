"""Compile XSTAR spectral FITS outputs into the iongrid point schema."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from iongrid import SCHEMA_VERSION
from .grid import XstarPoint

D_1KPC_CM = 3.085677581491367e21
ERG_PER_EV = 1.602176634e-12


def _read_spectrum(path: Path):
    from astropy.io import fits

    with fits.open(path) as hdul:
        data = hdul["XSTAR_SPECTRA"].data
        arrays = {key: np.asarray(data[key], dtype=float) for key in
                  ("energy", "incident", "transmitted", "emit_inward", "emit_outward")}
        params = hdul["PARAMETERS"].data
        values = {(name.decode("ascii") if isinstance(name, bytes) else str(name)).strip().lower(): float(value) for name, value in
                  zip(params["parameter"], params["value"])}
    return arrays, values


def _bin_widths(energy: np.ndarray) -> np.ndarray:
    edges = np.empty(len(energy) + 1)
    edges[1:-1] = (energy[1:] + energy[:-1]) / 2
    edges[0] = energy[0] - (energy[1] - energy[0]) / 2
    edges[-1] = energy[-1] + (energy[-1] - energy[-2]) / 2
    if edges[0] <= 0:
        edges[0] = energy[0] * np.sqrt(energy[0] / energy[1])
    return np.diff(edges)


def compile_node(node_dir: str | Path, point: XstarPoint) -> Path:
    """Pack total optical depth and outward emission at XSTAR's native energy sampling.

    Emission follows the predecessor xstar2xspec convention: bin-integrated
    photon flux normalized to 10^38 erg/s at a reference distance of 1 kpc.
    """
    node_dir = Path(node_dir)
    total, parameters = _read_spectrum(node_dir / "xout_spect1.fits")
    continuum, _ = _read_spectrum(node_dir / "xout_cont1.fits")
    energy_ev = total["energy"]
    if len(energy_ev) < 2 or np.any(~np.isfinite(energy_ev)) or np.any(np.diff(energy_ev) <= 0):
        raise ValueError("XSTAR energy axis must be finite and increasing")
    if not np.array_equal(energy_ev, continuum["energy"]):
        raise ValueError("XSTAR total and continuum energy grids differ")
    luminosity_38 = parameters.get("rlrad38")
    if luminosity_38 is None or not np.isfinite(luminosity_38) or luminosity_38 <= 0:
        raise ValueError("XSTAR output lacks a positive rlrad38 parameter")
    incident = total["incident"]
    transmitted = total["transmitted"]
    if np.any(~np.isfinite(incident)) or np.any(~np.isfinite(transmitted)) or np.any(incident < 0):
        raise ValueError("Invalid XSTAR incident or transmitted spectrum")
    transmission = np.divide(transmitted, incident, out=np.ones_like(incident), where=incident > 0)
    tau_total = -np.log(np.clip(transmission, 1e-30, 1.0))

    # Preserve the conversion used by the predecessor XSTAR-to-XSPEC workflow.
    width_erg = _bin_widths(energy_ev) * ERG_PER_EV
    energy_erg = energy_ev * ERG_PER_EV
    factor = (1e38 / (4 * np.pi * D_1KPC_CM**2)) / luminosity_38 * width_erg / energy_erg
    emitted = total["emit_outward"] * factor
    raw_continuum = continuum["emit_outward"] * factor
    if np.any(~np.isfinite(emitted)) or np.any(emitted < 0) or np.any(~np.isfinite(raw_continuum)):
        raise ValueError("Invalid XSTAR emission spectrum")
    # The two rounded FITS products can differ slightly in continuum-only bins.
    # Keep all exported components nonnegative and their sum equal to total.
    emt_continuum = np.minimum(np.maximum(raw_continuum, 0), emitted)
    emt_lines = emitted - emt_continuum
    metadata = np.array(
        (point.point_id, point.logxi, point.vturb, point.log_nh, point.log_density, luminosity_38),
        dtype=[("point_id", "U60"), ("logxi", "f8"), ("vturb", "f8"),
               ("log_nh", "f8"), ("log_density", "f8"), ("luminosity_38", "f8")],
    )
    destination = node_dir / f"{point.point_id}.npz"
    np.savez_compressed(
        destination, schema_version=np.int16(SCHEMA_VERSION),
        energy=(energy_ev / 1000).astype(np.float32),
        tau_total=tau_total.astype(np.float32),
        emt_total=emitted.astype(np.float32),
        emt_continuum=emt_continuum.astype(np.float32),
        emt_lines=emt_lines.astype(np.float32),
        incident=incident.astype(np.float32), transmitted=transmitted.astype(np.float32),
        metadata=metadata,
        emission_reference_kpc=np.float32(1),
        emission_reference_luminosity_erg_s=np.float64(1e38),
    )
    return destination
