"""Generate small iongrid and OGIP tables for the interactive inspector."""

from pathlib import Path

import numpy as np
from astropy.io import fits

from iongrid import SCHEMA_VERSION
from iongrid.ogip import write_ogip_table


output = Path(__file__).resolve().parent / "products"
output.mkdir(exist_ok=True)
cube_path = output / "demo-tau.fits"
table_path = output / "demo-transmission.mmod"

energy = np.geomspace(0.5, 20.0, 160)
logxi = np.array([3.5, 4.0, 4.5])
vturb = np.array([500.0, 2000.0])
log_nh = np.array([22.0, 23.0])
tau = np.empty((len(logxi), len(vturb), len(log_nh), len(energy)), dtype=np.float32)
for i, xi in enumerate(logxi):
    for j, turbulence in enumerate(vturb):
        for k, nh in enumerate(log_nh):
            center = 6.5 + 0.4 * (xi - 4.0)
            width = 0.08 + turbulence / 25000.0
            tau[i, j, k] = (10 ** (nh - 22.0) / 5.0) * np.exp(
                -0.5 * ((energy - center) / width) ** 2
            )

primary = fits.PrimaryHDU()
primary.header["ORIGIN"] = "iongrid"
primary.header["ENGINE"] = "DEMO"
primary.header["IGFMTVER"] = SCHEMA_VERSION
primary.header["NPZKEY"] = "tau_total"
primary.header["QUANTITY"] = "Optical depth"
spectra = fits.ImageHDU(tau, name="SPECTRA")
spectra.header["BUNIT"] = "dimensionless"
spectra.header["ORDER"] = "(logxi, vturb, log_nh, energy)"
grid = fits.BinTableHDU.from_columns([
    fits.Column(name="ENERGY", format=f"{len(energy)}D", unit="keV", array=[energy]),
    fits.Column(name="LOGXI", format=f"{len(logxi)}D", array=[logxi]),
    fits.Column(name="VTURB", format=f"{len(vturb)}D", unit="km s-1", array=[vturb]),
    fits.Column(name="LOG_NH", format=f"{len(log_nh)}D", array=[log_nh]),
], name="GRID")
fits.HDUList([primary, spectra, grid]).writeto(cube_path, overwrite=True)
write_ogip_table(cube_path, table_path, model_name="demo", model_type="multiplicative",
                 redshift=False, overwrite=True)
print(cube_path)
print(table_path)
