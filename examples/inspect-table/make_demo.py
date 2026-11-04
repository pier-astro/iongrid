"""Generate synthetic point and OGIP tables for the inspector."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
from astropy.io import fits

from iongrid import POINTS_VERSION
from iongrid.ogip import write_ogip_table
from iongrid.sampling import cartesian

output = Path(__file__).resolve().parent / "products"
output.mkdir(exist_ok=True)
point_table = output / "demo-tau.fits"
ogip_table = output / "demo-transmission.mmod"

energy = np.geomspace(0.5, 20.0, 160)
rows = cartesian(logxi=[3.5, 4.0, 4.5], vturb=[500.0, 2000.0], lognh=[22.0, 23.0])
tau = np.empty((len(rows), len(energy)), dtype=np.float32)
for index, row in enumerate(rows):
    center = 6.5 + 0.4 * (row["logxi"] - 4.0)
    width = 0.08 + row["vturb"] / 25000.0
    tau[index] = (10 ** (row["lognh"] - 22.0) / 5.0) * np.exp(
        -0.5 * ((energy - center) / width) ** 2
    )

primary = fits.PrimaryHDU()
primary.header["ORIGIN"] = "iongrid"
primary.header["IGFMTVER"] = POINTS_VERSION
primary.header["QUANTITY"] = "Optical depth"
columns = [fits.Column(name="POINT_ID", format="8A", array=[f"p{i}" for i in range(len(rows))])]
for name in rows[0]:
    columns.append(fits.Column(name=name.upper(), format="D", array=[row[name] for row in rows]))
params = fits.BinTableHDU.from_columns(columns, name="PARAMS")
energies = fits.ImageHDU(energy, name="ENERGY")
energies.header["BUNIT"] = "keV"
spectra = fits.ImageHDU(tau, name="SPECTRA")
spectra.header["BUNIT"] = "dimensionless"
fits.HDUList([primary, params, energies, spectra]).writeto(point_table, overwrite=True)
write_ogip_table(point_table, ogip_table, model_name="demo", model_type="multiplicative",
                 redshift=False, overwrite=True)
print(point_table)
print(ogip_table)
