#!/usr/bin/env python3
"""Set XSTAR and target luminosities below; print the XSPEC emission norm."""

from astropy.cosmology import Planck18

# Copy this recipe with the run. Match L_ION_SIM to settings.py LUMINOSITY_38 * 1e38.
L_ION_SIM = 1e38             # XSTAR input 1-1000 Ryd luminosity, erg/s
L_ION_TARGET = 1e45          # incident 1-1000 Ryd luminosity, erg/s (illustrative)
REDSHIFT = 0.184             # set to None when using DISTANCE_MPC
DISTANCE_MPC = None          # luminosity distance; used only if REDSHIFT is None
COVERING_FRACTION = 1.0     # global solid-angle fraction; 1 if WINE applies it

D_L_MPC = Planck18.luminosity_distance(REDSHIFT).value if REDSHIFT is not None else DISTANCE_MPC
# iongrid already divides XSTAR emission by the input luminosity and exports
# it for 1e38 erg/s at 1 kpc. L_ION_SIM cancels from the XSPEC scalar.
norm = COVERING_FRACTION * (L_ION_TARGET / 1e38) * (0.001 / D_L_MPC) ** 2

print(f"XSTAR L_ion,sim = {L_ION_SIM:.6e} erg/s")
print(f"XSPEC emission norm = {norm:.6e}")
