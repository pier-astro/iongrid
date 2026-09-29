#!/usr/bin/env python3
"""Set the CLOUDY input and target below; print the full-shell XSPEC emission norm."""

from astropy.cosmology import Planck18

CLOUDY_INPUT = "luminosity"   # "luminosity" or "xi"; match the simulation deck
L_ION_SIM_INPUT = 1e48        # 1-1000 Ryd erg/s for "luminosity"
LOGXI = None                  # needed only for "xi"
LOG_DENSITY = None            # needed only for "xi": log10(n_H / cm^-3)
RADIUS_CM = None              # needed only for "xi"
L_ION_TARGET = 1e45          # incident 1-1000 Ryd luminosity, erg/s (illustrative)
REDSHIFT = 0.184             # set to None when using DISTANCE_MPC
DISTANCE_MPC = None          # luminosity distance; used only if REDSHIFT is None
COVERING_FRACTION = 1.0     # global solid-angle fraction; 1 if WINE applies it

if CLOUDY_INPUT == "xi":
    if LOGXI is None or LOG_DENSITY is None or RADIUS_CM is None:
        raise ValueError("Set LOGXI, LOG_DENSITY and RADIUS_CM for CLOUDY_INPUT='xi'")
    L_ION_SIM = 10 ** (LOGXI + LOG_DENSITY) * RADIUS_CM**2
elif CLOUDY_INPUT == "luminosity":
    L_ION_SIM = L_ION_SIM_INPUT
else:
    raise ValueError("CLOUDY_INPUT must be 'xi' or 'luminosity'")
D_L_MPC = Planck18.luminosity_distance(REDSHIFT).value if REDSHIFT is not None else DISTANCE_MPC
norm = COVERING_FRACTION * (L_ION_TARGET / L_ION_SIM) * (0.01 / D_L_MPC) ** 2

print(f"CLOUDY L_ion,sim = {L_ION_SIM:.6e} erg/s")
print(f"XSPEC full-shell emission norm = {norm:.6e}")
