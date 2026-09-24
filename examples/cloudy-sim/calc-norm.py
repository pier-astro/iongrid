#!/usr/bin/env python3
"""Calculate the starting XSPEC normalization for a 10 kpc emission table."""

import sys
from astropy.cosmology import Planck18

if len(sys.argv) > 2:
    raise SystemExit("Usage: python calc-norm.py [REDSHIFT]")
# Preserve the original example when no argument is supplied.
redshift = float(sys.argv[1]) if len(sys.argv) == 2 else 0.184
if redshift <= 0:
    raise SystemExit("REDSHIFT must be positive")

# Reference distance used to scale the grid is 10 kpc (0.01 Mpc)
d_ref_mpc = 0.01 # Mpc

# Calculate luminosity distance in Mpc
d_L_mpc = Planck18.luminosity_distance(redshift).value

# XSPEC additive table model normalization factor is (d_ref / d_L)^2
norm = (d_ref_mpc / d_L_mpc) ** 2

print(f"Redshift: {redshift}")
print(f"Luminosity Distance: {d_L_mpc:.2f} Mpc")
print(f"Suggested XSPEC starting normalization: {norm:.2e}")
