"""Edit these values in the copied campaign directory."""

import numpy as np 

from pathlib import Path

from iongrid.sampling import cartesian, lhs
from iongrid.sed import native, table

ROOT = Path(__file__).resolve().parent
ENGINE = "xstar"
EXECUTABLE = "/data-fast/dragon/software/heasoft-6.36/x86_64-pc-linux-gnu-libc2.39/bin/xstar"
TEMPLATE = ROOT / "xstar.par"
# SED configuration:
# SED = native()
ENERGY = np.logspace(-4, 3, 1000)   # 0.1 eV to 1 MeV
# Constants:
K_BOLTZ = 8.617333262e-8            # Boltzmann constant (keV / K)
H_PLANCK = 4.1356677e-18            # Placnk constant (keV s)
C = 2.997925e+8                     # Speed of light (m / s)
# Values fixed for SED:
T_IR = 1e+3                         # IR temperature (K) setting the low-energy cut-off of the BBB
LOW_XRAY_E = 1e-1                   # Lower energy limit (keV) to X-ray power law
# SED variables:
# alpha_uv                          --> Energy index of the optical-to-UV power law (BBB in low-energy side)
# tbb                               --> Temperature (K) setting the high-energy cut-off of the BBB
# X-ray power law
# alpha_ox                          --> Energy index of the optical-to-X-ray power law between 2500 Å (~ 0.00496 keV) and 2 keV
# gamma_x                           --> Photon index of the X-ray power law
# high_xray_break                   --> High energy (keV) break of X-ray power law
def special_sed(alpha_uv, logtbb, alpha_ox, gamma_x, high_xray_break):
    # Energy values:
    e = ENERGY
    # Convert logtbb to linear values:
    tbb = 10**logtbb
    # BBB:
    def _bbb(e):
        return e**alpha_uv * np.exp(-e / (K_BOLTZ*tbb)) * np.exp(-(K_BOLTZ*T_IR) / e)
    # Energy index derived from photon index of X-ray power law (gamma_x):
    alpha_x = 1.0 - gamma_x 
    # 2500 Å (~ 0.00496 keV) to keV:
    e_2500a = (H_PLANCK*C) / (2500e-10)
    # Flux at 2 keV obtained from BBB's flux at 2500 Angstrom:
    f_2kev = _bbb(e_2500a) * (2 / e_2500a) ** alpha_ox
    # Shape of the X-ray power law as a broken power law (flux per frequency) normalized to 1 at 2 keV:
    shape = np.where(e <= high_xray_break, (e / 2) ** alpha_x, (high_xray_break / 2) ** alpha_x * (e / high_xray_break) ** -2.0)
    # X-ray range of the SED:
    xray = f_2kev * shape * np.exp(-LOW_XRAY_E / e + LOW_XRAY_E / 2)
    f_freq = _bbb(e) + xray
    return np.array([e, f_freq / e])   # table() wants photons per keV, so divide F_freq by E
SED = table(special_sed, axes=("alpha_uv", "logtbb", "alpha_ox", "gamma_x", "high_xray_break"))
SAMPLING = "cartesian"

if SAMPLING == "cartesian":
    PARAMS = cartesian(
        logxi=[3.0, 4.0],
        vturb=[100.0, 1000.0],
        lognh=[22.0, 23.0],
        feabund=[0.5, 1.5],
        alpha_uv=[-1, -2.5],
        logtbb=[5, 6],
        alpha_ox=[-1.1, -1.8],
        gamma_x=[1.4, 2.6],
        high_xray_break=[100, 300],
    )
elif SAMPLING == "lhs":
    PARAMS = (
        lhs(
            bounds={
                "logxi": (2.0, 7.0),
                "vturb": (50.0, 10000.0),
                "lognh": (20.0, 25.0),
                "feabund": (0.5, 2.0),
                "gamma": (1.4, 2.6)},
            samples=20,
            seed=12345,
            anchors=True,
            logarithmic=("vturb",)
            )
        + 
        lhs(
            bounds={
                "logxi": (2.0, 7.0), 
                "vturb": (50.0, 10000.0), 
                "lognh": (22.5, 25.0),
                "feabund": (0.5, 2.0), 
                "gamma": (1.4, 2.6)},
            samples=10,
            seed=777, 
            anchors=False, 
            logarithmic=("vturb",),
            )
        + 
        lhs(
            bounds={
                "logxi": (2.0, 5.0), 
                "vturb": (50.0, 10000.0), 
                "lognh": (20.0, 25.0),
                "feabund": (0.5, 2.0), 
                "gamma": (1.4, 2.6)},
            samples=20,
            seed=34562, 
            anchors=False, 
            logarithmic=("vturb",),
            )
    )
else:
    raise ValueError("SAMPLING must be 'cartesian' or 'lhs'")

CLEAN = "both"  # "raw", "clean", or "both".
EPS_T = 1e-5
EPS_E = 1e-5
