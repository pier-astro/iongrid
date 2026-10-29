"""Removal of XSTAR numerical-noise features from derived spectra.

XSTAR writes spectra as 32-bit floats (relative resolution ~1.2e-7) and the
quantities we difference or divide (total vs continuum emission, transmitted
vs incident) carry independent round-off / single-precision noise. The "line"
spectrum of a line-free bin is therefore not 0 but +-eps*continuum, and T of an
unabsorbed bin is not 1 but 1 +- eps. These helpers remove only what is
indistinguishable from that noise floor; every other bin is left bit-for-bit
unchanged. Thresholds are *relative*, so they do not depend on flux units or
on the (per-bin) XSTAR->XSPEC conversion factor.

Two ways to use them:

* ``clean_tau`` / ``clean_lines``: act on the packed products (``tau`` and
  ``emt_lines`` + ``emt_total`` in the point .npz files). Plug in through
  ``make_postprocess`` and the ``postprocess`` argument of the cube writers.
* ``clean_transmission`` / ``clean_line_emission`` / ``noise_report``: act on the
  raw XSTAR arrays (spect1 / cont1). Use ``noise_report`` on a sample of raw
  spectra to calibrate eps before choosing EPS_T / EPS_E.
"""

from __future__ import annotations

import numpy as np

DEFAULT_EPS_T = 1e-5
DEFAULT_EPS_E = 1e-5
DEFAULT_LINE_FLOOR = 0.0


# ---------------------------------------------------------------------------
# Packed-product versions (tau, emt_lines/emt_total)
# ---------------------------------------------------------------------------

def clean_tau(tau, eps_T=DEFAULT_EPS_T):
    """Set tau to exactly 0 where the transmission deviates from 1 by < eps_T.

    Equivalent to snapping |1 - T| < eps_T to T = 1 (T = exp(-tau) <= 1 in the
    packed products because the ratio is clipped to the physical interval).
    """
    tau = np.asarray(tau, dtype=np.float64)
    out = tau.copy()
    with np.errstate(invalid="ignore"):
        out[-np.expm1(-tau) < eps_T] = 0.0
    return out


def clean_lines(total, lines, eps_E=DEFAULT_EPS_E, floor=DEFAULT_LINE_FLOOR):
    """Zero (or ``floor``) line bins whose excess is below eps_E * total.

    Valid for packed products where ``lines = max(total - continuum, 0)``: then
    ``max(total, continuum) == total`` wherever lines > 0, so the test is
    identical to the raw-array version. ``total`` itself is not modified.
    """
    total = np.asarray(total, dtype=np.float64)
    lines = np.asarray(lines, dtype=np.float64)
    out = lines.copy()
    with np.errstate(invalid="ignore"):
        out[~(lines > eps_E * np.abs(total))] = floor
    return out


def make_postprocess(kind, *, eps_T=DEFAULT_EPS_T, eps_E=DEFAULT_EPS_E,
                     line_floor=DEFAULT_LINE_FLOOR, total_key="emt_total"):
    """Build a ``postprocess(npz_data, spectrum)`` hook for the cube writers.

    kind: "tau" (``tau_total``; uses the npz's raw ``incident``/``transmitted``
    when present) or "lines" (``emt_lines``; the npz must also contain
    ``total_key``, default ``emt_total``).
    Returns ``(postprocess, extra_header)``; the header stamps the cube with
    NOISCLN = True and the thresholds used.
    """
    if kind == "tau":
        def post(data, spectrum):
            # Preferred: redo the ratio from the raw incident/transmitted spectra
            # stored in the node npz, exactly as compile_node does (T clipped to
            # [1e-30, 1]) but with noise-level |1-T| snapped to 1 first.
            if "incident" in data and "transmitted" in data:
                T = clean_transmission(data["incident"], data["transmitted"], eps_T, clip_upper=True)
                return -np.log(np.clip(T, 1e-30, 1.0))
            return clean_tau(spectrum, eps_T)
        header = {"NOISCLN": (True, "XSTAR noise cleaning applied"),
                  "EPS_T": (float(eps_T), "rel. noise floor, transmission")}
    elif kind == "lines":
        def post(data, spectrum):
            return clean_lines(data[total_key], spectrum, eps_E, line_floor)
        header = {"NOISCLN": (True, "XSTAR noise cleaning applied"),
                  "EPS_E": (float(eps_E), "rel. noise floor, line emission"),
                  "LINEFLR": (float(line_floor), "value of no-emission bins")}
    else:
        raise ValueError("kind must be 'tau' or 'lines'")
    return post, header


# ---------------------------------------------------------------------------
# Raw-array versions (XSTAR spect1 / cont1)
# ---------------------------------------------------------------------------

def clean_transmission(incident, transmitted, eps_T=DEFAULT_EPS_T, clip_upper=False):
    """T = transmitted/incident with |T-1| < eps_T snapped to 1 (T=1 where incident <= 0)."""
    inc = np.asarray(incident, dtype=np.float64)
    trn = np.asarray(transmitted, dtype=np.float64)
    T = np.ones(inc.shape, dtype=np.float64)
    ok = np.isfinite(inc) & np.isfinite(trn) & (inc > 0)
    np.divide(trn, inc, out=T, where=ok)
    T[ok & (np.abs(T - 1.0) < eps_T)] = 1.0
    np.maximum(T, 0.0, out=T)
    if clip_upper:
        np.minimum(T, 1.0, out=T)
    return T


def clean_line_emission(spect, cont, eps_E=DEFAULT_EPS_E, floor=DEFAULT_LINE_FLOOR):
    """(spect - cont), with bins below eps_E * max(spect, cont) replaced by floor."""
    s = np.asarray(spect, dtype=np.float64)
    c = np.asarray(cont, dtype=np.float64)
    diff = s - c
    scale = np.fmax(np.abs(s), np.abs(c))
    with np.errstate(invalid="ignore"):
        real = np.isfinite(diff) & (diff > eps_E * scale)
    out = np.full(diff.shape, floor, dtype=np.float64)
    out[real] = diff[real]
    return out


def noise_report(spect, cont, incident=None, transmitted=None, quantile=0.999, verbose=True):
    """Estimate noise floors from raw arrays (shape (n_spectra, n_energy)); modifies nothing.

    Real line emission only ADDS flux to the continuum, so negative values of
    (spect - cont) are pure noise; for symmetric noise their size measures the
    positive noise too. Likewise T > 1 is pure noise if ``transmitted``
    contains no emission. Suggested eps = 2 x the chosen quantile of that tail.
    ``quantile_in_f32eps`` gives it in units of float32 epsilon (1.19e-7): a few
    => plain float32 round-off of the files; tens to hundreds => noise from the
    single-precision calculation itself.
    """
    f32eps = float(np.finfo(np.float32).eps)

    def tail(x):
        if x.size == 0:
            return {"n": 0, "quantile": np.nan, "max": np.nan, "quantile_in_f32eps": np.nan, "suggested": np.nan}
        q = float(np.quantile(x, quantile))
        return {"n": int(x.size), "quantile": q, "max": float(x.max()),
                "quantile_in_f32eps": q / f32eps, "suggested": 2.0 * q}

    s = np.asarray(spect, dtype=np.float64)
    c = np.asarray(cont, dtype=np.float64)
    diff, scale = s - c, np.fmax(np.abs(s), np.abs(c))
    ok = np.isfinite(diff) & (scale > 0)
    rel = diff[ok] / scale[ok]
    report = {"emission": tail(-rel[rel < 0])}
    report["emission"]["frac_exactly_zero"] = float(np.mean(rel == 0)) if rel.size else np.nan
    report["emission"]["frac_negative"] = float(np.mean(rel < 0)) if rel.size else np.nan
    report["eps_E_suggested"] = report["emission"]["suggested"]
    if incident is not None and transmitted is not None:
        inc = np.asarray(incident, dtype=np.float64)
        trn = np.asarray(transmitted, dtype=np.float64)
        okT = np.isfinite(inc) & np.isfinite(trn) & (inc > 0)
        T = trn[okT] / inc[okT]
        report["transmission"] = tail(T[T > 1.0] - 1.0)
        report["transmission"]["frac_above_one"] = float(np.mean(T > 1.0)) if T.size else np.nan
        report["eps_T_suggested"] = report["transmission"]["suggested"]
    if verbose:
        for key in ("emission", "transmission"):
            if key in report:
                r = report[key]
                print(f"{key:12s}: n_noise_bins={r['n']}, noise@q={r['quantile']:.2e} "
                      f"({r['quantile_in_f32eps']:.1f} x f32eps), max={r['max']:.2e}, "
                      f"suggested eps={r['suggested']:.2e}")
    return report
