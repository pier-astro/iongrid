"""Small, script-configured XSTAR grids on the common iongrid axes.

Two grid modes are supported:

* Cartesian: the outer product of per-axis value lists.
* LHS: an explicit list of scattered points read from a CSV file, e.g. the one written by the Latin-hypercube generator (make-lhs.py).

Optional axes (backward compatible: ``None`` means "not part of this grid", and existing run directories keep identical point IDs and XSTAR parameters):
* ``feabund``: iron abundance relative to selcted abundance table, written to XSTAR as ``ABUNDANCE_KEY`` (default ``fe``).
* ``gamma``: photon index of XSTAR's built-in power law, written as the energy index ``trad = 1 - gamma`` (gamma = 2 <-> trad = -1, the default).
             Not combinable with a custom ``sed_file``.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np

MANIFEST_NAME = "manifest.csv"
POINTS_DIRNAME = "points"

# XSTAR parameter name for the iron abundance (check against XSTAR build).
ABUNDANCE_KEY = "fe"

MANIFEST_FIELDS = (
    "point_id", "logxi", "vturb", "log_nh", "log_density",
    "status", "exec_time_s", "feabund", "gamma",
)

# Accepted column names in an LHS CSV (first match wins).
LHS_COLUMNS = {
    "logxi": ("logxi",),
    "vturb": ("vturb",),
    "log_nh": ("log_nh", "lognH"),
    "feabund": ("feabund",),
    "gamma": ("gamma",),
}


@dataclass
class XstarPoint:
    point_id: str
    logxi: float
    vturb: float
    log_nh: float
    log_density: float
    feabund: float | None = None
    gamma: float | None = None
    status: str = "ready"
    exec_time_s: float | None = None

    def dirpath(self, root: str | Path) -> Path:
        return Path(root) / POINTS_DIRNAME / self.point_id


def gamma_to_trad(gamma: float) -> float:
    """Photon index -> XSTAR power-law energy index (Gamma=2 -> trad=-1)."""
    return 1.0 - float(gamma)


def point_id(logxi: float, vturb: float, log_nh: float,
             feabund: float | None = None, gamma: float | None = None) -> str:
    pid = f"xi{logxi:.3f}_vt{vturb:.3f}_nh{log_nh:.3f}"
    if feabund is not None:
        pid += f"_fe{feabund:.3f}"
    if gamma is not None:
        pid += f"_g{gamma:.3f}"
    return pid.replace(".", "p")


def _opt_float(value) -> float | None:
    return float(value) if value not in (None, "") else None


def read_manifest(path: str | Path) -> list[XstarPoint]:
    """Read a manifest; files written before feabund/gamma existed still load."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [XstarPoint(
            point_id=row["point_id"], logxi=float(row["logxi"]),
            vturb=float(row["vturb"]), log_nh=float(row["log_nh"]),
            log_density=float(row["log_density"]), status=row["status"],
            exec_time_s=_opt_float(row.get("exec_time_s")),
            feabund=_opt_float(row.get("feabund")),
            gamma=_opt_float(row.get("gamma")),
        ) for row in csv.DictReader(handle)]


def write_manifest(points: list[XstarPoint], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(MANIFEST_FIELDS)
        for pt in points:
            writer.writerow((
                pt.point_id, pt.logxi, pt.vturb, pt.log_nh, pt.log_density,
                pt.status, "" if pt.exec_time_s is None else f"{pt.exec_time_s:.3f}",
                "" if pt.feabund is None else pt.feabund,
                "" if pt.gamma is None else pt.gamma,
            ))
    return path


# --------------------------------------------------------------------------
# Point generation
# --------------------------------------------------------------------------

def _finite_axis(values, name):
    axis = tuple(map(float, values))
    if not axis or not np.all(np.isfinite(axis)):
        raise ValueError(f"grid axis {name} must contain finite values")
    return axis


def cartesian_points(logxi_values, vturb_values, log_nh_values, log_density,
                     feabund_values=None, gamma_values=None) -> list[XstarPoint]:
    axes = [_finite_axis(logxi_values, "logxi"), _finite_axis(vturb_values, "vturb"),
            _finite_axis(log_nh_values, "log_nh")]
    fe_axis = _finite_axis(feabund_values, "feabund") if feabund_values is not None else (None,)
    g_axis = _finite_axis(gamma_values, "gamma") if gamma_values is not None else (None,)
    points, ids = [], set()
    for xi, vt, nh, fe, g in product(*axes, fe_axis, g_axis):
        pid = point_id(xi, vt, nh, fe, g)
        if pid in ids:
            raise ValueError(f"Grid axes produce duplicate point ID: {pid}")
        ids.add(pid)
        points.append(XstarPoint(pid, xi, vt, nh, float(log_density), feabund=fe, gamma=g))
    return points


def _find_column(fieldnames, key):
    for candidate in LHS_COLUMNS[key]:
        if candidate in fieldnames:
            return candidate
    return None


def lhs_points(lhs_file: str | Path, log_density: float) -> list[XstarPoint]:
    """Read scattered grid points from a CSV (one XSTAR run per row).

    Required columns: logxi, vturb, log_nh (or lognH). Optional: feabund,
    gamma. Point IDs are ``lhs<row index>``; the existing
    "parameters changed for completed point" check protects against a CSV
    that is edited or reordered after points have been run.
    """
    path = Path(lhs_file)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        names = reader.fieldnames or []
        cols = {key: _find_column(names, key) for key in LHS_COLUMNS}
        missing = [k for k in ("logxi", "vturb", "log_nh") if cols[k] is None]
        if missing:
            raise ValueError(f"{path}: missing required column(s) {missing}; found {names}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    width = max(5, len(str(len(rows) - 1)))
    points = []
    for i, row in enumerate(rows):
        vals = {k: _opt_float(row[c]) if c else None
                for k, c in cols.items()}
        for k in ("logxi", "vturb", "log_nh"):
            if vals[k] is None or not np.isfinite(vals[k]):
                raise ValueError(f"{path}, row {i + 2}: invalid {k}")
        if vals["vturb"] <= 0:
            raise ValueError(f"{path}, row {i + 2}: vturb must be positive")
        points.append(XstarPoint(
            f"lhs{i:0{width}d}", vals["logxi"], vals["vturb"], vals["log_nh"],
            float(log_density), feabund=vals["feabund"], gamma=vals["gamma"],
        ))
    return points


# --------------------------------------------------------------------------
# Setup / run
# --------------------------------------------------------------------------

def setup_grid(
    root: str | Path, *, logxi_values=None, vturb_values=None, log_nh_values=None,
    feabund_values=None, gamma_values=None, lhs_file: str | Path | None = None,
    log_density: float, luminosity_38: float, xstar_options: dict | None = None,
    sed_file: str | Path | None = None,
) -> Path:
    """Write a manifest and one recorded XSTAR parameter set per grid point.

    Give either the three Cartesian axes (optionally with ``feabund_values`` and
    ``gamma_values``) or ``lhs_file``.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cartesian = [logxi_values, vturb_values, log_nh_values]
    if lhs_file is not None:
        if any(v is not None for v in cartesian + [feabund_values, gamma_values]):
            raise ValueError("lhs_file cannot be combined with Cartesian axis values")
    elif any(v is None for v in cartesian):
        raise ValueError("give logxi_values, vturb_values and log_nh_values, or lhs_file")
    if not np.isfinite(log_density) or not np.isfinite(luminosity_38) or luminosity_38 <= 0:
        raise ValueError("density and luminosity must be finite; luminosity_38 must be positive")
    old = {pt.point_id: pt for pt in read_manifest(root / MANIFEST_NAME)} if (root / MANIFEST_NAME).exists() else {}
    options = dict(xstar_options or {})
    reserved = {"density", "column", "rlogxi", "vturbi", "rlrad38", "lcpres", "modelname"}
    if reserved & options.keys():
        raise ValueError(f"XSTAR grid controls cannot be overridden: {sorted(reserved & options.keys())}")
    sed = Path(sed_file).expanduser().resolve() if sed_file is not None else None
    if sed is not None and not sed.is_file():
        raise FileNotFoundError(sed)

    if lhs_file is not None:
        points = lhs_points(lhs_file, log_density)
    else:
        points = cartesian_points(logxi_values, vturb_values, log_nh_values, log_density,
                                  feabund_values, gamma_values)

    uses_gamma = any(pt.gamma is not None for pt in points)
    uses_fe = any(pt.feabund is not None for pt in points)
    if uses_gamma and sed is not None:
        raise ValueError("gamma is set through XSTAR's power law and cannot be combined with sed_file")
    if uses_gamma and "trad" in options:
        raise ValueError("'trad' is set from gamma; remove it from xstar_options")
    if uses_fe and ABUNDANCE_KEY in options:
        raise ValueError(f"{ABUNDANCE_KEY!r} is set from feabund; remove it from xstar_options")

    for pt in points:
        pid = pt.point_id
        run_dir = pt.dirpath(root)
        run_dir.mkdir(parents=True, exist_ok=True)
        params = {
            "spectrum": "pow", "trad": -1.0, "spectun": 0,
            "temperature": 400, "pressure": 0.03,
            "density": 10**log_density, "column": 10**pt.log_nh,
            "rlogxi": pt.logxi, "vturbi": pt.vturb, "rlrad38": luminosity_38,
            "cfrac": 1.0, "lcpres": 0,
            "nsteps": 3, "niter": 0, "ncn2": 9999,
            "lwrite": 0, "lprint": 0, "mode": "ql", "modelname": pid,
        }
        params.update(options)
        if pt.gamma is not None:
            params["trad"] = gamma_to_trad(pt.gamma)
        if pt.feabund is not None:
            params[ABUNDANCE_KEY] = pt.feabund
        if sed is not None:
            params.update(spectrum="file", spectrum_file=sed.name, spectun=0)
            target = run_dir / sed.name
            if target.resolve() != sed:
                if pid in old and old[pid].status == "ok" and target.exists() and target.read_bytes() != sed.read_bytes():
                    raise ValueError(f"SED changed for completed point {pid}; use a new run directory")
                if not target.exists() or target.read_bytes() != sed.read_bytes():
                    shutil.copy2(sed, target)
        params_path = run_dir / "xstar-params.json"
        encoded = json.dumps(params, indent=2, sort_keys=True) + "\n"
        if pid in old and old[pid].status == "ok" and params_path.exists() and params_path.read_text() != encoded:
            raise ValueError(f"Parameters changed for completed point {pid}; use a new run directory")
        params_path.write_text(encoded, encoding="utf-8")
        if pid in old:
            pt.status, pt.exec_time_s = old[pid].status, old[pid].exec_time_s
    return write_manifest(points, root / MANIFEST_NAME)


def run_single_point(pt: XstarPoint, root: str | Path, xstar_path: str = "xstar", timeout: float | None = None) -> XstarPoint:
    """Run one XSTAR process in its own directory without touching the shared manifest."""
    directory = pt.dirpath(root)
    params = json.loads((directory / "xstar-params.json").read_text(encoding="utf-8"))
    command = [xstar_path, *(f"{key}={value}" for key, value in params.items())]
    for filename in ("xout_spect1.fits", "xout_cont1.fits"):
        (directory / filename).unlink(missing_ok=True)
    start = perf_counter()
    try:
        with (directory / "xstar.stdout").open("w") as out, (directory / "xstar.stderr").open("w") as err:
            result = subprocess.run(command, cwd=directory, stdout=out, stderr=err, timeout=timeout, check=False)
        code = result.returncode
    except (OSError, subprocess.TimeoutExpired) as error:
        (directory / "xstar.stderr").write_text(str(error) + "\n", encoding="utf-8")
        code = 1
    pt.exec_time_s = perf_counter() - start
    (directory / "xstar.exitcode").write_text(f"{code}\n", encoding="ascii")
    pt.status = "ok" if code == 0 and _has_outputs(directory) else "failed"
    return pt


def _has_outputs(directory: Path) -> bool:
    return all((directory / filename).is_file() for filename in ("xout_spect1.fits", "xout_cont1.fits"))


def refresh_grid(root: str | Path) -> Path:
    root = Path(root)
    path = root / MANIFEST_NAME
    points = read_manifest(path)
    for pt in points:
        directory = pt.dirpath(root)
        marker = directory / "xstar.exitcode"
        if marker.exists():
            pt.status = "ok" if marker.read_text().strip() == "0" and _has_outputs(directory) else "failed"
        elif _has_outputs(directory):
            pt.status = "ok"  # Import existing completed XSTAR runs.
    return write_manifest(points, path)


def pack_grid_npz(root: str | Path) -> Path:
    from .spectra import compile_node
    root = Path(root)
    path = root / MANIFEST_NAME
    for pt in read_manifest(path):
        if pt.status == "ok":
            compile_node(pt.dirpath(root), pt)
    return path


def run_grid(root: str | Path, action: str, *, workers: int = 1, limit: int | None = None,
             xstar_path: str = "xstar", timeout: float | None = None, **setup_options) -> Path:
    """Use the same setup/run/refresh/packnpz actions as the CLOUDY recipes."""
    root = Path(root)
    if action == "setup":
        return setup_grid(root, **setup_options)
    if action == "refresh":
        return refresh_grid(root)
    if action == "packnpz":
        return pack_grid_npz(root)
    if action != "run":
        raise ValueError("action must be setup, run, refresh, or packnpz")
    path = setup_grid(root, **setup_options)
    pending = [pt for pt in read_manifest(path) if pt.status != "ok"]
    if limit is not None:
        pending = pending[:limit]
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        results = [run_single_point(pt, root, xstar_path, timeout) for pt in pending]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = [future.result() for future in as_completed([
                pool.submit(run_single_point, pt, root, xstar_path, timeout) for pt in pending
            ])]
    updated = {pt.point_id: pt for pt in results}
    write_manifest([updated.get(pt.point_id, pt) for pt in read_manifest(path)], path)
    return path