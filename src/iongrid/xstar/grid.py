"""Small, script-configured XSTAR grids on the common iongrid axes."""

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


@dataclass
class XstarPoint:
    point_id: str
    logxi: float
    vturb: float
    log_nh: float
    log_density: float
    status: str = "ready"
    exec_time_s: float | None = None

    def dirpath(self, root: str | Path) -> Path:
        return Path(root) / POINTS_DIRNAME / self.point_id


def point_id(logxi: float, vturb: float, log_nh: float) -> str:
    return f"xi{logxi:.3f}_vt{vturb:.3f}_nh{log_nh:.3f}".replace(".", "p")


def read_manifest(path: str | Path) -> list[XstarPoint]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [XstarPoint(
            point_id=row["point_id"], logxi=float(row["logxi"]),
            vturb=float(row["vturb"]), log_nh=float(row["log_nh"]),
            log_density=float(row["log_density"]), status=row["status"],
            exec_time_s=float(row["exec_time_s"]) if row["exec_time_s"] else None,
        ) for row in csv.DictReader(handle)]


def write_manifest(points: list[XstarPoint], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("point_id", "logxi", "vturb", "log_nh", "log_density", "status", "exec_time_s"))
        for pt in points:
            writer.writerow((pt.point_id, pt.logxi, pt.vturb, pt.log_nh, pt.log_density,
                             pt.status, "" if pt.exec_time_s is None else f"{pt.exec_time_s:.3f}"))
    return path


def setup_grid(
    root: str | Path, *, logxi_values, vturb_values, log_nh_values,
    log_density: float, luminosity_38: float, xstar_options: dict | None = None,
    sed_file: str | Path | None = None,
) -> Path:
    """Write a manifest and one recorded XSTAR parameter set per grid point."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    axes = [tuple(map(float, values)) for values in (logxi_values, vturb_values, log_nh_values)]
    if any(not values or not np.all(np.isfinite(values)) for values in axes):
        raise ValueError("grid axes must contain finite values")
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
    ids = set()
    points = []
    for xi, vt, nh in product(*axes):
        pid = point_id(xi, vt, nh)
        if pid in ids:
            raise ValueError(f"Grid axes produce duplicate point ID: {pid}")
        ids.add(pid)
        pt = XstarPoint(pid, xi, vt, nh, float(log_density))
        run_dir = pt.dirpath(root)
        run_dir.mkdir(parents=True, exist_ok=True)
        params = {
            "spectrum": "pow", "trad": -1.0, "spectun": 0,
            "temperature": 400, "pressure": 0.03,
            "density": 10**log_density, "column": 10**nh,
            "rlogxi": xi, "vturbi": vt, "rlrad38": luminosity_38,
            "cfrac": 1.0, "lcpres": 0,
            "nsteps": 3, "niter": 0, "ncn2": 9999,
            "lwrite": 0, "lprint": 0, "mode": "ql", "modelname": pid,
        }
        params.update(options)
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
        points.append(pt)
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
