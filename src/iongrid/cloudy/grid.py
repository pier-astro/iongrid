from __future__ import annotations

import csv
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

DEFAULT_CLOUDY = "cloudy"
DEFAULT_ROOT = Path(".")
MANIFEST_NAME = "manifest.csv"
TEMPLATE_NAME = "cloudy-template.in"
POINTS_DIRNAME = "points"


@dataclass
class CloudyPoint:
    """One local Cloudy plasma state plus metadata needed to locate products."""

    point_id: str
    logxi: float
    vturb: float
    log_density: float
    log_nh: float
    status: str = "ready"
    exec_time_s: float | None = None

    def dirpath(self, root: str | Path = DEFAULT_ROOT) -> Path:
        """Return the directory that stores this point's raw Cloudy files."""
        return Path(root) / POINTS_DIRNAME / self.point_id

    def filepath(self, suffix: str, root: str | Path = DEFAULT_ROOT) -> Path:
        """Return the path to one standard Cloudy product for this point."""
        return self.dirpath(root) / f"{self.point_id}.{suffix}"


def point_id(logxi: float, vturb: float, log_nh: float) -> str:
    """Build a short stable identifier used for folder and file names."""
    xi = f"{logxi:.3f}".replace(".", "p")
    vt = int(round(vturb))
    nh = f"{log_nh:.3f}".replace(".", "p")
    return f"xi{xi}_vt{vt:04d}_nh{nh}"


def expected_files(point: CloudyPoint, root: str | Path = DEFAULT_ROOT) -> dict[str, Path]:
    """Map one local plasma point to its retained standard Cloudy products."""
    return {
        "input": point.filepath("in", root),
        "stdout": point.filepath("out", root),
        "overview": point.filepath("ovr", root),
        "species_abundance": point.filepath("spc", root),
        "line_list": point.filepath("lin", root),
        "line_labels": point.filepath("llab", root),
        "continuum": point.filepath("con", root),
        "raw": point.filepath("raw", root),
        "optical_depth": point.filepath("opt", root),
        "fine_optical_depth": point.filepath("fopt", root),
    }



def make_local_points(
    *,
    logxi_values=None,
    vturb_values=None,
    log_nh_values=None,
    log_density: float = 10.0,
) -> list[CloudyPoint]:
    """Create local plasma grid points before anything is written."""
    xi_vals = logxi_values if logxi_values is not None else np.linspace(3.0, 6.5, 5).tolist()
    vt_vals = vturb_values if vturb_values is not None else np.linspace(100.0, 1000.0, 5).tolist()
    nh_vals = log_nh_values if log_nh_values is not None else np.linspace(21.0, 24.0, 4).tolist()
    return [
        CloudyPoint(
            point_id=point_id(float(xi), float(vt), float(nh)),
            logxi=float(xi),
            vturb=float(vt),
            log_density=float(log_density),
            log_nh=float(nh),
        )
        for xi in xi_vals
        for vt in vt_vals
        for nh in nh_vals
    ]


def read_manifest(manifest_path: str | Path) -> list[CloudyPoint]:
    """Read the manifest CSV file into a list of CloudyPoint objects."""
    points = []
    with Path(manifest_path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            points.append(CloudyPoint(
                point_id=row["point_id"],
                logxi=float(row["logxi"]),
                vturb=float(row["vturb"]),
                log_density=float(row["log_density"]),
                log_nh=float(row["log_nh"]),
                status=row["status"],
                exec_time_s=float(row["exec_time_s"]) if row.get("exec_time_s") else None,
            ))
    return points


def write_manifest(points: list[CloudyPoint], manifest_path: str | Path) -> Path:
    """Write the compact manifest that indexes the point directories."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["point_id", "logxi", "vturb", "log_density", "log_nh", "status", "exec_time_s"])
        writer.writeheader()
        for pt in points:
            writer.writerow({
                "point_id": pt.point_id,
                "logxi": f"{pt.logxi:.6g}",
                "vturb": f"{pt.vturb:.6g}",
                "log_density": f"{pt.log_density:.6g}",
                "log_nh": f"{pt.log_nh:.6g}",
                "status": pt.status,
                "exec_time_s": "" if pt.exec_time_s is None else f"{pt.exec_time_s:.3f}",
            })
    return manifest_path


def write_grid(
    root: str | Path,
    logxi_values: list[float] | None = None,
    vturb_values: list[float] | None = None,
    log_nh_values: list[float] | None = None,
    log_density: float = 10.0,
    template_path: str | Path | None = None,
    *,
    sed_slope: float | None = None,
    sed_file: str | Path | None = None,
) -> Path:
    """Write the Cloudy input files and copy default template locally if missing."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    # Determine the SED command to replace the {sed_command} template placeholder
    if sed_file is not None:
        sed_file_path = Path(sed_file)
        if not sed_file_path.exists():
            raise FileNotFoundError(f"SED file not found: {sed_file_path}")

        sed_filename = sed_file_path.name
        local_sed_path = Path(root) / sed_filename
        if sed_file_path.resolve() != local_sed_path.resolve():
            import shutil
            shutil.copy2(sed_file_path, local_sed_path)

        sed_command = f'table SED "{sed_filename}"'
    elif sed_slope is not None:
        sed_command = f'power law, slope={sed_slope:.6g} hi cut 1000 Ryd low cut 1 Ryd'
    else:
        sed_command = 'power law, slope=-0.8 hi cut 1000 Ryd low cut 1 Ryd'

    # Localize template: check root/cloudy-template.in, copy package default if missing
    if template_path is None:
        local_template = root / TEMPLATE_NAME
        if not local_template.exists():
            default_template = Path(__file__).parent / TEMPLATE_NAME
            local_template.write_text(default_template.read_text(encoding="utf-8"), encoding="utf-8")
        template_path = local_template

    template_text = Path(template_path).read_text(encoding="utf-8")

    # Read existing manifest to preserve statuses of unmodified points
    existing_status = {}
    manifest_path = root / MANIFEST_NAME
    if manifest_path.exists():
        for pt in read_manifest(manifest_path):
            # Map by (rounded_logxi, rounded_vturb, rounded_log_nh)
            key = (round(pt.logxi, 5), round(pt.vturb, 3), round(pt.log_nh, 5))
            existing_status[key] = (pt.status, pt.exec_time_s, pt.point_id)

    # Process input values
    if logxi_values is None:
        xi_vals = np.linspace(3.0, 6.5, 5).tolist()
    elif isinstance(logxi_values, (int, float)):
        xi_vals = [float(logxi_values)]
    else:
        xi_vals = [float(xi) for xi in logxi_values]

    if vturb_values is None:
        vt_vals = np.linspace(100.0, 1000.0, 5).tolist()
    elif isinstance(vturb_values, (int, float)):
        vt_vals = [float(vturb_values)]
    else:
        vt_vals = [float(vt) for vt in vturb_values]

    if log_nh_values is None:
        nh_vals = np.linspace(21.0, 24.0, 4).tolist()
    elif isinstance(log_nh_values, (int, float)):
        nh_vals = [float(log_nh_values)]
    else:
        nh_vals = [float(nh) for nh in log_nh_values]

    # Generate points and write inputs
    points = []
    for xi in xi_vals:
        for vt in vt_vals:
            for nh in nh_vals:
                pid = point_id(xi, vt, nh)
                key = (round(float(xi), 5), round(float(vt), 3), round(float(nh), 5))
                status, exec_time, old_pid = existing_status.get(key, ("ready", None, None))

                # Handle renaming and directory tracking for grid extensions / updates
                if old_pid is not None:
                    old_dir = root / POINTS_DIRNAME / old_pid
                    new_dir = root / POINTS_DIRNAME / pid
                    if new_dir.exists():
                        pass
                    elif old_dir.exists():
                        old_dir.rename(new_dir)
                        # Rename all files starting with old_pid in the renamed directory
                        for f in new_dir.iterdir():
                            if f.is_file() and f.name.startswith(old_pid):
                                suffix = f.name[len(old_pid):]
                                f.rename(new_dir / f"{pid}{suffix}")
                    else:
                        # Reset status if files are missing physically
                        status = "ready"
                        exec_time = None

                pt = CloudyPoint(
                    point_id=pid,
                    logxi=float(xi),
                    vturb=float(vt),
                    log_density=float(log_density),
                    log_nh=float(nh),
                    status=status,
                    exec_time_s=exec_time,
                )
                points.append(pt)

                run_dir = pt.dirpath(root)
                run_dir.mkdir(parents=True, exist_ok=True)



                ctx = {
                    "title": pid,
                    "point_id": pid,
                    "logxi": f"{pt.logxi:.6g}",
                    "log_density": f"{pt.log_density:.6g}",
                    "vturb": f"{pt.vturb:.6g}",
                    "log_nh": f"{pt.log_nh:.6g}",
                    "overview_file": f"{pid}.ovr",
                    "species_abundance_file": f"{pid}.spc",
                    "line_list_file": f"{pid}.lin",
                    "line_labels_file": f"{pid}.llab",
                    "continuum_file": f"{pid}.con",
                    "raw_file": f"{pid}.raw",
                    "optical_depth_file": f"{pid}.opt",
                    "fine_optical_depth_file": f"{pid}.fopt",
                    "sed_command": sed_command,
                }
                pt.filepath("in", root).write_text(template_text.format_map(ctx), encoding="utf-8")

    write_manifest(points, manifest_path)
    return manifest_path

def _find_cloudy_data_dir(cloudy_path: str | Path) -> Path | None:
    path = Path(cloudy_path).resolve()
    for parent in [path.parent, path.parent.parent, path.parent.parent.parent, path.parent.parent.parent.parent]:
        candidate = parent / "data"
        if candidate.is_dir() and (candidate / "checksums.dat").exists():
            return candidate
    return None


def run_single_point(
    pt: CloudyPoint,
    grid_dir: Path,
    cloudy_path: str,
    verbose: bool = True,
) -> CloudyPoint:
    """Run a single Cloudy simulation point.

    Args:
        pt (CloudyPoint): The point configuration.
        grid_dir (Path): Root directory of the grid.
        cloudy_path (str): Path to the Cloudy executable.
        verbose (bool): If True, prints failures. Defaults to True.

    Returns:
        CloudyPoint: Updated point containing status and execution time.
    """
    in_path = pt.filepath("in", grid_dir)
    out_path = pt.filepath("out", grid_dir)
    start = perf_counter()
    try:
        import os
        env = os.environ.copy()
        grid_dir_resolved = Path(grid_dir).resolve().as_posix()
        existing_path = env.get("CLOUDY_DATA_PATH", "")
        if not existing_path:
            default_data = _find_cloudy_data_dir(cloudy_path)
            if default_data:
                existing_path = default_data.as_posix()

        if existing_path:
            env["CLOUDY_DATA_PATH"] = f"{grid_dir_resolved}{os.pathsep}{existing_path}"
        else:
            env["CLOUDY_DATA_PATH"] = grid_dir_resolved

        with in_path.open("rb") as stdin, out_path.open("wb") as stdout:
            subprocess.run(
                [cloudy_path],
                stdin=stdin,
                stdout=stdout,
                stderr=subprocess.STDOUT,
                cwd=pt.dirpath(grid_dir),
                env=env,
                check=True,
            )
        exec_time = perf_counter() - start
        out_text = out_path.read_text(encoding="utf-8", errors="ignore")
        status = "ok" if "Cloudy exited OK" in out_text else "failed"
        match = re.search(r"ExecTime\(s\)\s+([0-9]+(?:\.[0-9]+)?)", out_text)
        if match:
            exec_time = float(match.group(1))
        return CloudyPoint(
            point_id=pt.point_id,
            logxi=pt.logxi,
            vturb=pt.vturb,
            log_density=pt.log_density,
            log_nh=pt.log_nh,
            status=status,
            exec_time_s=exec_time,
        )
    except Exception as e:
        if verbose:
            print(f"[cloudy] Failed running {pt.point_id}: {e}")
        return CloudyPoint(
            point_id=pt.point_id,
            logxi=pt.logxi,
            vturb=pt.vturb,
            log_density=pt.log_density,
            log_nh=pt.log_nh,
            status="failed",
            exec_time_s=None,
        )


def run_grid(
    grid_dir: str | Path,
    action: str = "run",
    *,
    logxi_values: list[float] | None = None,
    vturb_values: list[float] | None = None,
    log_nh_values: list[float] | None = None,
    log_density: float = 10.0,
    sed_slope: float | None = None,
    sed_file: str | Path | None = None,
    template_path: str | Path | None = None,
    cloudy_path: str = DEFAULT_CLOUDY,
    workers: int = 1,
    limit: int | None = None,
    verbose: bool = True,
    make_binaries: bool = True,
    energy_grid: np.ndarray | None = None,
) -> Path:
    """Setup, run, or refresh a Cloudy simulation grid.

    The grid setup creates a template input file for each combination of
    log(xi), vturb, and log(NH). Execution runs Cloudy across each point
    in the grid. Refreshing updates the manifest with execution times,
    statuses, and generates optional binaries.

    Parameters
    ----------
    grid_dir : str or Path
        Root directory of the grid.
    action : {"run", "setup", "refresh"}, default "run"
        Action to perform:
        * "setup": Create template input files for each combination of
          `logxi_values`, `vturb_values`, and `log_nh_values`.
        * "run": Run Cloudy for each point in the grid.
        * "refresh": Update the manifest with the execution times and
          statuses of each point.
    logxi_values : list of float, optional
        List of log(xi) values.
    vturb_values : list of float, optional
        List of vturb values.
    log_nh_values : list of float, optional
        List of log(NH) values.
    log_density : float, default 10.0
        log(density) value.
    sed_slope : float, optional
        Power law slope (default -0.8).
        Relation to Photon Index (PI) is: PI = 1 - alpha.
    sed_file : str or Path, optional
        Path to a custom SED file (overrides sed_slope if provided).
    template_path : str or Path, optional
        Path to the template input file.
    cloudy_path : str, default DEFAULT_CLOUDY
        Path to the Cloudy executable.
    workers : int, default 1
        Number of concurrent worker processes to use for running the grid.
    limit : int, optional
        Limit the number of simulation points to process.
    verbose : bool, default True
        If True, prints execution progress and statistics to stdout.
    make_binaries : bool, default True
        If True, produces `.npz` files containing optical depths and diffuse
        emission including micro-turbulence effects.
    energy_grid : np.ndarray, optional
        Energy grid used to generate binaries if `make_binaries` is True.
        If None, the default energy grid in cloudy.spectra is used.

    Returns
    -------
    Path
        Path to the generated or updated manifest file.

    Raises
    ------
    ValueError
        If an invalid `action` string is provided.
    """
    grid_dir = Path(grid_dir)
    manifest_path = grid_dir / MANIFEST_NAME

    if action == "setup":
        return write_grid(
            root=grid_dir,
            logxi_values=logxi_values,
            vturb_values=vturb_values,
            log_nh_values=log_nh_values,
            log_density=log_density,
            sed_slope=sed_slope,
            sed_file=sed_file,
            template_path=template_path,
        )

    if action == "run":
        write_grid(
            root=grid_dir,
            logxi_values=logxi_values,
            vturb_values=vturb_values,
            log_nh_values=log_nh_values,
            log_density=log_density,
            sed_slope=sed_slope,
            sed_file=sed_file,
            template_path=template_path,
        )

        points = read_manifest(manifest_path)
        pending = [pt for pt in points if pt.status != "ok"]
        if limit is not None:
            pending = pending[:limit]

        if pending:
            updated = {}
            if workers <= 1:
                for pt in pending:
                    res = run_single_point(pt, grid_dir, cloudy_path, verbose)
                    updated[res.point_id] = res
                    if verbose:
                        exec_str = f"{res.exec_time_s:.2f}s" if res.exec_time_s else "?"
                        print(f"[cloudy] {res.point_id} {res.status} ({exec_str})")
            else:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    futures = {executor.submit(run_single_point, pt, grid_dir, cloudy_path, verbose): pt for pt in pending}
                    for fut in as_completed(futures):
                        res = fut.result()
                        updated[res.point_id] = res
                        if verbose:
                            exec_str = f"{res.exec_time_s:.2f}s" if res.exec_time_s else "?"
                            print(f"[cloudy] {res.point_id} {res.status} ({exec_str})")

            # Merge results back into manifest
            final_points = []
            for pt in points:
                final_points.append(updated.get(pt.point_id, pt))
            write_manifest(final_points, manifest_path)

        manifest_path = refresh_grid(grid_dir, make_binaries=False, energy_grid=energy_grid)
        if make_binaries:
            manifest_path = pack_grid_npz(grid_dir, energy_grid=energy_grid)
        return manifest_path

    if action == "refresh":
        return refresh_grid(grid_dir, make_binaries=False, energy_grid=energy_grid)

    if action == "packnpz":
        return pack_grid_npz(grid_dir, energy_grid=energy_grid)

    raise ValueError(f"unknown action: {action!r} (expected 'setup', 'run', 'refresh', or 'packnpz')")


def refresh_grid(root: str | Path, make_binaries: bool = False, energy_grid: np.ndarray | None = None) -> Path:
    """Update point statuses and execution times in the manifest CSV from existing stdout log files."""
    root = Path(root)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    points = read_manifest(manifest_path)
    for pt in points:
        out_path = pt.filepath("out", root)
        if out_path.exists():
            out_text = out_path.read_text(encoding="utf-8", errors="ignore")
            pt.status = "ok" if "Cloudy exited OK" in out_text else "failed"
            match = re.search(r"ExecTime\(s\)\s+([0-9]+(?:\.[0-9]+)?)", out_text)
            if match:
                pt.exec_time_s = float(match.group(1))

    manifest_path = write_manifest(points, manifest_path)

    if make_binaries:
        manifest_path = pack_grid_npz(root, energy_grid=energy_grid)

    return manifest_path


def pack_grid_npz(root: str | Path, energy_grid: np.ndarray | None = None) -> Path:
    """Compile simulated grid nodes into binary .npz files for all successful grid points."""
    root = Path(root)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    points = read_manifest(manifest_path)
    from .spectra import compile_node, ENERGY_GRID
    if energy_grid is None:
        energy_grid = ENERGY_GRID

    manifest_changed = False
    for pt in points:
        if pt.status == "ok":
            try:
                compile_node(pt.dirpath(root), energy_grid=energy_grid)
            except Exception:
                pt.status = "failed"
                manifest_changed = True

    if manifest_changed:
        manifest_path = write_manifest(points, manifest_path)

    return manifest_path


def cloud_exited_ok(out_path: str | Path) -> bool:
    """Return whether a Cloudy stdout log reports a successful exit."""
    return "Cloudy exited OK" in Path(out_path).read_text(encoding="utf-8", errors="ignore")


def parse_exec_time(out_path: str | Path) -> float | None:
    """Extract ExecTime(s) from a Cloudy stdout log when present."""
    text = Path(out_path).read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"ExecTime\(s\)\s+([0-9]+(?:\.[0-9]+)?)", text)
    return float(match.group(1)) if match else None
