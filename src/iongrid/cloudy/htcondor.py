from __future__ import annotations

import random
from pathlib import Path

from .grid import DEFAULT_CLOUDY, MANIFEST_NAME, read_manifest, run_single_point, write_manifest

CONDOR_DIRNAME = "htc_submit"


def _resolve_python_bin(python_path: str | None) -> str:
    """Return a Python executable path for the wrapper script.

    Accepts either a direct executable path, a directory containing a Python
    interpreter, or a bare command name.
    """
    if python_path is None:
        return "${PYTHON:-python3}"

    path = Path(python_path).expanduser()
    if path.is_dir():
        for candidate in (path / "python", path / "python3", path / "python3.12"):
            if candidate.exists():
                return str(candidate)
        return str(path / "python")

    if path.exists():
        return str(path)

    return str(path)


def run_condor_batch(
    grid_dir: str | Path,
    batch_file: str | Path,
    cloudy_path: str = DEFAULT_CLOUDY,
    *,
    make_binaries: bool = True,
    results_path: str | Path | None = None,
    verbose: bool = True,
) -> Path:
    """Run one HTCondor batch file containing one point id per line.

    Worker jobs write Cloudy outputs into the normal ``points/`` directories.
    They do not update the shared manifest; run ``refresh`` after the cluster
    jobs finish to recover the ordered manifest from the Cloudy stdout files.
    """
    grid_dir = Path(grid_dir)
    batch_file = Path(batch_file)
    manifest_path = grid_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    if not batch_file.exists():
        raise FileNotFoundError(f"Batch file not found: {batch_file}")

    point_ids = [line.strip() for line in batch_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    points = {pt.point_id: pt for pt in read_manifest(manifest_path)}
    missing = [pid for pid in point_ids if pid not in points]
    if missing:
        raise ValueError(f"Batch file contains point ids missing from manifest: {missing[:5]}")

    results = []
    for pid in point_ids:
        res = run_single_point(points[pid], grid_dir, cloudy_path, verbose=verbose)
        if res.status == "ok" and make_binaries:
            try:
                from .spectra import compile_node
                compile_node(res.dirpath(grid_dir))
            except Exception as e:
                if verbose:
                    print(f"[cloudy] Failed compiling {res.point_id}: {e}")
                res.status = "failed"
        results.append(res)
        if verbose:
            exec_str = f"{res.exec_time_s:.2f}s" if res.exec_time_s else "?"
            print(f"[cloudy] {res.point_id} {res.status} ({exec_str})")

    if results_path is None:
        if batch_file.parent.name == "batches":
            results_path = batch_file.parent.parent / "results" / f"{batch_file.stem}.csv"
        else:
            results_path = batch_file.parent / "results" / f"{batch_file.stem}.csv"
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(results, results_path)
    return results_path


def write_condor_submission(
    grid_dir: str | Path,
    *,
    cloudy_path: str = DEFAULT_CLOUDY,
    batch_size: int = 1,
    shuffle: bool = True,
    seed: int = 0,
    submit_dir: str | Path | None = None,
    python_path: str | None = None,
    request_cpus: int = 1,
    request_memory: str = "3G",
    requirements: str | None = None,
    getenv: str = "PYTHON",
    limit: int | None = None,
) -> Path:
    """Write HTCondor submit files for the current ordered manifest.

    The manifest itself is never shuffled. A shuffled copy of the pending point
    list is used only to write ``htc_submit/batches/batch_*.txt``.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if request_cpus < 1:
        raise ValueError("request_cpus must be >= 1")

    grid_dir = Path(grid_dir).resolve()
    manifest_path = grid_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    submit_dir = Path(submit_dir).resolve() if submit_dir is not None else grid_dir / CONDOR_DIRNAME
    batches_dir = submit_dir / "batches"
    logs_dir = submit_dir / "logs"
    results_dir = submit_dir / "results"
    batches_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    pending = [pt for pt in read_manifest(manifest_path) if pt.status != "ok"]
    if limit is not None:
        pending = pending[:limit]
    batch_points = list(pending)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(batch_points)

    batch_count = 0
    for start in range(0, len(batch_points), batch_size):
        points = batch_points[start:start + batch_size]
        batch_path = batches_dir / f"batch_{batch_count:06d}.txt"
        batch_path.write_text("\n".join(pt.point_id for pt in points) + "\n", encoding="utf-8")
        batch_count += 1

    wrapper_path = submit_dir / "cloudy_condor_wrapper.sh"
    python_bin = _resolve_python_bin(python_path)
    python_line = f"PYTHON_BIN={python_bin!r}\n"

    wrapper_path.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "\n"
        "PROCESS_ID=$1\n"
        f"GRID_DIR={grid_dir.as_posix()!r}\n"
        f"CLOUDY_PATH={cloudy_path!r}\n"
        f"{python_line}"
        f"SUBMIT_DIR={submit_dir.as_posix()!r}\n"
        "BATCH_FILE=$(printf \"%s/batches/batch_%06d.txt\" \"$SUBMIT_DIR\" \"$PROCESS_ID\")\n"
        "RESULTS_FILE=$(printf \"%s/results/batch_%06d.csv\" \"$SUBMIT_DIR\" \"$PROCESS_ID\")\n"
        "\n"
        "if [ ! -f \"$BATCH_FILE\" ]; then\n"
        "    echo \"ERROR: Batch file not found: $BATCH_FILE\"\n"
        "    exit 1\n"
        "fi\n"
        "\n"
        "echo \"Running Cloudy batch $PROCESS_ID from $BATCH_FILE\"\n"
        "\"$PYTHON_BIN\" -c \"from iongrid.cloudy.htcondor import run_condor_batch; import sys; run_condor_batch(sys.argv[1], sys.argv[2], cloudy_path=sys.argv[3], results_path=sys.argv[4])\" \\\n"
        "    \"$GRID_DIR\" \"$BATCH_FILE\" \"$CLOUDY_PATH\" \"$RESULTS_FILE\"\n",
        encoding="utf-8",
    )
    wrapper_path.chmod(0o755)

    submit_path = submit_dir / "grid_submission.sub"
    requirements_line = f"requirements = {requirements}\n" if requirements else ""
    submit_path.write_text(
        "# HTCondor submission file for Cloudy photoionization grid\n"
        f"executable   = {wrapper_path.as_posix()}\n"
        f"getenv       = {getenv}\n"
        f"output       = {logs_dir.as_posix()}/out.$(ClusterId).$(Process)\n"
        f"error        = {logs_dir.as_posix()}/err.$(ClusterId).$(Process)\n"
        f"log          = {logs_dir.as_posix()}/log.$(ClusterId).$(Process)\n"
        f"request_cpus = {request_cpus}\n"
        f"request_memory = {request_memory}\n"
        f"{requirements_line}"
        "when_to_transfer_output = on_exit\n"
        "\n"
        "arguments = $(Process)\n"
        f"queue {batch_count}\n",
        encoding="utf-8",
    )

    return submit_path
