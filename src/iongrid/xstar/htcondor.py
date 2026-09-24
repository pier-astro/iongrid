"""Generate HTCondor batches for independent XSTAR point runs."""

from __future__ import annotations

import random
import shlex
from pathlib import Path

from .grid import MANIFEST_NAME, read_manifest, run_single_point, write_manifest


def run_condor_batch(root: str | Path, batch_file: str | Path, *,
                     xstar_path: str = "xstar", results_path: str | Path | None = None) -> Path:
    """Run a batch without writing the shared manifest; refresh it afterwards."""
    root = Path(root)
    batch_file = Path(batch_file)
    points = {pt.point_id: pt for pt in read_manifest(root / MANIFEST_NAME)}
    point_ids = [line.strip() for line in batch_file.read_text().splitlines() if line.strip()]
    if any(pid not in points for pid in point_ids):
        raise ValueError("Batch refers to a point absent from manifest")
    results = [run_single_point(points[pid], root, xstar_path) for pid in point_ids]
    for point in results:
        print(f"[xstar] {point.point_id}: {point.status} ({point.exec_time_s:.1f}s)")
    if results_path is None:
        results_path = batch_file.parent.parent / "results" / f"{batch_file.stem}.csv"
    return write_manifest(results, results_path)


def write_condor_submission(
    root: str | Path, *, xstar_path: str = "xstar", batch_size: int = 1,
    shuffle: bool = True, seed: int = 0, limit: int | None = None,
    python_path: str = "python3", headas_init: str | Path | None = None,
    request_cpus: int = 1, request_memory: str = "3G",
    requirements: str | None = None,
) -> Path:
    """Write submit files for pending points on a shared filesystem."""
    if batch_size < 1 or request_cpus < 1:
        raise ValueError("batch_size and request_cpus must be positive")
    root = Path(root).resolve()
    pending = [pt for pt in read_manifest(root / MANIFEST_NAME) if pt.status != "ok"]
    if limit is not None:
        pending = pending[:limit]
    if shuffle:
        random.Random(seed).shuffle(pending)
    submit_dir = root / "htc_submit"
    for part in ("batches", "results", "logs"):
        (submit_dir / part).mkdir(parents=True, exist_ok=True)
    batch_count = 0
    for start in range(0, len(pending), batch_size):
        ids = [pt.point_id for pt in pending[start:start + batch_size]]
        (submit_dir / "batches" / f"batch_{batch_count:06d}.txt").write_text("\n".join(ids) + "\n")
        batch_count += 1
    if batch_count == 0:
        raise ValueError("No pending XSTAR points to submit")
    source_line = ""
    if headas_init is not None:
        init = Path(headas_init).expanduser().resolve()
        source_line = f"source {shlex.quote(str(init))}\n"
    wrapper = submit_dir / "xstar_condor_wrapper.sh"
    wrapper.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        + source_line
        + f"ROOT={shlex.quote(str(root))}\n"
        + f"SUBMIT={shlex.quote(str(submit_dir))}\n"
        + "BATCH=$(printf '%s/batches/batch_%06d.txt' \"$SUBMIT\" \"$1\")\n"
        + "RESULTS=$(printf '%s/results/batch_%06d.csv' \"$SUBMIT\" \"$1\")\n"
        + f"{shlex.quote(python_path)} -m iongrid.xstar.worker \"$ROOT\" \"$BATCH\" \"$RESULTS\" {shlex.quote(xstar_path)}\n"
    )
    wrapper.chmod(0o755)
    submit = submit_dir / "grid_submission.sub"
    lines = [
        f"executable = {wrapper}", "getenv = True",
        f"output = {submit_dir / 'logs' / 'out.$(ClusterId).$(Process)'}",
        f"error = {submit_dir / 'logs' / 'err.$(ClusterId).$(Process)'}",
        f"log = {submit_dir / 'logs' / 'log.$(ClusterId).$(Process)'}",
        f"request_cpus = {request_cpus}", f"request_memory = {request_memory}",
    ]
    if requirements:
        lines.append(f"requirements = {requirements}")
    lines += ["arguments = $(Process)", f"queue {batch_count}"]
    submit.write_text("\n".join(lines) + "\n")
    return submit
