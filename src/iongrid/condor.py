"""Write HTCondor batches for a campaign on a shared filesystem."""

import json
import random
import re
import shlex
import sys
from pathlib import Path
from tempfile import mkdtemp

from .grid import Grid


def write(grid, *, batch=16, fragment=None, text=None, init=None, select="ready",
          seed=0, executable=None, python="python3"):
    """Write a native submit file. The caller submits it with condor_submit."""
    if batch < 1 or (fragment is not None and text is not None):
        raise ValueError("Use a positive batch size and at most one submit fragment")
    directives = Path(fragment).read_text() if fragment is not None else text or ""
    if re.search(r"^\s*(?:queue\b|(?:executable|arguments|initialdir)\s*=)", directives, re.I | re.M):
        raise ValueError("The submit fragment must leave queue, executable, arguments and initialdir to iongrid")
    grid.refresh()
    rows = grid._select(select)
    if not rows:
        raise ValueError(f"No {select} points")
    random.Random(seed).shuffle(rows)
    directory = Path(mkdtemp(prefix="condor-", dir=grid.root))
    for number, start in enumerate(range(0, len(rows), batch)):
        ids = [row["point_id"] for row in rows[start:start + batch]]
        (directory / f"{number}.json").write_text(json.dumps(ids))

    wrapper = directory / "worker.sh"
    setup = f"source {shlex.quote(str(Path(init).resolve()))}\n" if init is not None else ""
    command = shlex.join([str(python), "-B", "-m", "iongrid.condor", str(grid.root)])
    options = shlex.join([select, str(executable or grid.config["engine"])])
    wrapper.write_text(f'#!/bin/bash\nset -e\n{setup}{command} "$1.json" {options}\n')
    wrapper.chmod(0o755)
    submit = directory / "grid.sub"
    submit.write_text(
        f'executable = "{wrapper}"\ninitialdir = "{directory}"\n'
        "arguments = $(Process)\n"
        "output = out.$(ClusterId).$(Process)\n"
        "error = err.$(ClusterId).$(Process)\n"
        "log = events.$(ClusterId).log\n"
        f"{directives.rstrip()}\nqueue {number + 1}\n"
    )
    return submit


def _worker(root, batch, select, executable):
    grid = Grid.open(root)
    rows = {row["point_id"]: row for row in grid._read()}
    for point_id in json.loads(Path(batch).read_text()):
        row = dict(rows[point_id], status=select)
        grid._run(row, executable=executable)


if __name__ == "__main__":
    _worker(*sys.argv[1:])
