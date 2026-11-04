"""Filesystem records shared by local and cluster execution."""

import hashlib
import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import fcntl


def write(path, text):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(text)
            stream.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


@contextmanager
def lock(path, *, blocking=True):
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield stream.fileno()


def outcome(directory):
    path = directory / "run.json"
    return json.loads(path.read_text()) if path.exists() else {"status": "ready"}


def execute(directory, command, *, lease, env, stdout, success, inputs=None, outputs=(), timeout=None):
    """Run with a fresh result marker and separate standard error."""
    marker = directory / "run.json"
    checksums = json.loads((directory / "inputs.json").read_text())
    for name, checksum in checksums.items():
        path = directory / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            result = {"status": "failed", "error": f"Prepared input changed: {path}"}
            write(marker, json.dumps(result))
            (directory / "collect.json").unlink(missing_ok=True)
            return result
    write(marker, json.dumps({"status": "running"}))
    for name in outputs:
        (directory / name).unlink(missing_ok=True)
    (directory / f"{directory.name}.npz").unlink(missing_ok=True)
    (directory / "collect.json").unlink(missing_ok=True)
    start = perf_counter()
    code = None
    with (directory / stdout).open("wb") as out, (directory / "stderr.log").open("wb") as err:
        try:
            code = subprocess.run(command, cwd=directory, env=env, stdin=inputs or subprocess.DEVNULL,
                                  stdout=out, stderr=err, timeout=timeout, pass_fds=(lease,), check=False).returncode
        except (OSError, subprocess.TimeoutExpired) as error:
            err.write(f"{error}\n".encode())
    result = {"status": "completed" if code == 0 and success() else "failed",
              "seconds": perf_counter() - start, "returncode": code}
    write(marker, json.dumps(result))
    return result


def input_text(template, text, default):
    if template is not None and text is not None:
        raise ValueError("Choose either template or text")
    return text if text is not None else Path(template or default).read_text()


def install(directory, files):
    """Retain prepared files and protect existing results from changed inputs."""
    directory.mkdir(parents=True, exist_ok=True)
    checksums = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    record = (json.dumps(checksums, sort_keys=True, indent=2) + "\n").encode()
    files = dict(files, **{"inputs.json": record})
    with lock(directory / ".lock"):
        for name, content in files.items():
            path = directory / name
            if path.exists() and path.read_bytes() != content:
                raise ValueError(f"Prepared input changed: {path}")
        for name, content in files.items():
            path = directory / name
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
