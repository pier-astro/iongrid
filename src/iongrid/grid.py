"""A disk-backed campaign of independently executable points."""

import csv
import hashlib
import io
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ._run import lock, outcome, write

_UNSET = object()
_RECORD = ("point_id", "status", "collection", "seconds", "sedkey")


def _engine(name):
    if name == "cloudy":
        from .cloudy.point import CloudyPoint, DEFAULT
        return CloudyPoint, DEFAULT
    if name == "xstar":
        from .xstar.point import XstarPoint, DEFAULT
        return XstarPoint, DEFAULT
    raise ValueError(f"Unknown engine {name!r}")


def _key(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _coordinates(row):
    return {name: value for name, value in row.items() if name not in _RECORD}


class Grid:
    """Open or prepare a campaign directory with a CSV manifest."""

    def __init__(self, root, config):
        self.root = root
        self.config = config
        self.point, _ = _engine(config["engine"])

    @classmethod
    def create(cls, root, *, engine, params, template=None, sed=None):
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        _, default = _engine(engine)
        deck = Path(template or default).read_text()
        config = {"engine": engine, "file_sed": sed is not None, "sedaxes": list(sed.axes) if sed else [],
                  "template_sha256": hashlib.sha256(deck.encode()).hexdigest()}
        grid = cls(root, config)
        with lock(root / ".iongrid.lock"):
            if (root / "campaign.json").exists() or grid.manifest.exists():
                raise FileExistsError(f"Campaign already exists: {root}")
            snapshot = root / "input.template"
            metadata = root / "campaign.json"
            if snapshot.exists():
                raise FileExistsError(f"Campaign template snapshot already exists: {snapshot}")
            try:
                write(snapshot, deck)
                write(metadata, json.dumps(config, indent=2) + "\n")
                grid._extend([], params, deck, sed)
            except BaseException:
                snapshot.unlink(missing_ok=True)
                metadata.unlink(missing_ok=True)
                raise
        return grid

    @classmethod
    def open(cls, root):
        root = Path(root).resolve()
        if not (root / "manifest.csv").is_file():
            raise FileNotFoundError(f"No campaign manifest in {root}")
        return cls(root, json.loads((root / "campaign.json").read_text()))

    @property
    def manifest(self):
        return self.root / "manifest.csv"

    def _read(self):
        with self.manifest.open(newline="") as stream:
            rows = csv.DictReader(line for line in stream if not line.startswith("#"))
            return [{name: float(value) if name not in _RECORD or name == "seconds" else value
                     for name, value in row.items() if value != ""} for row in rows]

    def _write(self, rows):
        counts = self._counts(rows)
        stream = io.StringIO(newline="")
        for name in ("total", "completed", "failed"):
            stream.write(f"# {name}: {counts[name]}\n")
        writer = csv.DictWriter(stream, [*_RECORD, *sorted(_coordinates(rows[0]))])
        writer.writeheader()
        writer.writerows(rows)
        write(self.manifest, stream.getvalue())

    @staticmethod
    def _counts(rows):
        completed = sum(row["status"] == "completed" for row in rows)
        failed = sum(row["status"] == "failed" for row in rows)
        return {"total": len(rows), "completed": completed, "failed": failed,
                "pending": len(rows) - completed - failed,
                "collected": sum(row["collection"] == "completed" for row in rows)}

    def status(self):
        return self._counts(self._read())

    def directory(self, point_id):
        return self.root / "points" / point_id

    def _point(self, params):
        names = {field.name for field in fields(self.point)}
        unknown = params.keys() - names - set(self.config["sedaxes"])
        if unknown:
            raise ValueError(f"Unknown parameters for {self.config['engine']}: {sorted(unknown)}")
        return self.point(**{name: value for name, value in params.items() if name in names})

    def extend(self, params, *, sed=_UNSET):
        """Append unique rows. Supply a tabulated source for previously unseen SED keys."""
        if sed is not _UNSET and (not self.config["file_sed"] or sed is None
                                   or list(sed.axes) != self.config["sedaxes"]):
            raise ValueError("Extension must use the campaign's SED mode and axes")
        with lock(self.root / ".iongrid.lock"):
            deck = (self.root / "input.template").read_text()
            if hashlib.sha256(deck.encode()).hexdigest() != self.config["template_sha256"]:
                raise ValueError("The saved campaign template changed")
            self._extend(self._read(), params, deck, sed)
        return self

    def _extend(self, rows, params, deck, sed):
        existing = {_key(_coordinates(row)): row for row in rows}
        columns = set(_coordinates(rows[0])) if rows else None
        files = self.root / "seds"
        files.mkdir(exist_ok=True)
        points = self.root / "points"
        points.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=self.root) as scratch:
            scratch = Path(scratch)
            staged = scratch / "seds"
            staged.mkdir()
            added = []
            seen = set()
            sources = {}
            prepared = {}
            gamma = rows[0].get("gamma") if rows else _UNSET
            cloudy = self.config["engine"] == "cloudy"
            options = {"strict": True} if cloudy else {}
            sedname = "SED/incident.sed" if cloudy else "incident.dat"
            for row in params:
                values = {name: float(value) or 0.0 for name, value in row.items()}
                if values.keys() & set(_RECORD) or not all(map(math.isfinite, values.values())):
                    raise ValueError("Parameters must be finite numbers with nonreserved names")
                if not values:
                    raise ValueError("A campaign row requires explicit parameters")
                if columns is None:
                    columns = set(values)
                if values.keys() != columns:
                    raise ValueError("Use the same parameter columns throughout a campaign")
                key = _key(values)
                if key in seen:
                    continue
                seen.add(key)
                point = self._point(values)
                sedkey, source = "", None
                if self.config["file_sed"]:
                    axes = self.config["sedaxes"]
                    missing = set(axes) - values.keys()
                    if missing:
                        raise ValueError(f"Missing SED coordinates: {sorted(missing)}")
                    if "gamma" not in axes:
                        if gamma is _UNSET:
                            gamma = values.get("gamma")
                        if values.get("gamma") != gamma:
                            raise ValueError("Varying gamma with a table SED requires gamma in the SED axes")
                    sedkey = hashlib.sha256(_key([values[name] for name in axes]).encode()).hexdigest()
                    if sedkey not in sources:
                        sources[sedkey] = self._sed(values, sed, files, staged, sedkey)
                    source = sources[sedkey]
                    if source.suffix == ".npz":
                        with np.load(source) as arrays:
                            source = arrays["energy"], arrays["photons"]
                if key in existing:
                    continue
                point_id = f"p{len(rows) + len(added) + 1:09d}"
                directory = scratch / point_id
                point.prepare(directory, text=deck, sed=source, **options)
                if not cloudy:
                    par = directory / "xstar.par"
                    par.unlink()
                    par.hardlink_to(self.root / "input.template")
                if sedkey:
                    path = directory / sedname
                    if sedkey in prepared:
                        path.unlink()
                        path.hardlink_to(prepared[sedkey])
                    else:
                        prepared[sedkey] = path
                added.append(dict(values, point_id=point_id, status="ready", collection="ready", sedkey=sedkey))
            if not rows and not added:
                raise ValueError("A campaign requires at least one parameter row")
            for row in added:
                destination = self.directory(row["point_id"])
                if destination.exists():
                    raise FileExistsError(f"Unindexed point directory: {destination}")
            moves = [(source, files / source.name) for source in staged.iterdir()]
            moves.extend((scratch / row["point_id"], self.directory(row["point_id"])) for row in added)
            moved = []
            try:
                for source, destination in moves:
                    source.rename(destination)
                    moved.append((source, destination))
                self._write(rows + added)
            except BaseException:
                for source, destination in reversed(moved):
                    destination.rename(source)
                raise

    @staticmethod
    def _sed(values, sed, files, staged, key):
        def stored(directory):
            native = directory / f"{key}.native"
            array = directory / f"{key}.npz"
            source = native if native.exists() else array if array.exists() else None
            if source is not None:
                checksum = (directory / f"{key}.sha256").read_text()
                if hashlib.sha256(source.read_bytes()).hexdigest() != checksum:
                    raise ValueError(f"Staged SED changed: {source}")
            return source

        source = stored(staged)
        if source is not None:
            return source
        previous = stored(files)
        if sed is _UNSET:
            if previous is None:
                raise ValueError("Supply sed=table(...) for the new SED coordinates")
            return previous

        from .sed import native_file, spectrum

        source = sed.resolve(values)
        if native_file(source):
            source = Path(source)
            content = source.read_bytes()
            if previous is not None:
                if not native_file(previous) or previous.read_bytes() != content:
                    raise ValueError("Tabulated SED changed for an existing coordinate key")
                return previous
            destination = staged / f"{key}.native"
            destination.write_bytes(content)
            (staged / f"{key}.sha256").write_text(hashlib.sha256(content).hexdigest())
            return destination
        energy, photons = spectrum(source)
        if previous is not None:
            if previous.suffix == ".npz":
                with np.load(previous) as arrays:
                    same = np.array_equal(arrays["energy"], energy) and np.array_equal(arrays["photons"], photons)
                if same:
                    return previous
            raise ValueError("Tabulated SED changed for an existing coordinate key")
        destination = staged / f"{key}.npz"
        np.savez(destination, energy=energy, photons=photons)
        (staged / f"{key}.sha256").write_text(hashlib.sha256(destination.read_bytes()).hexdigest())
        return destination

    def refresh(self):
        """Reconcile outcomes recorded by point runners."""
        with lock(self.root / ".iongrid.lock"):
            rows = self._read()
            for row in rows:
                directory = self.directory(row["point_id"])
                result = outcome(directory)
                if result["status"] == "running":
                    try:
                        with lock(directory / ".lock", blocking=False):
                            result = outcome(directory)
                            if result["status"] == "running":
                                result = {"status": "failed", "error": "Interrupted execution"}
                                write(directory / "run.json", json.dumps(result))
                    except BlockingIOError:
                        pass
                row["status"] = result["status"]
                row.pop("seconds", None)
                if "seconds" in result:
                    row["seconds"] = result["seconds"]
                record = directory / "collect.json"
                row["collection"] = json.loads(record.read_text())["status"] if record.exists() else "ready"
            self._write(rows)
        return self.status()

    def _select(self, select):
        if select not in {"ready", "failed"}:
            raise ValueError("select must be 'ready' or 'failed'")
        return [row for row in self._read() if row["status"] == select]

    def _run(self, row, **options):
        directory = self.directory(row["point_id"])
        try:
            return self._point(_coordinates(row)).run(directory, expected=row["status"], **options)
        except BlockingIOError:
            return None

    def run(self, *, workers=1, select="ready", **options):
        self.refresh()
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(lambda row: self._run(row, **options), self._select(select)))
        finally:
            self.refresh()
        return self.status()

    def collect(self):
        """Compile successful simulations and record extraction failures separately."""
        self.refresh()
        errors = {}
        for row in self._read():
            if row["status"] != "completed" or row["collection"] == "completed":
                continue
            directory = self.directory(row["point_id"])
            with lock(directory / ".lock"):
                if outcome(directory)["status"] != "completed":
                    continue
                path = directory / "collect.json"
                if path.exists() and json.loads(path.read_text())["status"] == "completed":
                    continue
                record = {"status": "completed"}
                try:
                    self._point(_coordinates(row)).collect(directory)
                except Exception as error:
                    record = {"status": "failed", "error": str(error)}
                    errors[row["point_id"]] = str(error)
                write(path, json.dumps(record))
        self.refresh()
        return errors

    def pack(self, output, *, key, quantity, unit, overwrite=False, postprocess=None, header=None):
        """Write collected rows in the common point-by-energy FITS format."""
        from ._fits import write_fits_table

        self.refresh()
        campaign = self._read()
        rows = [row for row in campaign if row["collection"] == "completed"]
        parameters = {row["point_id"]: _coordinates(row) for row in rows}
        xstar = self.config["engine"] == "xstar"
        metadata = dict(header or {})
        metadata["NPLANNED"] = (len(campaign), "Number of points in campaign")
        return write_fits_table(
            parameters, self.root / "points", output, npz_key=key, quantity=quantity, unit=unit,
            engine=self.config["engine"], reference_distance_kpc=1.0 if xstar else 10.0,
            reference_luminosity_erg_s=1e38 if xstar else None,
            overwrite=overwrite, postprocess=postprocess,
            extra_header=metadata,
        )
