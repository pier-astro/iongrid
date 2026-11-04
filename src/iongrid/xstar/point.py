"""Prepare and execute one XSTAR parameter set."""

import csv
import json
import os
import shutil
from dataclasses import dataclass
from math import log10
from pathlib import Path
from tempfile import TemporaryDirectory

from .._run import execute, input_text, install, lock, outcome
from ..sed import native_file, spectrum

DEFAULT = Path(__file__).with_name("xstar.par")
PRODUCTS = ("xout_spect1.fits", "xout_cont1.fits")


def _defaults(par):
    return {row[0]: row[3] for row in csv.reader(
        line for line in par.splitlines() if line.strip() and not line.lstrip().startswith("#")
    )}


@dataclass(frozen=True)
class XstarPoint:
    lognh: float | None = None
    logxi: float | None = None
    vturb: float | None = None
    logdensity: float | None = None
    logluminosity: float | None = None
    gamma: float | None = None
    feabund: float | None = None

    def prepare(self, directory, *, template=None, text=None, sed=None):
        directory = Path(directory)
        par = input_text(template, text, DEFAULT)
        defaults = _defaults(par)
        if self.logxi is not None and int(defaults["lcpres"]) != 0:
            raise ValueError("The logxi coordinate requires XSTAR constant-density mode")
        overrides = {"modelname": directory.name, "mode": "h"}
        for name, value in {
            "column": None if self.lognh is None else 10 ** self.lognh,
            "density": None if self.logdensity is None else 10 ** self.logdensity,
            "rlrad38": None if self.logluminosity is None else 10 ** (self.logluminosity - 38),
            "rlogxi": self.logxi,
            "vturbi": self.vturb,
            "feabund": self.feabund,
        }.items():
            if value is not None:
                overrides[name] = value
        files = {"xstar.par": par.encode()}

        if sed is not None:
            if native_file(sed):
                content = Path(sed).read_bytes()
                units = int(defaults["spectun"])
            else:
                energy, photons = spectrum(sed)
                rows = [f"{e * 1000:.17g} {p:.17g}" for e, p in zip(energy, photons)]
                content = ("\n".join([str(len(rows)), *rows]) + "\n").encode()
                units = 1
            files["incident.dat"] = content
            overrides.update(spectrum="file", spectrum_file="incident.dat", spectun=units)

        elif self.gamma is not None:
            if defaults.get("spectrum") != "pow":
                raise ValueError("Native gamma requires spectrum=pow in xstar.par")
            overrides["trad"] = 1 - self.gamma

        elif defaults.get("spectrum") == "file":
            raise ValueError("Select a file SED explicitly with table(...)")
        
        if overrides.keys() - defaults.keys():
            raise ValueError(f"XSTAR parameters missing in .par: {sorted(overrides.keys() - defaults.keys())}")
        files["overrides.json"] = (json.dumps(overrides, sort_keys=True, indent=2) + "\n").encode()
        install(directory, files)
        return directory / "xstar.par"

    def run(self, directory, *, executable="xstar", timeout=None, expected=None):
        directory = Path(directory).resolve()
        overrides = json.loads((directory / "overrides.json").read_text())
        env = os.environ.copy()
        system = env.get("PFILES", "").partition(";")[2]
        if not system and env.get("HEADAS"):
            system = str(Path(env["HEADAS"]) / "syspfiles")
        resolved = shutil.which(executable)
        executable = str(Path(resolved).resolve()) if resolved else executable
        command = [executable, *(f"{name}={value}" for name, value in overrides.items())]
        with lock(directory / ".lock", blocking=False) as lease, TemporaryDirectory(prefix="iongrid-pfiles-") as pfiles:
            if expected is not None and outcome(directory)["status"] != expected:
                return None
            shutil.copyfile(directory / "xstar.par", Path(pfiles) / "xstar.par")
            env["PFILES"] = f"{pfiles};{system}" if system else pfiles
            return execute(directory, command, lease=lease, env=env, stdout="xstar.stdout",
                           success=lambda: all((directory / name).is_file() for name in PRODUCTS),
                           outputs=PRODUCTS, timeout=timeout)

    def collect(self, directory):
        from .spectra import compile_node

        directory = Path(directory)
        params = _defaults((directory / "xstar.par").read_text())
        params.update(json.loads((directory / "overrides.json").read_text()))

        return compile_node(
            directory, logxi=float(params["rlogxi"]), vturb=float(params["vturbi"]),
            lognh=log10(float(params["column"])), logdensity=log10(float(params["density"])),
        )
