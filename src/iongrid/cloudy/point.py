"""Prepare and execute one CLOUDY input deck."""

import os
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from string import Formatter

from .._run import execute, input_text, install, lock, outcome
from ..sed import native_file, spectrum

DEFAULT = Path(__file__).with_name("cloudy-template.in")
PRODUCTS = dict(overview="ovr", species_abundance="spc", line_list="lin",
                line_labels="llab", continuum="con", raw="raw",
                optical_depth="opt", fine_optical_depth="fopt")


@dataclass(frozen=True)
class CloudyPoint:
    lognh: float | None = None
    logxi: float | None = None
    vturb: float | None = None
    logdensity: float | None = None
    logluminosity: float | None = None
    gamma: float | None = None
    slope: float | None = None
    feabund: float | None = None

    def prepare(self, directory, *, template=None, text=None, sed=None, strict=False):
        directory = Path(directory)
        deck = input_text(template, text, DEFAULT)
        fields = {name for _, name, _, _ in Formatter().parse(deck) if name}
        values = {"title": directory.name, "point_id": directory.name}
        values.update({f"{name}_file": f"{directory.name}.{suffix}" for name, suffix in PRODUCTS.items()})
        for name, aliases in {
            "lognh": ("lognh", "log_nh"), "logdensity": ("logdensity", "log_density"),
            "logluminosity": ("logluminosity", "log_luminosity"), "vturb": ("vturb",),
            "logxi": ("logxi", "log_radius", "logradius"),
        }.items():
            value = getattr(self, name)
            if strict and value is not None and not fields.intersection(aliases):
                raise ValueError(f"CLOUDY template does not use {name}")
            if value is not None:
                values.update(dict.fromkeys(aliases, value))
        if None not in (self.logluminosity, self.logxi, self.logdensity):
            logradius = (self.logluminosity - self.logxi - self.logdensity) / 2
            values.update(log_radius=logradius, logradius=logradius)

        files = {}
        if sed is not None:
            if self.slope is not None:
                raise ValueError("CLOUDY slope is a native SED control")
            if "sed_command" not in fields:
                raise ValueError("A tabulated CLOUDY SED requires {sed_command}")
            if native_file(sed):
                content = Path(sed).read_bytes()
            else:
                energy, photons = spectrum(sed)
                # F_nu is proportional to E times photon density. CLOUDY interpolates log F_nu.
                if (photons <= 0).any():
                    raise ValueError("CLOUDY table SED requires strictly positive photon density")
                flux = energy * photons
                rows = [f"{e:.17g} {f:.17g}" for e, f in zip(energy, flux)]
                rows[0] += " units keV"
                content = ("\n".join(rows) + "\n").encode()
            files["SED/incident.sed"] = content
            values["sed_command"] = 'table SED "incident.sed"'
        else:
            if self.slope is not None and self.gamma is not None:
                raise ValueError("Choose either native CLOUDY slope or gamma")
            if (strict or fields) and not fields.intersection({"sed_command", "slope"}):
                if self.gamma is not None or self.slope is not None:
                    raise ValueError("CLOUDY template does not use the native power-law index")
            slope = self.slope if self.slope is not None else -0.8
            if self.gamma is not None:
                slope = 1 - self.gamma
            values["slope"] = slope
            values["sed_command"] = f"power law, slope={slope:.16g} hi cut 1000 Ryd low cut 1 Ryd"
        if (strict or fields) and self.feabund is not None and not fields.intersection({"iron", "feabund"}):
            raise ValueError("CLOUDY template does not use feabund")
        values["feabund"] = self.feabund
        values["iron"] = ("# Use the selected abundance mixture for iron" if self.feabund is None
                          else f"element iron scale {self.feabund:.16g} linear")
        path = directory / f"{directory.name}.in"
        try:
            files[path.name] = deck.format_map(values).encode()
        except KeyError as error:
            raise ValueError(f"Missing CLOUDY template value {error.args[0]!r}") from error
        install(directory, files)
        return path

    def run(self, directory, *, executable="cloudy", timeout=None, expected=None):
        directory = Path(directory).resolve()
        resolved = which(executable)
        executable = str(Path(resolved).resolve()) if resolved else executable
        env = os.environ.copy()
        data = env.get("CLOUDY_DATA_PATH") or _find_cloudy_data_dir(executable)
        env["CLOUDY_DATA_PATH"] = os.pathsep.join(str(path) for path in (directory, data) if path)
        prefix = directory.name
        with lock(directory / ".lock", blocking=False) as lease, (directory / f"{prefix}.in").open("rb") as deck:
            if expected is not None and outcome(directory)["status"] != expected:
                return None
            return execute(directory, [executable], lease=lease, env=env, stdout=f"{prefix}.out",
                           success=lambda: cloud_exited_ok(directory / f"{prefix}.out"),
                           inputs=deck, outputs=[f"{prefix}.{s}" for s in PRODUCTS.values()],
                           timeout=timeout)

    def collect(self, directory):
        from .spectra import compile_node

        directory = Path(directory)
        if None in (self.logxi, self.vturb, self.lognh):
            raise ValueError("Collection requires logxi, vturb and lognh")
        return compile_node(directory, logxi=self.logxi, vturb=self.vturb, lognh=self.lognh)



def _find_cloudy_data_dir(cloudy_path: str | Path) -> Path | None:
    path = Path(cloudy_path).resolve()
    for parent in [path.parent, path.parent.parent, path.parent.parent.parent, path.parent.parent.parent.parent]:
        candidate = parent / "data"
        if candidate.is_dir() and (candidate / "checksums.dat").exists():
            return candidate
    return None


def cloud_exited_ok(out_path: str | Path) -> bool:
    """Return whether a Cloudy stdout log reports a successful exit."""
    return "Cloudy exited OK" in Path(out_path).read_text(encoding="utf-8", errors="ignore")
