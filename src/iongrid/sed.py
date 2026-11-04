"""Explicit native and tabulated incident spectra."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class _Table:
    source: object
    axes: tuple[str, ...]

    def resolve(self, params):
        key = tuple(params[name] for name in self.axes)
        if callable(self.source):
            return self.source(*key)
        if isinstance(self.source, Mapping):
            lookup = key[0] if len(key) == 1 else key
            try:
                return self.source[lookup]
            except KeyError as error:
                raise ValueError(f"No tabulated SED for {dict(zip(self.axes, key))}") from error
        return self.source


def native():
    """Use the engine's native SED parameters."""
    return None


def table(source, *, axes=()):
    """Select a native file or an (energy_keV, relative photons/keV) spectrum."""
    axes = tuple(axes)
    if len(set(axes)) != len(axes):
        raise ValueError("SED axes must be unique")
    if isinstance(source, Mapping) and not axes:
        raise ValueError("A SED mapping requires axes")
    if axes and not (callable(source) or isinstance(source, Mapping)):
        raise ValueError("SED axes require a mapping or callable")
    return _Table(source, axes)


def spectrum(source):
    """Validate the common array convention before engine-specific conversion."""
    arrays = np.asarray(source, dtype=float)
    if arrays.ndim != 2 or arrays.shape[0] != 2:
        raise ValueError("A tabulated SED requires two one-dimensional arrays")
    energy, photons = arrays
    if (len(energy) < 2 or not np.all(np.isfinite([energy, photons]))
            or np.any(energy <= 0) or np.any(np.diff(energy) <= 0)
            or np.any(photons < 0) or not np.any(photons > 0)):
        raise ValueError("Expected increasing positive keV and finite nonnegative photon density")
    return energy, photons


def native_file(source):
    return isinstance(source, (str, Path))
