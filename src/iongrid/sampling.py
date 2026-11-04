"""Generate parameter rows independently of the simulation engine."""

from itertools import product

import numpy as np
from scipy.stats import qmc


def cartesian(**params):
    """Form the product of explicit axis values and scalar constants."""
    axes = [np.atleast_1d(values) for values in params.values()]
    if any(axis.ndim != 1 or not len(axis) for axis in axes):
        raise ValueError("Each parameter needs a scalar or a nonempty one-dimensional axis")
    return [dict(zip(params, row)) for row in product(*(axis.tolist() for axis in axes))]


def lhs(bounds, *, samples, seed=None, anchors=True, logarithmic=(), **fixed):
    """Sample bounds, counting two optional anchors per dimension in samples."""
    names = tuple(bounds)
    if not names or set(names) & fixed.keys() or set(logarithmic) - set(names):
        raise ValueError("Supply sampled bounds, distinct fixed parameters, and valid logarithmic names")
    limits = np.asarray(list(bounds.values()), dtype=float)
    if limits.shape != (len(names), 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("Each sampled parameter needs finite, increasing bounds")
    log = np.array([name in logarithmic for name in names])
    if np.any(limits[log] <= 0):
        raise ValueError("Logarithmic sampling requires positive bounds")
    count = 2 * len(names) if anchors else 0
    if samples < max(1, count):
        raise ValueError(f"samples must be at least {max(1, count)}")
    interior = samples - count
    unit = (qmc.LatinHypercube(len(names), seed=seed).random(interior)
            if interior else np.empty((0, len(names))))
    if anchors:
        rng = np.random.default_rng(seed)
        boundary = rng.random((count, len(names)))
        columns = np.arange(len(names))
        boundary[2 * columns, columns] = 0
        boundary[2 * columns + 1, columns] = 1
        unit = np.vstack((unit, boundary))
        rng.shuffle(unit)
    transformed = limits.copy()
    transformed[log] = np.log10(limits[log])
    values = transformed[:, 0] + unit * np.diff(transformed, axis=1).ravel()
    values[:, log] = 10 ** values[:, log]
    values = np.where(unit == 0, limits[:, 0], values)
    values = np.where(unit == 1, limits[:, 1], values)
    return [dict(fixed, **dict(zip(names, row))) for row in values.tolist()]
