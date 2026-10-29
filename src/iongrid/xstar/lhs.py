"""Latin-hypercube parameter grids for XSTAR runs.

``generate_lhs_grid`` writes a CSV with one row per XSTAR simulation, which
``setup_grid(lhs_file=...)`` (iongrid.xstar.grid) turns into a manifest. The
recipe script ``make-lhs.py`` passes it the LHS_* values of ``settings.py``.

Needs pandas, scipy and matplotlib (not required by the rest of iongrid).

Sampling populations: each population inherits the full bounds and may override
some of them (e.g. oversample high column densities). Every population includes
exact lower/upper boundary anchors for each parameter, i.e. 2 * n_parameters of
its ``n_samples`` rows are anchors and the rest is an ordinary scrambled LHS.

vturb sampling: "linear" is uniform in km/s; "log" is uniform in log10(vturb),
giving more samples at low velocity.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.stats import qmc
except ImportError as error:  # pragma: no cover
    raise ImportError("iongrid.xstar.lhs needs pandas, scipy and matplotlib") from error

Bounds = Dict[str, Tuple[float, float]]


# =============================================================================
# Validation
# =============================================================================

def validate_configuration(bounds: Bounds, vturb_sampling: str, populations: Sequence[dict]) -> None:
    if vturb_sampling not in {"linear", "log"}:
        raise ValueError(f"vturb_sampling must be 'linear' or 'log'. Received: {vturb_sampling!r}")
    if "vturb" not in bounds:
        raise ValueError("'vturb' must be present in the bounds.")
    if vturb_sampling == "log" and bounds["vturb"][0] <= 0:
        raise ValueError("Logarithmic vturb sampling requires a strictly positive lower bound.")
    for parameter, (lower, upper) in bounds.items():
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError(f"Non-finite bounds for {parameter}: {(lower, upper)}")
        if lower >= upper:
            raise ValueError(f"Invalid bounds for {parameter}: {(lower, upper)}")
    names = [spec["name"] for spec in populations]
    if len(names) != len(set(names)):
        raise ValueError("Population names must be unique.")
    for spec in populations:
        if int(spec["n_samples"]) <= 0:
            raise ValueError(f"Invalid sample count for {spec['name']!r}.")
        if int(spec["n_samples"]) < 2 * len(bounds):
            raise ValueError(
                f"Population {spec['name']!r} requires at least {2 * len(bounds)} samples "
                "to include all boundary anchors."
            )


def build_population_bounds(full_bounds: Bounds, overrides: Bounds) -> Bounds:
    """Population bounds: unspecified parameters keep their full-space bounds."""
    bounds = dict(full_bounds)
    for parameter, limits in overrides.items():
        if parameter not in full_bounds:
            raise ValueError(
                f"Unknown parameter {parameter!r} in population bounds. "
                f"Available parameters: {list(full_bounds)}"
            )
        lower, upper = limits
        full_lower, full_upper = full_bounds[parameter]
        if lower >= upper:
            raise ValueError(f"Invalid bounds for {parameter}: {limits}")
        if lower < full_lower or upper > full_upper:
            raise ValueError(
                f"Bounds {limits} for {parameter} fall outside its full range {(full_lower, full_upper)}."
            )
        bounds[parameter] = (float(lower), float(upper))
    return bounds


# =============================================================================
# Latin hypercube sampling
# =============================================================================

def scale_unit_samples(unit_samples: np.ndarray, bounds: Bounds, vturb_sampling: str) -> "pd.DataFrame":
    """Map [0, 1] samples to physical values (vturb in log10 if requested; still km/s)."""
    names = list(bounds)
    scaled = np.empty_like(unit_samples, dtype=float)
    for i, parameter in enumerate(names):
        lower, upper = bounds[parameter]
        u = unit_samples[:, i]
        if parameter == "vturb" and vturb_sampling == "log":
            log_lower, log_upper = np.log10(lower), np.log10(upper)
            scaled[:, i] = 10.0 ** (log_lower + u * (log_upper - log_lower))
        else:
            scaled[:, i] = lower + u * (upper - lower)
    return pd.DataFrame(scaled, columns=names)


def latin_hypercube_sample(n_samples: int, bounds: Bounds, seed: int, vturb_sampling: str) -> "pd.DataFrame":
    """LHS of n_samples rows: (n_samples - 2 n_params) ordinary LHS rows plus
    2 n_params boundary anchors (one at each lower and upper bound per parameter,
    other coordinates random), shuffled."""
    names = list(bounds)
    n_parameters = len(names)
    n_anchors = 2 * n_parameters
    n_lhs = n_samples - n_anchors
    if n_lhs < 0:
        raise ValueError(
            f"n_samples={n_samples} is too small for {n_parameters} parameters. "
            f"At least {n_anchors} samples are required."
        )
    sampler = qmc.LatinHypercube(d=n_parameters, scramble=True, optimization="random-cd", seed=seed)
    unit_lhs = sampler.random(n=n_lhs)
    rng = np.random.default_rng(seed + 1)
    anchor_unit = rng.random((n_anchors, n_parameters))
    for i in range(n_parameters):
        anchor_unit[2 * i, i] = 0.0
        anchor_unit[2 * i + 1, i] = 1.0
    result = pd.concat(
        [scale_unit_samples(unit_lhs, bounds, vturb_sampling),
         scale_unit_samples(anchor_unit, bounds, vturb_sampling)],
        ignore_index=True,
    )
    return result.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def check_exact_boundaries(df: "pd.DataFrame", bounds: Bounds, label: str) -> None:
    """Every lower and upper bound must be represented in the sample."""
    print(f"\nChecking parameter boundaries: {label}")
    for parameter, (lower, upper) in bounds.items():
        values = df[parameter].to_numpy()
        if not np.any(np.isclose(values, lower, rtol=1e-12, atol=1e-12)):
            raise AssertionError(f"{label}: lower boundary {lower} missing for {parameter}.")
        if not np.any(np.isclose(values, upper, rtol=1e-12, atol=1e-12)):
            raise AssertionError(f"{label}: upper boundary {upper} missing for {parameter}.")
        print(f"  {parameter:10s}: minimum={df[parameter].min():.8g}, maximum={df[parameter].max():.8g}  [OK]")


def validate_parameter_ranges(df: "pd.DataFrame", bounds: Bounds, label: str) -> None:
    """All sampled values must be finite and inside their permitted ranges."""
    for parameter, (lower, upper) in bounds.items():
        values = df[parameter].to_numpy()
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{label}: non-finite values found for {parameter}.")
        tolerance = 1e-10 * max(1.0, abs(lower), abs(upper))
        if np.any(values < lower - tolerance):
            raise ValueError(f"{label}: values below the lower bound for {parameter}.")
        if np.any(values > upper + tolerance):
            raise ValueError(f"{label}: values above the upper bound for {parameter}.")
    print(f"Parameter-range validation passed: {label}")


def print_parameter_ranges(df: "pd.DataFrame", parameters: Sequence[str], label: str) -> None:
    print(f"\n{label}")
    for parameter in parameters:
        print(f"  {parameter:10s}: {df[parameter].min():.8g} -- {df[parameter].max():.8g}")
    print(f"  Number of samples: {len(df)}")


# =============================================================================
# Existing grid / duplicates
# =============================================================================

def load_existing_grid(path, parameters: Sequence[str], column_map: Optional[dict] = None,
                       fill_values: Optional[dict] = None) -> Optional["pd.DataFrame"]:
    """Parameter columns of an existing grid (extra columns ignored).

    ``column_map`` maps parameter -> column name in that file (default: same
    name); ``fill_values`` supplies a constant for parameters the file lacks.
    """
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Existing grid file does not exist: {path}")
    column_map, fill_values = column_map or {}, fill_values or {}
    existing = pd.read_csv(path)
    selected = {}
    for parameter in parameters:
        column = column_map.get(parameter, parameter)
        if column in existing.columns:
            selected[parameter] = existing[column].to_numpy()
        elif parameter in fill_values:
            selected[parameter] = np.full(len(existing), fill_values[parameter])
        else:
            raise ValueError(f"Column {column!r} for parameter {parameter!r} is missing from {path}.")
    result = pd.DataFrame(selected)
    for parameter in parameters:
        result[parameter] = pd.to_numeric(result[parameter], errors="raise")
    result = result.dropna(subset=list(parameters))
    print(f"\nLoaded {len(result)} existing parameter combinations from {path}")
    return result


def parameter_keys(df: "pd.DataFrame", parameters: Sequence[str]) -> "pd.Series":
    """Hashable key per parameter combination (rounded to 10 decimals)."""
    return df[list(parameters)].round(10).astype(str).agg("|".join, axis=1)


def remove_duplicates(new_df: "pd.DataFrame", parameters: Sequence[str],
                      existing_df: Optional["pd.DataFrame"] = None) -> "pd.DataFrame":
    """Drop points already in ``existing_df``, then duplicates between populations (first kept)."""
    result = new_df.copy()
    print("\nDuplicate handling")
    print(f"  New samples before filtering: {len(result)}")
    if existing_df is not None and not existing_df.empty:
        already = parameter_keys(result, parameters).isin(set(parameter_keys(existing_df, parameters)))
        result = result.loc[~already].copy()
        print(f"  Removed points already in existing grid: {int(already.sum())}")
    before = len(result)
    result["_parameter_key"] = parameter_keys(result, parameters)
    result = result.drop_duplicates(subset="_parameter_key", keep="first")
    result = result.drop(columns="_parameter_key").reset_index(drop=True)
    print(f"  Duplicate new points removed: {before - len(result)}")
    print(f"  Samples remaining: {len(result)}")
    return result


# =============================================================================
# Summaries and plots
# =============================================================================

def make_population_summary(df: "pd.DataFrame") -> "pd.DataFrame":
    summary = df.groupby("population", sort=False).size().rename("n_samples").reset_index()
    total = pd.DataFrame({"population": ["TOTAL"], "n_samples": [len(df)]})
    return pd.concat([summary, total], ignore_index=True)


def make_parameter_summary(df: "pd.DataFrame", parameters: Sequence[str]) -> "pd.DataFrame":
    rows = []
    for parameter in parameters:
        for population, subset in df.groupby("population", sort=False):
            values = subset[parameter]
            rows.append({"population": population, "parameter": parameter, "minimum": values.min(),
                         "maximum": values.max(), "mean": values.mean(), "std": values.std()})
    return pd.DataFrame(rows)


def _finish_figure(fig, path: Path, save: bool, show: bool) -> None:
    fig.tight_layout()
    if save:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_1d_distributions(df, parameters, vturb_sampling, bins, output_dir, save, show) -> None:
    ncols = 2
    nrows = int(np.ceil(len(parameters) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.5 * nrows), squeeze=False)
    populations = df["population"].drop_duplicates().tolist()
    for i, parameter in enumerate(parameters):
        ax = axes.flat[i]
        for population in populations:
            ax.hist(df.loc[df["population"] == population, parameter], bins=bins, alpha=0.45, label=population)
        ax.hist(df[parameter], bins=bins, histtype="step", color="black", linewidth=2.5, label="Total", zorder=10)
        ax.set_xlabel(parameter)
        ax.set_ylabel("Number of samples")
        ax.set_title(f"Distribution of {parameter}")
        ax.grid(alpha=0.25)
        if parameter == "vturb" and vturb_sampling == "log":
            ax.set_xscale("log")
        ax.legend(fontsize=8)
    for ax in axes.flat[len(parameters):]:
        ax.set_visible(False)
    fig.suptitle(f"Parameter distributions — vturb sampling: {vturb_sampling}", y=1.01)
    _finish_figure(fig, output_dir / "parameter_distributions.png", save, show)


def plot_pairwise_projections(df, parameters, vturb_sampling, output_dir, save, show) -> None:
    pairs = list(combinations(parameters, 2))
    ncols = 3
    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
    populations = df["population"].drop_duplicates().tolist()
    for i, (x_parameter, y_parameter) in enumerate(pairs):
        ax = axes.flat[i]
        for population in populations:
            subset = df[df["population"] == population]
            ax.scatter(subset[x_parameter], subset[y_parameter], s=7, alpha=0.35, label=population)
        ax.set_xlabel(x_parameter)
        ax.set_ylabel(y_parameter)
        ax.grid(alpha=0.2)
        if vturb_sampling == "log":
            if x_parameter == "vturb":
                ax.set_xscale("log")
            if y_parameter == "vturb":
                ax.set_yscale("log")
        if i == 0:
            ax.legend(fontsize=7)
    for ax in axes.flat[len(pairs):]:
        ax.set_visible(False)
    fig.suptitle("Pairwise parameter projections", y=1.01)
    _finish_figure(fig, output_dir / "pairwise_projections.png", save, show)


def plot_lognh_logxi(df, output_dir, save, show) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    for population in df["population"].drop_duplicates().tolist():
        subset = df[df["population"] == population]
        ax.scatter(subset["log_nh"], subset["logxi"], s=9, alpha=0.4, label=population)
    ax.set_xlabel("log_nh")
    ax.set_ylabel("logxi")
    ax.set_title("Density-ionization parameter coverage")
    ax.grid(alpha=0.25)
    ax.legend()
    _finish_figure(fig, output_dir / "log_nh_logxi_coverage.png", save, show)


# =============================================================================
# Main entry point
# =============================================================================

def generate_lhs_grid(
    bounds: Bounds,
    populations: Sequence[dict],
    output_dir,
    *,
    vturb_sampling: str = "log",
    existing_grid_file=None,
    existing_column_map: Optional[dict] = None,
    existing_fill_values: Optional[dict] = None,
    save_plots: bool = True,
    show_plots: bool = False,
    histogram_bins: int = 40,
) -> "pd.DataFrame":
    """Generate, validate, combine and save the LHS grid; return the final table.

    Writes to ``output_dir``: ``new_parameter_grid_<population>.csv`` per
    population, the combined ``new_parameter_grid.csv`` (parameter columns only,
    the file read by setup), ``population_summary.csv``, ``parameter_summary.csv``
    and diagnostic plots.
    """
    bounds = dict(bounds)
    parameters = list(bounds)
    output_dir = Path(output_dir)
    validate_configuration(bounds, vturb_sampling, populations)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("XSTAR LHS PARAMETER GRID GENERATION")
    print("=" * 72)
    print("\nSampled parameters:")
    for parameter, limits in bounds.items():
        print(f"  {parameter:10s}: {limits}")
    print("\nFixed parameters:")
    print("  luminosity: read from the XSTAR simulation configuration")
    if "gamma" not in bounds:
        print("  gamma:      read from the SED configuration")
    if "feabund" not in bounds:
        print("  feabund:    XSTAR default")
    print(f"\nvturb sampling strategy: {vturb_sampling}")
    print("  Uniform in physical velocity (km/s)." if vturb_sampling == "linear"
          else "  Uniform in log10(vturb); more samples per km/s at low velocity.")
    print("\nPopulation configuration:")
    for spec in populations:
        print(f"  {spec['name']:12s}: {spec['n_samples']:6d} samples, seed={spec['seed']}")

    population_dfs: List["pd.DataFrame"] = []
    population_files: List[Path] = []
    for spec in populations:
        name, n_samples, seed = spec["name"], int(spec["n_samples"]), int(spec["seed"])
        pop_bounds = build_population_bounds(bounds, spec.get("bounds", {}))
        print("\n" + "=" * 72)
        print(f"Generating population: {name}")
        print("=" * 72)
        print("Population bounds:")
        for parameter, limits in pop_bounds.items():
            print(f"  {parameter:10s}: {limits}")
        df = latin_hypercube_sample(n_samples, pop_bounds, seed, vturb_sampling)
        check_exact_boundaries(df, pop_bounds, f"{name} population")  # before duplicate removal
        validate_parameter_ranges(df, pop_bounds, f"{name} population")
        df = df.assign(population=name)
        print_parameter_ranges(df, parameters, f"{name.upper()} SAMPLE")
        population_file = output_dir / f"new_parameter_grid_{name}.csv"
        df.to_csv(population_file, index=False)
        population_files.append(population_file)
        population_dfs.append(df)
        print(f"Saved: {population_file}")

    new_df = pd.concat(population_dfs, ignore_index=True)
    requested_total = sum(int(spec["n_samples"]) for spec in populations)
    print("\n" + "=" * 72)
    print("COMBINED GRID")
    print("=" * 72)
    print(f"Requested total samples: {requested_total}")
    print(f"Combined before filtering: {len(new_df)}")

    existing_df = load_existing_grid(existing_grid_file, parameters, existing_column_map, existing_fill_values)
    final_df = remove_duplicates(new_df, parameters, existing_df)
    validate_parameter_ranges(final_df, bounds, "final combined grid")

    final_df = final_df.sort_values(by=parameters + ["population"], kind="mergesort").reset_index(drop=True)
    combined_file = output_dir / "new_parameter_grid.csv"
    final_df[parameters].to_csv(combined_file, index=False)  # physical parameters only
    print(f"Saved combined grid: {combined_file}")

    population_summary = make_population_summary(final_df)
    parameter_summary = make_parameter_summary(final_df, parameters)
    population_summary_file = output_dir / "population_summary.csv"
    parameter_summary_file = output_dir / "parameter_summary.csv"
    population_summary.to_csv(population_summary_file, index=False)
    parameter_summary.to_csv(parameter_summary_file, index=False)
    print(f"Saved population summary: {population_summary_file}")
    print(f"Saved parameter summary:  {parameter_summary_file}")
    print("\nFinal population counts:")
    print(population_summary.to_string(index=False))
    print_parameter_ranges(final_df, parameters, "FINAL GRID")

    plot_files = []
    if not final_df.empty and (save_plots or show_plots):
        plot_1d_distributions(final_df, parameters, vturb_sampling, histogram_bins, output_dir, save_plots, show_plots)
        plot_files.append(output_dir / "parameter_distributions.png")
        if len(parameters) >= 2:
            plot_pairwise_projections(final_df, parameters, vturb_sampling, output_dir, save_plots, show_plots)
            plot_files.append(output_dir / "pairwise_projections.png")
        if {"log_nh", "logxi"} <= set(parameters):
            plot_lognh_logxi(final_df, output_dir, save_plots, show_plots)
            plot_files.append(output_dir / "log_nh_logxi_coverage.png")
        if save_plots:
            print(f"\nPlots saved in: {output_dir.resolve()}")

    print("\n" + "=" * 72)
    print("GRID GENERATION COMPLETE")
    print("=" * 72)
    print(f"Sampling strategy for vturb: {vturb_sampling}")
    print(f"Requested samples:           {requested_total}")
    print(f"Final unique samples:        {len(final_df)}")
    print(f"Output directory:            {output_dir.resolve()}")
    print("\nGenerated files:")
    for path in [*population_files, combined_file, population_summary_file, parameter_summary_file,
                 *(plot_files if save_plots else [])]:
        print(f"  {path}")
    return final_df
