from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from iongrid import SCHEMA_VERSION

# Constants
C_KM_S = 299792.458
SQRT_2 = np.sqrt(2.0)
# UV/Opt (1 eV - 50 eV): Coarse 20-point log spacing for broadband SED photometry
e_uv = np.geomspace(0.001, 0.05, 20)
# Core X-ray (50 eV - 15 keV): 7500-point log spacing. Yields ~0.4 eV at 0.5 keV (RGS/HETG) & ~4.5 eV at 6 keV (XRISM/Athena)
e_core = np.geomspace(0.05, 15.0, 7501)[1:]
# Hard X-ray (15 keV - 100 keV): Coarse 200-point log spacing for NuSTAR
e_hard = np.geomspace(15.0, 100.0, 201)[1:]

ENERGY_GRID = np.concatenate([e_uv, e_core, e_hard]).astype(np.float32)


def bin_edges(centers: np.ndarray) -> np.ndarray:
    """Return contiguous bin edges halfway between strictly increasing centers."""
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or len(centers) < 2:
        raise ValueError("energy centers must be a one-dimensional array of length >= 2")
    if not np.all(np.isfinite(centers)) or np.any(centers <= 0.0) or np.any(np.diff(centers) <= 0.0):
        raise ValueError("energy centers must be finite, positive, and strictly increasing")

    edges = np.empty(len(centers) + 1)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    if edges[0] <= 0.0:
        edges[0] = centers[0] * np.sqrt(centers[0] / centers[1])
    return edges


def bin_average(x_src: np.ndarray, y_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    """Average a sampled density into destination bins, conserving its integral."""
    x_src = np.asarray(x_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    if y_src.shape != x_src.shape or not np.all(np.isfinite(y_src)):
        raise ValueError("source values must be finite and match the source energy grid")

    x_src_edges = bin_edges(x_src)
    x_dst_edges = bin_edges(x_dst)
    src_widths = np.diff(x_src_edges)
    dst_widths = np.diff(x_dst_edges)
    cdf_src = np.concatenate(([0.0], np.cumsum(y_src * src_widths)))
    cdf_dst = np.interp(x_dst_edges, x_src_edges, cdf_src, left=0.0, right=cdf_src[-1])
    return np.diff(cdf_dst) / dst_widths

def read_continuum(con_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a Cloudy .con file into: energy_kev, incident, transmitted."""
    data = np.loadtxt(con_path, comments="#", usecols=(0, 1, 2))
    return data[:, 0], data[:, 1], data[:, 2]


def read_optical_depth(opt_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse a Cloudy .opt file into energy and optical depths.

    Returns
    -------
    energy : np.ndarray
    total : np.ndarray
    absorption : np.ndarray
    scattering : np.ndarray
    """
    data = np.loadtxt(opt_path, comments="#", usecols=(0, 1, 2, 3))
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]


def read_fine_optical_depth(fopt_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a Cloudy .fopt file (fine optical depth) into energy and line optical depth."""
    try:
        data = np.loadtxt(fopt_path, comments="#", usecols=(0, 1))
        if data.ndim == 1:
            if len(data) == 0:
                return np.empty(0, dtype=float), np.empty(0, dtype=float)
            return np.array([data[0]], dtype=float), np.array([data[1]], dtype=float)
        return data[:, 0], data[:, 1]
    except (FileNotFoundError, IOError, ValueError, IndexError):
        return np.empty(0, dtype=float), np.empty(0, dtype=float)


def read_slab_continuum(con_path: str | Path, raw_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a Cloudy .con and .raw files to extract the slab-only diffuse continuum."""
    con_data = np.loadtxt(con_path, comments="#", usecols=(0, 3, 5))
    if con_data.ndim == 1:
        con_data = con_data[np.newaxis, :]
    energy_con = con_data[:, 0]
    diff_out = con_data[:, 1]
    reflc = con_data[:, 2]

    raw_data = np.loadtxt(raw_path, comments="#", usecols=(0, 5, 6))
    if raw_data.ndim == 1:
        raw_data = raw_data[np.newaxis, :]
    energy_raw = raw_data[:, 0]
    con_ref_incid = raw_data[:, 1]
    con_emit_reflec = raw_data[:, 2]

    reflected_total_raw = con_ref_incid + con_emit_reflec
    reflected_diffuse_fraction = np.divide(
        con_emit_reflec,
        reflected_total_raw,
        out=np.zeros_like(con_emit_reflec),
        where=reflected_total_raw > 0.0,
    )

    continuum_out_density = np.divide(diff_out, energy_con)
    continuum_reflected_diffuse_density = np.divide(
        reflc * reflected_diffuse_fraction,
        energy_con,
    )
    continuum_total_density = continuum_out_density + continuum_reflected_diffuse_density

    return energy_con, continuum_total_density


def read_lines_intensity_validated(lin_path: str | Path, llab_path: str | Path) -> list[tuple[float, float]]:
    """Extract line intensities validated by label and wavelength matching in the line labels file."""
    lin_path = Path(lin_path)
    llab_path = Path(llab_path)

    line_types = {}
    with llab_path.open("r", encoding="utf-8", errors="ignore") as handle:
        try:
            next(handle)
        except StopIteration:
            pass
        for line in handle:
            if line.startswith("####") or not line.strip():
                continue

            columns = line.rstrip("\n").split("\t")
            if len(columns) < 2:
                continue

            line_index = columns[0].strip()
            if not line_index.isdigit() or line_index == "0":
                continue

            left = columns[1]
            comment = columns[-1].strip()
            parts = left.split()
            if len(parts) < 2:
                continue

            label = " ".join(parts[:-1]).strip()
            wavelength_token = parts[-1].rstrip("?")
            if "type: t" in comment and "index=" in comment:
                line_types[(label, wavelength_token)] = True

    line_catalog = []
    with lin_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if "begin emission lines" in line:
                break

        for line in handle:
            if line.startswith("***") or not line.strip():
                break

            left, separator, right = line.partition("\t")
            if not separator:
                continue

            try:
                flux = float(right)
            except ValueError:
                continue
            if flux <= 0.0:
                continue

            parts = left.split()
            if len(parts) < 2:
                continue

            label = " ".join(parts[:-1]).strip()
            if label.endswith(" Inwd") or label.endswith(" Coll") or label.endswith(" Pump") or label.endswith(" Heat"):
                continue

            wavelength_token = parts[-1].rstrip("?")
            if not line_types.get((label, wavelength_token), False):
                continue

            suffix = wavelength_token[-1]
            try:
                if suffix == "A":
                    wavelength_angstrom = float(wavelength_token[:-1])
                elif suffix == "m":
                    wavelength_angstrom = float(wavelength_token[:-1]) * 1.0e4
                elif suffix == "c":
                    wavelength_angstrom = float(wavelength_token[:-1]) * 1.0e8
                else:
                    wavelength_angstrom = float(wavelength_token)
            except ValueError:
                continue

            if wavelength_angstrom <= 0.0:
                continue

            energy_kev = 12.3984193 / wavelength_angstrom
            line_catalog.append((energy_kev, flux))

    return line_catalog


def broaden_lines(
    energy_grid: np.ndarray,
    line_catalog: list[tuple[float, float]],
    vturb: float,
    sigma_window: float = 5.0,
) -> np.ndarray:
    """Broaden line intensities onto a target energy grid using Gaussian profiles."""
    line_density = np.zeros_like(energy_grid)
    if not line_catalog:
        return line_density

    if len(energy_grid) == 1:
        for line_energy, line_flux in line_catalog:
            sigma = line_energy * (vturb / C_KM_S) / SQRT_2
            if sigma <= 0.0:
                continue
            idx = np.argmin(np.abs(energy_grid - line_energy))
            line_density[idx] += line_flux
        return line_density

    dE = np.diff(energy_grid)
    dE = np.append(dE, dE[-1])
    edges = np.zeros(len(energy_grid) + 1)
    edges[0] = energy_grid[0] - dE[0] / 2.0
    edges[1:] = energy_grid + dE / 2.0
    bin_width = np.diff(edges)

    for line_energy, line_flux in line_catalog:
        sigma = line_energy * (vturb / C_KM_S) / SQRT_2
        if sigma <= 0.0:
            continue

        energy_low = line_energy - sigma_window * sigma
        energy_high = line_energy + sigma_window * sigma
        low_idx = np.searchsorted(energy_grid, energy_low, side="left")
        high_idx = np.searchsorted(energy_grid, energy_high, side="right")

        if high_idx <= low_idx:
            closest_idx = np.argmin(np.abs(energy_grid - line_energy))
            line_density[closest_idx] += line_flux / bin_width[closest_idx]
            continue

        local_energy = energy_grid[low_idx:high_idx]
        local_width = bin_width[low_idx:high_idx]
        profile = np.exp(-0.5 * ((local_energy - line_energy) / sigma) ** 2)
        normalization = np.sum(profile * local_width)

        if normalization > 0.0:
            line_density[low_idx:high_idx] += line_flux * profile / normalization

    return line_density


def compile_node(node_dir: str | Path, energy_grid: np.ndarray = ENERGY_GRID):
    """Compile raw Cloudy output ASCII files of a node into a structured compressed binary .npz file."""
    node_dir = Path(node_dir)
    point_id = node_dir.name

    # 1. Parse parameters from directory name
    parts = point_id.split('_')
    if len(parts) != 3 or not parts[0].startswith("xi") or not parts[1].startswith("vt") or not parts[2].startswith("nh"):
        raise ValueError(f"Directory name {point_id} does not match expected format xi[value]_vt[value]_nh[value]")

    logxi = float(parts[0][2:].replace('p', '.'))
    vturb = float(parts[1][2:])
    log_nh = float(parts[2][2:].replace('p', '.'))

    # 2. Parse log_density from input deck file (.in)
    in_file = node_dir / f"{point_id}.in"
    if not in_file.exists():
        raise FileNotFoundError(f"Missing input deck file: {in_file}")
    in_text = in_file.read_text(encoding="utf-8")
    out_file = node_dir / f"{point_id}.out"
    emission_input = None
    if out_file.exists():
        out_text = out_file.read_text(encoding="utf-8", errors="replace")
        reported_units = {unit.lower() for unit in re.findall(
            r"^\s*(Intensity \(erg/s/cm\^2\)|Luminosity \(erg/s\))(?=\s|\.|$)",
            out_text, re.IGNORECASE | re.MULTILINE,
        )}
        if len(reported_units) > 1:
            raise ValueError(f"Conflicting CLOUDY emission units in {out_file}")
        if reported_units:
            emission_input = "intensity" if next(iter(reported_units)).startswith("intensity") else "luminosity"
    if emission_input is None:
        commands = re.findall(r"^\s*(xi|luminosity)\s+", in_text, re.IGNORECASE | re.MULTILINE)
        if len(commands) == 1:
            emission_input = "intensity" if commands[0].lower() == "xi" else "luminosity"
        else:
            marker = re.search(r"^\s*#\s*iongrid emission input:\s*(intensity|luminosity)\s*$", in_text, re.IGNORECASE | re.MULTILINE)
            if marker is None or len(commands) > 1:
                raise ValueError(f"Cannot identify CLOUDY emission units in {out_file} or {in_file}; add '# iongrid emission input: intensity' (or luminosity) to the deck")
            emission_input = marker.group(1).lower()
    hden_match = re.search(r"hden\s+([0-9.-]+)\s+log", in_text)
    if not hden_match:
        raise ValueError(f"Could not parse hden from {in_file}")
    log_density = float(hden_match.group(1))

    # 3. Read average temperature from overview file (.ovr)
    ovr_path = node_dir / f"{point_id}.ovr"
    if not ovr_path.exists():
        raise FileNotFoundError(f"Missing overview file: {ovr_path}")
    ovr_data = np.loadtxt(ovr_path, comments="#", usecols=1)
    temperature = float(ovr_data) if ovr_data.ndim == 0 else float(np.mean(ovr_data))

    # 4. Read optical depths (.opt & .fopt)
    opt_path = node_dir / f"{point_id}.opt"
    if not opt_path.exists():
        raise FileNotFoundError(f"Missing optical depth file: {opt_path}")
    e_opt, _, opt_abs, opt_scat = read_optical_depth(opt_path)

    fopt_path = node_dir / f"{point_id}.fopt"
    e_fopt, fopt_tau = read_fine_optical_depth(fopt_path)

    # 5. Read continuum spectra (.con & .raw)
    con_path = node_dir / f"{point_id}.con"
    raw_path = node_dir / f"{point_id}.raw"
    if not con_path.exists() or not raw_path.exists():
        raise FileNotFoundError(f"Missing required continuum files: {con_path} or {raw_path}")
    e_con, incident, transmitted = read_continuum(con_path)
    e_slab, continuum = read_slab_continuum(con_path, raw_path)

    # 6. Setup temporary uniform log-space grid for physical broadening
    # Spacing should be at least as fine as the core X-ray grid spacing.
    # Core spacing: d_lnE ~ (ln(15.0) - ln(0.05)) / 7500 = 0.0007605
    # Total range: ln(102.0) - ln(0.0008) = 11.755
    # Required points: 11.755 / 0.0007605 ~ 15457. Let's use 16000 points.
    e_high = np.geomspace(0.0008, 102.0, 16000)

    # 7. Interpolate raw inputs to e_high
    abs_opt_high = np.interp(e_high, e_opt, opt_abs, left=0.0, right=0.0)
    tau_scattering_high = np.interp(e_high, e_opt, opt_scat, left=0.0, right=0.0)

    if len(e_fopt) > 0:
        fine_lines_high = np.interp(e_high, e_fopt, fopt_tau, left=0.0, right=0.0)
    else:
        fine_lines_high = np.zeros_like(e_high)

    tau_absorption_high = abs_opt_high + fine_lines_high

    inc_high = np.interp(e_high, e_con, incident, left=0.0, right=0.0)
    trans_high = np.interp(e_high, e_con, transmitted, left=0.0, right=0.0)
    emt_continuum_high = np.interp(e_high, e_slab, continuum, left=0.0, right=0.0)

    # 8. Place emission lines on e_high as delta functions
    lin_path = node_dir / f"{point_id}.lin"
    llab_path = node_dir / f"{point_id}.llab"

    emt_lines_high = np.zeros_like(e_high)
    line_catalog = []
    if lin_path.exists() and llab_path.exists():
        line_catalog = read_lines_intensity_validated(lin_path, llab_path)
        if len(line_catalog) > 0:
            dE_high = np.zeros_like(e_high)
            dE_high[:-1] = np.diff(e_high)
            dE_high[-1] = dE_high[-2]

            for E0, flux in line_catalog:
                if E0 < e_high[0] or E0 > e_high[-1]:
                    continue
                idx = np.searchsorted(e_high, E0)
                idx = np.clip(idx, 0, len(e_high) - 1)
                emt_lines_high[idx] += flux / dE_high[idx]

    # 9. Convolve optical depth and emission for turbulent velocity on e_high
    d_lnE_high = np.log(e_high[1]) - np.log(e_high[0])
    sigma_lnE = vturb / (SQRT_2 * C_KM_S)
    sigma_pix_high = sigma_lnE / d_lnE_high

    if sigma_pix_high > 0.0:
        tau_absorption_high = gaussian_filter1d(tau_absorption_high, sigma_pix_high, mode="nearest")
        # tau_scattering_high = gaussian_filter1d(tau_scattering_high, sigma_pix_high, mode="nearest") # already smooth
        emt_continuum_high = gaussian_filter1d(emt_continuum_high, sigma_pix_high, mode="nearest")
        if len(line_catalog) > 0:
            emt_lines_high = gaussian_filter1d(emt_lines_high, sigma_pix_high, mode="constant", cval=0.0)

    # 10. Downsample from e_high to target energy_grid using bin-averaging
    tau_absorption = bin_average(e_high, tau_absorption_high, energy_grid).astype(np.float32)
    tau_scattering = bin_average(e_high, tau_scattering_high, energy_grid).astype(np.float32)
    tau_total = tau_absorption + tau_scattering

    radius_match = re.search(r"^\s*radius\s+([0-9.eE+-]+)([^\n]*)", in_text, re.IGNORECASE | re.MULTILINE)
    if radius_match is None:
        raise ValueError(f"A CLOUDY radius is required to export full-shell emission: {in_file}")
    radius_options = radius_match.group(2).split("#", 1)[0].lower()
    radius_value = float(radius_match.group(1))
    r_in = radius_value if "linear" in radius_options else 10.0**radius_value
    if "parsec" in radius_options:
        r_in *= 3.085677581491367e18
    if not np.isfinite(r_in) or r_in <= 0:
        raise ValueError(f"Invalid CLOUDY radius in {in_file}")

    D_10KPC_CM = 3.085677581491367e22
    C_KEV_TO_ERG = 1.602176634e-9
    if emission_input == "intensity":
        dilution = (r_in / D_10KPC_CM) ** 2
    else:
        dilution = 1.0 / (4.0 * np.pi * D_10KPC_CM**2)

    bin_widths = np.diff(bin_edges(energy_grid))
    photon_conversion_high = dilution / (e_high * C_KEV_TO_ERG)
    emt_continuum = (
        bin_average(e_high, emt_continuum_high * photon_conversion_high, energy_grid)
        * bin_widths
    ).astype(np.float32)
    emt_lines = (
        bin_average(e_high, emt_lines_high * photon_conversion_high, energy_grid)
        * bin_widths
    ).astype(np.float32)
    emt_total = emt_continuum + emt_lines

    diagnostic_dtype = np.float64 if emission_input == "luminosity" else np.float32
    inc_interp = bin_average(e_high, inc_high, energy_grid).astype(diagnostic_dtype)
    trans_interp = bin_average(e_high, trans_high, energy_grid).astype(diagnostic_dtype)

    # 11. Create metadata record
    metadata = np.array(
        (point_id, logxi, vturb, log_nh, log_density, temperature),
        dtype=[
            ('point_id', 'U50'),
            ('logxi', 'f8'),
            ('vturb', 'f8'),
            ('log_nh', 'f8'),
            ('log_density', 'f8'),
            ('temperature', 'f8')
        ]
    )

    # 12. Save compressed arrays
    npz_path = node_dir / f"{point_id}.npz"
    np.savez_compressed(
        npz_path,
        schema_version=np.int16(SCHEMA_VERSION),
        energy=energy_grid,
        tau_absorption=tau_absorption,
        tau_scattering=tau_scattering,
        tau_total=tau_total,
        emt_continuum=emt_continuum,
        emt_lines=emt_lines,
        emt_total=emt_total,
        incident=inc_interp,
        transmitted=trans_interp,
        metadata=metadata,
    )
