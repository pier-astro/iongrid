#!/usr/bin/env python3
"""Generate the Latin-hypercube parameter grid defined by the LHS_* block of settings.py.

Writes LHS_OUTPUT_DIR/new_parameter_grid.csv (read by `python sim-local.py setup`
when GRID_MODE = "lhs"), plus per-population CSVs, summaries and diagnostic plots.
Needs pandas, scipy and matplotlib.
"""

from iongrid.xstar.lhs import generate_lhs_grid
from settings import (
    LHS_BOUNDS, LHS_EXISTING_COLUMN_MAP, LHS_EXISTING_FILL_VALUES, LHS_EXISTING_GRID_FILE,
    LHS_HISTOGRAM_BINS, LHS_OUTPUT_DIR, LHS_POPULATIONS, LHS_SAVE_PLOTS, LHS_SHOW_PLOTS,
    LHS_VTURB_SAMPLING,
)

if __name__ == "__main__":
    generate_lhs_grid(
        LHS_BOUNDS, LHS_POPULATIONS, LHS_OUTPUT_DIR,
        vturb_sampling=LHS_VTURB_SAMPLING,
        existing_grid_file=LHS_EXISTING_GRID_FILE,
        existing_column_map=LHS_EXISTING_COLUMN_MAP,
        existing_fill_values=LHS_EXISTING_FILL_VALUES,
        save_plots=LHS_SAVE_PLOTS, show_plots=LHS_SHOW_PLOTS,
        histogram_bins=LHS_HISTOGRAM_BINS,
    )
