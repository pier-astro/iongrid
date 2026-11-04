"""Prepare and run the campaign locally."""

import sys

sys.dont_write_bytecode = True

import settings
from iongrid import Grid

WORKERS = 4
SELECT = "ready"  # Use "failed" to retry failed simulations.
TIMEOUT = None

if (settings.ROOT / "manifest.csv").is_file():
    grid = Grid.open(settings.ROOT)
else:
    grid = Grid.create(settings.ROOT,
                       engine=settings.ENGINE,
                       template=settings.TEMPLATE,
                       params=settings.PARAMS,
                       sed=settings.SED)

print(
    grid.run(
        workers=WORKERS,
        select=SELECT, 
        executable=settings.EXECUTABLE,
        timeout=TIMEOUT,
        )
    )

for point, error in grid.collect().items():
    print(f"{point}: {error}")
print(grid.status())
