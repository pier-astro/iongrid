"""Prepare HTCondor batches or refresh a submitted campaign."""

import sys
sys.dont_write_bytecode = True

import settings
from iongrid import Grid, condor

ACTION = "write"  # "write" or "refresh".

BATCH = 4
SELECT = "ready"  # Use "failed" to retry failed simulations.

INIT = None  # Path to a shell script that configures the engine on workers.

DIRECTIVES = """\
request_cpus = 1
request_memory = 3G
"""

if ACTION == "write":
    if (settings.ROOT / "manifest.csv").is_file():
        grid = Grid.open(settings.ROOT)

    else:
        grid = Grid.create(settings.ROOT,
                           engine=settings.ENGINE,
                           template=settings.TEMPLATE,
                           params=settings.PARAMS,
                           sed=settings.SED)
        
    submit = condor.write(grid,
                          batch=BATCH,
                          text=DIRECTIVES,
                          init=INIT,
                          select=SELECT,
                          executable=settings.EXECUTABLE)
    print(f'condor_submit "{submit}"')

elif ACTION == "refresh":
    print(Grid.open(settings.ROOT).refresh())

else:
    raise ValueError("ACTION must be 'write' or 'refresh'")
