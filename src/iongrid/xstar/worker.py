"""Entry point used by generated HTCondor wrappers."""

import sys
from .htcondor import run_condor_batch


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit("Usage: python -m iongrid.xstar.worker ROOT BATCH RESULTS XSTAR")
    run_condor_batch(sys.argv[1], sys.argv[2], results_path=sys.argv[3], xstar_path=sys.argv[4])


if __name__ == "__main__":
    main()
