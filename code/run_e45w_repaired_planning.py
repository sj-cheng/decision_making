"""Fresh complete wall-budget evaluation; no mixing with failed W-v2 cells."""
from pathlib import Path
from learning.e45w_support_recovery import install, REVISION
import run_e45w_planning_evaluation as base

ORIGINAL_CONFIGURE = base.configure


def configure(output):
    ORIGINAL_CONFIGURE(output)
    install(Path(output)/'belief_failures_repair')


base.configure = configure


if __name__=='__main__':
    base.main()
