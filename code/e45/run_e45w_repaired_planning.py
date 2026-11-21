"""Fresh complete wall-budget evaluation; no mixing with failed W-v2 cells."""
import sys
from pathlib import Path
_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
import repo_paths
repo_paths.install()
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
