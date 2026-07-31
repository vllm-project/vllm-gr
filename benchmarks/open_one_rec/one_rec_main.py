import logging

# isort: off
# Import patch to register onerec dataset and GR beam-search backends.
# This monkey-patches vllm.benchmarks.datasets.get_samples etc. before they're used.
# Must be imported before any vllm.benchmarks module usage — do NOT reorder.
import benchmarks.open_one_rec.one_rec_patch  # noqa: F401
from vllm.entrypoints.cli.main import main as vllm_main
# isort: on

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    """Entry point for vllm-gr benchmark."""
    vllm_main()
