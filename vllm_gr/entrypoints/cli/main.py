# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
import os
import sys

from vllm.entrypoints.cli.main import main as vllm_main

logger = logging.getLogger(__name__)


def _has_lmcache_connector_in_argv(argv: list[str]) -> bool:
    for idx, arg in enumerate(argv):
        if arg.startswith("--kv-transfer-config"):
            if "LMCacheConnectorV1" in arg:
                return True
            if arg == "--kv-transfer-config" and idx + 1 < len(argv):
                return "LMCacheConnectorV1" in argv[idx + 1]
    return False


def main():
    if "serve" in sys.argv and "--gr" in sys.argv:
        logger.info(
            "vllm detected --gr flag in serve command. Launching vllm-gr specialized serving path."
        )
        if _has_lmcache_connector_in_argv(sys.argv):
            os.environ["VLLM_GR_LMCACHE_PATCH"] = "1"

        # Launch vLLM-gr specialized serving path
        from vllm.entrypoints.utils import cli_env_setup

        from vllm_gr.patch import run_patch

        run_patch()
        cli_env_setup()
        sys.argv.remove("--gr")

    # Delegate to original vLLM CLI
    vllm_main()


if __name__ == "__main__":
    main()
