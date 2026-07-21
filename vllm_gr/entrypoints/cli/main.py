# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.metadata
import sys

from vllm_gr.logger import init_logger

logger = init_logger(__name__)


def _version() -> str:
    try:
        return importlib.metadata.version("vllm-gr")
    except importlib.metadata.PackageNotFoundError:
        from vllm_gr import __version__

        return __version__


def _maybe_launch_beam_search_serve() -> bool:
    """Check for --gr flag in serve command and launch beam search serving.

    Returns True if the --gr flag was handled and the original vLLM main
    should be invoked afterwards.
    """
    if "serve" not in sys.argv or "--gr" not in sys.argv:
        return False

    logger.info(
        "vllm detected --gr flag in serve command. Launching vllm-gr specialized serving path."
    )

    # Check for LMCache connector in args
    for idx, arg in enumerate(sys.argv):
        if arg.startswith("--kv-transfer-config"):
            if "LMCacheConnectorV1" in arg:
                import os

                os.environ["VLLM_GR_LMCACHE_PATCH"] = "1"
            if arg == "--kv-transfer-config" and idx + 1 < len(sys.argv):
                if "LMCacheConnectorV1" in sys.argv[idx + 1]:
                    import os

                    os.environ["VLLM_GR_LMCACHE_PATCH"] = "1"

    # Beam search patches are already applied through the unified
    # _apply_gr_beam_patches() in patch.py via the plugin system.
    # Remove the --gr flag so vLLM's parser doesn't complain.
    sys.argv.remove("--gr")
    return True


def main() -> None:
    # Configure this before importing vLLM. The parent value is inherited by
    # EngineCore and Worker subprocesses launched by the serve command.
    from vllm_gr._env_check import configure_model_runner_default

    configure_model_runner_default()

    from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG, cli_env_setup
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    import vllm_gr.entrypoints.cli.check_env as gr_check_env

    # Handle --gr flag for beam search serving mode
    _maybe_launch_beam_search_serve()

    cli_env_setup()

    parser = FlexibleArgumentParser(
        description="vLLM-GR CLI",
        epilog=VLLM_SUBCMD_PARSER_EPILOG.format(subcmd="[subcommand]"),
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=_version(),
    )

    subparsers = parser.add_subparsers(required=False, dest="subparser")
    cmds = {}

    # Keep the diagnostic path independent from the heavy serving import
    # graph. A broken optional serving dependency must not prevent check-env
    # from explaining the runtime problem.
    command_list = gr_check_env.cmd_init()
    if len(sys.argv) < 2 or sys.argv[1] != "check-env":
        import vllm_gr.entrypoints.cli.serve as gr_serve

        command_list = [*gr_serve.cmd_init(), *command_list]

    for cmd in command_list:
        cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
        cmds[cmd.name] = cmd

    args = parser.parse_args()
    if args.subparser in cmds:
        cmds[args.subparser].validate(args)

    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
