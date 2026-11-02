# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from logging import Logger
from string import Template

from vllm import envs
from vllm.entrypoints.serve.utils import api_utils
from vllm.logger import current_formatter_type


def _log_version_and_model(lgr: Logger, version: str, model_name: str) -> None:
    def _version() -> str:
        import importlib.metadata

        try:
            return importlib.metadata.version("vllm-gr")
        except importlib.metadata.PackageNotFoundError:
            from vllm_gr import __version__
        return __version__

    if envs.VLLM_DISABLE_LOG_LOGO or (formatter := current_formatter_type(lgr)) is None:
        message = "vLLM server version %s, serving model %s"
    else:
        logo_template = Template(
            "\n       ${w}█     █     █▄   ▄█${r}   ${b}▄▄▄▄   ▄▄▄${r}\n"
            " ${o}▄▄${r} ${b}▄█${r} ${w}█     █     █ ▀▄▀ █${r} ${w}▄${r} ${b}█  ▄▄  █▄▄▀${r}  version ${w}%s${r}\n"
            "  ${o}█${r}${b}▄█▀${r} ${w}█     █     █     █${r}   ${b}▀▄▄█   █  █${r}  model ${w}%s${r}\n"
            "   ${b}▀▀${r}  ${w}▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀${r}\n"
        )
        colors = {
            "w": "\033[97;1m",  # white
            "o": "\033[93m",  # orange
            "b": "\033[94m",  # blue
            "r": "\033[0m",  # reset
        }
        if formatter != "color":
            # monochrome logo (no ansi escape codes)
            colors = dict.fromkeys(colors, "")

        message = logo_template.substitute(colors)

    lgr.info(message, _version(), model_name)


def patch_banner():
    api_utils.log_version_and_model = _log_version_and_model
