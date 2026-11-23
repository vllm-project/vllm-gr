# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

import sys

from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_gr.config.parser import read_settings
from vllm_gr.engine.startup import argument_parsers, native_settings


class GRArgumentParser(FlexibleArgumentParser):
    """Apply typed YAML defaults while preserving upstream CLI processing."""

    def parse_args(self, args=None, namespace=None):
        argv = list(sys.argv[1:] if args is None else args)
        owners = list(argument_parsers(self))
        snapshots = [
            (owner, dict(owner._defaults), [(action, action.default) for action in owner._actions])
            for owner in owners
        ]
        option_actions = {
            option: action
            for owner in owners
            for option, action in owner._option_string_actions.items()
        }
        explicit = set(vars(namespace)) if namespace is not None else set()
        for token in argv:
            option = token.split("=", 1)[0].split(".", 1)[0].replace("_", "-")
            if option in option_actions:
                explicit.add(option_actions[option].dest)
        self._gr_yaml_fields: set[str] = set()
        try:
            result = super().parse_args(argv, namespace)
            result._gr_explicit_native_fields = explicit | self._gr_yaml_fields
            return result
        finally:
            for owner, defaults, actions in snapshots:
                owner._defaults.clear()
                owner._defaults.update(defaults)
                for action, default in actions:
                    action.default = default
            self._gr_yaml_fields = set()

    def load_config_file(self, file_path: str) -> list[str]:
        if not file_path.endswith((".yaml", ".yml")):
            raise ValueError("--config requires a YAML file")
        document = read_settings(file_path)
        values = native_settings(document, parser=self)
        self._gr_yaml_fields.update(values)
        if "gr" in document:
            values["gr"] = [document["gr"]]
        for owner in argument_parsers(self):
            names = {action.dest for action in owner._actions}
            owner.set_defaults(**{name: value for name, value in values.items() if name in names})
        # Upstream uses this option to recognize a model supplied by YAML.
        # Other values stay typed defaults, avoiding stringification of null,
        # false and empty lists. Explicit argv still overrides them.
        model = values.get("model")
        return ["--model", str(model)] if model is not None else []
