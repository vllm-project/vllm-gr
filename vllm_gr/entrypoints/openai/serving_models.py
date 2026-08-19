# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.models.serving import (
    BaseModelPath,
    LoRAModulePath,
    OpenAIServingModels,
)

from vllm_gr.config import BeamControlMode, ConstraintBackend, GRConfig
from vllm_gr.constraints import (
    BeamConstraintCoordinator,
    TrieConstraintCompiler,
    create_constraint_strategy,
)
from vllm_gr.constraints.strategies import TrieConstraintStrategy
from vllm_gr.constraints.trie_resource import trie_resource_to_wire

_original_init = OpenAIServingModels.__init__


def init_gr(
    self,
    engine_client: EngineClient,
    base_model_paths: list[BaseModelPath],
    *,
    lora_modules: list[LoRAModulePath] | None = None,
):
    _original_init(self, engine_client, base_model_paths, lora_modules=lora_modules)
    self.catalog = None
    self.constraint_strategy = None
    self.constraint_coordinator = BeamConstraintCoordinator(None)
    self.beam_constraint_resource = None
    self.beam_constraint_resource_wire = None
    tokenizer = self.input_processor.tokenizer
    vllm_config = getattr(engine_client, "vllm_config", None)
    gr_config = GRConfig.from_vllm_config(vllm_config)
    if gr_config.catalog_path:
        if (
            gr_config.beam_control_mode
            in (
                BeamControlMode.ENGINE_PER_STEP_REQUEST,
                BeamControlMode.ENGINE_PERSISTENT_REQUEST,
            )
            and gr_config.constraint_backend is ConstraintBackend.TRIE
        ):
            # Serving performs the tokenizer-dependent compilation once, then
            # sends only an immutable token-ID resource across the EngineCore
            # boundary. No Catalog/tokenizer object is retained for decisions.
            resource = TrieConstraintCompiler.from_catalog(tokenizer, gr_config.catalog_path)
            self.beam_constraint_resource = resource
            self.beam_constraint_resource_wire = trie_resource_to_wire(resource)
        else:
            self.constraint_strategy = create_constraint_strategy(
                backend=gr_config.constraint_backend,
                tokenizer=tokenizer,
                catalog_path=gr_config.catalog_path,
            )
            self.constraint_coordinator = BeamConstraintCoordinator(self.constraint_strategy)
            if isinstance(self.constraint_strategy, TrieConstraintStrategy):
                # Compatibility for callers using the staged Serving mode.
                self.catalog = self.constraint_strategy.catalog


def patch_OpenAIServingModels_init():
    OpenAIServingModels.__init__ = init_gr
