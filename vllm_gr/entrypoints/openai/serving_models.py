# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import asyncio

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.models.serving import (
    BaseModelPath,
    LoRAModulePath,
    OpenAIServingModels,
)

from vllm_gr.config import GRConfig
from vllm_gr.constraints import BeamConstraintCoordinator, create_constraint_strategy
from vllm_gr.constraints.driver_inject import inject_constraint_resources_async
from vllm_gr.constraints.startup import prepare_constraint_table
from vllm_gr.constraints.strategies import (
    ConstraintTableStrategy,
    TrieConstraintStrategy,
)

_original_init = OpenAIServingModels.__init__


def _model_vocab_size(vllm_config: object | None) -> int:
    """Resolve vocab size across vLLM ModelConfig API versions."""

    model_config = getattr(vllm_config, "model_config", None)
    get_vocab_size = getattr(model_config, "get_vocab_size", None)
    vocab_size = (
        get_vocab_size() if callable(get_vocab_size) else getattr(model_config, "vocab_size", None)
    )
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError(
            "constraint_backend='constraint_table' requires a positive "
            "engine_client.vllm_config.model_config vocab size"
        )
    return vocab_size


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
    tokenizer = self.input_processor.tokenizer
    vllm_config = getattr(engine_client, "vllm_config", None)
    gr_config = GRConfig.from_vllm_config(vllm_config)
    canonical_table = gr_config.constraint_table
    if canonical_table is not None and canonical_table.enabled:
        self.constraint_strategy = ConstraintTableStrategy(canonical_table.path or "")
        self.constraint_strategy.digest = canonical_table.artifact_digest
        self.constraint_coordinator = BeamConstraintCoordinator(self.constraint_strategy)
    elif gr_config.catalog_path:
        self.constraint_strategy = create_constraint_strategy(
            backend=gr_config.constraint_backend,
            tokenizer=tokenizer,
            catalog_path=gr_config.catalog_path,
        )
        self.constraint_coordinator = BeamConstraintCoordinator(self.constraint_strategy)
        if isinstance(self.constraint_strategy, TrieConstraintStrategy):
            # Compatibility for existing callers that inspect ``models.catalog``.
            self.catalog = self.constraint_strategy.catalog
        elif isinstance(self.constraint_strategy, ConstraintTableStrategy):
            # Compile synchronously (cheap, no workers touched); the collective
            # RPC broadcast is deferred to the first beam_search so the async
            # client cannot race it (issue #246 §7).
            vocab_size = _model_vocab_size(vllm_config)
            resource = prepare_constraint_table(
                catalog_path=gr_config.catalog_path,
                tokenizer=tokenizer,
                vocab_size=vocab_size,
                gr_config=gr_config,
            )
            self.constraint_strategy.digest = resource.digest
            self._constraint_resource = resource
            self._constraint_tables_injected = False
            self._constraint_injection_lock = asyncio.Lock()


async def _ensure_constraint_tables_injected(self) -> None:
    """Broadcast the compiled resource to every worker exactly once.

    Idempotent and serialized by an ``asyncio.Lock`` so concurrent beam_search
    calls cannot double-inject or observe a half-injected worker fleet.

    The method is mounted globally on ``OpenAIServingModels``; Trie / no-catalog
    instances never initialize the constraint fields, so this is a no-op for
    them (callers additionally gate on ``is_constrained`` before invoking).
    """
    if not getattr(self, "_constraint_resource", None):
        return
    if getattr(self, "_constraint_tables_injected", False):
        return
    async with self._constraint_injection_lock:
        if getattr(self, "_constraint_tables_injected", False):
            return
        await inject_constraint_resources_async(self.engine_client, self._constraint_resource)
        self._constraint_tables_injected = True


def patch_OpenAIServingModels_init():
    OpenAIServingModels.__init__ = init_gr
    OpenAIServingModels._ensure_constraint_tables_injected = _ensure_constraint_tables_injected
