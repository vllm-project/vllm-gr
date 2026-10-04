# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.models.serving import (
    BaseModelPath,
    LoRAModulePath,
    OpenAIServingModels,
)

from vllm_gr.config import GRConfig
from vllm_gr.utils.catalog import Catalog

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
    tokenizer = self.input_processor.tokenizer
    vllm_config = getattr(engine_client, "vllm_config", None)
    catalog_path = GRConfig.from_vllm_config(vllm_config).catalog_path
    self.catalog_path = catalog_path
    if catalog_path:
        self.catalog = Catalog(tokenizer)
        self.catalog.load(catalog_path)

    # EngineCore has already inherited the BeamProc IPC configuration by the
    # time OpenAIServingModels is created. The tokenizer and catalog are now
    # available as well, so an explicitly enabled Rust sidecar can be fully
    # prepared and started before the API server reports itself ready. This
    # keeps sidecar startup out of the first benchmark/warmup request.
    from vllm_gr.entrypoints.openai.beam_proc_manager import (
        beam_proc_enabled,
        ensure_beam_proc_started,
        prepare_beam_proc_catalog,
    )

    if beam_proc_enabled():
        prepare_beam_proc_catalog(
            getattr(self, "model_config", None),
            tokenizer,
            catalog_path=catalog_path,
        )
        ensure_beam_proc_started()
        from vllm_gr.entrypoints.openai.serving_engine import (
            _ensure_beamproc_apiserver_output,
        )

        # init_gr runs on the API event loop before requests are accepted.
        _ensure_beamproc_apiserver_output(engine_client, "startup")


def patch_OpenAIServingModels_init():
    OpenAIServingModels.__init__ = init_gr
