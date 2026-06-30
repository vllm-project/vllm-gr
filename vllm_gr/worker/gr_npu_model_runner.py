# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ascend NPU GR model runner."""

from copy import deepcopy

from vllm.distributed.kv_transfer import (
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

from vllm_gr.adapters.registry import (
    create_adapter,
    get_model_type_from_config,
    is_adapter_registered,
)
from vllm_gr.config.gr_config import GRConfig
from vllm_gr.worker.gr_runner_mixin import GRFeatureRunnerMixin


class GRNPUModelRunner(GRFeatureRunnerMixin, NPUModelRunner):
    """NPU model runner that injects GR features into HSTU inputs."""

    def __init__(self, vllm_config, device, gr_config=None):
        model_type = get_model_type_from_config(vllm_config)
        if not is_adapter_registered(model_type):
            raise RuntimeError(
                "GRNPUModelRunner must not be used for LLM passthrough models. "
                "Use the default vllm-ascend NPU worker (do not set GR worker_cls)."
            )

        gr_config = gr_config or GRConfig()
        adapter = create_adapter(model_type, vllm_config, gr_config)
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            adapter=adapter,
        )

    def _use_hstu_generic_kv_cache_init(self) -> bool:
        model = getattr(self, "model", None)
        if model is None or not getattr(model, "_hstu_kv_cache_layers", None):
            return False

        from vllm_gr.adapters.registry import is_adapter_registered

        model_type = get_model_type_from_config(self.vllm_config)
        return is_adapter_registered(model_type)

    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        self._apply_hstu_runtime_policy()
        self._sync_hstu_static_forward_context()
        return result

    def get_kv_cache_spec(self):
        self._sync_hstu_static_forward_context()

        kv_cache_spec = super().get_kv_cache_spec()
        if kv_cache_spec:
            return kv_cache_spec

        return self._get_hstu_kv_cache_spec_fallback()

    def initialize_kv_cache(self, kv_cache_config):
        self._sync_hstu_static_forward_context()
        if not self._use_hstu_generic_kv_cache_init():
            return super().initialize_kv_cache(kv_cache_config)

        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config
        self.may_add_encoder_only_layers_to_kv_cache_config()
        self.maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        GPUModelRunner.initialize_attn_backend(self, kv_cache_config)
        self.use_hybrid_blocks = len(self.attn_groups) > 1
        self.need_accepted_tokens = False

        kernel_block_sizes = GPUModelRunner._prepare_kernel_block_sizes(
            self,
            kv_cache_config,
        )
        GPUModelRunner.initialize_metadata_builders(
            self,
            kv_cache_config,
            kernel_block_sizes,
        )
        self.may_reinitialize_input_batch(kv_cache_config)
        kv_cache_raw_tensors = GPUModelRunner._allocate_kv_cache_tensors(
            self,
            kv_cache_config,
        )
        kv_caches = GPUModelRunner._reshape_kv_cache_tensors(
            self,
            kv_cache_config,
            kv_cache_raw_tensors,
            kernel_block_sizes,
        )
        for layer_name, target_layer_name in getattr(
            self,
            "shared_kv_cache_layers",
            {},
        ).items():
            kv_caches[layer_name] = kv_caches[target_layer_name]

        num_attn_module = 2 if self.model_config.hf_config.model_type == "longcat_flash" else 1
        from vllm.v1.worker.utils import bind_kv_cache

        compilation_config = getattr(
            self,
            "compilation_config",
            getattr(self.vllm_config, "compilation_config", None),
        )
        static_forward_context = getattr(
            compilation_config,
            "static_forward_context",
            {},
        )

        bind_kv_cache(
            kv_caches,
            static_forward_context,
            getattr(self, "kv_caches", []),
            num_attn_module,
        )

        if has_kv_transfer_group():
            kv_transfer_group = get_kv_transfer_group()
            if self.cross_layers_kv_cache is not None:
                assert self.cross_layers_attn_backend is not None
                kv_transfer_group.register_cross_layers_kv_cache(
                    self.cross_layers_kv_cache,
                    self.cross_layers_attn_backend,
                )
            else:
                kv_transfer_group.register_kv_caches(kv_caches)
