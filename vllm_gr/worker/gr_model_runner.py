# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""GPU GR model runner."""

from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_gr.adapters.registry import (
    create_adapter,
    get_model_type_from_config,
    is_adapter_registered,
)
from vllm_gr.config.gr_config import GRConfig
from vllm_gr.worker.gr_runner_mixin import GRFeatureRunnerMixin


class GRModelRunner(GRFeatureRunnerMixin, GPUModelRunner):
    """GPU model runner that injects GR features into HSTU inputs."""

    def __init__(self, vllm_config, device, gr_config=None):
        gr_config = gr_config or GRConfig()
        model_type = get_model_type_from_config(vllm_config)
        if not is_adapter_registered(model_type):
            raise ValueError(
                f"GRModelRunner requires a registered adapter, got model_type={model_type!r}"
            )
        adapter = create_adapter(model_type, vllm_config, gr_config)
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            adapter=adapter,
        )

    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        self._apply_hstu_runtime_policy()
        return result
