# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""HSTU model adapter package.

Importing this package triggers Layer-1 family registration (pure Python,
no heavy imports).  The adapter class is registered separately in
``adapter.py`` when it is imported (Layer-2).
"""

from vllm_gr.adapters.registry import register_family

register_family(
    model_type="hstu",
    gpu_worker="vllm_gr.worker.gr_worker.GRWorker",
    npu_worker="vllm_gr.worker.gr_npu_worker.GRNPUWorker",
    defaults_fn="vllm_gr.adapters.hstu._defaults:apply_hstu_defaults",
    payload_type="vllm_gr.engine:GRFeaturesPayload",
)
