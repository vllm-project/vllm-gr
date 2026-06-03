# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
GRLLM — user-facing entry point for generative recommendation inference.

Inherits from vLLM's LLM class and exposes a recommend() API that accepts
GRPrompt inputs. Internally creates a GRLLMEngine (which has GR-specific
processors) and drives the engine loop to collect GRRequestOutput results.
"""

import gc
import time
from typing import List, Optional, Sequence, Union

from tqdm.auto import tqdm
from vllm.entrypoints.llm import LLM
from vllm.pooling_params import PoolingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils.counter import Counter

from vllm_gr.config.gr_config import GRConfig
from vllm_gr.engine.arg_utils import GREngineArgs
from vllm_gr.engine.gr_engine import GRLLMEngine
from vllm_gr.inputs.data import GRPrompt, is_gr_prompt
from vllm_gr.logger import init_logger
from vllm_gr.outputs import GRRequestOutput
from vllm_gr.utils.npu_adapter import release_accelerator_memory

logger = init_logger(__name__)


class GRLLM(LLM):
    """High-level recommendation interface built on top of vLLM's LLM.

    Does not call super().__init__() because the base LLM creates a
    standard LLMEngine; instead we create a GRLLMEngine directly.
    """

    def __init__(
        self,
        model: str = "",
        model_family: Optional[str] = None,
        max_seq_len: int = 512,
        max_candidates: int = 1000,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        gr_config: Optional[GRConfig] = None,
        **kwargs,
    ):
        hstu_runtime_policy = {
            "enable_npu_fused_prefill": kwargs.pop("enable_npu_fused_prefill", None),
            "force_hammer_npu": kwargs.pop("force_hammer_npu", None),
        }
        resolved_gr_config = gr_config or GRConfig(
            max_seq_len=max_seq_len,
            max_candidates=max_candidates,
        )
        for policy_name, policy_value in hstu_runtime_policy.items():
            if policy_value is not None and hasattr(resolved_gr_config, policy_name):
                setattr(resolved_gr_config, policy_name, policy_value)
        engine_args = GREngineArgs(
            model=model,
            model_family=model_family,
            max_seq_len=resolved_gr_config.max_seq_len,
            max_candidates=resolved_gr_config.max_candidates,
            enable_npu_fused_prefill=getattr(resolved_gr_config, "enable_npu_fused_prefill", None),
            force_hammer_npu=getattr(resolved_gr_config, "force_hammer_npu", None),
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            **kwargs,
        )

        self.llm_engine = GRLLMEngine.from_engine_args(
            engine_args=engine_args,
            usage_context=UsageContext.LLM_CLASS,
        )
        self.engine_class = type(self.llm_engine)

        self.model = model
        self.model_family = engine_args.model_family
        self.max_seq_len = resolved_gr_config.max_seq_len
        self.max_candidates = resolved_gr_config.max_candidates
        self.gr_config = resolved_gr_config
        self.request_counter = Counter()
        self.default_sampling_params = None
        self.supported_tasks = ("recommendation",)
        self.io_processor = None

        logger.info(
            "GRLLM initialized: model=%s, model_family=%s, max_seq_len=%d, max_candidates=%d",
            model,
            model_family,
            self.max_seq_len,
            self.max_candidates,
        )

    def recommend(
        self,
        prompts: Union[GRPrompt, Sequence[GRPrompt]],
        *,
        use_tqdm: bool = True,
    ) -> List[GRRequestOutput]:
        """Generate recommendation results for one or more GRPrompts."""
        start_time = time.time()

        if not isinstance(prompts, (list, tuple)):
            prompts = [prompts]

        for prompt in prompts:
            if not is_gr_prompt(prompt):
                raise ValueError(
                    f"Expected GRPrompt, got {type(prompt)}. "
                    "GRPrompt must contain 'user_id', 'user_history', "
                    "and 'candidates'."
                )

        request_ids = [f"gr-{next(self.request_counter)}" for _ in prompts]

        for prompt, request_id in zip(prompts, request_ids):
            pooling_params = PoolingParams(task="embed")
            self.llm_engine.add_request(
                request_id=request_id,
                prompt=prompt,
                params=pooling_params,
            )

        outputs = self._run_recommendation_engine(
            request_ids=request_ids,
            use_tqdm=use_tqdm,
        )

        elapsed = time.time() - start_time
        logger.debug(
            "Recommendation completed in %.3fs for %d prompts",
            elapsed,
            len(prompts),
        )
        return outputs

    def _run_recommendation_engine(
        self,
        request_ids: list[str],
        use_tqdm: bool,
    ) -> List[GRRequestOutput]:
        outputs_by_request_id: dict[str, GRRequestOutput] = {}

        if use_tqdm:
            pbar = tqdm(
                total=len(request_ids),
                desc="Generating recommendations",
            )
        else:
            pbar = None

        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()

            for output in step_outputs:
                if isinstance(output, GRRequestOutput) and output.finished:
                    if output.request_id not in outputs_by_request_id and pbar is not None:
                        pbar.update(1)
                    outputs_by_request_id[output.request_id] = output

        if pbar is not None:
            pbar.close()

        missing_request_ids = [
            request_id for request_id in request_ids if request_id not in outputs_by_request_id
        ]
        if missing_request_ids:
            raise RuntimeError(f"Missing GRRequestOutput for request ids: {missing_request_ids!r}")

        return [outputs_by_request_id[request_id] for request_id in request_ids]

    def recommend_batch(
        self,
        prompts: Sequence[GRPrompt],
        batch_size: int = 32,
    ) -> List[GRRequestOutput]:
        """Process prompts in fixed-size batches for large-scale inference."""
        all_outputs = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            outputs = self.recommend(batch, use_tqdm=False)
            all_outputs.extend(outputs)
        return all_outputs

    def get_engine(self) -> GRLLMEngine:
        return self.llm_engine

    def get_prefix_cache_counters(self) -> dict[str, int]:
        return self.llm_engine.get_prefix_cache_counters()

    def shutdown(self) -> None:
        if hasattr(self, "llm_engine") and self.llm_engine is not None:
            self.llm_engine.shutdown()
            self.llm_engine = None
        self.default_sampling_params = None
        self.io_processor = None
        gc.collect()
        release_accelerator_memory()
        logger.info("GRLLM shutdown complete")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False
