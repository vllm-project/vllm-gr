# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Model-specific policies around the shared offline beam-search trunk."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm_gr.entrypoints.beam_search_config import (
    BeamSearchModel,
    BeamSearchStep,
    PreparedBeamInputs,
)
from vllm_gr.logger import init_logger

if TYPE_CHECKING:
    from vllm.logprobs import FlatLogprobs

    from vllm_gr.entrypoints.gr import GRLLM

logger = init_logger(__name__)


@dataclass
class AdaptedBeamSearchRequest:
    """Model-specific inputs and policies consumed by the shared search."""

    inputs: PreparedBeamInputs
    steps: list[BeamSearchStep]
    ignore_eos: bool
    catalog: Any | None


class ModelAdapter(ABC):
    """Interface between one model family and the shared beam-search trunk."""

    supports_custom_attention = True

    @abstractmethod
    def adapt_beam_search_request(
        self,
        llm: "GRLLM",
        prompts: list[Any],
        params: Any,
    ) -> AdaptedBeamSearchRequest:
        """Prepare model inputs and per-step search policies."""

    @abstractmethod
    def finalize_beam_search_outputs(
        self,
        instances: list[Any],
        prepared: PreparedBeamInputs,
        initial_logprobs: "FlatLogprobs",
        beam_width: int,
    ) -> list[Any]:
        """Convert shared beam state into model-specific outputs."""


class OneRecModelAdapter(ModelAdapter):
    def adapt_beam_search_request(
        self,
        llm: "GRLLM",
        prompts: list[Any],
        params: Any,
    ) -> AdaptedBeamSearchRequest:
        from vllm_gr.models.one_rec import (
            build_beam_search_steps,
            prepare_beam_search_inputs,
        )

        prepared = prepare_beam_search_inputs(llm, prompts, params)
        pre_calculated = int(prepared.begin_token_id is not None) + int(
            prepared.end_token_id is not None
        )
        max_tokens = params.max_tokens
        if max_tokens < pre_calculated:
            logger.warning(
                "max_tokens is less than the number of pre-calculated "
                "tokens; using the pre-calculated token count"
            )
            max_tokens = pre_calculated
        return AdaptedBeamSearchRequest(
            inputs=prepared,
            steps=build_beam_search_steps(
                max_tokens - pre_calculated,
                params.beam_width,
            ),
            ignore_eos=params.ignore_eos,
            catalog=llm.catalog,
        )

    def finalize_beam_search_outputs(
        self,
        instances: list[Any],
        prepared: PreparedBeamInputs,
        initial_logprobs: "FlatLogprobs",
        beam_width: int,
    ) -> list[Any]:
        from vllm_gr.models.one_rec import finalize_beam_search_outputs

        return finalize_beam_search_outputs(
            instances,
            prepared,
            initial_logprobs,
            beam_width,
        )


class RecifModelAdapter(ModelAdapter):
    supports_custom_attention = False

    def adapt_beam_search_request(
        self,
        llm: "GRLLM",
        prompts: list[Any],
        params: Any,
    ) -> AdaptedBeamSearchRequest:
        from vllm_gr.models.recif import (
            build_beam_search_steps,
            prepare_beam_search_inputs,
        )

        return AdaptedBeamSearchRequest(
            inputs=prepare_beam_search_inputs(llm, prompts),
            steps=build_beam_search_steps(params),
            ignore_eos=True,
            catalog=None,
        )

    def finalize_beam_search_outputs(
        self,
        instances: list[Any],
        prepared: PreparedBeamInputs,
        initial_logprobs: "FlatLogprobs",
        beam_width: int,
    ) -> list[Any]:
        from vllm_gr.models.recif import finalize_beam_search_outputs

        return finalize_beam_search_outputs(
            instances,
            prepared,
            initial_logprobs,
            beam_width,
        )


def create_model_adapter(model: BeamSearchModel | str) -> ModelAdapter:
    model = BeamSearchModel(model)
    if model is BeamSearchModel.RECIF:
        return RecifModelAdapter()
    return OneRecModelAdapter()


__all__ = [
    "AdaptedBeamSearchRequest",
    "ModelAdapter",
    "OneRecModelAdapter",
    "RecifModelAdapter",
    "create_model_adapter",
]
