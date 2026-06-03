# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Ported to vllm-GR with modifications.

"""Multitask module for HSTU model.

This fork keeps the inference-time prediction path and intentionally omits the
full upstream supervision-label / weight / loss computation pipeline. Training
parity is out of scope for the inference-only recommendation runtime.
"""

import abc
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from vllm_gr.config.multitask_types import MultitaskTaskType
from vllm_gr.logger import init_logger
from vllm_gr.models.dlrm_v3.common import HammerModule

logger = init_logger(__name__)


@dataclass
class TaskConfig:
    task_name: str
    task_weight: int
    task_type: MultitaskTaskType


class MultitaskModule(HammerModule):
    """Abstract base class for multitask modules."""

    @abc.abstractmethod
    def forward(
        self,
        encoded_user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        supervision_labels: Dict[str, torch.Tensor],
        supervision_weights: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        pass


def _compute_pred_and_logits(
    prediction_module: torch.nn.Module,
    encoded_user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    task_offsets: List[int],
    has_multiple_task_types: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute predictions and logits for all tasks."""
    mt_logits = prediction_module(encoded_user_embeddings * item_embeddings).transpose(0, 1)
    mt_preds_list: List[torch.Tensor] = []

    for task_type in MultitaskTaskType:
        logits = mt_logits[task_offsets[task_type] : task_offsets[task_type + 1], :]
        if task_offsets[task_type + 1] - task_offsets[task_type] > 0:
            if task_type == MultitaskTaskType.REGRESSION:
                mt_preds_list.append(logits)
            else:
                mt_preds_list.append(F.sigmoid(logits))

    if has_multiple_task_types:
        mt_preds: torch.Tensor = torch.concat(mt_preds_list, dim=0)
    else:
        mt_preds: torch.Tensor = mt_preds_list[0]

    return mt_preds, mt_logits


class DefaultMultitaskModule(MultitaskModule):
    """Default implementation of multitask module.

    Only prediction tensors are produced. Training labels, weights, and losses
    are not materialized in this inference-oriented fork.
    """

    def __init__(
        self,
        task_configs: List[TaskConfig],
        embedding_dim: int,
        prediction_fn: Callable[[int, int], torch.nn.Module],
        causal_multitask_weights: float,
    ) -> None:
        super().__init__()

        assert sorted(task_configs, key=lambda x: x.task_type) == task_configs, (
            "task_configs must be sorted by task_type."
        )
        assert len(task_configs) > 0, "task_configs must be non-empty."

        self._task_configs: List[TaskConfig] = task_configs
        self._task_offsets: List[int] = [0] * (len(MultitaskTaskType) + 1)

        for task in self._task_configs:
            self._task_offsets[task.task_type + 1] += 1

        self._has_multiple_task_types: bool = self._task_offsets.count(0) < len(MultitaskTaskType)
        self._task_offsets[1:] = np.cumsum(self._task_offsets[1:]).tolist()
        self._causal_multitask_weights: float = causal_multitask_weights
        self._prediction_module: torch.nn.Module = prediction_fn(embedding_dim, len(task_configs))

    def forward(
        self,
        encoded_user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        mt_preds, mt_logits = _compute_pred_and_logits(
            prediction_module=self._prediction_module,
            encoded_user_embeddings=encoded_user_embeddings,
            item_embeddings=item_embeddings,
            task_offsets=self._task_offsets,
            has_multiple_task_types=self._has_multiple_task_types,
        )

        return mt_preds
