# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""A host-only Prefill runner constructed through GR's shared factory."""

from vllm_gr.platforms.graph import graph_for
from vllm_gr.v1.worker.prefill_graph_common import BasePrefillGraphRunner


class ExamplePrefillRunner(BasePrefillGraphRunner):
    def __init__(self, model_runner, buckets=None):
        self.model_runner = model_runner
        self.buckets = list(buckets or [])
        self.captured_buckets = ()

    def capture_graphs(self, runner):
        assert runner is self.model_runner
        self.captured_buckets = graph_for(runner).capture("prefill", tuple, self.buckets)

    def execute_model_prefill(
        self, bucket, input_ids, positions, slot_mapping, num_scheduled_tokens
    ):
        if bucket in self.captured_buckets:
            return graph_for(self.model_runner).replay("prefill", tuple, input_ids)
        return None
