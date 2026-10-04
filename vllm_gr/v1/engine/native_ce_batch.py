# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Snapshot Python CE state for direct submission to native TX.

Python retains context/cache bookkeeping. Rust expands grouped rows, packs
numeric arrays, restores sliced token IDs, and encodes the final wire header.
No Python objects or borrowed array pointers survive the native submission call.
"""

import numpy as np


def send_engine_outputs(engine, bridge, payload):
    from vllm_gr.ops.sliced_logits import get_vocab_slice_config
    from vllm_gr.v1.engine import core

    contexts = payload.get("output_contexts")
    vocab_offset, vocab_size, _ = get_vocab_slice_config()
    grouped_states = getattr(engine, "beamproc_grouped_output_states", {})
    for client_index, batch in payload["outputs"].items():
        descriptors, buffers, consumed = [], [], []
        for output in batch.outputs:
            if not output.request_id or (
                contexts is not None and output.request_id not in contexts
            ):
                continue
            group = grouped_states.get(output.request_id)
            logprobs = output.new_logprobs
            if logprobs is not None:
                token_ids = np.ascontiguousarray(logprobs.logprob_token_ids, dtype="<i4")
                scores = np.ascontiguousarray(logprobs.logprobs, dtype="<f4")
                if token_ids.shape != scores.shape:
                    raise ValueError("CE candidate token/logprob shapes differ")
                if group is not None:
                    width = len(group["child_ids"])
                    if token_ids.ndim == 1:
                        token_ids = token_ids.reshape(1, -1)
                        scores = scores.reshape(1, -1)
                    if (
                        width
                        and token_ids.ndim == 2
                        and token_ids.shape[0] == 1
                        and token_ids.shape[1] % width == 0
                    ):
                        token_ids = token_ids.reshape(width, token_ids.shape[1] // width)
                        scores = scores.reshape(token_ids.shape)
                    if not width or token_ids.ndim != 2 or token_ids.shape[0] != width:
                        # Match the existing serializer's fallback for unsplittable groups.
                        group = None
                if token_ids.ndim != 2:
                    raise ValueError("CE candidate arrays must have two dimensions")
                rows, cols = token_ids.shape
            else:
                group = None
                token_ids, scores = b"", b""
                rows, cols = 0, 0

            if group is not None:
                context = dict(group["trace_context"])
                context.pop("parent_token_ids", None)
                context["prompt_token_count"] = 0
                context["use_shadow_parent_tokens"] = True
                child_ids = group["child_ids"]
                ranks, new_tokens = b"", b""
                consumed.append((output.request_id, group))
            else:
                context = contexts.get(output.request_id) if contexts is not None else None
                if context is None:
                    context = core._beamproc_context_for_output(engine, output)
                child_ids = None
                ranks = (
                    np.ascontiguousarray(logprobs.sampled_token_ranks, dtype="<i4")
                    if logprobs is not None
                    else b""
                )
                new_tokens = np.asarray(output.new_token_ids, dtype="<i4")
            descriptors.append(
                {
                    "request_id": output.request_id,
                    "child_ids": child_ids,
                    "context": context,
                    "finish_reason": core._beamproc_value(output.finish_reason),
                    "stop_reason": core._beamproc_value(output.stop_reason),
                    "num_cached_tokens": core._beamproc_num_cached_tokens(output),
                    "rows": rows,
                    "cols": cols,
                }
            )
            buffers.extend((token_ids, scores, ranks, new_tokens))
        if not descriptors:
            continue
        bridge.send_batch(
            {
                "engine_index": payload.get("engine_index", engine.engine_index),
                "client_index": client_index,
                "correlation_base": payload.get("correlation_base", 0),
                "vocab_offset": vocab_offset,
                "vocab_size": vocab_size,
                "outputs": descriptors,
            },
            buffers,
        )
        # Submission has copied all data. Keep state mutation on the CE caller;
        # native workers never touch Python cache or scheduler objects.
        for request_id, group in consumed:
            if grouped_states.get(request_id) is group:
                grouped_states.pop(request_id)
            if group.get("is_final_step"):
                core._cleanup_beamproc_root_state(engine, group["root_request_id"], request_id)
