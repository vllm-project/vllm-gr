# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA Graph Runner for GPU prefill execution."""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_gr.utils.npu_adapter import is_cuda_available
from vllm_gr.v1.worker.prefill_graph_common import (
    ENV_ENABLE_PREFILL_GRAPH,
    SUPPORTED_NUM_REQS,
    BasePrefillGraphRunner,
    is_prefill_graph_enabled,
)

logger = init_logger(__name__)


class GPUPrefillGraphRunner(BasePrefillGraphRunner):
    """Manages CUDA graph capture and replay on NVIDIA GPUs."""

    def __init__(
        self,
        model_runner: Any,
        buckets: list[int] | None = None,
        max_pad_ratio: float | None = None,
        max_pad_tokens: int | None = None,
    ) -> None:
        super().__init__(
            model_runner=model_runner,
            buckets=buckets,
            max_pad_ratio=max_pad_ratio,
            max_pad_tokens=max_pad_tokens,
        )
        logger.info(
            "GPUPrefillGraphRunner initialized on device %s with buckets %s",
            self.device,
            self.buckets,
        )

    def capture_graphs(self, runner: Any) -> None:
        """Captures CUDA graphs for all prefill bucket sizes and supported batch sizes (1..8 requests)."""
        if not is_prefill_graph_enabled():
            logger.info(
                "Prefill CUDA Graph capture is disabled via environment variable (%s=0).",
                ENV_ENABLE_PREFILL_GRAPH,
            )
            return

        if not is_cuda_available():
            logger.info("CUDA not available; skipping CUDA graph capture for prefill.")
            return

        from tqdm import tqdm
        from vllm.config import CUDAGraphMode
        from vllm.distributed.device_communicators.pynccl_allocator import (
            set_graph_pool_id,
        )
        from vllm.forward_context import set_forward_context
        from vllm.platforms import current_platform

        logger.info(
            "Beginning Prefill CUDA graph capture for buckets: %s (batch sizes: %s)",
            self.buckets,
            SUPPORTED_NUM_REQS,
        )

        pool = current_platform.get_global_graph_pool()
        if pool is not None:
            set_graph_pool_id(pool)
        else:
            set_graph_pool_id(current_platform.graph_pool_handle())

        sorted_buckets = sorted(self.buckets, reverse=True)

        for bucket in tqdm(sorted_buckets, desc="Capturing Prefill CUDA graphs (FULL)"):
            buf = self.buffers[bucket]

            # Adjust positions_buf shape for mrope / xdrope if needed
            if getattr(runner, "uses_mrope", False):
                buf.positions_buf = torch.zeros(
                    (3, bucket), dtype=torch.int64, device=runner.device
                )
                buf.positions_buf[:] = torch.arange(bucket, dtype=torch.int64, device=runner.device)
            elif getattr(runner, "uses_xdrope_dim", 0) > 0:
                buf.positions_buf = torch.zeros(
                    (2, bucket), dtype=torch.int64, device=runner.device
                )
                buf.positions_buf[:] = torch.arange(bucket, dtype=torch.int64, device=runner.device)
            else:
                buf.positions_buf = torch.arange(bucket, dtype=torch.int64, device=runner.device)

            buf.input_ids_buf.fill_(buf.pad_token_id)
            buf.slot_mapping_buf.fill_(-1)
            slot_mappings_by_group = {0: buf.slot_mapping_buf}
            # Per-layer mapping for vllm's unified KV-cache update op. Point
            # every attention layer at the static buffer: the graph bakes the
            # pointer and execute_model_prefill() refreshes the contents
            # before each replay. Without this the op silently skips KV
            # writes and replays read an empty KV cache.
            slot_mappings_by_layer: dict[str, torch.Tensor] = {}
            kv_cache_config = getattr(runner, "kv_cache_config", None)
            if kv_cache_config is not None and len(kv_cache_config.kv_cache_groups) > 0:
                for layer_name in kv_cache_config.kv_cache_groups[0].layer_names:
                    slot_mappings_by_layer[layer_name] = buf.slot_mapping_buf

            for num_reqs in SUPPORTED_NUM_REQS:
                if bucket < num_reqs:
                    continue

                try:
                    # Distribute bucket tokens evenly across num_reqs for capture
                    base_len = bucket // num_reqs
                    rem = bucket % num_reqs
                    req_lens = [base_len + (1 if i < rem else 0) for i in range(num_reqs)]

                    # Set up seq_lens and query_start_loc for (bucket, num_reqs)
                    if hasattr(runner, "seq_lens") and hasattr(runner, "optimistic_seq_lens_cpu"):
                        for i, rlen in enumerate(req_lens):
                            runner.optimistic_seq_lens_cpu[i] = rlen
                        runner.optimistic_seq_lens_cpu[num_reqs:].fill_(0)
                        runner.seq_lens.copy_(runner.optimistic_seq_lens_cpu, non_blocking=True)

                    if hasattr(runner, "query_start_loc"):
                        runner.query_start_loc.np[0] = 0
                        curr = 0
                        for i, rlen in enumerate(req_lens):
                            curr += rlen
                            runner.query_start_loc.np[i + 1] = curr
                        runner.query_start_loc.np[num_reqs + 1 :] = 0
                        runner.query_start_loc.copy_to_gpu()

                    # Build attention metadata for (bucket, num_reqs)
                    attn_metadata, _ = runner._build_attention_metadata(
                        num_tokens=bucket,
                        num_tokens_padded=bucket,
                        num_reqs=num_reqs,
                        max_query_len=bucket,
                        ubatch_slices=None,
                        for_cudagraph_capture=True,
                        slot_mappings=slot_mappings_by_group,
                    )

                    # The captured flash-attention call bakes max_seqlen_k from
                    # this metadata scalar; the replay then admits requests
                    # whose KV length (num_computed + scheduled) fits within
                    # 2x the bucket. Requests beyond that bound fall back to
                    # eager (checked in execute_model_prefill).
                    bound = 2 * bucket
                    self._apply_max_seq_len_bound(attn_metadata, bound)
                    buf.max_seq_len_bounds[num_reqs] = bound

                    # Rebind the block table, seq_lens, and query_start_loc
                    # to static per-bucket buffers: a captured graph bakes the
                    # pointer address, so the tensor at that address must stay
                    # stable for the lifetime of the graph. The runner's live
                    # per-batch tensors are overwritten (or reallocated) for
                    # the next batch, so the graph must never bake them.
                    bt_metas = (
                        list(attn_metadata.values())
                        if isinstance(attn_metadata, dict)
                        else [attn_metadata]
                    )
                    for meta in bt_metas:
                        if not hasattr(meta, "block_table"):
                            continue
                        bt_shape = meta.block_table.shape
                        # Pre-allocated at init from the model config; re-size
                        # only if the metadata's width exceeds it (should not
                        # happen — config-derived widths match).
                        if (
                            buf.block_table_buf is None
                            or bt_shape[1] > buf.block_table_buf.shape[1]
                        ):
                            buf.block_table_buf = torch.zeros(
                                (SUPPORTED_NUM_REQS[-1], bt_shape[1]),
                                dtype=meta.block_table.dtype,
                                device=runner.device,
                            )
                        meta.block_table = buf.block_table_buf[: bt_shape[0], : bt_shape[1]]
                        # Rebind seq_lens / query_start_loc to the static
                        # buffers pre-allocated at init: the replay refreshes
                        # them before every execution (see execute_model_prefill).
                        if hasattr(meta, "seq_lens") and meta.seq_lens is not None:
                            meta.seq_lens = buf.seq_lens_buf[: meta.seq_lens.shape[0]]
                        if hasattr(meta, "query_start_loc") and meta.query_start_loc is not None:
                            meta.query_start_loc = buf.query_start_loc_buf[
                                : meta.query_start_loc.shape[0]
                            ]

                    if hasattr(attn_metadata, "is_prefill_graph_exec"):
                        attn_metadata.is_prefill_graph_exec = True

                    # Warmup passes
                    for _ in range(2):
                        with set_forward_context(
                            attn_metadata,
                            runner.vllm_config,
                            num_tokens=bucket,
                            cudagraph_runtime_mode=CUDAGraphMode.NONE,
                            slot_mapping=slot_mappings_by_layer,
                        ):
                            runner.model(
                                input_ids=buf.input_ids_buf,
                                positions=buf.positions_buf,
                            )
                    torch.accelerator.synchronize()

                    # Capture CUDA graph
                    graph = torch.cuda.CUDAGraph()
                    with (
                        torch.cuda.graph(graph, pool=pool, stream=torch.cuda.current_stream()),
                        set_forward_context(
                            attn_metadata,
                            runner.vllm_config,
                            num_tokens=bucket,
                            cudagraph_runtime_mode=CUDAGraphMode.NONE,
                            slot_mapping=slot_mappings_by_layer,
                        ),
                    ):
                        output = runner.model(
                            input_ids=buf.input_ids_buf,
                            positions=buf.positions_buf,
                        )

                    torch.accelerator.synchronize()
                    buf.captured_graphs[num_reqs] = graph
                    buf.output_bufs[num_reqs] = output
                    buf.static_attn_metadatas[num_reqs] = attn_metadata

                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Failed to capture Prefill CUDA graph for bucket %d (num_reqs=%d): %r.",
                        bucket,
                        num_reqs,
                        e,
                    )

        captured_summary = {
            b: list(self.buffers[b].captured_graphs.keys())
            for b in self.buckets
            if self.buffers[b].captured_graphs
        }
        msg = f"Prefill CUDA Graphs capture completed. Captured (bucket -> req_counts): {captured_summary}"
        logger.info(msg)

    def has_captured_graph(self, bucket: int, num_reqs: int = 1) -> bool:
        """Returns True if a Graph has been captured for this bucket and num_reqs."""
        buf = self.buffers.get(bucket)
        return buf is not None and buf.has_captured_graph(num_reqs)

    @staticmethod
    def _apply_max_seq_len_bound(attn_metadata: Any, bound: int) -> None:
        """Bind the per-layer max_seq_len scalar to the replay KV-length bound.

        The captured flash-attention call bakes max_seqlen_k from this scalar;
        the runner only replays batches whose per-request KV lengths stay
        within ``bound`` (2x the bucket). 0 keeps the dummy value.
        """
        if not bound:
            return
        if isinstance(attn_metadata, dict):
            for meta in attn_metadata.values():
                if hasattr(meta, "max_seq_len"):
                    meta.max_seq_len = bound
        elif hasattr(attn_metadata, "max_seq_len"):
            attn_metadata.max_seq_len = bound

    def _max_scheduled_kvlen(self, num_scheduled_tokens: list[int] | dict[Any, int]) -> int | None:
        """Max per-request KV length (num_computed + scheduled) in the batch.

        Resolves each scheduled request ID through the InputBatch row mapping
        (num_scheduled_tokens is req-id-keyed). Returns None when the
        resolution is unavailable (legacy list form or missing arrays).
        """
        if not isinstance(num_scheduled_tokens, dict):
            return None
        input_batch = getattr(self.model_runner, "input_batch", None)
        if input_batch is None:
            return None
        req_id_to_index = getattr(input_batch, "req_id_to_index", None)
        computed = getattr(input_batch, "num_computed_tokens_cpu", None)
        if req_id_to_index is None or computed is None:
            return None
        max_kvlen = 0
        for req_id, count in num_scheduled_tokens.items():
            row = req_id_to_index.get(req_id)
            if row is None:
                return None
            max_kvlen = max(max_kvlen, int(computed[row]) + int(count))
        return max_kvlen

    def execute_model_prefill(
        self,
        bucket: int,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        slot_mapping: torch.Tensor | None,
        num_scheduled_tokens: list[int] | dict[Any, int],
    ) -> torch.Tensor | None:
        """Populates static device buffers, replays captured CUDA Graph for num_reqs, and returns output."""
        token_counts = (
            list(num_scheduled_tokens.values())
            if isinstance(num_scheduled_tokens, dict)
            else list(num_scheduled_tokens)
        )
        num_reqs = len(token_counts)
        total_actual = sum(token_counts)
        buf = self.buffers.get(bucket)
        if buf is None or not buf.has_captured_graph(num_reqs):
            return None

        # Never replay with a missing slot mapping: the captured
        # reshape-and-cache would skip or misdirect the current KV writes.
        if slot_mapping is None:
            logger.warning(
                "GPU prefill graph replay: no current slot mapping; falling back to eager prefill."
            )
            return None

        # The captured flash-attention bound (max_seqlen_k) is baked from
        # 2x the bucket; reject batches whose real per-request KV lengths
        # exceed it (the scalar cannot change at replay).
        bound = buf.max_seq_len_bounds.get(num_reqs)
        max_kvlen = self._max_scheduled_kvlen(num_scheduled_tokens)
        if bound is not None and max_kvlen is not None and max_kvlen > bound:
            logger.debug(
                "GPU prefill graph replay: batch max KV length %d exceeds the "
                "captured bound %d; falling back to eager prefill.",
                max_kvlen,
                bound,
            )
            return None

        if input_ids is None or positions is None:
            logger.warning(
                "GPU prefill graph replay: missing live inputs; falling back to eager prefill."
            )
            return None

        # Refresh the static buffers with plain on-stream D2D copies from the
        # live per-batch tensors. _prepare_inputs has already computed all of
        # them (positions, slot mapping, seq_lens, query start loc) for THIS
        # batch on the current stream, and the next batch's writes are
        # enqueued strictly after these copies on the same stream (a CUDA
        # stream is FIFO), so reading the live tensors here can never observe
        # the next batch's data. No host-side recomputation or allocations.
        if total_actual < bucket:
            buf.input_ids_buf[total_actual:].fill_(buf.pad_token_id)
            buf.positions_buf[..., total_actual:].fill_(0)
            buf.slot_mapping_buf[total_actual:].fill_(-1)
        buf.input_ids_buf[:total_actual].copy_(input_ids[:total_actual], non_blocking=True)
        if positions.ndim == buf.positions_buf.ndim == 1:
            buf.positions_buf[:total_actual].copy_(positions[:total_actual], non_blocking=True)
        else:
            buf.positions_buf[..., :total_actual].copy_(
                positions[..., :total_actual], non_blocking=True
            )
        buf.slot_mapping_buf[:total_actual].copy_(slot_mapping[:total_actual], non_blocking=True)

        # Per-request attention metadata: copy the live GPU tensors
        # _prepare_inputs just computed, in the InputBatch row order.
        qsl_gpu = getattr(self.model_runner, "query_start_loc", None)
        if qsl_gpu is None or not hasattr(qsl_gpu, "gpu"):
            logger.warning(
                "GPU prefill graph replay: live query_start_loc unavailable; "
                "falling back to eager prefill."
            )
            return None
        buf.query_start_loc_buf[: num_reqs + 1].copy_(
            qsl_gpu.gpu[: num_reqs + 1], non_blocking=True
        )
        seq_gpu = getattr(self.model_runner, "seq_lens", None)
        if seq_gpu is None:
            logger.warning(
                "GPU prefill graph replay: live seq_lens unavailable; "
                "falling back to eager prefill."
            )
            return None
        buf.seq_lens_buf[:num_reqs].copy_(seq_gpu[:num_reqs], non_blocking=True)

        # Refresh the static block table (baked pointer) from the live GPU
        # block table, which the runner keeps current for the batch. vLLM
        # pre-allocates the block table at a fixed width (max_model_len /
        # block_size), so the live and capture widths are equal in practice;
        # the explicit column slice keeps the copy correct if that ever
        # changes upstream (copy_ requires matching shapes).
        if buf.block_table_buf is not None:
            try:
                live_bt = self.model_runner.input_batch.block_table[0].block_table.gpu
                if live_bt is None:
                    raise AttributeError("live block table is None")
            except (AttributeError, IndexError, TypeError):
                logger.warning(
                    "GPU prefill graph replay: live block table unavailable; "
                    "falling back to eager prefill."
                )
                return None
            if live_bt.shape[0] < num_reqs or live_bt.shape[1] > buf.block_table_buf.shape[1]:
                logger.warning(
                    "GPU prefill graph replay: block table shape %s exceeds the "
                    "staged buffer; falling back to eager prefill.",
                    tuple(live_bt.shape),
                )
                return None
            num_cols = min(live_bt.shape[1], buf.block_table_buf.shape[1])
            buf.block_table_buf[:num_reqs, :num_cols].copy_(
                live_bt[:num_reqs, :num_cols], non_blocking=True
            )

        # Replay the captured graph for num_reqs.
        graph = buf.captured_graphs[num_reqs]
        graph.replay()

        return buf.output_bufs[num_reqs]
