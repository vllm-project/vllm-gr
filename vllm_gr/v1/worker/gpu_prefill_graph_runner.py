# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA Graph Runner for GPU unified Prefill + 2 Decodes (P2D) execution."""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_gr.utils.npu_adapter import is_cuda_available
from vllm_gr.v1.worker.prefill_graph_common import (
    DEFAULT_DECODE_STEPS,
    ENV_DISABLE_P2D,
    ENV_ENABLE_PREFILL_GRAPH,
    PREFILL_GRAPH_MAX_BUCKET_DEFAULT,
    BasePrefillGraphRunner,
    P2DBufferPool,
    P2DMetadata,
    SIDSlices,
    _get_block_table_cols,
    check_p2d_vocab_size,
    get_kv_capture_bound,
    get_max_context_len,
    get_p2d_beam_widths,
    get_p2d_max_beams,
    get_p2d_step_beam_widths_config,
    get_p2d_supported_num_reqs,
    is_p2d_enabled,
    is_prefill_graph_enabled,
    resolve_p2d_pool_kv_bound,
    resolve_sid_slices_from_runner,
    run_decode_pass_eager,
    run_decode_passes,
    run_p2d_stepwise_tail,
    stage_p2d_block_table,
    step0_select_topk,
)

logger = init_logger(__name__)

GPU_PREFILL_CAPTURE_NUM_REQS = (1,)


def _gpu_decode_forward_ctx(runner: Any, attn_meta: P2DMetadata, num_tokens: int):
    """GPU forward-context manager for the shared decode passes."""
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import set_forward_context

    vllm_config = getattr(runner, "vllm_config", None)
    if vllm_config is None or not hasattr(vllm_config, "parallel_config"):
        from types import SimpleNamespace

        vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=1, is_moe_model=False),
            compilation_config=SimpleNamespace(
                fast_moe_cold_start=False, static_forward_context=[]
            ),
        )
    return set_forward_context(
        attn_meta,
        vllm_config,
        num_tokens=num_tokens,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )


class GPUPrefillGraphRunner(BasePrefillGraphRunner):
    """Manages unified Prefill + 2 Decodes (P2D) CUDA graph capture and replay on GPUs."""

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

        model_config = getattr(model_runner, "model_config", None)
        hf_config = getattr(model_config, "hf_text_config", None) or getattr(
            model_config, "hf_config", None
        )
        self.num_layers = int(getattr(hf_config, "num_hidden_layers", 28))
        self.num_kv_heads = int(getattr(hf_config, "num_key_value_heads", 8))
        self.head_dim = int(getattr(hf_config, "head_dim", 128))
        self.vocab_size = int(getattr(hf_config, "vocab_size", 176256))
        self.dtype = getattr(model_runner, "dtype", torch.bfloat16)
        self.sid_slices: SIDSlices | None = resolve_sid_slices_from_runner(model_runner)

        # 2D CUDA Graphs cache and dynamic eager output buffer
        self.captured_2d_graphs: dict[tuple[int, int], Any] = {}
        # Decoupled single-pass decode graphs: (pass_idx, num_reqs, beam_width).
        self.captured_decode_step_graphs: dict[tuple[int, int, int], Any] = {}
        self.eager_output_buf: torch.Tensor | None = None
        # CPU-only diagnostic for the most recent replay attempt; never reads
        # device state or changes the stream ordering.
        self.last_replay_reason: str | None = None

        # Allocate static contiguous buffer pool on CUDA device lazily. The
        # P2D kill switch skips the multi-GB pool entirely: dispatch is gated
        # by is_p2d_enabled(), so nothing ever touches an unallocated pool.
        self.p2d_pool: P2DBufferPool | None = None
        if is_p2d_enabled() and is_cuda_available() and self.device.type == "cuda":
            self._ensure_p2d_pool()

        # Assign layer_idx to each attention layer in the model
        if hasattr(model_runner, "model"):
            layers = getattr(getattr(model_runner.model, "model", None), "layers", None)
            if layers is not None:
                for idx, layer in enumerate(layers):
                    if hasattr(layer, "self_attn"):
                        layer.self_attn.layer_idx = idx
                        if hasattr(layer.self_attn, "impl"):
                            layer.self_attn.impl.layer_idx = idx
                            impl_head_size = getattr(layer.self_attn.impl, "head_size", None)
                            if impl_head_size is not None:
                                assert self.head_dim == impl_head_size, (
                                    f"head_dim ({self.head_dim}) != impl.head_size "
                                    f"({impl_head_size})"
                                )

        logger.info(
            "GPUPrefillGraphRunner initialized on device %s with buckets %s (layers=%d, kv_heads=%d, head_dim=%d)",
            self.device,
            self.buckets,
            self.num_layers,
            self.num_kv_heads,
            self.head_dim,
        )

    def _ensure_p2d_pool(self) -> P2DBufferPool:
        """Lazily initialize the static contiguous P2D buffer pool on CUDA."""
        backend = getattr(self.model_runner, "constraint_backend", None)
        if self.p2d_pool is not None:
            if getattr(self.p2d_pool, "constraint_backend", None) is backend:
                return self.p2d_pool
            # The runner is constructed before load_model publishes the
            # constraint backend; rebuild so the pool geometry (logits width,
            # constrained execution) matches the server-wide catalog config.
            self.p2d_pool = None
        if self.p2d_pool is None:
            max_ctx = get_max_context_len(self.model_runner)
            # KV extent for the pool and its bucket-less 2D/decode-step
            # graphs: defaults to the model context so every prompt rides the
            # P2D tiers; VLLM_GR_P2D_KV_BOUND overrides it.
            max_b = resolve_p2d_pool_kv_bound(max_ctx)
            max_b = max(self.max_bucket, max_b)
            cols = _get_block_table_cols(self.model_runner) or 2048
            max_blocks = max(2048, cols, (max_b + 15) // 16)
            # Logits-buffer width must match what compute_logits actually
            # produces: with an active constraint backend the LogitsProcessor
            # is bound to the backend's catalog window (None on TP>1 → full
            # vocab, and the legacy SID slice must never be consulted); the
            # legacy vocabulary slice only applies without a backend.
            pool_vocab_size = self.vocab_size
            if backend is not None:
                window = getattr(backend, "vocab_slice", None)
                if window is not None and len(window) == 2:
                    pool_vocab_size = int(window[1])
            else:
                from vllm_gr.ops.sliced_logits import (
                    get_vocab_slice_config,
                    is_vocab_slice_enabled,
                )

                if is_vocab_slice_enabled():
                    _, slice_size, _ = get_vocab_slice_config()
                    if slice_size is not None and slice_size > 0:
                        pool_vocab_size = slice_size
            self.p2d_pool = P2DBufferPool(
                max_batch=max(get_p2d_supported_num_reqs()),
                max_bucket=max_b,
                max_beams=get_p2d_max_beams(
                    int(
                        getattr(
                            getattr(self.model_runner, "model_config", None),
                            "max_logprobs",
                            0,
                        )
                        or 0
                    ),
                    runner=self.model_runner,
                ),
                num_decode_steps=DEFAULT_DECODE_STEPS,
                num_layers=self.num_layers,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                vocab_size=pool_vocab_size,
                dtype=self.dtype,
                device=self.device,
                alloc_prompt_kv=False,
                max_blocks=max_blocks,
                sid_slices=self.sid_slices,
                constraint_backend=backend,
            )
        return self.p2d_pool

    def _validate_constraint_slabs(self, p2d_pool: P2DBufferPool) -> None:
        """Run one validated constrained Top-K into the static slabs.

        Exercises the full public validation ABI (``select_out``) once at
        capture start so the subsequent unchecked launches inside the graph
        bodies are guaranteed to use a geometry the backend already accepted.
        """
        from vllm_gr.constraints.backend import ConstraintMetadata

        backend = getattr(p2d_pool, "constraint_backend", None)
        if backend is None or not getattr(backend, "is_ready", False):
            return
        top_k = min(4, p2d_pool.max_beams)
        rows = 1
        logits = torch.zeros((rows, p2d_pool.vocab_size), dtype=p2d_pool.dtype, device=self.device)
        out_tokens = p2d_pool.cand_tokens_flat[: rows * top_k].view(rows, top_k)
        out_logprobs = p2d_pool.cand_logprobs_flat[: rows * top_k].view(rows, top_k)
        out_mask = p2d_pool.cand_mask_flat[: rows * top_k].view(rows, top_k)
        out_counts = p2d_pool.cand_valid_counts[:rows]
        backend.select_out(
            logits,
            ConstraintMetadata(
                prefix_token_ids=p2d_pool.prefixes_0[:rows],
                temperatures=p2d_pool.temperatures,
                current_step=0,
                top_k=top_k,
            ),
            out_token_ids=out_tokens,
            out_logprobs=out_logprobs,
            out_valid_mask=out_mask,
            out_valid_counts=out_counts,
        )
        logger.info("P2D constrained Top-K slab validation passed (top_k=%d).", top_k)

    @staticmethod
    def _apply_max_seq_len_bound(attn_metadata: Any, bound: int) -> None:
        """Helper to bind max_seq_len on attention metadata objects or dictionaries."""
        if not bound:
            return
        metas = list(attn_metadata.values()) if isinstance(attn_metadata, dict) else [attn_metadata]
        for meta in metas:
            if hasattr(meta, "max_seq_len"):
                meta.max_seq_len = bound

    @staticmethod
    def _kv_capture_bound(bucket: int) -> int:
        """KV bound associated with this bucket's graph.

        The returned value feeds the recorded replay gate
        (``buf.max_seq_len_bounds``) and any captured attention scalar, so the
        two can never disagree. The override only ever widens it: buckets
        whose ``2 * bucket`` already covers their contexts keep the default.
        """
        bound = 2 * bucket
        kv_bound = get_kv_capture_bound()
        if kv_bound is not None and kv_bound > bound:
            bound = kv_bound
        return bound

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

    def capture_pure_prefill_graphs(self, runner: Any, pool: Any = None) -> None:
        """Capture one-request pure-prefill graphs for every bucket.

        The P2D tiers fuse Prefill with two Decodes and therefore can never
        serve Asynchronous-scheduling stage dispatches (which schedule the
        Decode stages separately) or the standard prefill-graph fallback.
        This capture restores the plain prefill graph per bucket, keyed by
        ``num_reqs``, which ``execute_model_prefill`` replays.
        """
        if not is_prefill_graph_enabled() or not is_cuda_available():
            return
        from tqdm import tqdm
        from vllm.config import CUDAGraphMode
        from vllm.distributed.device_communicators.pynccl_allocator import (
            set_graph_pool_id,
        )
        from vllm.forward_context import set_forward_context
        from vllm.platforms import current_platform

        if pool is None:
            gpool = current_platform.get_global_graph_pool()
            if gpool is not None:
                set_graph_pool_id(gpool)
                pool = gpool
            else:
                pool = current_platform.graph_pool_handle()
                set_graph_pool_id(pool)

        logger.info(
            "Beginning pure Prefill CUDA graph capture for buckets: %s (batch sizes: %s)",
            self.buckets,
            GPU_PREFILL_CAPTURE_NUM_REQS,
        )

        for bucket in tqdm(
            sorted(self.buckets, reverse=True), desc="Capturing Prefill CUDA graphs"
        ):
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

            for num_reqs in GPU_PREFILL_CAPTURE_NUM_REQS:
                if bucket < num_reqs or buf.has_captured_graph(num_reqs):
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
                    # this bound. Requests beyond it fall back to eager
                    # (checked in execute_model_prefill).
                    bound = self._kv_capture_bound(bucket)
                    if bound < bucket:
                        logger.warning(
                            "Prefill graph KV bound %d is below bucket %d; replays "
                            "for this bucket can never cover their own query extent.",
                            bound,
                            bucket,
                        )
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
                        if (
                            buf.block_table_buf is None
                            or bt_shape[1] > buf.block_table_buf.shape[1]
                        ):
                            buf.block_table_buf = torch.zeros(
                                (GPU_PREFILL_CAPTURE_NUM_REQS[-1], bt_shape[1]),
                                dtype=meta.block_table.dtype,
                                device=runner.device,
                            )
                        meta.block_table = buf.block_table_buf[: bt_shape[0], : bt_shape[1]]
                        if hasattr(meta, "seq_lens") and meta.seq_lens is not None:
                            static_seq_lens = buf.seq_lens_buf[: meta.seq_lens.shape[0]]
                            static_seq_lens.copy_(meta.seq_lens)
                            meta.seq_lens = static_seq_lens
                        if hasattr(meta, "query_start_loc") and meta.query_start_loc is not None:
                            static_query_start_loc = buf.query_start_loc_buf[
                                : meta.query_start_loc.shape[0]
                            ]
                            static_query_start_loc.copy_(meta.query_start_loc)
                            meta.query_start_loc = static_query_start_loc

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
                        "Failed to capture pure Prefill CUDA graph for bucket %d (num_reqs=%d): %r.",
                        bucket,
                        num_reqs,
                        e,
                    )

        captured_summary = {
            b: [key for key in self.buffers[b].captured_graphs if isinstance(key, int)]
            for b in self.buckets
            if any(isinstance(key, int) for key in self.buffers[b].captured_graphs)
        }
        logger.info("Pure Prefill CUDA Graphs capture completed. Captured: %s", captured_summary)

    def capture_graphs(self, runner: Any) -> None:
        """Captures unified P2D CUDA graphs for all prefill bucket sizes and supported batch/beam sizes."""
        if not is_prefill_graph_enabled():
            logger.info(
                "Prefill CUDA Graph capture is disabled via environment variable (%s=0).",
                ENV_ENABLE_PREFILL_GRAPH,
            )
            return

        if not is_p2d_enabled():
            # The kill switch is read once at import (P2DEnv.disabled): skip
            # the entire P2D capture (monolithic, 2D, decode-step graphs) so
            # disabled deployments pay no graph time or memory.
            logger.info(
                "P2D CUDA graph capture skipped (kill switch %s=1).",
                ENV_DISABLE_P2D,
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
            "Beginning P2D CUDA graph capture for buckets: %s (batch sizes: %s, beam widths: %s)",
            self.buckets,
            get_p2d_supported_num_reqs(),
            get_p2d_beam_widths(),
        )

        pool = current_platform.get_global_graph_pool()
        if pool is not None:
            set_graph_pool_id(pool)
        else:
            pool = current_platform.graph_pool_handle()
            set_graph_pool_id(pool)

        sorted_buckets = sorted(self.buckets, reverse=True)
        p2d_pool = self._ensure_p2d_pool()

        # One validated constrained Top-K launch locks the slab geometry
        # (contiguity, dtypes, shapes) before the capture bodies switch to the
        # unchecked out-variant.
        backend = getattr(p2d_pool, "constraint_backend", None)
        if backend is not None and getattr(backend, "is_ready", False):
            self._validate_constraint_slabs(p2d_pool)

        # Capture-time assert: head_dim == impl.head_size across attention layers
        if hasattr(runner, "model"):
            layers = getattr(getattr(runner.model, "model", None), "layers", None)
            if layers is not None:
                for idx, layer in enumerate(layers):
                    impl = getattr(getattr(layer, "self_attn", None), "impl", None)
                    if impl is not None and hasattr(impl, "head_size"):
                        assert self.head_dim == impl.head_size, (
                            f"Capture-time assert failed: layer {idx} head_dim ({self.head_dim}) "
                            f"!= impl.head_size ({impl.head_size})"
                        )

        for bucket in tqdm(sorted_buckets, desc="Capturing P2D CUDA graphs (FULL)"):
            buf = self.buffers[bucket]
            if buf.output_buf is None:
                hidden_size = (
                    runner.model_config.get_hidden_size()
                    if hasattr(runner.model_config, "get_hidden_size")
                    else getattr(runner.model_config, "hidden_size", 2048)
                )
                buf.output_buf = torch.zeros(
                    (bucket, hidden_size),
                    dtype=runner.model_config.dtype,
                    device=self.device,
                )

            for num_reqs in get_p2d_supported_num_reqs():
                if bucket < num_reqs:
                    continue

                for beam_width in get_p2d_beam_widths():
                    assert beam_width <= p2d_pool.max_beams, (
                        f"Capture-time assert failed: beam_width ({beam_width}) "
                        f"exceeds p2d_pool.max_beams ({p2d_pool.max_beams})"
                    )
                    try:
                        # Set up capture dimensions
                        total_prompt_tokens = bucket
                        total_beam_tokens = num_reqs * beam_width

                        # Set up static prompt seqlens and indices for capture
                        p2d_pool.cu_seqlens_q_0[0] = 0
                        p2d_pool.cu_seqlens_k_prompt[0] = 0
                        p2d_pool.cu_seqlens_q_prefix[0] = 0

                        # Evenly distribute the bucket across requests (the
                        # remainder goes to the first requests), keeping the
                        # capture fully generic for any num_reqs.
                        base_len = bucket // num_reqs
                        rem = bucket % num_reqs
                        cum = 0
                        for r in range(num_reqs):
                            chunk_len = base_len + (1 if r < rem else 0)
                            cum += chunk_len
                            p2d_pool.cu_seqlens_q_0[r + 1] = cum
                            p2d_pool.cu_seqlens_k_prompt[r + 1] = cum
                            p2d_pool.cu_seqlens_q_prefix[r + 1] = (r + 1) * beam_width
                            p2d_pool.logits_indices_0[r] = cum - 1
                            p2d_pool.seqused_k[r] = chunk_len
                            seg_start = cum - chunk_len
                            p2d_pool.positions_0[seg_start:cum].copy_(
                                torch.arange(chunk_len, dtype=torch.int64, device=self.device)
                            )
                            r_start = r * beam_width
                            r_end = r_start + beam_width
                            p2d_pool.positions_1[r_start:r_end].fill_(chunk_len)
                            p2d_pool.positions_2[r_start:r_end].fill_(chunk_len + 1)

                        # Suffix query seqlens
                        p2d_pool.cu_seqlens_q_suffix[: total_beam_tokens + 1].copy_(
                            torch.arange(
                                total_beam_tokens + 1, dtype=torch.int32, device=self.device
                            )
                        )
                        p2d_pool.cu_seqlens_k_suffix_1[: total_beam_tokens + 1].copy_(
                            torch.arange(
                                total_beam_tokens + 1, dtype=torch.int32, device=self.device
                            )
                        )
                        p2d_pool.cu_seqlens_k_suffix_2[: total_beam_tokens + 1].copy_(
                            torch.arange(
                                0,
                                2 * total_beam_tokens + 1,
                                2,
                                dtype=torch.int32,
                                device=self.device,
                            )
                        )

                        p2d_pool.input_ids_0[:total_prompt_tokens].fill_(buf.pad_token_id)
                        p2d_pool.slot_mapping_0[:total_prompt_tokens].copy_(
                            torch.arange(total_prompt_tokens, dtype=torch.int64, device=self.device)
                        )

                        # Build P2D metadata for each pass
                        attn_meta_0 = P2DMetadata(
                            p2d_pool=p2d_pool,
                            p2d_pass=0,
                            bucket=bucket,
                            num_reqs=num_reqs,
                            beam_width=beam_width,
                            total_prompt_tokens=total_prompt_tokens,
                            cu_seqlens_q_0=p2d_pool.cu_seqlens_q_0[: num_reqs + 1],
                            block_table=p2d_pool.block_table[:num_reqs],
                            slot_mapping=p2d_pool.slot_mapping_0[:total_prompt_tokens],
                            seqused_k=p2d_pool.seqused_k[:num_reqs],
                        )

                        attn_meta_1 = P2DMetadata(
                            p2d_pool=p2d_pool,
                            p2d_pass=1,
                            bucket=bucket,
                            num_reqs=num_reqs,
                            beam_width=beam_width,
                            total_prompt_tokens=total_prompt_tokens,
                            cu_seqlens_q_prefix=p2d_pool.cu_seqlens_q_prefix[: num_reqs + 1],
                            cu_seqlens_k_prompt=p2d_pool.cu_seqlens_k_prompt[: num_reqs + 1],
                            cu_seqlens_q_suffix=p2d_pool.cu_seqlens_q_suffix[
                                : total_beam_tokens + 1
                            ],
                            cu_seqlens_k_suffix_1=p2d_pool.cu_seqlens_k_suffix_1[
                                : total_beam_tokens + 1
                            ],
                            block_table=p2d_pool.block_table[:num_reqs],
                            seqused_k=p2d_pool.seqused_k[:num_reqs],
                        )

                        attn_meta_2 = P2DMetadata(
                            p2d_pool=p2d_pool,
                            p2d_pass=2,
                            bucket=bucket,
                            num_reqs=num_reqs,
                            beam_width=beam_width,
                            total_prompt_tokens=total_prompt_tokens,
                            cu_seqlens_q_prefix=p2d_pool.cu_seqlens_q_prefix[: num_reqs + 1],
                            cu_seqlens_k_prompt=p2d_pool.cu_seqlens_k_prompt[: num_reqs + 1],
                            cu_seqlens_q_suffix=p2d_pool.cu_seqlens_q_suffix[
                                : total_beam_tokens + 1
                            ],
                            cu_seqlens_k_suffix_2=p2d_pool.cu_seqlens_k_suffix_2[
                                : total_beam_tokens + 1
                            ],
                            block_table=p2d_pool.block_table[:num_reqs],
                            seqused_k=p2d_pool.seqused_k[:num_reqs],
                        )

                        def run_p2d_passes(
                            _meta0: P2DMetadata = attn_meta_0,
                            _meta1: P2DMetadata = attn_meta_1,
                            _meta2: P2DMetadata = attn_meta_2,
                            _prompt_toks: int = total_prompt_tokens,
                            _beam_toks: int = total_beam_tokens,
                            _reqs: int = num_reqs,
                            _bw: int = beam_width,
                        ) -> None:
                            # -------------------------------------------------------------
                            # Pass 0: Prefill Forward
                            # -------------------------------------------------------------
                            with set_forward_context(
                                _meta0,
                                runner.vllm_config,
                                num_tokens=_prompt_toks,
                                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                            ):
                                hidden_0 = runner.model(
                                    input_ids=p2d_pool.input_ids_0[:_prompt_toks],
                                    positions=p2d_pool.positions_0[:_prompt_toks],
                                )

                            sample_hidden_0 = hidden_0[p2d_pool.logits_indices_0[:_reqs]]
                            logits_0 = runner.model.compute_logits(sample_hidden_0)
                            check_p2d_vocab_size(p2d_pool, logits_0, "graph capture")
                            p2d_pool.logits_0[:_reqs].copy_(logits_0)

                            # GPU Top-W selection for step 0
                            step0_select_topk(
                                logits_0=p2d_pool.logits_0,
                                num_reqs=_reqs,
                                beam_width=_bw,
                                pool=p2d_pool,
                            )

                            # -------------------------------------------------------------
                            # Passes 1 & 2: Decodes 1 & 2
                            # -------------------------------------------------------------
                            run_decode_passes(
                                runner=runner,
                                p2d_pool=p2d_pool,
                                attn_meta_1=_meta1,
                                attn_meta_2=_meta2,
                                total_beam_tokens=_beam_toks,
                                num_reqs=_reqs,
                                beam_width=_bw,
                                num_layers=self.num_layers,
                                num_kv_heads=self.num_kv_heads,
                                head_dim=self.head_dim,
                                forward_ctx_fn=_gpu_decode_forward_ctx,
                            )

                        # Warmup
                        for _ in range(2):
                            run_p2d_passes()
                        torch.accelerator.synchronize()

                        # Capture CUDA graph
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, pool=pool, stream=torch.cuda.current_stream()):
                            run_p2d_passes()

                        torch.accelerator.synchronize()
                        buf.captured_graphs[(num_reqs, beam_width)] = graph
                        logger.info(
                            "Captured P2D CUDA graph: bucket=%d, num_reqs=%d, beam_width=%d",
                            bucket,
                            num_reqs,
                            beam_width,
                        )

                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "Failed to capture P2D CUDA graph for bucket %d (num_reqs=%d, beam_width=%d): %r.",
                            bucket,
                            num_reqs,
                            beam_width,
                            e,
                        )

        # Capture 2D CUDA Graphs (Decodes 1 & 2) for Eager-P + 2D execution
        self.capture_2d_graphs(runner, pool=pool)
        # Capture the decoupled single-pass decode graphs for the stepwise tier.
        self.capture_decode_step_graphs(runner, pool=pool)

        captured_summary = {
            b: list(self.buffers[b].captured_graphs.keys())
            for b in self.buckets
            if self.buffers[b].captured_graphs
        }
        logger.info(
            "P2D CUDA Graphs capture completed. Captured (bucket -> configurations): %s, 2D: %s, "
            "steps: %s",
            captured_summary,
            list(self.captured_2d_graphs.keys()),
            sorted(self.captured_decode_step_graphs.keys()),
        )

    def execute_p2d(
        self,
        bucket: int,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        num_scheduled_tokens: list[int] | dict[Any, int],
        beam_width: int = 128,
        block_table: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
        computed_tokens: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Populates static device buffers, replays captured P2D CUDA Graph, and returns (final_tokens, final_cum_lps)."""
        token_counts = (
            list(num_scheduled_tokens.values())
            if isinstance(num_scheduled_tokens, dict)
            else list(num_scheduled_tokens)
        )
        num_reqs = len(token_counts)
        total_actual = sum(token_counts)
        if input_ids is None or input_ids.shape[0] < total_actual or total_actual > bucket:
            return None
        if num_reqs not in get_p2d_supported_num_reqs():
            return None
        buf = self.buffers.get(bucket)
        if buf is None or not buf.has_captured_graph(num_reqs, beam_width):
            return None

        graph = buf.captured_graphs.get((num_reqs, beam_width))
        if graph is None or not hasattr(graph, "replay"):
            return None

        p2d_pool = self._ensure_p2d_pool()
        assert beam_width <= p2d_pool.max_beams, (
            f"beam_width ({beam_width}) exceeds p2d_pool.max_beams ({p2d_pool.max_beams})"
        )
        if getattr(p2d_pool, "prompt_kv", None) is None and (
            block_table is None or slot_mapping is None
        ):
            return None

        c_tokens = computed_tokens or [0] * num_reqs

        # Refresh static buffers with live prompt inputs (generic over any
        # number of requests; the tail absorbs the bucket padding).
        p2d_pool.cu_seqlens_q_0[0] = 0
        p2d_pool.cu_seqlens_k_prompt[0] = 0
        p2d_pool.cu_seqlens_q_prefix[0] = 0
        cum = 0
        for r, sched in enumerate(token_counts):
            c0 = c_tokens[r] if r < len(c_tokens) else 0
            T_r = c0 + sched
            seg_start = cum
            cum += sched
            p2d_pool.input_ids_0[seg_start:cum].copy_(input_ids[seg_start:cum], non_blocking=True)
            p2d_pool.positions_0[seg_start:cum].copy_(positions[seg_start:cum], non_blocking=True)
            p2d_pool.logits_indices_0[r] = cum - 1
            p2d_pool.cu_seqlens_q_0[r + 1] = cum
            p2d_pool.cu_seqlens_k_prompt[r + 1] = cum
            p2d_pool.cu_seqlens_q_prefix[r + 1] = (r + 1) * beam_width
            p2d_pool.seqused_k[r] = T_r
            r_start = r * beam_width
            r_end = r_start + beam_width
            p2d_pool.positions_1[r_start:r_end].fill_(T_r)
            p2d_pool.positions_2[r_start:r_end].fill_(T_r + 1)
        if cum < bucket:
            p2d_pool.input_ids_0[cum:bucket].fill_(buf.pad_token_id)
            p2d_pool.positions_0[cum:bucket].fill_(0)

        # Stage live block table and slot mapping into static pool
        if block_table is not None:
            stage_p2d_block_table(p2d_pool.block_table, block_table, num_reqs)
        if slot_mapping is not None:
            num_slots = min(total_actual, slot_mapping.shape[0])
            p2d_pool.slot_mapping_0[:num_slots].copy_(slot_mapping[:num_slots], non_blocking=True)
            if num_slots < bucket:
                p2d_pool.slot_mapping_0[num_slots:bucket].fill_(-1)

        # Replay the captured CUDA graph
        graph.replay()
        logger.debug(
            "P2D execute_p2d replayed graph successfully, tokens shape=%s",
            p2d_pool.final_tokens[:num_reqs, :beam_width].shape,
        )

        return (
            p2d_pool.final_tokens[:num_reqs, :beam_width],
            p2d_pool.final_cum_lps[:num_reqs, :beam_width],
            p2d_pool.valid_beam_counts[:num_reqs],
            p2d_pool.step_scores[:num_reqs, :beam_width],
        )

    def has_2d_graph(self, num_reqs: int = 1, beam_width: int = 128) -> bool:
        """Returns True if a 2D Graph has been captured for this configuration."""
        if not is_p2d_enabled():
            return False
        return (num_reqs, beam_width) in self.captured_2d_graphs

    def capture_2d_graphs(self, runner: Any, pool: Any = None) -> None:
        """Captures standalone 2D (Decodes 1 & 2) CUDA Graphs for Eager-P + 2D execution."""
        if not is_p2d_enabled():
            return
        from vllm.distributed.device_communicators.pynccl_allocator import (
            set_graph_pool_id,
        )
        from vllm.platforms import current_platform

        p2d_pool = self._ensure_p2d_pool()
        max_b = getattr(p2d_pool, "max_bucket", PREFILL_GRAPH_MAX_BUCKET_DEFAULT)

        if pool is None:
            gpool = current_platform.get_global_graph_pool()
            if gpool is not None:
                set_graph_pool_id(gpool)
                pool = gpool
            else:
                pool = current_platform.graph_pool_handle()
                set_graph_pool_id(pool)

        for num_reqs in get_p2d_supported_num_reqs():
            for beam_width in get_p2d_beam_widths():
                if beam_width > p2d_pool.max_beams:
                    continue
                total_beam_tokens = num_reqs * beam_width
                try:
                    p2d_pool.cu_seqlens_q_prefix[0] = 0
                    p2d_pool.cu_seqlens_k_prompt[0] = 0

                    # Evenly distributed capture-time prompt lengths (generic
                    # over any num_reqs); replays rebind the live lengths.
                    base_len = max_b // num_reqs
                    rem = max_b % num_reqs
                    cum = 0
                    for r in range(num_reqs):
                        chunk_len = base_len + (1 if r < rem else 0)
                        cum += chunk_len
                        p2d_pool.cu_seqlens_q_prefix[r + 1] = (r + 1) * beam_width
                        p2d_pool.cu_seqlens_k_prompt[r + 1] = cum
                        p2d_pool.seqused_k[r] = chunk_len
                        r_start = r * beam_width
                        r_end = r_start + beam_width
                        p2d_pool.positions_1[r_start:r_end].fill_(chunk_len)
                        p2d_pool.positions_2[r_start:r_end].fill_(chunk_len + 1)

                    p2d_pool.cu_seqlens_q_suffix[: total_beam_tokens + 1].copy_(
                        torch.arange(total_beam_tokens + 1, dtype=torch.int32, device=self.device)
                    )
                    p2d_pool.cu_seqlens_k_suffix_1[: total_beam_tokens + 1].copy_(
                        torch.arange(total_beam_tokens + 1, dtype=torch.int32, device=self.device)
                    )
                    p2d_pool.cu_seqlens_k_suffix_2[: total_beam_tokens + 1].copy_(
                        torch.arange(
                            0,
                            2 * total_beam_tokens + 1,
                            2,
                            dtype=torch.int32,
                            device=self.device,
                        )
                    )

                    attn_meta_1 = P2DMetadata(
                        p2d_pool=p2d_pool,
                        p2d_pass=1,
                        bucket=max_b,
                        num_reqs=num_reqs,
                        beam_width=beam_width,
                        total_prompt_tokens=max_b,
                        cu_seqlens_q_prefix=p2d_pool.cu_seqlens_q_prefix[: num_reqs + 1],
                        cu_seqlens_k_prompt=p2d_pool.cu_seqlens_k_prompt[: num_reqs + 1],
                        cu_seqlens_q_suffix=p2d_pool.cu_seqlens_q_suffix[: total_beam_tokens + 1],
                        cu_seqlens_k_suffix_1=p2d_pool.cu_seqlens_k_suffix_1[
                            : total_beam_tokens + 1
                        ],
                        block_table=p2d_pool.block_table[:num_reqs],
                        seqused_k=p2d_pool.seqused_k[:num_reqs],
                    )

                    attn_meta_2 = P2DMetadata(
                        p2d_pool=p2d_pool,
                        p2d_pass=2,
                        bucket=max_b,
                        num_reqs=num_reqs,
                        beam_width=beam_width,
                        total_prompt_tokens=max_b,
                        cu_seqlens_q_prefix=p2d_pool.cu_seqlens_q_prefix[: num_reqs + 1],
                        cu_seqlens_k_prompt=p2d_pool.cu_seqlens_k_prompt[: num_reqs + 1],
                        cu_seqlens_q_suffix=p2d_pool.cu_seqlens_q_suffix[: total_beam_tokens + 1],
                        cu_seqlens_k_suffix_2=p2d_pool.cu_seqlens_k_suffix_2[
                            : total_beam_tokens + 1
                        ],
                        block_table=p2d_pool.block_table[:num_reqs],
                        seqused_k=p2d_pool.seqused_k[:num_reqs],
                    )

                    def run_2d_passes(
                        _m1: P2DMetadata = attn_meta_1,
                        _m2: P2DMetadata = attn_meta_2,
                        _tokens: int = total_beam_tokens,
                        _reqs: int = num_reqs,
                        _width: int = beam_width,
                    ) -> None:
                        run_decode_passes(
                            runner=runner,
                            p2d_pool=p2d_pool,
                            attn_meta_1=_m1,
                            attn_meta_2=_m2,
                            total_beam_tokens=_tokens,
                            num_reqs=_reqs,
                            beam_width=_width,
                            num_layers=self.num_layers,
                            num_kv_heads=self.num_kv_heads,
                            head_dim=self.head_dim,
                            forward_ctx_fn=_gpu_decode_forward_ctx,
                        )

                    for _ in range(2):
                        run_2d_passes()
                    torch.accelerator.synchronize()

                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=pool, stream=torch.cuda.current_stream()):
                        run_2d_passes()

                    torch.accelerator.synchronize()
                    self.captured_2d_graphs[(num_reqs, beam_width)] = graph
                    logger.info(
                        "Captured 2D CUDA graph: num_reqs=%d, beam_width=%d",
                        num_reqs,
                        beam_width,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Failed to capture 2D CUDA graph (num_reqs=%d, beam_width=%d): %r",
                        num_reqs,
                        beam_width,
                        e,
                    )

    def has_decode_step_graph(self, pass_idx: int, num_reqs: int, beam_width: int) -> bool:
        """True when a single-pass decode graph is captured for this configuration."""
        if not is_p2d_enabled():
            return False
        return (pass_idx, num_reqs, beam_width) in self.captured_decode_step_graphs

    def capture_decode_step_graphs(self, runner: Any, pool: Any = None) -> None:
        """Captures decoupled single-pass decode graphs (forward + LM head).

        Keyed by (pass_idx, num_reqs, beam_width): pass 1 graphs are captured
        for widths in P_BW, pass 2 graphs for widths in D0_BW (W2 shapes no
        graph — the step-2 pruning runs eagerly on-stream).
        """
        if not is_p2d_enabled():
            return
        from vllm.distributed.device_communicators.pynccl_allocator import (
            set_graph_pool_id,
        )
        from vllm.platforms import current_platform

        p2d_pool = self._ensure_p2d_pool()
        max_b = getattr(p2d_pool, "max_bucket", PREFILL_GRAPH_MAX_BUCKET_DEFAULT)

        if pool is None:
            gpool = current_platform.get_global_graph_pool()
            if gpool is not None:
                set_graph_pool_id(gpool)
                pool = gpool
            else:
                pool = current_platform.graph_pool_handle()
                set_graph_pool_id(pool)

        p_bw, d0_bw, _ = get_p2d_step_beam_widths_config()
        for pass_idx, widths in ((1, p_bw), (2, d0_bw)):
            for num_reqs in get_p2d_supported_num_reqs():
                for beam_width in widths:
                    if beam_width > p2d_pool.max_beams:
                        continue
                    total_beam_tokens = num_reqs * beam_width
                    try:
                        p2d_pool.cu_seqlens_q_prefix[0] = 0
                        p2d_pool.cu_seqlens_k_prompt[0] = 0
                        base_len = max_b // num_reqs
                        rem = max_b % num_reqs
                        cum = 0
                        for r in range(num_reqs):
                            chunk_len = base_len + (1 if r < rem else 0)
                            cum += chunk_len
                            p2d_pool.cu_seqlens_q_prefix[r + 1] = (r + 1) * beam_width
                            p2d_pool.cu_seqlens_k_prompt[r + 1] = cum
                            p2d_pool.seqused_k[r] = chunk_len
                            r_start = r * beam_width
                            r_end = r_start + beam_width
                            p2d_pool.positions_1[r_start:r_end].fill_(chunk_len)
                            p2d_pool.positions_2[r_start:r_end].fill_(chunk_len + 1)

                        suffix_kwargs: dict[str, Any]
                        if pass_idx == 1:
                            p2d_pool.cu_seqlens_q_suffix[: total_beam_tokens + 1].copy_(
                                torch.arange(
                                    total_beam_tokens + 1, dtype=torch.int32, device=self.device
                                )
                            )
                            p2d_pool.cu_seqlens_k_suffix_1[: total_beam_tokens + 1].copy_(
                                torch.arange(
                                    total_beam_tokens + 1, dtype=torch.int32, device=self.device
                                )
                            )
                            suffix_kwargs = {
                                "cu_seqlens_k_suffix_1": p2d_pool.cu_seqlens_k_suffix_1[
                                    : total_beam_tokens + 1
                                ]
                            }
                        else:
                            p2d_pool.cu_seqlens_k_suffix_2[: total_beam_tokens + 1].copy_(
                                torch.arange(
                                    0,
                                    2 * total_beam_tokens + 1,
                                    2,
                                    dtype=torch.int32,
                                    device=self.device,
                                )
                            )
                            suffix_kwargs = {
                                "cu_seqlens_k_suffix_2": p2d_pool.cu_seqlens_k_suffix_2[
                                    : total_beam_tokens + 1
                                ]
                            }

                        attn_meta = P2DMetadata(
                            p2d_pool=p2d_pool,
                            p2d_pass=pass_idx,
                            bucket=max_b,
                            num_reqs=num_reqs,
                            beam_width=beam_width,
                            total_prompt_tokens=max_b,
                            cu_seqlens_q_prefix=p2d_pool.cu_seqlens_q_prefix[: num_reqs + 1],
                            cu_seqlens_k_prompt=p2d_pool.cu_seqlens_k_prompt[: num_reqs + 1],
                            cu_seqlens_q_suffix=p2d_pool.cu_seqlens_q_suffix[
                                : total_beam_tokens + 1
                            ],
                            block_table=p2d_pool.block_table[:num_reqs],
                            seqused_k=p2d_pool.seqused_k[:num_reqs],
                            **suffix_kwargs,
                        )

                        def run_step(
                            _meta: P2DMetadata = attn_meta,
                            _pass: int = pass_idx,
                            _tokens: int = total_beam_tokens,
                        ) -> None:
                            run_decode_pass_eager(
                                runner,
                                p2d_pool,
                                _meta,
                                _pass,
                                _tokens,
                                _gpu_decode_forward_ctx,
                            )

                        for _ in range(2):
                            run_step()
                        torch.accelerator.synchronize()

                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, pool=pool, stream=torch.cuda.current_stream()):
                            run_step()
                        torch.accelerator.synchronize()

                        self.captured_decode_step_graphs[(pass_idx, num_reqs, beam_width)] = graph
                        logger.info(
                            "Captured P2D decode-step CUDA graph: pass=%d, num_reqs=%d, "
                            "beam_width=%d",
                            pass_idx,
                            num_reqs,
                            beam_width,
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "Failed to capture P2D decode-step CUDA graph "
                            "(pass=%d, num_reqs=%d, beam_width=%d): %r",
                            pass_idx,
                            num_reqs,
                            beam_width,
                            e,
                        )

    def _stage_p2d_step_inputs(
        self,
        p2d_pool: P2DBufferPool,
        token_counts: list[int],
        computed_tokens: list[int] | None,
        w0: int,
        w1: int,
        block_table: torch.Tensor | None,
    ) -> None:
        """Stages per-request T0 seqlens, decode positions, and the block table
        for the stepwise/eager P2D tiers."""
        num_reqs = len(token_counts)
        c_tokens = computed_tokens or [0] * num_reqs
        for r, sched in enumerate(token_counts):
            T_r = (c_tokens[r] if r < len(c_tokens) else 0) + sched
            p2d_pool.seqused_k[r] = T_r
            p2d_pool.positions_1[r * w0 : (r + 1) * w0].fill_(T_r)
            p2d_pool.positions_2[r * w1 : (r + 1) * w1].fill_(T_r + 1)
        if block_table is not None:
            stage_p2d_block_table(p2d_pool.block_table, block_table, num_reqs)

    def execute_stepwise_p2d(
        self,
        runner: Any,
        hidden: torch.Tensor,
        sample_indices: list[int],
        num_scheduled_tokens: list[int] | dict[Any, int],
        computed_tokens: list[int] | None,
        step_bws: Any,
        block_table: torch.Tensor | None,
        slot_mapping: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Tier 2: prefill (already executed) + decoupled decode-step graphs,
        chained on-stream with the pruning primitives (zero host sync)."""
        token_counts = (
            list(num_scheduled_tokens.values())
            if isinstance(num_scheduled_tokens, dict)
            else list(num_scheduled_tokens)
        )
        num_reqs = len(token_counts)
        total_actual = sum(token_counts)
        if num_reqs not in get_p2d_supported_num_reqs() or total_actual <= 0:
            return None
        w0, w1, w2 = step_bws
        graph_1 = self.captured_decode_step_graphs.get((1, num_reqs, w0))
        graph_2 = self.captured_decode_step_graphs.get((2, num_reqs, w1))
        if graph_1 is None or graph_2 is None:
            return None
        p2d_pool = self._ensure_p2d_pool()
        if getattr(p2d_pool, "prompt_kv", None) is None and (
            block_table is None or slot_mapping is None
        ):
            return None

        self._stage_p2d_step_inputs(p2d_pool, token_counts, computed_tokens, w0, w1, block_table)

        run_p2d_stepwise_tail(
            runner=runner,
            p2d_pool=p2d_pool,
            hidden=hidden,
            sample_indices=sample_indices,
            attn_meta_1=None,
            attn_meta_2=None,
            num_reqs=num_reqs,
            w0=w0,
            w1=w1,
            w2=w2,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            forward_ctx_fn=_gpu_decode_forward_ctx,
            decode1_fn=lambda _tokens: graph_1.replay(),
            decode2_fn=lambda _tokens: graph_2.replay(),
        )

        return (
            p2d_pool.final_tokens[:num_reqs, :w2],
            p2d_pool.final_cum_lps[:num_reqs, :w2],
            p2d_pool.valid_beam_counts[:num_reqs],
            p2d_pool.step_scores[:num_reqs, :w2],
        )

    def execute_eager_p2d_stepped(
        self,
        runner: Any,
        hidden: torch.Tensor,
        sample_indices: list[int],
        num_scheduled_tokens: list[int] | dict[Any, int],
        computed_tokens: list[int] | None,
        step_bws: Any,
        block_table: torch.Tensor | None,
        slot_mapping: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Tier 3: on-device eager P2D — the same stepwise chain with eager
        single-pass forwards instead of graph replays."""
        token_counts = (
            list(num_scheduled_tokens.values())
            if isinstance(num_scheduled_tokens, dict)
            else list(num_scheduled_tokens)
        )
        num_reqs = len(token_counts)
        total_actual = sum(token_counts)
        if num_reqs not in get_p2d_supported_num_reqs() or total_actual <= 0:
            return None
        w0, w1, w2 = step_bws
        p2d_pool = self._ensure_p2d_pool()
        if getattr(p2d_pool, "prompt_kv", None) is None and (
            block_table is None or slot_mapping is None
        ):
            return None

        self._stage_p2d_step_inputs(p2d_pool, token_counts, computed_tokens, w0, w1, block_table)

        # Per-pass metadata for the eager attention dispatch.
        p2d_pool.cu_seqlens_q_prefix[0] = 0
        for r in range(num_reqs):
            p2d_pool.cu_seqlens_q_prefix[r + 1] = (r + 1) * w0
        p2d_pool.cu_seqlens_q_suffix[: num_reqs * w0 + 1].copy_(
            torch.arange(num_reqs * w0 + 1, dtype=torch.int32, device=self.device)
        )
        p2d_pool.cu_seqlens_k_suffix_1[: num_reqs * w0 + 1].copy_(
            torch.arange(num_reqs * w0 + 1, dtype=torch.int32, device=self.device)
        )
        p2d_pool.cu_seqlens_k_suffix_2[: num_reqs * w1 + 1].copy_(
            torch.arange(0, 2 * num_reqs * w1 + 1, 2, dtype=torch.int32, device=self.device)
        )
        attn_meta_1 = P2DMetadata(
            p2d_pool=p2d_pool,
            p2d_pass=1,
            bucket=total_actual,
            num_reqs=num_reqs,
            beam_width=w0,
            total_prompt_tokens=total_actual,
            cu_seqlens_q_prefix=p2d_pool.cu_seqlens_q_prefix[: num_reqs + 1],
            cu_seqlens_k_prompt=p2d_pool.seqused_k[:num_reqs],
            cu_seqlens_q_suffix=p2d_pool.cu_seqlens_q_suffix[: num_reqs * w0 + 1],
            cu_seqlens_k_suffix_1=p2d_pool.cu_seqlens_k_suffix_1[: num_reqs * w0 + 1],
            block_table=p2d_pool.block_table[:num_reqs],
            seqused_k=p2d_pool.seqused_k[:num_reqs],
        )
        p2d_pool.cu_seqlens_q_prefix[0] = 0
        for r in range(num_reqs):
            p2d_pool.cu_seqlens_q_prefix[r + 1] = (r + 1) * w1
        attn_meta_2 = P2DMetadata(
            p2d_pool=p2d_pool,
            p2d_pass=2,
            bucket=total_actual,
            num_reqs=num_reqs,
            beam_width=w1,
            total_prompt_tokens=total_actual,
            cu_seqlens_q_prefix=p2d_pool.cu_seqlens_q_prefix[: num_reqs + 1],
            cu_seqlens_k_prompt=p2d_pool.seqused_k[:num_reqs],
            cu_seqlens_q_suffix=p2d_pool.cu_seqlens_q_suffix[: num_reqs * w1 + 1],
            cu_seqlens_k_suffix_2=p2d_pool.cu_seqlens_k_suffix_2[: num_reqs * w1 + 1],
            block_table=p2d_pool.block_table[:num_reqs],
            seqused_k=p2d_pool.seqused_k[:num_reqs],
        )

        run_p2d_stepwise_tail(
            runner=runner,
            p2d_pool=p2d_pool,
            hidden=hidden,
            sample_indices=sample_indices,
            attn_meta_1=attn_meta_1,
            attn_meta_2=attn_meta_2,
            num_reqs=num_reqs,
            w0=w0,
            w1=w1,
            w2=w2,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            forward_ctx_fn=_gpu_decode_forward_ctx,
            decode1_fn=lambda _tokens: run_decode_pass_eager(
                runner, p2d_pool, attn_meta_1, 1, num_reqs * w0, _gpu_decode_forward_ctx
            ),
            decode2_fn=lambda _tokens: run_decode_pass_eager(
                runner, p2d_pool, attn_meta_2, 2, num_reqs * w1, _gpu_decode_forward_ctx
            ),
        )

        return (
            p2d_pool.final_tokens[:num_reqs, :w2],
            p2d_pool.final_cum_lps[:num_reqs, :w2],
            p2d_pool.valid_beam_counts[:num_reqs],
            p2d_pool.step_scores[:num_reqs, :w2],
        )

    def execute_eager_p2d(
        self,
        runner: Any,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        num_scheduled_tokens: list[int] | dict[Any, int],
        beam_width: int = 128,
        block_table: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
        computed_tokens: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Executes Eager Prefill followed by captured 2D CUDA Graph replay."""
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import set_forward_context

        token_counts = (
            list(num_scheduled_tokens.values())
            if isinstance(num_scheduled_tokens, dict)
            else list(num_scheduled_tokens)
        )
        num_reqs = len(token_counts)
        total_actual = sum(token_counts)
        if input_ids is None or positions is None or total_actual <= 0:
            return None

        p2d_pool = self._ensure_p2d_pool()
        # The captured 2D decode graphs bake attention extents bounded by the
        # pool's capture bucket; longer KV lengths are handled by the engine
        # path (the dispatch skips the P2D tiers beyond this bound).
        max_b = getattr(p2d_pool, "max_bucket", self.max_bucket)
        if input_ids.shape[0] < total_actual or total_actual > max_b:
            return None
        if num_reqs not in get_p2d_supported_num_reqs():
            return None

        graph_2d = self.captured_2d_graphs.get((num_reqs, beam_width))
        if graph_2d is None or not hasattr(graph_2d, "replay"):
            return None

        p2d_pool = self._ensure_p2d_pool()
        assert beam_width <= p2d_pool.max_beams, (
            f"beam_width ({beam_width}) exceeds p2d_pool.max_beams ({p2d_pool.max_beams})"
        )
        if getattr(p2d_pool, "prompt_kv", None) is None and (
            block_table is None or slot_mapping is None
        ):
            return None

        c_tokens = computed_tokens or [0] * num_reqs
        # Refresh dynamic seqlens and positions (generic over any num_reqs).
        p2d_pool.cu_seqlens_q_0[0] = 0
        p2d_pool.cu_seqlens_k_prompt[0] = 0
        p2d_pool.cu_seqlens_q_prefix[0] = 0
        sample_indices: list[int] = []
        cum = 0
        cum_T = 0
        for r, sched in enumerate(token_counts):
            c0 = c_tokens[r] if r < len(c_tokens) else 0
            T_r = c0 + sched
            cum += sched
            cum_T += T_r
            p2d_pool.cu_seqlens_q_0[r + 1] = cum
            p2d_pool.cu_seqlens_k_prompt[r + 1] = cum_T
            p2d_pool.cu_seqlens_q_prefix[r + 1] = (r + 1) * beam_width
            r_start = r * beam_width
            r_end = r_start + beam_width
            p2d_pool.positions_1[r_start:r_end].fill_(T_r)
            p2d_pool.positions_2[r_start:r_end].fill_(T_r + 1)
            sample_indices.append(cum - 1)
            if hasattr(p2d_pool, "seqused_k"):
                p2d_pool.seqused_k[r] = T_r

        # Step 1: Eager Prefill Forward (writes KV directly to pool.prompt_kv)
        # Stage live block table for 2D decodes
        if block_table is not None and hasattr(p2d_pool, "block_table"):
            stage_p2d_block_table(p2d_pool.block_table, block_table, num_reqs)

        # Step 1: Eager Prefill Forward
        p2d_block_table = (
            p2d_pool.block_table[:num_reqs] if hasattr(p2d_pool, "block_table") else None
        )
        p2d_seqused_k = p2d_pool.seqused_k[:num_reqs] if hasattr(p2d_pool, "seqused_k") else None
        attn_meta_0 = P2DMetadata(
            p2d_pool=p2d_pool,
            p2d_pass=0,
            bucket=total_actual,
            num_reqs=num_reqs,
            beam_width=beam_width,
            total_prompt_tokens=total_actual,
            cu_seqlens_q_0=p2d_pool.cu_seqlens_q_0[: num_reqs + 1],
            slot_mapping=slot_mapping,
            block_table=p2d_block_table,
            seqused_k=p2d_seqused_k,
        )
        vllm_config = getattr(runner, "vllm_config", None)
        if vllm_config is None or not hasattr(vllm_config, "parallel_config"):
            from types import SimpleNamespace

            vllm_config = SimpleNamespace(
                parallel_config=SimpleNamespace(data_parallel_size=1, is_moe_model=False),
                compilation_config=SimpleNamespace(
                    fast_moe_cold_start=False, static_forward_context=[]
                ),
            )
        with set_forward_context(
            attn_meta_0,
            vllm_config,
            num_tokens=total_actual,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        ):
            hidden_0 = runner.model(
                input_ids=input_ids[:total_actual],
                positions=positions[:total_actual],
            )

        # Step 2: GPU Step 0 Top-W Sampler
        sample_hidden_0 = hidden_0[sample_indices]
        logits_0 = runner.model.compute_logits(sample_hidden_0)
        check_p2d_vocab_size(p2d_pool, logits_0, "eager")
        p2d_pool.logits_0[:num_reqs].copy_(logits_0, non_blocking=True)
        step0_select_topk(
            logits_0=p2d_pool.logits_0,
            num_reqs=num_reqs,
            beam_width=beam_width,
            pool=p2d_pool,
        )

        # Step 3: Replay 2D CUDA Graph (Decodes 1 & 2)
        graph_2d.replay()
        logger.debug(
            "P2D execute_eager_p2d replayed 2D graph successfully (tokens=%d, reqs=%d, bw=%d)",
            total_actual,
            num_reqs,
            beam_width,
        )

        return (
            p2d_pool.final_tokens[:num_reqs, :beam_width],
            p2d_pool.final_cum_lps[:num_reqs, :beam_width],
            p2d_pool.valid_beam_counts[:num_reqs],
            p2d_pool.step_scores[:num_reqs, :beam_width],
        )

    def get_output_buf(
        self,
        total_tokens: int,
        model_config: Any,
    ) -> torch.Tensor:
        """Returns or allocates a device tensor of shape (N >= total_tokens, hidden_size)."""
        hidden_size = (
            model_config.get_hidden_size()
            if hasattr(model_config, "get_hidden_size")
            else getattr(model_config, "hidden_size", 2048)
        )
        needed_tokens = max(self.max_bucket, total_tokens)
        if (
            self.eager_output_buf is None
            or self.eager_output_buf.shape[0] < needed_tokens
            or self.eager_output_buf.shape[1] != hidden_size
        ):
            self.eager_output_buf = torch.zeros(
                (needed_tokens, hidden_size),
                dtype=model_config.dtype,
                device=self.device,
            )
        return self.eager_output_buf

    def execute_model_prefill(
        self,
        bucket: int,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        slot_mapping: torch.Tensor | None,
        num_scheduled_tokens: list[int] | dict[Any, int],
    ) -> torch.Tensor | None:
        """Populate static buffers, replay the pure-prefill graph, and return output."""
        token_counts = (
            list(num_scheduled_tokens.values())
            if isinstance(num_scheduled_tokens, dict)
            else list(num_scheduled_tokens)
        )
        num_reqs = len(token_counts)
        total_actual = sum(token_counts)
        self.last_replay_reason = "graph_not_captured"
        buf = self.buffers.get(bucket)
        if buf is None or not buf.has_captured_graph(num_reqs):
            return None

        # Never replay with a missing slot mapping: the captured
        # reshape-and-cache would skip or misdirect the current KV writes.
        if slot_mapping is None:
            self.last_replay_reason = "missing_slot_mapping"
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
            self.last_replay_reason = "kv_length_exceeds_bound"
            logger.debug(
                "GPU prefill graph replay: batch max KV length %d exceeds the "
                "captured bound %d; falling back to eager prefill.",
                max_kvlen,
                bound,
            )
            return None

        if input_ids is None or positions is None:
            self.last_replay_reason = "missing_inputs"
            logger.warning(
                "GPU prefill graph replay: missing live inputs; falling back to eager prefill."
            )
            return None
        if input_ids.shape[0] < total_actual:
            self.last_replay_reason = "input_too_small"
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
            self.last_replay_reason = "missing_query_start_loc"
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
            self.last_replay_reason = "missing_seq_lens"
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
                self.last_replay_reason = "missing_block_table"
                logger.warning(
                    "GPU prefill graph replay: live block table unavailable; "
                    "falling back to eager prefill."
                )
                return None
            if live_bt.shape[0] < num_reqs or live_bt.shape[1] > buf.block_table_buf.shape[1]:
                self.last_replay_reason = "block_table_shape"
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
        self.last_replay_reason = "replay"

        return buf.output_bufs[num_reqs]
