# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend NPU Graph Runner for NPU prefill execution."""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_gr.utils.npu_adapter import is_npu_available
from vllm_gr.v1.worker.prefill_graph_common import (
    DEFAULT_DECODE_STEPS,
    ENV_DISABLE_P2D,
    ENV_ENABLE_PREFILL_GRAPH,
    SUPPORTED_NUM_REQS,
    BasePrefillGraphRunner,
    P2DBufferPool,
    P2DMetadata,
    PrefillBucketBuffers,
    SIDSlices,
    check_p2d_vocab_size,
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

# The graph task-group update may rewrite the FIA host lists to a DIFFERENT
# request count and re-bind the block table, so a single captured graph per
# bucket covers every supported num_reqs (1..8). The dummy capture uses the
# max request count so the FIA workspace is sized conservatively.
CAPTURE_DUMMY_NUM_REQS = SUPPORTED_NUM_REQS[-1]


def build_update_lists(token_counts: list[int], bucket: int) -> tuple[list[int], list[int]]:
    """Fresh per-request host seq-lens lists for a task-group update.

    ``qlen_list`` is the per-request query-token count (cumsum without the
    leading zero, the FIA-v2 convention); ``kvlen_list`` is the per-request
    KV length. The last request absorbs the padding so both lists always sum
    to the bucket (GPU-parity semantics: padded positions are causal-sandboxed
    behind the real tokens of the last request).

    Fallback when the current batch's metadata lists are unavailable: assumes
    no cached tokens, so KV length equals the scheduled chunk length.
    """
    qlen_list = []
    curr = 0
    for cnt in token_counts:
        curr += cnt
        qlen_list.append(curr)
    kvlen_list = list(token_counts)
    pad = bucket - sum(token_counts)
    if pad > 0:
        qlen_list[-1] += pad
        kvlen_list[-1] += pad
    return qlen_list, kvlen_list


def build_update_lists_from_metadata(
    meta: Any, token_counts: list[int], bucket: int
) -> tuple[list[int], list[int]]:
    """Cached-aware host seq-lens lists copied from the current batch metadata.

    ``seq_lens_cpu`` carries the real per-request KV lengths (including
    already-computed tokens for cached/chunked prefill); ``query_start_loc_cpu``
    is the chunk-lens cumsum with the leading zero dropped. The last request
    absorbs the padding. Falls back to ``build_update_lists`` when the
    metadata lists are absent.
    """
    qsl_cpu = getattr(meta, "query_start_loc_cpu", None)
    sl_cpu = getattr(meta, "seq_lens_cpu", None)
    num_reqs = int(getattr(meta, "num_reqs", 0))
    if (
        qsl_cpu is not None
        and sl_cpu is not None
        and 1 <= num_reqs <= len(sl_cpu)
        and num_reqs + 1 <= len(qsl_cpu)
    ):
        qlen_list = list(qsl_cpu[1 : num_reqs + 1])
        kvlen_list = list(sl_cpu[:num_reqs])
        total_actual = int(getattr(meta, "num_actual_tokens", sum(token_counts)))
        pad = bucket - total_actual
        if pad > 0 and qlen_list:
            qlen_list[-1] += pad
            kvlen_list[-1] += pad
        return qlen_list, kvlen_list
    return build_update_lists(token_counts, bucket)


class NPUPrefillGraphRunner(BasePrefillGraphRunner):
    """Manages Ascend Graph capture and execution on Huawei Ascend NPUs.

    Capture recipe (validated by the NPU probe):
      - static ops (incl. reshape_and_cache) are plain-captured;
      - each layer's FIA-v2 call is wrapped in graph_task_group_begin/end
        with host seq-lens lists pre-built outside the capture region;
      - replay: populate static buffers, rewrite every task group's params
        (fresh lists + current block table) on a dedicated update stream,
        order the serving stream after the update stream
        (``wait_stream``), then ``graph.replay()``.
    """

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
        self._update_stream: Any = None
        self.entries: dict[int, Any] = {}
        self._replays_by_bucket: dict[int, int] = {}
        # FIA workspace shared across every layer AND every bucket: only one
        # prefill graph executes at a time (sequential batches, sequential
        # layers, update stream ordered before replay), and the workspace
        # getter is monotonic in num_tokens. Buckets are captured
        # largest-first, so the first successful capture establishes the max
        # size and smaller buckets reuse it.
        self.shared_workspace: Any = None
        self._logged_first_replay: set[int] = set()
        # --- P2D (Prefill + 2 Decodes) decode-graph state -------------------
        # Four decode graphs: (num_reqs, beam_width) in {1,2} x {32,128}.
        # The decode graph has NO bucket dimension: the prefix attention
        # reads the paged KV cache and its host lists + block table are
        # rebound per replay, so prompt length is fully variable.
        self.captured_2d_graphs: dict[tuple[int, int], tuple[Any, Any]] = {}
        # Decoupled single-pass decode graphs: (pass_idx, num_reqs, beam_width).
        self.captured_decode_step_graphs: dict[tuple[int, int, int], tuple[Any, Any]] = {}
        # Monolithic P2D graphs: (bucket, num_reqs, beam_width) -> (graph, entry).
        self.monolithic_graphs: dict[tuple[int, int, int], tuple[Any, Any]] = {}
        self.p2d_pool: P2DBufferPool | None = None
        self.prefix_block_table_buf: torch.Tensor | None = None
        self.eager_output_buf: torch.Tensor | None = None
        self._decode_layer_names: list[str] = []
        self.sid_slices: SIDSlices | None = resolve_sid_slices_from_runner(model_runner)
        logger.info(
            "NPUPrefillGraphRunner initialized on device %s with buckets %s",
            self.device,
            self.buckets,
        )

    def log_replay(self, bucket: int, num_reqs: int, total_tokens: int) -> None:
        """Logs on the first replay of each bucket, then every 100 replays."""
        self._replays_by_bucket[bucket] = self._replays_by_bucket.get(bucket, 0) + 1
        count = self._replays_by_bucket[bucket]
        if bucket not in self._logged_first_replay:
            self._logged_first_replay.add(bucket)
            logger.info(
                "Prefill NPU Graph Replay active for bucket %d (reqs=%d, %d actual tokens).",
                bucket,
                num_reqs,
                total_tokens,
            )
        elif count % 100 == 0:
            logger.info(
                "Prefill NPU Graph Replay: bucket %d reached %d replays.",
                bucket,
                count,
            )

    def _get_update_stream(self) -> Any:
        if self._update_stream is None:
            self._update_stream = torch.npu.Stream(device=self.device)
        return self._update_stream

    def _bind_entry_workspace(self, entry: Any) -> None:
        """Reuse the cross-bucket shared workspace when it is already sized.

        Buckets are captured largest-first, so the first successful capture
        establishes the max workspace and every smaller bucket reuses it
        (the getter is monotonic in num_tokens).
        """
        if self.shared_workspace is not None:
            entry.workspace = self.shared_workspace

    @staticmethod
    def _get_max_kvlen(runner: Any) -> int:
        """Maximum supported per-request KV length from the model config.

        Used to size the FIA workspace for the largest possible
        cached/chunked-prefill context; 0 when the config is unavailable.
        """
        return get_max_context_len(runner)

    @staticmethod
    def _get_attn_builder_and_layers(runner: Any) -> tuple[Any, list[str]]:
        attn_groups = getattr(runner, "attn_groups", None)
        if not attn_groups:
            raise AttributeError("runner.attn_groups is unavailable")
        group = attn_groups[0][0]
        builder = group.get_metadata_builder(0)
        return builder, list(group.layer_names)

    @staticmethod
    def _forward_context(runner: Any, meta_dict: dict, num_tokens: int):
        from vllm.config import CUDAGraphMode

        try:
            from vllm_ascend.ascend_forward_context import set_ascend_forward_context

            return set_ascend_forward_context(
                meta_dict,
                runner.vllm_config,
                num_tokens=num_tokens,
                aclgraph_runtime_mode=CUDAGraphMode.NONE,
                model_instance=runner.model,
            )
        except ImportError:
            from vllm.forward_context import set_forward_context

            return set_forward_context(
                meta_dict,
                runner.vllm_config,
                num_tokens=num_tokens,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )

    def _prepare_capture_buffers(self, buf: PrefillBucketBuffers, bucket: int, runner: Any) -> None:
        if getattr(runner, "uses_mrope", False):
            buf.positions_buf = torch.zeros((3, bucket), dtype=torch.int64, device=self.device)
            buf.positions_buf[:] = torch.arange(bucket, dtype=torch.int64, device=self.device)
        elif getattr(runner, "uses_xdrope_dim", 0) > 0:
            buf.positions_buf = torch.zeros((2, bucket), dtype=torch.int64, device=self.device)
            buf.positions_buf[:] = torch.arange(bucket, dtype=torch.int64, device=self.device)
        else:
            buf.positions_buf = torch.arange(bucket, dtype=torch.int64, device=self.device)
        buf.input_ids_buf.fill_(buf.pad_token_id)
        # NPU's ATB ReshapeAndCache op requires int32 slot mappings (the GPU
        # shared buffer is int64). Capture with valid sequential slots: the
        # probe proved all-(-1) is a degenerate setup case, and the real
        # slots are copied in by execute_model_prefill before every replay.
        buf.slot_mapping_buf = torch.arange(bucket, dtype=torch.int32, device=self.device)
        # Static block-table staging buffer: a captured graph bakes the
        # pointer address, so the tensor at that address must stay stable
        # for the lifetime of the graph. The runner's live block table is
        # overwritten (or reallocated) for the next batch, so the graph
        # must never bake it; the task update binds this static buffer and
        # execute_model_prefill refreshes its contents before every replay.
        block_size = 16
        kv_cache_config = getattr(runner, "kv_cache_config", None)
        if kv_cache_config is not None and len(kv_cache_config.kv_cache_groups) > 0:
            block_size = kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
        max_kvlen = self._get_max_kvlen(runner)
        max_blocks = (
            max((max_kvlen + block_size - 1) // block_size, bucket // block_size + 1)
            if (max_kvlen > 0)
            else (bucket // block_size + 1)
        )
        # Reuse the init-pre-allocated buffer unless it is missing or too
        # narrow for this bucket's sizing.
        if buf.block_table_buf is None or buf.block_table_buf.shape[1] < max_blocks:
            buf.block_table_buf = torch.zeros(
                (SUPPORTED_NUM_REQS[-1], max_blocks), dtype=torch.int32, device=self.device
            )

    def _build_dummy_metadata(self, bucket: int, builder: Any, buf: PrefillBucketBuffers) -> Any:
        from vllm_gr.v1.attention.backends.beam_attn_metadata import (
            BeamAttentionMetadata,
        )

        # Every dummy request needs at least one token: with fewer tokens than
        # the dummy width (small buckets), spreading them over the full width
        # would leave zero-length requests. Cap the width by the bucket.
        num_reqs = min(SUPPORTED_NUM_REQS[-1], CAPTURE_DUMMY_NUM_REQS, bucket)
        base_len = bucket // num_reqs
        rem = bucket % num_reqs
        req_lens = [base_len + (1 if i < rem else 0) for i in range(num_reqs)]
        qsl_cpu = [0]
        curr = 0
        for rlen in req_lens:
            curr += rlen
            qsl_cpu.append(curr)
        qsl_dev = torch.tensor(qsl_cpu, dtype=torch.int32, device=self.device)
        seq_lens_dev = torch.tensor(req_lens, dtype=torch.int32, device=self.device)

        kv_cache_config = getattr(self.model_runner, "kv_cache_config", None)
        num_blocks = getattr(kv_cache_config, "num_blocks", None) if kv_cache_config else None
        # Size the table for the longest request, not the average: an uneven
        # split (e.g. bucket 129 -> 17,16,...,16) needs an extra column, and
        # the FIA input contract requires the block table to cover seq_len.
        max_req_len = max(req_lens)
        blocks_per_req = (max_req_len + builder.block_size - 1) // builder.block_size
        if num_blocks is None:
            num_blocks = blocks_per_req * num_reqs
        block_table = torch.zeros((num_reqs, blocks_per_req), dtype=torch.int32, device=self.device)
        blk = 0
        for r in range(num_reqs):
            for c in range(blocks_per_req):
                if blk >= num_blocks:
                    break
                block_table[r, c] = blk
                blk += 1

        return BeamAttentionMetadata(
            num_actual_tokens=bucket,
            max_query_len=bucket,
            query_start_loc=qsl_dev,
            max_seq_len=bucket,
            seq_lens=seq_lens_dev,
            block_table=block_table,
            slot_mapping=buf.slot_mapping_buf,
            causal=True,
            npu_causal_atten_mask=getattr(builder, "npu_causal_atten_mask", None),
            is_prefill_graph_exec=True,
            num_reqs=num_reqs,
            num_heads=builder.num_heads_q,
            num_kv_heads=builder.num_kv_heads,
            block_size=builder.block_size,
            query_start_loc_cpu=qsl_cpu,
            seq_lens_cpu=req_lens,
        )

    def capture_graphs(self, runner: Any) -> None:
        """Captures one Ascend ACL graph per prefill bucket.

        A single graph per bucket covers all supported batch sizes because
        the task-group update rewrites the per-request host lists at replay.
        """
        if not is_prefill_graph_enabled():
            logger.info(
                "Prefill NPU Graph capture is disabled via environment variable (%s=0).",
                ENV_ENABLE_PREFILL_GRAPH,
            )
            return

        if not is_npu_available():
            logger.info("NPU not available; skipping NPU graph capture for prefill.")
            return

        import torch_npu  # noqa: F401

        from vllm_gr.v1.attention.backends.beam_attn_npu import (
            NPUPrefillGraphEntry,
            prefill_graph_capture,
        )

        # Create the task-update stream BEFORE any capture (vllm-ascend
        # creates its update_stream in load_model, ahead of graph capture).
        if self._update_stream is None:
            self._update_stream = torch.npu.Stream(device=self.device)

        try:
            builder, layer_names = self._get_attn_builder_and_layers(runner)
        except Exception as e:  # noqa: BLE001
            logger.warning("NPU prefill graph capture: cannot locate attention builder: %r.", e)
            return

        logger.info(
            "Beginning Prefill NPU graph capture for buckets: %s (batch sizes: %s)",
            self.buckets,
            SUPPORTED_NUM_REQS,
        )

        from tqdm import tqdm

        graph_pool = torch.npu.graph_pool_handle()
        sorted_buckets = sorted(self.buckets, reverse=True)
        for bucket in tqdm(sorted_buckets, desc="Capturing CUDA graphs (prefill, FULL)"):
            try:
                # Per-bucket setup (buffer preparation and dummy metadata
                # allocation) lives inside the same guard as the capture:
                # an allocation failure here must fall back to eager instead
                # of aborting model initialization.
                buf = self.buffers[bucket]
                self._prepare_capture_buffers(buf, bucket, runner)
                meta = self._build_dummy_metadata(bucket, builder, buf)
                meta_dict = {name: meta for name in layer_names}
                entry = NPUPrefillGraphEntry(bucket=bucket, max_kvlen=self._get_max_kvlen(runner))
                self._bind_entry_workspace(entry)
                # Mirror vllm-ascend's _dummy_run: the model's persistent
                # static buffers were created under inference_mode during
                # vllm's decode-graph capture, so warmup and capture must
                # also run under inference_mode (in-place updates to
                # inference tensors are rejected outside it).
                with torch.inference_mode():
                    for _ in range(2):
                        with self._forward_context(runner, meta_dict, bucket):
                            runner.model(input_ids=buf.input_ids_buf, positions=buf.positions_buf)
                    torch.npu.synchronize()

                    graph = torch.npu.NPUGraph()
                    with (
                        prefill_graph_capture(entry),
                        self._forward_context(runner, meta_dict, bucket),
                        torch.npu.graph(graph, pool=graph_pool, capture_error_mode="global"),
                    ):
                        output = runner.model(
                            input_ids=buf.input_ids_buf, positions=buf.positions_buf
                        )
                    torch.npu.synchronize()
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Failed to capture Prefill NPU graph for bucket %d: %r. "
                    "Falling back to eager prefill for this and all remaining "
                    "buckets (a failed NPU capture poisons the in-process "
                    "capture state; retries require a restart).",
                    bucket,
                    e,
                )
                break

            buf.captured_graphs[1] = graph
            buf.output_bufs[1] = output
            buf.static_attn_metadatas[1] = meta
            self.entries[bucket] = entry
            if self.shared_workspace is None:
                self.shared_workspace = entry.workspace
            logger.info(
                "Captured Prefill NPU graph for bucket %d (%d attention layers).",
                bucket,
                len(entry.layers),
            )

        captured_summary = {b: list(self.buffers[b].captured_graphs.keys()) for b in self.buckets}
        msg = f"Prefill NPU Graph capture completed. Captured (bucket -> graph): {captured_summary}"
        logger.info(msg)

        # P2D is gated off on NPU (platform gate) and additionally by the
        # universal kill switch; both skip every P2D capture (monolithic, 2D,
        # and decode-step graphs) while retaining the standard prefill graphs
        # captured above.
        if not is_p2d_enabled():
            logger.info(
                "P2D NPU graph capture skipped (NPU platform gate or kill switch %s=1).",
                ENV_DISABLE_P2D,
            )
            return

        # Capture the monolithic P2D graphs (Pass 0 + 2 Decodes) after the
        # prefill buckets, then the standalone 2D graphs (Eager-P + 2D tier).
        self.capture_monolithic_p2d_graphs(runner)
        logger.info(
            "P2D Monolithic NPU graph capture completed. Captured configs: %s",
            sorted(self.monolithic_graphs.keys()),
        )

        # Capture the P2D decode graphs (Decodes 1 & 2) after the prefill
        # buckets: 4 static-shape graphs with no bucket dimension.
        self.capture_2d_graphs(runner)
        logger.info(
            "P2D 2D NPU graph capture completed. Captured configs: %s",
            sorted(self.captured_2d_graphs.keys()),
        )

        # Decoupled single-pass decode graphs for the stepwise tier.
        self.capture_decode_step_graphs(runner)
        logger.info(
            "P2D decode-step NPU graph capture completed. Captured configs: %s",
            sorted(self.captured_decode_step_graphs.keys()),
        )

    def has_captured_graph(
        self, bucket: int, num_reqs: int = 1, beam_width: int | None = None
    ) -> bool:
        """Returns True if a graph was captured for this bucket (covers all num_reqs)."""
        buf = self.buffers.get(bucket)
        return (
            buf is not None
            and buf.has_captured_graph(1)
            and bucket in self.entries
            and 1 <= num_reqs <= SUPPORTED_NUM_REQS[-1]
        )

    def execute_model_prefill(
        self,
        bucket: int,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        slot_mapping: torch.Tensor | None,
        num_scheduled_tokens: list[int] | dict[Any, int],
    ) -> torch.Tensor | None:
        """Populates static buffers, rewrites task-group params, and replays the graph."""
        token_counts = (
            list(num_scheduled_tokens.values())
            if isinstance(num_scheduled_tokens, dict)
            else list(num_scheduled_tokens)
        )
        num_reqs = len(token_counts)
        buf = self.buffers.get(bucket)
        entry = self.entries.get(bucket)
        if (
            buf is None
            or entry is None
            or not buf.has_captured_graph(1)
            or not 1 <= num_reqs <= SUPPORTED_NUM_REQS[-1]
        ):
            return None

        # Refresh the static buffers with plain on-stream D2D copies from the
        # live per-batch tensors. _prepare_inputs has already computed all of
        # them for THIS batch on the current stream, and the next batch's
        # writes are enqueued strictly after these copies on the same stream
        # (an NPU stream is FIFO), so reading the live tensors here can never
        # observe the next batch's data. No host-side recomputation or
        # allocations.
        total_actual = sum(token_counts)
        if input_ids is None or positions is None:
            logger.warning(
                "NPU prefill graph replay: missing live inputs; falling back to eager prefill."
            )
            return None
        # Never replay with a missing slot mapping: the graph writes the KV
        # cache at the captured slot addresses.
        if slot_mapping is None:
            logger.warning(
                "NPU prefill graph replay: no current slot mapping; falling back to eager prefill."
            )
            return None
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

        attn_metadata = None
        try:
            from vllm.forward_context import get_forward_context

            attn_metadata = getattr(get_forward_context(), "attn_metadata", None)
        except Exception as e:  # noqa: BLE001
            logger.debug("NPU prefill graph replay: no forward context: %r", e)
        if not isinstance(attn_metadata, dict) or not attn_metadata:
            logger.warning(
                "NPU prefill graph replay: no current attention metadata; "
                "falling back to eager prefill."
            )
            return None
        try:
            meta = next(iter(attn_metadata.values()))
        except (StopIteration, AttributeError):
            logger.warning(
                "NPU prefill graph replay: attention metadata unavailable; "
                "falling back to eager prefill."
            )
            return None

        # Cached-aware host lists copied from the current metadata (the
        # task-update re-binds them along with the block table).
        qlen_list, kvlen_list = build_update_lists_from_metadata(meta, token_counts, bucket)

        # Refresh the static block table (baked pointer) from the live
        # metadata tensor with a plain on-stream D2D copy (see the stream-FIFO
        # rationale above). The graph's FIA tasks read the static buffer
        # during replay; the update stream waits on the serving stream before
        # reading it, so the copy is ordered before the replay's reads.
        live_bt = meta.block_table
        bt_buf = buf.block_table_buf
        if bt_buf is None or live_bt.shape[1] > bt_buf.shape[1]:
            logger.warning(
                "NPU prefill graph replay: block table shape %s exceeds the "
                "staged buffer; falling back to eager prefill.",
                tuple(live_bt.shape),
            )
            return None
        num_cols = min(live_bt.shape[1], bt_buf.shape[1])
        bt_staged = bt_buf[: live_bt.shape[0], :num_cols]
        bt_staged.copy_(live_bt[: live_bt.shape[0], :num_cols], non_blocking=True)

        from vllm_gr.v1.attention.backends.beam_attn_npu import update_graph_params

        try:
            update_ok = update_graph_params(
                self._get_update_stream(), entry, qlen_list, kvlen_list, bt_staged
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "NPU prefill graph replay: task-group update failed (%r); "
                "falling back to eager prefill.",
                e,
            )
            return None
        if not update_ok:
            # Never replay after a failed update: a partially-updated graph
            # would execute with stale per-layer params. Fall back to eager.
            logger.warning(
                "NPU prefill graph replay: task-group update failed; falling back to eager prefill."
            )
            return None
        # Ordering chain (no host-side synchronize):
        #   (1) update_stream.wait_stream(serving_stream) inside the update —
        #       the update's device work is ordered after the previous
        #       replay's completion (the serving-stream tail includes it), so
        #       the update never rewrites tasks that are still executing;
        #   (2) this serving_stream.wait_stream(update_stream) — replay after
        #       the update's device work;
        #   (3) host program order (params rewritten before replay enqueued).
        # The host never blocks, so it can pipeline the next batch while the
        # device still executes this replay.
        torch.npu.current_stream().wait_stream(self._get_update_stream())

        graph = buf.captured_graphs[1]
        graph.replay()
        return buf.output_bufs[1]

    # ------------------------------------------------------------------
    # P2D: Eager prefill (existing prefill graph or eager fallback) + 2D
    # decode graph. The decode passes read the prompt prefix from the
    # paged KV cache via a block-table FIA task group, so no contiguous
    # prompt pool is needed (alloc_prompt_kv=False) and no bucket
    # dimension exists for the decode graph.
    # ------------------------------------------------------------------

    def _ensure_p2d_pool(self) -> P2DBufferPool:
        """Lazily initialize the NPU P2D pool (no contiguous prompt KV)."""
        if self.p2d_pool is None:
            model_config = getattr(self.model_runner, "model_config", None)
            hf_config = getattr(model_config, "hf_text_config", None) or getattr(
                model_config, "hf_config", None
            )
            num_layers = int(getattr(hf_config, "num_hidden_layers", 28))
            num_kv_heads = int(getattr(hf_config, "num_key_value_heads", 8))
            head_dim = int(getattr(hf_config, "head_dim", 128))
            vocab_size = int(getattr(hf_config, "vocab_size", 176256))
            pool_vocab_size = vocab_size
            from vllm_gr.ops.sliced_logits import (
                get_vocab_slice_config,
                is_vocab_slice_enabled,
            )

            if is_vocab_slice_enabled():
                _, slice_size, _ = get_vocab_slice_config()
                if slice_size is not None and slice_size > 0:
                    pool_vocab_size = slice_size
            dtype = getattr(self.model_runner, "dtype", torch.bfloat16)
            # KV extent for the pool and its bucket-less decode graphs:
            # defaults to the model context; VLLM_GR_P2D_KV_BOUND overrides it.
            max_b = max(
                self.max_bucket,
                resolve_p2d_pool_kv_bound(get_max_context_len(self.model_runner)),
            )
            self.p2d_pool = P2DBufferPool(
                max_batch=max(get_p2d_supported_num_reqs()),
                max_bucket=max_b,
                max_beams=get_p2d_max_beams(
                    int(getattr(model_config, "max_logprobs", 0) or 0), runner=self.model_runner
                ),
                num_decode_steps=DEFAULT_DECODE_STEPS,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                vocab_size=pool_vocab_size,
                dtype=dtype,
                device=self.device,
                alloc_prompt_kv=False,
                sid_slices=self.sid_slices,
                slot_mapping_dtype=torch.int32,
            )
        return self.p2d_pool

    def _get_prefix_block_table_buf(self, max_blocks: int) -> torch.Tensor:
        """Beam-replicated block-table staging buffer (W_total x max_blocks)."""
        max_logprobs = int(
            getattr(
                getattr(self.model_runner, "model_config", None),
                "max_logprobs",
                0,
            )
            or 0
        )
        rows = max(get_p2d_supported_num_reqs()) * get_p2d_max_beams(max_logprobs)
        if (
            self.prefix_block_table_buf is None
            or self.prefix_block_table_buf.shape[0] < rows
            or self.prefix_block_table_buf.shape[1] < max_blocks
        ):
            self.prefix_block_table_buf = torch.zeros(
                (rows, max_blocks), dtype=torch.int32, device=self.device
            )
        return self.prefix_block_table_buf

    def _npu_decode_forward_ctx(self, runner: Any, attn_meta: P2DMetadata, num_tokens: int):
        """NPU forward-context manager for the shared decode passes."""
        from vllm.config import CUDAGraphMode

        if not self._decode_layer_names:
            raise RuntimeError("NPU P2D decode capture: attention layer names are unavailable")
        meta_dict = {name: attn_meta for name in self._decode_layer_names}
        try:
            from vllm_ascend.ascend_forward_context import set_ascend_forward_context

            return set_ascend_forward_context(
                meta_dict,
                runner.vllm_config,
                num_tokens=num_tokens,
                aclgraph_runtime_mode=CUDAGraphMode.NONE,
                model_instance=runner.model,
            )
        except ImportError:
            from vllm.forward_context import set_forward_context

            return set_forward_context(
                meta_dict,
                runner.vllm_config,
                num_tokens=num_tokens,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )

    def _build_decode_dummy_metadata(
        self,
        runner: Any,
        num_reqs: int,
        beam_width: int,
        p2d_pass: int,
    ) -> P2DMetadata:
        """Dummy P2DMetadata for a decode pass with the NPU paged-cache fields."""
        builder, layer_names = self._get_attn_builder_and_layers(runner)
        self._decode_layer_names = layer_names
        total_beams = num_reqs * beam_width
        max_blocks = self._max_blocks_for(runner)
        block_table = torch.zeros((total_beams, max_blocks), dtype=torch.int32, device=self.device)
        # Capture-time prompt lengths are dummies; the task update rebinds
        # the per-beam kvlens at every replay and the workspace is sized by
        # max_kvlen (the model's max context), not these values.
        dummy_kvlen = min(self._get_max_kvlen(runner) or 16, 16)
        return P2DMetadata(
            p2d_pool=self._ensure_p2d_pool(),
            p2d_pass=p2d_pass,
            bucket=0,
            num_reqs=num_reqs,
            beam_width=beam_width,
            total_prompt_tokens=num_reqs * dummy_kvlen,
            block_table=block_table,
            num_heads=builder.num_heads_q,
            num_kv_heads=builder.num_kv_heads,
            block_size=builder.block_size,
            npu_causal_atten_mask=getattr(builder, "npu_causal_atten_mask", None),
            prefix_kvlen_list=[dummy_kvlen] * total_beams,
        )

    def _max_blocks_for(self, runner: Any) -> int:
        """Block-table column count for the decode graph's staged block table."""
        block_size = 16
        kv_cache_config = getattr(runner, "kv_cache_config", None)
        if kv_cache_config is not None and len(kv_cache_config.kv_cache_groups) > 0:
            block_size = kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
        max_kvlen = self._get_max_kvlen(runner)
        if max_kvlen > 0:
            return (max_kvlen + block_size - 1) // block_size
        return (self.max_bucket + block_size - 1) // block_size

    def has_2d_graph(self, num_reqs: int = 1, beam_width: int = 128) -> bool:
        """Returns True if a 2D (Decodes 1 & 2) Graph has been captured for this configuration."""
        if not is_p2d_enabled():
            return False
        return (num_reqs, beam_width) in self.captured_2d_graphs

    def has_decode_step_graph(self, pass_idx: int, num_reqs: int, beam_width: int) -> bool:
        """True when a single-pass decode graph is captured for this configuration."""
        if not is_p2d_enabled():
            return False
        return (pass_idx, num_reqs, beam_width) in self.captured_decode_step_graphs

    def capture_decode_step_graphs(self, runner: Any, pool: Any = None) -> None:
        """Captures decoupled single-pass decode graphs (forward + LM head).

        Keyed by (pass_idx, num_reqs, beam_width): pass 1 graphs for P_BW
        widths, pass 2 graphs for D0_BW widths (W2 shapes no graph). A failed
        capture is logged and skipped; the poisoned-capture rule breaks the
        loop.
        """
        if not is_p2d_enabled():
            return
        if not is_npu_available():
            logger.info("NPU not available; skipping NPU P2D decode-step graph capture.")
            return

        import torch_npu  # noqa: F401

        from vllm_gr.v1.attention.backends.beam_attn_npu import (
            NPUDecodeGraphEntry,
            decode_graph_capture,
        )

        p2d_pool = self._ensure_p2d_pool()
        if pool is None:
            pool = torch.npu.graph_pool_handle()

        p_bw, d0_bw, _ = get_p2d_step_beam_widths_config()
        for pass_idx, widths in ((1, p_bw), (2, d0_bw)):
            for num_reqs in get_p2d_supported_num_reqs():
                for beam_width in widths:
                    if beam_width > p2d_pool.max_beams:
                        continue
                    total_beam_tokens = num_reqs * beam_width
                    try:
                        attn_meta = self._build_decode_dummy_metadata(
                            runner, num_reqs, beam_width, pass_idx
                        )
                        p2d_pool.positions_1[:total_beam_tokens].fill_(16)
                        p2d_pool.positions_2[:total_beam_tokens].fill_(17)
                        entry = NPUDecodeGraphEntry(max_kvlen=self._get_max_kvlen(runner))

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
                                self._npu_decode_forward_ctx,
                            )

                        with torch.inference_mode():
                            for _ in range(2):
                                run_step()
                            torch.npu.synchronize()

                            graph = torch.npu.NPUGraph()
                            with (
                                decode_graph_capture(entry),
                                torch.npu.graph(graph, pool=pool, capture_error_mode="global"),
                            ):
                                run_step()
                            torch.npu.synchronize()

                        self.captured_decode_step_graphs[(pass_idx, num_reqs, beam_width)] = (
                            graph,
                            entry,
                        )
                        logger.info(
                            "Captured P2D decode-step NPU graph: pass=%d, num_reqs=%d, "
                            "beam_width=%d, tasks=%d",
                            pass_idx,
                            num_reqs,
                            beam_width,
                            len(entry.tasks),
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "Failed to capture P2D decode-step NPU graph "
                            "(pass=%d, num_reqs=%d, beam_width=%d): %r. Aborting decode-step "
                            "capture; the stepwise tier falls back to eager on-device.",
                            pass_idx,
                            num_reqs,
                            beam_width,
                            e,
                        )
                        return

    def _stage_p2d_step_inputs(
        self,
        p2d_pool: P2DBufferPool,
        token_counts: list[int],
        computed_tokens: list[int] | None,
        w0: int,
        w1: int,
    ) -> list[int]:
        """Stages per-request T0 decode positions; returns the T0 list."""
        num_reqs = len(token_counts)
        c_tokens = computed_tokens or [0] * num_reqs
        total_prompt_lens = [
            (c_tokens[r] if r < len(c_tokens) else 0) + sched
            for r, sched in enumerate(token_counts)
        ]
        cum = 0
        for r, sched in enumerate(token_counts):
            cum += sched
            p2d_pool.logits_indices_0[r] = cum - 1
        for r, T_r in enumerate(total_prompt_lens):
            p2d_pool.positions_1[r * w0 : (r + 1) * w0].fill_(T_r)
            p2d_pool.positions_2[r * w1 : (r + 1) * w1].fill_(T_r + 1)
        return total_prompt_lens

    def _make_stepwise_decode_fn(
        self,
        runner: Any,
        entry: Any,
        graph: Any,
        pass_idx: int,
        num_reqs: int,
        beam_width: int,
        total_prompt_lens: list[int],
        block_table: torch.Tensor,
    ) -> Any:
        """Returns a decode-step replay fn: stage block table -> task update ->
        wait_stream -> replay (no host sync)."""
        from vllm_gr.v1.attention.backends.beam_attn_npu import (
            update_decode_graph_params,
        )

        max_blocks = self._max_blocks_for(runner)

        def _replay(_tokens: int) -> None:
            bt_buf = self._get_prefix_block_table_buf(max_blocks)
            bt_staged = stage_p2d_block_table(
                bt_buf, block_table, num_reqs=num_reqs, beam_width=beam_width
            )
            prefix_qlen_list = list(range(1, num_reqs * beam_width + 1))
            prefix_kvlen_list: list[int] = []
            for T_r in total_prompt_lens:
                prefix_kvlen_list.extend([T_r] * beam_width)
            ok = update_decode_graph_params(
                self._get_update_stream(),
                entry,
                prefix_qlen_list,
                prefix_kvlen_list,
                bt_staged,
            )
            if not ok:
                raise RuntimeError(
                    f"NPU decode-step update failed (pass={pass_idx}, bw={beam_width})"
                )
            torch.npu.current_stream().wait_stream(self._get_update_stream())
            graph.replay()

        return _replay

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
        """Tier 2: prefill (already executed) + decoupled decode-step graphs."""
        token_counts = (
            list(num_scheduled_tokens.values())
            if isinstance(num_scheduled_tokens, dict)
            else list(num_scheduled_tokens)
        )
        num_reqs = len(token_counts)
        total_actual = sum(token_counts)
        if num_reqs not in get_p2d_supported_num_reqs() or total_actual <= 0:
            return None
        if block_table is None or slot_mapping is None:
            return None
        w0, w1, w2 = step_bws
        step1_entry = self.captured_decode_step_graphs.get((1, num_reqs, w0))
        step2_entry = self.captured_decode_step_graphs.get((2, num_reqs, w1))
        if step1_entry is None or step2_entry is None:
            return None

        p2d_pool = self._ensure_p2d_pool()
        total_prompt_lens = self._stage_p2d_step_inputs(
            p2d_pool, token_counts, computed_tokens, w0, w1
        )

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
            num_layers=p2d_pool.num_layers,
            num_kv_heads=p2d_pool.num_kv_heads,
            head_dim=p2d_pool.head_dim,
            forward_ctx_fn=self._npu_decode_forward_ctx,
            decode1_fn=self._make_stepwise_decode_fn(
                runner,
                step1_entry[1],
                step1_entry[0],
                1,
                num_reqs,
                w0,
                total_prompt_lens,
                block_table,
            ),
            decode2_fn=self._make_stepwise_decode_fn(
                runner,
                step2_entry[1],
                step2_entry[0],
                2,
                num_reqs,
                w1,
                total_prompt_lens,
                block_table,
            ),
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
        single-pass forwards."""
        token_counts = (
            list(num_scheduled_tokens.values())
            if isinstance(num_scheduled_tokens, dict)
            else list(num_scheduled_tokens)
        )
        num_reqs = len(token_counts)
        total_actual = sum(token_counts)
        if num_reqs not in get_p2d_supported_num_reqs() or total_actual <= 0:
            return None
        if block_table is None or slot_mapping is None:
            return None
        w0, w1, w2 = step_bws

        p2d_pool = self._ensure_p2d_pool()
        total_prompt_lens = self._stage_p2d_step_inputs(
            p2d_pool, token_counts, computed_tokens, w0, w1
        )

        def _eager_meta(pass_idx: int, bw: int) -> P2DMetadata:
            prefix_kvlen_list: list[int] = []
            for T_r in total_prompt_lens:
                prefix_kvlen_list.extend([T_r] * bw)
            return P2DMetadata(
                p2d_pool=p2d_pool,
                p2d_pass=pass_idx,
                bucket=total_actual,
                num_reqs=num_reqs,
                beam_width=bw,
                total_prompt_tokens=sum(total_prompt_lens),
                prefix_kvlen_list=prefix_kvlen_list,
            )

        attn_meta_1 = _eager_meta(1, w0)
        attn_meta_2 = _eager_meta(2, w1)

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
            num_layers=p2d_pool.num_layers,
            num_kv_heads=p2d_pool.num_kv_heads,
            head_dim=p2d_pool.head_dim,
            forward_ctx_fn=self._npu_decode_forward_ctx,
            decode1_fn=lambda _tokens: run_decode_pass_eager(
                runner, p2d_pool, attn_meta_1, 1, num_reqs * w0, self._npu_decode_forward_ctx
            ),
            decode2_fn=lambda _tokens: run_decode_pass_eager(
                runner, p2d_pool, attn_meta_2, 2, num_reqs * w1, self._npu_decode_forward_ctx
            ),
        )

        return (
            p2d_pool.final_tokens[:num_reqs, :w2],
            p2d_pool.final_cum_lps[:num_reqs, :w2],
            p2d_pool.valid_beam_counts[:num_reqs],
            p2d_pool.step_scores[:num_reqs, :w2],
        )

    def has_monolithic_graph(self, bucket: int, num_reqs: int, beam_width: int) -> bool:
        """Returns True if a monolithic P2D graph was captured for this config."""
        if not is_p2d_enabled():
            return False
        return (bucket, num_reqs, beam_width) in self.monolithic_graphs

    def capture_monolithic_p2d_graphs(self, runner: Any, pool: Any = None) -> None:
        """Captures the monolithic (Pass 0 + 2 Decodes) NPU P2D graphs.

        One graph per (bucket, num_reqs, beam_width) over the monolithic
        buckets x {1,2} x {32,128}. Pass 0 writes the paged KV cache (staged
        slot mapping) and runs paged full-context attention; passes 1/2 read
        the prompt from the paged cache via beam-replicated block tables.
        A failed capture is logged and skipped; the poisoned-capture rule
        breaks the loop (NPU captures must not continue after a failure).
        """
        if not is_p2d_enabled():
            return
        if not is_npu_available():
            logger.info("NPU not available; skipping NPU P2D monolithic graph capture.")
            return

        import torch_npu  # noqa: F401

        from vllm_gr.v1.attention.backends.beam_attn_npu import (
            NPUMonolithicGraphEntry,
            monolithic_graph_capture,
        )
        from vllm_gr.v1.worker.prefill_graph_common import MONOLITHIC_SUPPORTED_BUCKETS

        p2d_pool = self._ensure_p2d_pool()
        try:
            builder, layer_names = self._get_attn_builder_and_layers(runner)
            self._decode_layer_names = layer_names
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "NPU P2D monolithic graph capture: cannot locate attention builder: %r.", e
            )
            return

        if pool is None:
            pool = torch.npu.graph_pool_handle()

        for bucket in MONOLITHIC_SUPPORTED_BUCKETS:
            if bucket not in self.buffers or bucket > self.max_bucket:
                continue
            for num_reqs in get_p2d_supported_num_reqs():
                if bucket < num_reqs:
                    continue
                for beam_width in get_p2d_beam_widths():
                    total_beam_tokens = num_reqs * beam_width
                    try:
                        entry = NPUMonolithicGraphEntry(
                            bucket=bucket,
                            num_reqs=num_reqs,
                            beam_width=beam_width,
                            max_kvlen=self._get_max_kvlen(runner),
                        )
                        attn_meta_0, attn_meta_1, attn_meta_2 = (
                            self._build_monolithic_dummy_metadata(
                                runner,
                                builder,
                                p2d_pool,
                                bucket,
                                num_reqs,
                                beam_width,
                            )
                        )
                        p2d_pool.positions_1[:total_beam_tokens].fill_(bucket)
                        p2d_pool.positions_2[:total_beam_tokens].fill_(bucket + 1)

                        def run_monolithic_passes(
                            _m0: P2DMetadata = attn_meta_0,
                            _m1: P2DMetadata = attn_meta_1,
                            _m2: P2DMetadata = attn_meta_2,
                            _bucket: int = bucket,
                            _reqs: int = num_reqs,
                            _width: int = beam_width,
                            _tokens: int = total_beam_tokens,
                        ) -> None:
                            with self._npu_decode_forward_ctx(runner, _m0, _bucket):
                                hidden_0 = runner.model(
                                    input_ids=p2d_pool.input_ids_0[:_bucket],
                                    positions=p2d_pool.positions_0[:_bucket],
                                )
                            sample_hidden_0 = hidden_0[p2d_pool.logits_indices_0[:_reqs]]
                            logits_0 = runner.model.compute_logits(sample_hidden_0)
                            check_p2d_vocab_size(p2d_pool, logits_0, "graph capture")
                            p2d_pool.logits_0[:_reqs].copy_(logits_0)
                            step0_select_topk(
                                logits_0=p2d_pool.logits_0,
                                num_reqs=_reqs,
                                beam_width=_width,
                                pool=p2d_pool,
                            )
                            run_decode_passes(
                                runner=runner,
                                p2d_pool=p2d_pool,
                                attn_meta_1=_m1,
                                attn_meta_2=_m2,
                                total_beam_tokens=_tokens,
                                num_reqs=_reqs,
                                beam_width=_width,
                                num_layers=p2d_pool.num_layers,
                                num_kv_heads=p2d_pool.num_kv_heads,
                                head_dim=p2d_pool.head_dim,
                                forward_ctx_fn=self._npu_decode_forward_ctx,
                            )

                        with torch.inference_mode():
                            for _ in range(2):
                                run_monolithic_passes()
                            torch.npu.synchronize()

                            graph = torch.npu.NPUGraph()
                            with (
                                monolithic_graph_capture(entry),
                                torch.npu.graph(graph, pool=pool, capture_error_mode="global"),
                            ):
                                run_monolithic_passes()
                            torch.npu.synchronize()

                        self.monolithic_graphs[(bucket, num_reqs, beam_width)] = (graph, entry)
                        logger.info(
                            "Captured monolithic P2D NPU graph: bucket=%d, num_reqs=%d, "
                            "beam_width=%d, prefill_tasks=%d, decode_tasks=%d",
                            bucket,
                            num_reqs,
                            beam_width,
                            len(entry.prefill_tasks),
                            len(entry.decode_tasks),
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "Failed to capture monolithic P2D NPU graph "
                            "(bucket=%d, num_reqs=%d, beam_width=%d): %r. "
                            "Aborting monolithic capture; P2D falls back to Eager-P + 2D decode.",
                            bucket,
                            num_reqs,
                            beam_width,
                            e,
                        )
                        return

    def _build_monolithic_dummy_metadata(
        self,
        runner: Any,
        builder: Any,
        p2d_pool: P2DBufferPool,
        bucket: int,
        num_reqs: int,
        beam_width: int,
    ) -> tuple[P2DMetadata, P2DMetadata, P2DMetadata]:
        """Dummy metadata for the monolithic capture (passes 0, 1, 2)."""
        total_beams = num_reqs * beam_width
        max_blocks = self._max_blocks_for(runner)
        dummy_kvlen = min(self._get_max_kvlen(runner) or 16, 16)

        # Pass 0: chunk qlen cumsum + full-prompt kvlen dummies; request-row
        # block table and sequential staged slot mapping. Generic over any
        # num_reqs (the bucket is evenly distributed, remainder to the first
        # requests).
        base_len = bucket // num_reqs
        rem = bucket % num_reqs
        pass0_qlen_list: list[int] = []
        p2d_pool.cu_seqlens_q_0[0] = 0
        cum = 0
        for r in range(num_reqs):
            chunk_len = base_len + (1 if r < rem else 0)
            cum += chunk_len
            pass0_qlen_list.append(cum)
            p2d_pool.cu_seqlens_q_0[r + 1] = cum
            p2d_pool.logits_indices_0[r] = cum - 1
            p2d_pool.seqused_k[r] = bucket
        p2d_pool.input_ids_0[:bucket].fill_(
            self.buffers[bucket].pad_token_id if bucket in self.buffers else 0
        )
        p2d_pool.positions_0[:bucket].copy_(
            torch.arange(bucket, dtype=torch.int64, device=self.device)
        )
        p2d_pool.slot_mapping_0[:bucket].copy_(
            torch.arange(bucket, dtype=p2d_pool.slot_mapping_0.dtype, device=self.device)
        )

        attn_meta_0 = P2DMetadata(
            p2d_pool=p2d_pool,
            p2d_pass=0,
            bucket=bucket,
            num_reqs=num_reqs,
            beam_width=beam_width,
            total_prompt_tokens=bucket,
            cu_seqlens_q_0=p2d_pool.cu_seqlens_q_0[: num_reqs + 1],
            block_table=p2d_pool.block_table[:num_reqs],
            slot_mapping=p2d_pool.slot_mapping_0[:bucket],
            seqused_k=p2d_pool.seqused_k[:num_reqs],
            num_heads=builder.num_heads_q,
            num_kv_heads=builder.num_kv_heads,
            block_size=builder.block_size,
            npu_causal_atten_mask=getattr(builder, "npu_causal_atten_mask", None),
            pass0_qlen_list=pass0_qlen_list,
            pass0_kvlen_list=[dummy_kvlen] * num_reqs,
        )

        def _decode_meta(p2d_pass: int) -> P2DMetadata:
            return P2DMetadata(
                p2d_pool=p2d_pool,
                p2d_pass=p2d_pass,
                bucket=bucket,
                num_reqs=num_reqs,
                beam_width=beam_width,
                total_prompt_tokens=num_reqs * dummy_kvlen,
                block_table=torch.zeros(
                    (total_beams, max_blocks), dtype=torch.int32, device=self.device
                ),
                num_heads=builder.num_heads_q,
                num_kv_heads=builder.num_kv_heads,
                block_size=builder.block_size,
                npu_causal_atten_mask=getattr(builder, "npu_causal_atten_mask", None),
                prefix_kvlen_list=[dummy_kvlen] * total_beams,
            )

        return attn_meta_0, _decode_meta(1), _decode_meta(2)

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
        """Replays the captured monolithic P2D NPU graph (Pass 0 + 2 Decodes).

        Stages the live prompt inputs, slot mapping, and block tables (request
        rows for pass 0, beam-replicated rows for the decode prefix), rebinds
        the FIA task groups with T0 = computed + scheduled seqlens, and replays.
        """
        from vllm_gr.v1.attention.backends.beam_attn_npu import (
            update_monolithic_graph_params,
        )

        token_counts = (
            list(num_scheduled_tokens.values())
            if isinstance(num_scheduled_tokens, dict)
            else list(num_scheduled_tokens)
        )
        num_reqs = len(token_counts)
        total_actual = sum(token_counts)
        if (
            num_reqs not in get_p2d_supported_num_reqs()
            or total_actual <= 0
            or total_actual > bucket
        ):
            return None
        if input_ids is None or positions is None:
            return None
        if block_table is None or slot_mapping is None:
            return None

        graph_entry = self.monolithic_graphs.get((bucket, num_reqs, beam_width))
        if graph_entry is None:
            return None
        graph, entry = graph_entry
        if graph is None or not hasattr(graph, "replay"):
            return None

        p2d_pool = self._ensure_p2d_pool()
        total_beam_tokens = num_reqs * beam_width
        computed_counts = (
            computed_tokens
            if computed_tokens is not None and len(computed_tokens) == num_reqs
            else [0] * num_reqs
        )

        # Stage prompt inputs (last request absorbs the padding).
        p2d_pool.input_ids_0[:total_actual].copy_(input_ids[:total_actual], non_blocking=True)
        if total_actual < bucket:
            pad_id = self.buffers[bucket].pad_token_id if bucket in self.buffers else 0
            p2d_pool.input_ids_0[total_actual:bucket].fill_(pad_id)
        p2d_pool.positions_0[:total_actual].copy_(positions[:total_actual], non_blocking=True)
        if total_actual < bucket:
            p2d_pool.positions_0[total_actual:bucket].fill_(0)
        num_slots = min(total_actual, slot_mapping.shape[0])
        p2d_pool.slot_mapping_0[:num_slots].copy_(slot_mapping[:num_slots], non_blocking=True)
        if num_slots < bucket:
            p2d_pool.slot_mapping_0[num_slots:bucket].fill_(-1)

        # Chunk qlen cumsums (FIA convention: no leading zero) and full-prompt
        # kvlens (computed + scheduled) per request.
        pass0_qlen_list: list[int] = []
        pass0_kvlen_list: list[int] = []
        curr = 0
        for i, sched in enumerate(token_counts):
            curr += sched
            pass0_qlen_list.append(curr)
            pass0_kvlen_list.append(computed_counts[i] + sched)
        p2d_pool.cu_seqlens_q_0[0] = 0
        for r, sched in enumerate(token_counts):
            cum = sum(token_counts[: r + 1])
            p2d_pool.cu_seqlens_q_0[r + 1] = cum
            p2d_pool.logits_indices_0[r] = cum - 1

        # Decode positions and per-beam prefix kvlens (generic over any num_reqs).
        prefix_kvlen_list: list[int] = []
        for r, T_r in enumerate(pass0_kvlen_list):
            r_start = r * beam_width
            r_end = r_start + beam_width
            p2d_pool.positions_1[r_start:r_end].fill_(T_r)
            p2d_pool.positions_2[r_start:r_end].fill_(T_r + 1)
            prefix_kvlen_list.extend([T_r] * beam_width)
        prefix_qlen_list = list(range(1, total_beam_tokens + 1))

        # Stage block tables: request rows for pass 0, beam rows for decodes.
        max_blocks = self._max_blocks_for(self.model_runner)
        req_bt_buf = p2d_pool.block_table
        if block_table.shape[1] > req_bt_buf.shape[1]:
            logger.warning(
                "NPU monolithic P2D replay: block table shape %s exceeds the staged "
                "buffer; falling back to Eager-P + 2D.",
                tuple(block_table.shape),
            )
            return None
        stage_p2d_block_table(req_bt_buf, block_table, num_reqs=num_reqs)
        beam_bt_buf = self._get_prefix_block_table_buf(max_blocks)
        beam_bt_staged = stage_p2d_block_table(
            beam_bt_buf,
            block_table,
            num_reqs=num_reqs,
            beam_width=beam_width,
        )

        try:
            update_ok = update_monolithic_graph_params(
                self._get_update_stream(),
                entry,
                pass0_qlen_list,
                pass0_kvlen_list,
                prefix_qlen_list,
                prefix_kvlen_list,
                req_bt_buf[:num_reqs, : block_table.shape[1]],
                beam_bt_staged,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "NPU monolithic P2D replay: task-group update failed (%r); "
                "falling back to Eager-P + 2D.",
                e,
            )
            return None
        if not update_ok:
            logger.warning(
                "NPU monolithic P2D replay: task-group update failed; falling back to Eager-P + 2D."
            )
            return None

        torch.npu.current_stream().wait_stream(self._get_update_stream())
        graph.replay()

        return (
            p2d_pool.final_tokens[:num_reqs, :beam_width],
            p2d_pool.final_cum_lps[:num_reqs, :beam_width],
            p2d_pool.valid_beam_counts[:num_reqs],
            p2d_pool.step_scores[:num_reqs, :beam_width],
        )

    def capture_2d_graphs(self, runner: Any, pool: Any = None) -> None:
        """Captures the standalone 2D (Decodes 1 & 2) NPU graphs.

        Four graphs total — (num_reqs, beam_width) in {1,2} x {32,128} —
        with NO bucket dimension (prefix kvlens are rebound per replay).
        A failed capture is logged and skipped: eager beam decode (the
        cascade path) remains the fallback, and the poisoned-capture rule
        breaks the loop (NPU captures must not continue after a failure).
        """
        if not is_p2d_enabled():
            return
        if not is_npu_available():
            logger.info("NPU not available; skipping NPU P2D decode graph capture.")
            return

        import torch_npu  # noqa: F401

        from vllm_gr.v1.attention.backends.beam_attn_npu import (
            NPUDecodeGraphEntry,
            decode_graph_capture,
        )

        p2d_pool = self._ensure_p2d_pool()
        try:
            _, layer_names = self._get_attn_builder_and_layers(runner)
            self._decode_layer_names = layer_names
        except Exception as e:  # noqa: BLE001
            logger.warning("NPU P2D decode graph capture: cannot locate attention builder: %r.", e)
            return

        if pool is None:
            pool = torch.npu.graph_pool_handle()

        for num_reqs in get_p2d_supported_num_reqs():
            for beam_width in get_p2d_beam_widths():
                total_beam_tokens = num_reqs * beam_width
                try:
                    attn_meta_1 = self._build_decode_dummy_metadata(runner, num_reqs, beam_width, 1)
                    attn_meta_2 = self._build_decode_dummy_metadata(runner, num_reqs, beam_width, 2)

                    # Dummy decode positions; refreshed per replay.
                    p2d_pool.positions_1[:total_beam_tokens].fill_(16)
                    p2d_pool.positions_2[:total_beam_tokens].fill_(17)

                    entry = NPUDecodeGraphEntry(max_kvlen=self._get_max_kvlen(runner))

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
                            num_layers=p2d_pool.num_layers,
                            num_kv_heads=p2d_pool.num_kv_heads,
                            head_dim=p2d_pool.head_dim,
                            forward_ctx_fn=self._npu_decode_forward_ctx,
                        )

                    with torch.inference_mode():
                        for _ in range(2):
                            run_2d_passes()
                        torch.npu.synchronize()

                        graph = torch.npu.NPUGraph()
                        with (
                            decode_graph_capture(entry),
                            torch.npu.graph(graph, pool=pool, capture_error_mode="global"),
                        ):
                            run_2d_passes()
                        torch.npu.synchronize()

                    self.captured_2d_graphs[(num_reqs, beam_width)] = (graph, entry)
                    logger.info(
                        "Captured P2D 2D NPU graph: num_reqs=%d, beam_width=%d, tasks=%d",
                        num_reqs,
                        beam_width,
                        len(entry.tasks),
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Failed to capture 2D NPU graph (num_reqs=%d, beam_width=%d): %r. "
                        "P2D decode will fall back to eager beam decode.",
                        num_reqs,
                        beam_width,
                        e,
                    )
                    break

    def execute_decode_p2d(
        self,
        runner: Any,
        hidden: torch.Tensor,
        block_table: torch.Tensor,
        num_scheduled_tokens: list[int] | dict[Any, int],
        beam_width: int = 128,
        computed_tokens: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Runs step-0 top-k (eager) and replays the captured 2D decode graph.

        ``hidden`` holds the prefill's hidden states (prefill graph replay or
        eager prefill — real tokens keep their batch positions in both
        layouts); ``block_table`` is the live per-request block table from the
        batch's attention metadata (the decode prefix attention reads the
        paged cache through it).
        """
        from vllm_gr.v1.attention.backends.beam_attn_npu import (
            update_decode_graph_params,
        )

        token_counts = (
            list(num_scheduled_tokens.values())
            if isinstance(num_scheduled_tokens, dict)
            else list(num_scheduled_tokens)
        )
        num_reqs = len(token_counts)
        total_actual = sum(token_counts)
        if num_reqs not in get_p2d_supported_num_reqs() or total_actual <= 0:
            return None

        graph_entry = self.captured_2d_graphs.get((num_reqs, beam_width))
        if graph_entry is None:
            return None
        graph, entry = graph_entry
        if graph is None or not hasattr(graph, "replay"):
            return None

        p2d_pool = self._ensure_p2d_pool()
        total_beam_tokens = num_reqs * beam_width

        # Step 0: sample the end-of-prompt rows and select top-W on device.
        # Stage the indices into the pool buffer first: a fresh
        # ``torch.tensor(list, device=...)`` syncs the stream (pageable H2D).
        sample_indices = self.get_prefill_logit_indices(token_counts)
        for r, idx in enumerate(sample_indices):
            p2d_pool.logits_indices_0[r] = idx
        sample_hidden = hidden[p2d_pool.logits_indices_0[:num_reqs]]
        logits_0 = runner.model.compute_logits(sample_hidden)
        check_p2d_vocab_size(p2d_pool, logits_0, "eager")
        p2d_pool.logits_0[:num_reqs].copy_(logits_0, non_blocking=True)
        step0_select_topk(
            logits_0=p2d_pool.logits_0,
            num_reqs=num_reqs,
            beam_width=beam_width,
            pool=p2d_pool,
        )

        # Refresh decode positions and per-beam prefix kvlens (generic over
        # any num_reqs).
        prefix_kvlen_list: list[int] = []
        computed_counts = (
            computed_tokens
            if computed_tokens is not None and len(computed_tokens) == num_reqs
            else [0] * num_reqs
        )
        total_prompt_lens = [c + s for c, s in zip(computed_counts, token_counts)]

        for r, T_r in enumerate(total_prompt_lens):
            r_start = r * beam_width
            r_end = r_start + beam_width
            p2d_pool.positions_1[r_start:r_end].fill_(T_r)
            p2d_pool.positions_2[r_start:r_end].fill_(T_r + 1)
            prefix_kvlen_list.extend([T_r] * beam_width)
        prefix_qlen_list = list(range(1, total_beam_tokens + 1))

        # Stage the live block table with per-beam row replication: every
        # beam of request i points at request i's prompt blocks.
        bt_buf = self._get_prefix_block_table_buf(self._max_blocks_for(runner))
        if block_table is None or block_table.shape[1] > bt_buf.shape[1]:
            logger.warning(
                "NPU P2D decode replay: block table shape %s exceeds the staged "
                "buffer; falling back to eager beam decode.",
                tuple(block_table.shape) if block_table is not None else None,
            )
            return None
        bt_staged = stage_p2d_block_table(
            bt_buf,
            block_table,
            num_reqs=num_reqs,
            beam_width=beam_width,
        )

        try:
            update_ok = update_decode_graph_params(
                self._get_update_stream(),
                entry,
                prefix_qlen_list,
                prefix_kvlen_list,
                bt_staged,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "NPU P2D decode replay: task-group update failed (%r); "
                "falling back to eager beam decode.",
                e,
            )
            return None
        if not update_ok:
            logger.warning(
                "NPU P2D decode replay: task-group update failed; "
                "falling back to eager beam decode."
            )
            return None

        torch.npu.current_stream().wait_stream(self._get_update_stream())
        graph.replay()

        return (
            p2d_pool.final_tokens[:num_reqs, :beam_width],
            p2d_pool.final_cum_lps[:num_reqs, :beam_width],
            p2d_pool.valid_beam_counts[:num_reqs],
            p2d_pool.step_scores[:num_reqs, :beam_width],
        )

    def get_output_buf(self, total_tokens: int, model_config: Any) -> torch.Tensor:
        """Returns a reusable dummy hidden-states buffer for the p2d return value."""
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
