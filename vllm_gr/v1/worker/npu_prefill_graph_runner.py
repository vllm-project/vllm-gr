# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend NPU Graph Runner for NPU prefill execution."""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_gr.utils.npu_adapter import is_npu_available
from vllm_gr.v1.worker.prefill_graph_common import (
    ENV_ENABLE_PREFILL_GRAPH,
    SUPPORTED_NUM_REQS,
    BasePrefillGraphRunner,
    PrefillBucketBuffers,
    get_max_context_len,
    is_prefill_graph_enabled,
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
        (fresh lists + current block table) on a dedicated update stream with
        a per-layer event recorded after each task update, then
        ``graph.replay()`` — each in-graph layer's FIA blocks on its baked
        event.wait until the update signals it, so the replay pipelines with
        the update at layer granularity.
    """

    def __init__(
        self,
        model_runner: Any,
        buckets: list[int] | None = None,
    ) -> None:
        super().__init__(
            model_runner=model_runner,
            buckets=buckets,
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

    def _release_all_events(self, entry: Any) -> None:
        """Fire every per-layer event for ``entry`` on the update stream.

        Recovery for replay-first (Plan A') aborts: a replay already enqueued
        on the serving stream stalls at each layer's in-graph ``event.wait``.
        If the matching update is aborted (exception), the un-recorded layers
        would hang the replay forever. Recording all events lets the stalled
        replay drain; its output is discarded by the caller's eager fallback.
        """
        update_stream = self._get_update_stream()
        layers = getattr(entry, "layers", None)
        if not layers:
            return
        with torch.npu.stream(update_stream):
            for task in layers:
                event = getattr(task, "event", None)
                if event is not None:
                    event.record(update_stream)

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

    def has_captured_graph(self, bucket: int, num_reqs: int = 1) -> bool:
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

        graph = buf.captured_graphs[1]
        graph.replay()

        try:
            update_ok = update_graph_params(
                self._get_update_stream(),
                entry,
                qlen_list,
                kvlen_list,
                bt_staged,
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
        #   (2) NO whole-graph serving_stream.wait_stream(update_stream): each
        #       layer's FIA in the replayed graph blocks on its baked in-graph
        #       event.wait until the update stream records that layer's event
        #       right after its task update, so the replay's layer i overlaps
        #       the update's layer i+1 instead of waiting for the whole update;
        #   (3) host program order (params rewritten before replay enqueued).
        # The host never blocks, so it can pipeline the next batch while the
        # device still executes this replay.
        return buf.output_bufs[1]
