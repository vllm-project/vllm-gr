# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Ascend NPU Graph Runner for NPU prefill execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
from vllm.logger import init_logger

from vllm_gr.utils.npu_adapter import is_npu_available
from vllm_gr.v1.worker.prefill_graph_common import (
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

    ``qlen_list`` contains cumulative query-token counts without a leading
    zero; ``kvlen_list`` contains per-request KV lengths. For nonempty
    ``token_counts`` and a bucket at least as large as their total, padding
    is added to the last request so ``qlen_list[-1] == bucket`` and
    ``sum(kvlen_list) == bucket``. Padded positions follow the real tokens
    of the last request.

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


if TYPE_CHECKING:

    class _BSNDPrefillBuffers(PrefillBucketBuffers):
        bsnd_captured_graphs: dict[int, Any]
        bsnd_output_bufs: dict[int, torch.Tensor]
        bsnd_compact_output_bufs: dict[int, torch.Tensor]
        bsnd_static_attn_metadatas: dict[int, Any]


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
    ) -> None:
        super().__init__(
            model_runner=model_runner,
            buckets=buckets,
        )
        self._update_stream: Any = None
        # A failed device task update may leave the graph/stream unusable.
        # Retain its resources and error until the worker is restarted.
        self._task_update_error: Exception | None = None
        self.entries: dict[int, Any] = {}
        # Opt-in configuration is read once per runner; empty means TND only.
        self._BSND_CAPTURE_SPECS = parse_bsnd_capture_specs(
            self.gr_settings.npu_bsnd_capture_specs,
            self.buckets,
        )
        self.bsnd_entries: dict[tuple[int, int], Any] = {}
        self._replays_by_bucket: dict[int | tuple[int, int], int] = {}
        # FIA workspace shared across every layer AND every bucket: only one
        # prefill graph executes at a time (sequential batches, sequential
        # layers, update stream ordered before replay), and the workspace
        # getter is monotonic in num_tokens. Buckets are captured
        # largest-first, so the first successful capture establishes the max
        # size and smaller buckets reuse it.
        self.shared_workspace: Any = None
        self._logged_first_replay: set[int | tuple[int, int]] = set()
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
    def _supports_beam_prefill_capture(runner: Any) -> bool:
        """Only Beam builders can consume the metadata used by these graphs."""
        from vllm_gr.v1.attention.backends.beam_attn_metadata import (
            BeamAttentionMetadataBuilder,
        )

        attn_groups = getattr(runner, "attn_groups", None)
        if not attn_groups:
            return False
        found_group = False
        for groups in attn_groups:
            for group in groups:
                found_group = True
                if not isinstance(group.get_metadata_builder(0), BeamAttentionMetadataBuilder):
                    return False
        return found_group

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
        # Allocate int32 NPU slots once before any BSND/TND capture.
        # Allocate once: replacing these tensors invalidates the earlier graph's inputs.
        if not getattr(buf, "_npu_capture_inputs_prepared", False):
            position_shape: tuple[int, ...]
            if getattr(runner, "uses_mrope", False):
                position_shape = (3, bucket)
            elif getattr(runner, "uses_xdrope_dim", 0) > 0:
                position_shape = (2, bucket)
            else:
                position_shape = (bucket,)
            buf.positions_buf = torch.empty(position_shape, dtype=torch.int64, device=self.device)
            buf.slot_mapping_buf = torch.empty((bucket,), dtype=torch.int32, device=self.device)
            setattr(buf, "_npu_capture_inputs_prepared", True)
        buf.positions_buf.copy_(torch.arange(bucket, dtype=torch.int64, device=self.device))
        buf.input_ids_buf.fill_(buf.pad_token_id)
        # NPU's ATB ReshapeAndCache op requires int32 slot mappings (the GPU
        # shared buffer is int64). Capture with valid sequential slots: the
        # probe proved all-(-1) is a degenerate setup case, and the real
        # slots are copied in by execute_model_prefill before every replay.
        buf.slot_mapping_buf.copy_(torch.arange(bucket, dtype=torch.int32, device=self.device))
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
        if not is_prefill_graph_enabled(runner):
            logger.info(
                "Prefill NPU Graph capture is disabled by gr.prefill_graph_enabled.",
            )
            return

        if not is_npu_available():
            logger.info("NPU not available; skipping NPU graph capture for prefill.")
            return

        if not self._supports_beam_prefill_capture(runner):
            logger.info("Skipping Beam prefill graphs for native or mixed attention backends.")
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

        # Capture BSND first; TND retains its capture/update ordering.
        self.capture_bsnd_graphs(runner)
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
            except Exception as e:
                # Do not continue with BSND graphs after a failed device capture.
                if self._BSND_CAPTURE_SPECS:
                    raise
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

    def raise_if_failed(self) -> None:
        """Reject further execution after a task update has failed."""
        if self._task_update_error is not None:
            raise RuntimeError(
                "NPU prefill graph task update previously failed; "
                "restart the worker before further model execution."
            ) from self._task_update_error

    def has_captured_graph(self, bucket: int, num_reqs: int = 1) -> bool:
        """Returns True if a graph was captured for this bucket (covers all num_reqs)."""
        self.raise_if_failed()
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
        self.raise_if_failed()
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
            self._task_update_error = e
            logger.exception(
                "NPU prefill graph task update failed for bucket %d; worker restart required.",
                bucket,
            )
            # Do not end the failed update or release graph resources here:
            # the device update state is unknown. Preserve the original error.
            raise
        if not update_ok:
            # False is a preflight rejection before any task update begins.
            logger.warning(
                "NPU prefill graph replay: task-group update unavailable; "
                "falling back to eager prefill."
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

    def match_bsnd_bucket(self, num_reqs: int, max_len: int) -> tuple[int, int] | None:
        # Choose the smallest successfully captured fitting shape, not env order.
        for T, B in self._BSND_CAPTURE_SPECS:
            if B == num_reqs and T // B >= max_len and self.has_bsnd_captured_graph(T, B):
                return T, B
        return None

    def has_bsnd_captured_graph(self, T: int, B: int) -> bool:
        """BSND: True if the BSND graph for (T,B) is captured."""
        self.raise_if_failed()
        buf = self.buffers.get(T)
        return (
            buf is not None
            and getattr(buf, "bsnd_captured_graphs", {}).get(B) is not None
            and (T, B) in self.bsnd_entries
        )

    def _build_bsnd_dummy_metadata(
        self, T: int, B: int, builder: Any, buf: PrefillBucketBuffers
    ) -> Any:
        """BSND: dummy metadata for the BSND (T,B) capture: req_lens=[P]*B.

        qsl_cpu = [0, P, 2P, ...]; prefill_fia_layout="BSND"; slot_mapping
        uses the arange capture buffer (replay rewrites it per batch).
        """
        from vllm_gr.v1.attention.backends.beam_attn_metadata import (
            BeamAttentionMetadata,
        )

        P = T // B
        req_lens = [P] * B
        qsl_cpu = [i * P for i in range(B + 1)]
        qsl_dev = torch.tensor(qsl_cpu, dtype=torch.int32, device=self.device)
        seq_lens_dev = torch.tensor(req_lens, dtype=torch.int32, device=self.device)
        return BeamAttentionMetadata(
            num_actual_tokens=T,
            max_query_len=P,
            query_start_loc=qsl_dev,
            max_seq_len=P,
            seq_lens=seq_lens_dev,
            block_table=None,
            slot_mapping=buf.slot_mapping_buf,
            causal=True,
            npu_causal_atten_mask=getattr(builder, "npu_causal_atten_mask", None),
            is_prefill_graph_exec=True,
            prefill_fia_layout="BSND",
            num_reqs=B,
            num_heads=builder.num_heads_q,
            num_kv_heads=builder.num_kv_heads,
            block_size=builder.block_size,
            query_start_loc_cpu=qsl_cpu,
            seq_lens_cpu=req_lens,
        )

    def capture_bsnd_graphs(self, runner: Any) -> None:
        """BSND: capture the fixed BSND prefill graphs, one per (T,B) spec.

        No task groups / ExternalEvents / layer registration: the fixed FIA
        kwargs (input_layout="BSND", no seq-lens lists, no block_table) are
        shape-bound, so replay needs no per-layer host update.
        """
        # No allocation/capture when disabled; resources are graph-owned.
        if not self._BSND_CAPTURE_SPECS or not is_prefill_graph_enabled(runner):
            return
        if not is_npu_available():
            logger.info("NPU not available; skipping BSND prefill graph capture.")
            return
        if not self._supports_beam_prefill_capture(runner):
            logger.info("Skipping BSND prefill graphs for native or mixed attention backends.")
            return
        try:
            builder, layer_names = self._get_attn_builder_and_layers(runner)
        except Exception as e:  # noqa: BLE001
            logger.warning("BSND capture: cannot locate attention builder: %r.", e)
            return

        from tqdm import tqdm

        graph_pool = torch.npu.graph_pool_handle()
        # Keep BSND resources on this runner-owned bucket.
        # Do not add NPU-only fields to the common bucket definition.
        for T, B in tqdm(
            list(self._BSND_CAPTURE_SPECS), desc="Capturing NPU graphs (prefill BSND)"
        ):
            try:
                if T not in self.buffers:
                    logger.warning(
                        "BSND capture: T=%d not in prefill buckets %s; skipping (T,B)=(%d,%d).",
                        T,
                        self.buckets,
                        T,
                        B,
                    )
                    continue
                buf = cast("_BSNDPrefillBuffers", self.buffers[T])
                if not hasattr(buf, "bsnd_captured_graphs"):
                    buf.bsnd_captured_graphs = {}
                    buf.bsnd_output_bufs = {}
                    buf.bsnd_compact_output_bufs = {}
                    buf.bsnd_static_attn_metadatas = {}
                self._prepare_capture_buffers(buf, T, runner)
                meta = self._build_bsnd_dummy_metadata(T, B, builder, buf)
                meta_dict = {name: meta for name in layer_names}
                with torch.inference_mode():
                    for _ in range(2):
                        with self._forward_context(runner, meta_dict, T):
                            runner.model(input_ids=buf.input_ids_buf, positions=buf.positions_buf)
                    torch.npu.synchronize()

                    graph = torch.npu.NPUGraph()
                    with (
                        self._forward_context(runner, meta_dict, T),
                        torch.npu.graph(graph, pool=graph_pool, capture_error_mode="global"),
                    ):
                        output = runner.model(
                            input_ids=buf.input_ids_buf, positions=buf.positions_buf
                        )
                    torch.npu.synchronize()
                    # Persistent independent destination; no allocation or host
                    # index conversion is needed after graph.replay().
                    compact_output = torch.empty_like(output)
            except Exception as e:
                # Propagate device/capture failures through the startup hook.
                raise RuntimeError(f"BSND capture failed for {(T, B)}; restart required") from e

            buf.bsnd_captured_graphs[B] = graph
            buf.bsnd_output_bufs[B] = output
            buf.bsnd_compact_output_bufs[B] = compact_output
            buf.bsnd_static_attn_metadatas[B] = meta
            # Hold and validate the actual captured inputs.
            capture_inputs = {
                name: getattr(buf, name)
                for name in ("input_ids_buf", "positions_buf", "slot_mapping_buf")
            }
            self.bsnd_entries[(T, B)] = {
                "layout": "BSND",
                "T": T,
                "B": B,
                "capture_inputs": capture_inputs,
                "capture_descriptors": {
                    name: self._bsnd_tensor_descriptor(tensor)
                    for name, tensor in capture_inputs.items()
                },
            }
            logger.info("Captured BSND prefill graph for (T=%d, B=%d).", T, B)

        spec_summary = {k: 1 for k in self.bsnd_entries}
        logger.info(f"BSND prefill graph capture completed. Captured: {spec_summary}")

    @staticmethod
    def _bsnd_tensor_descriptor(tensor: torch.Tensor) -> tuple:
        return (
            tensor.data_ptr(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
            tensor.device,
        )

    def _validate_bsnd_capture_inputs(self, buf: PrefillBucketBuffers, T: int, B: int) -> None:
        entry = self.bsnd_entries[(T, B)]
        for name, original in entry["capture_inputs"].items():
            current = getattr(buf, name)
            if (
                current is not original
                or self._bsnd_tensor_descriptor(current) != entry["capture_descriptors"][name]
            ):
                raise RuntimeError(
                    f"BSND ({T}, {B}) captured input changed: {name}; recapture is required"
                )

    def execute_model_prefill_bsnd(
        self,
        T: int,
        B: int,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        slot_mapping: torch.Tensor | None,
        token_counts: list[int],
    ) -> torch.Tensor | None:
        """Scatter each request into [i*P, i*P+L_i), replay, then compact.

        Padding rows carry pad_token / position 0 / slot -1; the causal mask
        already isolates them (pad key k>=L_i > q is causal-masked), so no
        dynamic padding mask is needed. After replay, real rows are gathered
        into a separate persistent buffer so the caller's logits index extraction
        (cumsum-1) is unchanged.
        """
        self.raise_if_failed()
        buf = self.buffers.get(T)
        if (
            buf is None
            or not self.has_bsnd_captured_graph(T, B)
            or input_ids is None
            or positions is None
            or slot_mapping is None
        ):
            return None
        # Validate packed extents before launching any copies.
        actual = sum(token_counts)
        if input_ids.ndim != 1 or input_ids.shape[0] < actual:
            return None
        if positions.shape[-1] < actual or slot_mapping.ndim != 1 or slot_mapping.shape[0] < actual:
            return None
        if positions.ndim != buf.positions_buf.ndim:
            return None
        P = T // B
        if len(token_counts) != B or any(c <= 0 or c > P for c in token_counts):
            return None

        # Check before any device work; never replay stale pointers.
        self._validate_bsnd_capture_inputs(buf, T, B)
        # Initialize padding before copying the real request regions.
        buf.input_ids_buf.fill_(buf.pad_token_id)
        if buf.positions_buf.ndim == 1:
            buf.positions_buf.fill_(0)
        else:
            buf.positions_buf[..., :].fill_(0)
        buf.slot_mapping_buf.fill_(-1)
        o = 0
        for i, L_i in enumerate(token_counts):
            s = i * P
            buf.input_ids_buf[s : s + L_i].copy_(input_ids[o : o + L_i], non_blocking=True)
            o += L_i
        o = 0
        for i, L_i in enumerate(token_counts):
            s = i * P
            if buf.positions_buf.ndim == 1:
                buf.positions_buf[s : s + L_i].copy_(positions[o : o + L_i], non_blocking=True)
            else:
                buf.positions_buf[..., s : s + L_i].copy_(
                    positions[..., o : o + L_i], non_blocking=True
                )
            buf.slot_mapping_buf[s : s + L_i].copy_(slot_mapping[o : o + L_i], non_blocking=True)
            o += L_i

        buf = cast("_BSNDPrefillBuffers", buf)
        graph = buf.bsnd_captured_graphs[B]
        graph.replay()

        # Avoid Python-list advanced indexing: it may synchronously upload an
        # index tensor after replay, waiting for the entire preceding graph.
        out = buf.bsnd_output_bufs[B]
        compact_output = buf.bsnd_compact_output_bufs[B]
        self._compact_bsnd_output(out, compact_output, P, token_counts)
        return compact_output

    @staticmethod
    def _compact_bsnd_output(out, compact_output, P: int, token_counts: list[int]) -> None:
        """Same-stream D2D copies; source and destination must not alias.

        Only the first sum(token_counts) rows are valid. Copies follow graph
        completion on the replay stream without uploading an index tensor.
        """
        offset = 0
        for row, count in enumerate(token_counts):
            start = row * P
            compact_output[offset : offset + count].copy_(
                out[start : start + count], non_blocking=True
            )
            offset += count

    def log_replay_bsnd(self, T: int, B: int, total_tokens: int) -> None:
        """BSND replay logging (first + every 100)."""
        key = (T, B)
        self._replays_by_bucket.setdefault(key, 0)
        self._replays_by_bucket[key] += 1
        if key not in self._logged_first_replay:
            self._logged_first_replay.add(key)
            logger.info(
                "BSND prefill graph replay active for (T=%d, B=%d, %d actual tokens).",
                T,
                B,
                total_tokens,
            )
        elif self._replays_by_bucket[key] % 100 == 0:
            logger.info(
                "BSND prefill graph replay: (T=%d,B=%d) reached %d replays.",
                T,
                B,
                self._replays_by_bucket[key],
            )


# Parsing is dependency-free and deterministic; no implicit B=4 enablement.
def parse_bsnd_capture_specs(raw: str, buckets) -> tuple[tuple[int, int], ...]:
    specs = set()
    for item in raw.split(";"):
        if not item.strip():
            continue
        fields = item.strip().split(",")
        if len(fields) != 2:
            raise ValueError("BSND specs must use T,B;T,B format")
        try:
            total, batch = map(int, fields)
        except ValueError as error:
            raise ValueError("BSND T and B must be integers") from error
        if total <= 0 or batch not in (1, 2, 4) or total % batch:
            raise ValueError("BSND requires positive T, B in (1,2,4), and T divisible by B")
        if total not in buckets:
            raise ValueError(f"BSND total capacity {total} is not in configured prefill buckets")
        specs.add((total, batch))
    return tuple(sorted(specs))
