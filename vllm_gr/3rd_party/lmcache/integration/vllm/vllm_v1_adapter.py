# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from typing import Optional

from vllm.logger import init_logger

logger = init_logger(__name__)


def _temporary_filter_requests(metadata, predicate):
    if not hasattr(metadata, "requests"):
        return None

    original_requests = metadata.requests
    filtered_requests = [request for request in original_requests if not predicate(request)]
    if len(filtered_requests) == len(original_requests):
        return None

    metadata.requests = filtered_requests
    return original_requests


def patch_installed_lmcache_adapter() -> None:
    try:
        adapter = importlib.import_module("lmcache.integration.vllm.vllm_v1_adapter")
    except ModuleNotFoundError:
        logger.warning("Installed lmcache v1 adapter not found; skipping lmcache runtime patch.")
        return
    except Exception as exc:
        logger.exception(
            "Unable to import installed lmcache adapter for runtime patch.",
            exc_info=exc,
        )
        return

    def patch_request_tracker_from_new_request():
        def from_new_request(
            lmcache_config,
            new_request,
            num_tokens_to_compute,
            lmcache_cached_tokens,
            skip_save,
        ):
            unfolded_block_ids = []
            if len(new_request.block_ids) == 0:
                unfolded_block_ids = []
            elif not isinstance(new_request.block_ids[0], list):
                unfolded_block_ids = new_request.block_ids.copy()
            else:
                unfolded_block_ids = new_request.block_ids[0].copy()

            disagg_spec = adapter.tmp_disagg_tracker.pop(new_request.req_id, None)
            request_configs = adapter.extract_request_configs(new_request.sampling_params)
            mm_hashes, mm_positions = adapter.extract_mm_features(new_request, modify=True)

            return adapter.RequestTracker(
                req_id=new_request.req_id,
                prompt_len=len(new_request.prompt_token_ids),
                token_ids=new_request.prompt_token_ids[:num_tokens_to_compute].copy(),
                allocated_block_ids=unfolded_block_ids,
                num_saved_tokens=lmcache_cached_tokens,
                disagg_spec=disagg_spec,
                mm_hashes=mm_hashes,
                mm_positions=mm_positions,
                skip_save=skip_save,
                request_configs=request_configs,
            )

        adapter.RequestTracker.from_new_request = staticmethod(from_new_request)

    def patch_reqmeta_from_request_tracker():
        original = adapter.ReqMeta.from_request_tracker

        def from_request_tracker(
            tracker,
            block_size,
            lmcache_chunk_size=256,
            load_spec=None,
            discard_partial_chunks=True,
            save_decode_cache=False,
        ):
            if tracker.allocated_block_ids is None or len(tracker.allocated_block_ids) == 0:
                adapter.logger.debug(
                    "Request %s has no allocated blocks (likely a newly forked beam request). "
                    "Skipping KV cache management for this request.",
                    tracker.req_id,
                )
                return None
            return original(
                tracker,
                block_size,
                lmcache_chunk_size,
                load_spec,
                discard_partial_chunks,
                save_decode_cache,
            )

        adapter.ReqMeta.from_request_tracker = staticmethod(from_request_tracker)
        if hasattr(adapter.ReqMeta, "__annotations__"):
            adapter.ReqMeta.__annotations__["slot_mapping"] = Optional[object]

    def patch_lmcache_connector_v1_impl():
        original_start_load_kv = adapter.LMCacheConnectorV1Impl.start_load_kv
        original_save_kv_layer = adapter.LMCacheConnectorV1Impl.save_kv_layer
        original_wait_for_save = adapter.LMCacheConnectorV1Impl.wait_for_save

        def start_load_kv(self, forward_context, **kwargs):
            metadata = self._parent._get_connector_metadata()
            old_requests = _temporary_filter_requests(
                metadata,
                lambda request: request.slot_mapping is None
                and getattr(request, "load_spec", None) is not None
                and getattr(request.load_spec, "can_load", False),
            )
            try:
                return original_start_load_kv(self, forward_context, **kwargs)
            finally:
                if old_requests is not None:
                    metadata.requests = old_requests

        def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
            connector_metadata = self._parent._get_connector_metadata()
            old_requests = _temporary_filter_requests(
                connector_metadata,
                lambda request: request.slot_mapping is None,
            )
            try:
                return original_save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs)
            finally:
                if old_requests is not None:
                    connector_metadata.requests = old_requests

        def wait_for_save(self):
            connector_metadata = self._parent._get_connector_metadata()
            old_requests = _temporary_filter_requests(
                connector_metadata,
                lambda request: request.slot_mapping is None,
            )
            try:
                return original_wait_for_save(self)
            finally:
                if old_requests is not None:
                    connector_metadata.requests = old_requests

        adapter.LMCacheConnectorV1Impl.start_load_kv = start_load_kv
        adapter.LMCacheConnectorV1Impl.save_kv_layer = save_kv_layer
        adapter.LMCacheConnectorV1Impl.wait_for_save = wait_for_save

    patch_request_tracker_from_new_request()
    patch_reqmeta_from_request_tracker()
    patch_lmcache_connector_v1_impl()
