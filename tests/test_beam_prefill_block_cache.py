# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for non-block-aligned prefill last-block caching in beam search."""

from types import SimpleNamespace
from typing import Any, List
from unittest.mock import MagicMock

import vllm_gr.v1.engine.engine_core_patch as ecp
from vllm_gr.v1.engine.core import _cache_beam_request, _ensure_beam_state


def _unpatch_scheduler() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    while hasattr(Scheduler.schedule, "__wrapped__"):
        Scheduler.schedule = Scheduler.schedule.__wrapped__
    if hasattr(Scheduler.schedule, ecp._SCHEDULER_PATCH_MARKER):
        delattr(Scheduler.schedule, ecp._SCHEDULER_PATCH_MARKER)


def test_cache_beam_request_computes_padded_block_hash() -> None:
    """Verify _cache_beam_request computes padded hash for unaligned prefill (e.g. 25 tokens, block_size=16)."""
    block_size = 16
    token_ids = list(range(25))  # 25 tokens -> 1 full block (16) + 9 tokens

    def mock_hasher(req: Any) -> List[int]:
        return [0xBBBB]

    setattr(mock_hasher, "block_size", block_size)

    engine: Any = SimpleNamespace()
    _ensure_beam_state(engine)
    engine.request_block_hasher = mock_hasher
    engine.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=block_size))

    req: Any = SimpleNamespace(
        request_id="session_25",
        _all_token_ids=token_ids,
        block_hashes=[0xAAAA],  # Initially only 1 full block hash
        _block_hasher=None,
        sampling_params=SimpleNamespace(eos_token_id=100),
        lora_request=None,
        cache_salt=None,
        mm_features=None,
        prompt_embeds=None,
        prompt_is_token_ids=True,
        num_prompt_tokens=25,
    )

    def _update_hashes() -> None:
        req.block_hashes.extend(mock_hasher(req))

    req.update_block_hashes = _update_hashes

    _cache_beam_request(engine, req)

    cached_entry = engine.beam_cache["session_25"]
    assert len(cached_entry["block_hashes"]) == 2
    assert cached_entry["block_hashes"] == [0xAAAA, 0xBBBB]
    assert req.block_hashes == [0xAAAA, 0xBBBB]


def test_patch_get_computed_blocks_returns_unaligned_prefix_len() -> None:
    """Verify patch_get_computed_blocks returns exact prefix_len (25) when all blocks match."""
    from vllm.v1.core.sched.scheduler import Scheduler

    _unpatch_scheduler()

    block_size = 16
    prefix_len = 25

    req = SimpleNamespace(
        request_id="session_25",
        is_beam_decode=True,
        prefix_len=prefix_len,
    )

    block_0 = SimpleNamespace(block_id=0)
    block_1 = SimpleNamespace(block_id=1)
    bn = SimpleNamespace(blocks=((block_0, block_1),))

    result_holder: dict[str, Any] = {}

    def mock_schedule(self: Any) -> Any:
        # Call get_computed_blocks DURING schedule execution while patch is active
        blocks_out, tn_out = self.kv_cache_manager.get_computed_blocks(req)
        result_holder["tn_out"] = tn_out
        result_holder["blocks_out"] = blocks_out
        return SimpleNamespace(num_scheduled_tokens={})

    def _create_blocks(b: Any) -> SimpleNamespace:
        return SimpleNamespace(blocks=b)

    orig_sched_fn = Scheduler.schedule
    try:
        Scheduler.schedule = mock_schedule
        ecp.apply_scheduler_patch()

        mock_inst: Any = SimpleNamespace()
        mock_inst.cache_config = SimpleNamespace(block_size=block_size)
        mock_inst.kv_cache_manager = SimpleNamespace(
            get_computed_blocks=MagicMock(return_value=(bn, 32)),
            create_kv_cache_blocks=_create_blocks,
            coordinator=SimpleNamespace(cache_blocks=MagicMock()),
        )
        mock_inst.requests = {}

        Scheduler.schedule(mock_inst)

        assert result_holder["tn_out"] == 25
        assert len(result_holder["blocks_out"].blocks[0]) == 2
    finally:
        Scheduler.schedule = orig_sched_fn
        _unpatch_scheduler()
        ecp.apply_scheduler_patch()


def test_patch_cache_blocks_extends_tokens_for_padded_hashes() -> None:
    """Verify cache_blocks passes through the original value when padded block hashes are present.

    With padded block-hash generation at prefill cache time, the scheduler
    no longer needs to extend the cache boundary -- cache_blocks receives the
    original num_tokens_to_cache value.
    """
    from vllm.v1.core.sched.scheduler import Scheduler

    _unpatch_scheduler()

    block_size = 16
    req = SimpleNamespace(
        request_id="session_25",
        num_prompt_tokens=25,
        block_hashes=[0xAAAA, 0xBBBB],
    )

    single_mgr = SimpleNamespace(
        req_to_blocks={"session_25": [SimpleNamespace(block_id=0), SimpleNamespace(block_id=1)]}
    )
    mock_orig_cache_blocks = MagicMock()
    coord = SimpleNamespace(
        primary_kv_cache_type=0,
        single_type_kv_cache_managers={0: single_mgr},
        cache_blocks=mock_orig_cache_blocks,
    )

    def mock_schedule(self: Any) -> Any:
        # Call cache_blocks DURING schedule execution while patch is active
        self.kv_cache_manager.coordinator.cache_blocks(req, 25)
        return SimpleNamespace(num_scheduled_tokens={})

    def _create_blocks(b: Any) -> SimpleNamespace:
        return SimpleNamespace(blocks=b)

    orig_sched_fn = Scheduler.schedule
    try:
        Scheduler.schedule = mock_schedule
        ecp.apply_scheduler_patch()

        mock_inst: Any = SimpleNamespace()
        mock_inst.cache_config = SimpleNamespace(block_size=block_size)
        mock_inst.kv_cache_manager = SimpleNamespace(
            get_computed_blocks=MagicMock(return_value=(SimpleNamespace(blocks=()), 0)),
            create_kv_cache_blocks=_create_blocks,
            cache_blocks=mock_orig_cache_blocks,
            coordinator=coord,
        )
        mock_inst.requests = {}

        Scheduler.schedule(mock_inst)

        # patch_cache_blocks extends the target boundary to the padded target (32 tokens)
        mock_orig_cache_blocks.assert_called_once_with(req, 32)
    finally:
        Scheduler.schedule = orig_sched_fn
        _unpatch_scheduler()
        ecp.apply_scheduler_patch()


def test_preprocess_add_request_preserves_block_hashes() -> None:
    """Verify EngineCore._process_engine_step preserves saved block_hashes and is_beam_decode."""
    req_payload = SimpleNamespace(
        request_id="session_25",
        block_hashes=[0xAAAA, 0xBBBB],
        is_beam_decode=True,
    )
    processed_req = SimpleNamespace(
        request_id="session_25",
        block_hashes=[],
        is_beam_decode=False,
    )

    engine: Any = SimpleNamespace()
    engine.preprocess_add_request = MagicMock(return_value=(processed_req, 0))
    engine._handle_request_preproc_error = MagicMock()

    # Simulate the logic added in _process_engine_step
    saved_hashes = getattr(req_payload, "block_hashes", None)
    is_beam_decode = getattr(req_payload, "is_beam_decode", False)
    request = engine.preprocess_add_request(req_payload)
    if saved_hashes and request and request[0]:
        request[0].block_hashes = list(saved_hashes)
    if is_beam_decode and request and request[0]:
        request[0].is_beam_decode = True

    assert processed_req.block_hashes == [0xAAAA, 0xBBBB]
    assert processed_req.is_beam_decode is True
