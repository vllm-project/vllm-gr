# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime installer for sliced-vocabulary logits projection."""

from vllm.logger import init_logger

from vllm_gr.ops.sliced_logits import (
    compute_sliced_logits,
    is_vocab_slice_enabled,
    set_vocab_slice_config,
)

logger = init_logger(__name__)

_SLICED_LOGITS_PATCHED = False


def _restore_vocab_slice_config() -> None:
    from vllm.config import get_current_vllm_config_or_none

    vllm_config = get_current_vllm_config_or_none()
    additional_config = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional_config, dict):
        return

    offset = additional_config.get("vllm_gr_vocab_offset")
    slice_size = additional_config.get("vllm_gr_vocab_slice_size")
    if offset is not None and slice_size is not None:
        set_vocab_slice_config(offset, slice_size)


def apply_sliced_logits_patch() -> bool:
    """Install sliced-logits hooks and report whether this call applied them."""
    global _SLICED_LOGITS_PATCHED

    if _SLICED_LOGITS_PATCHED:
        return False

    try:
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from vllm.v1.sample.sampler import Sampler

        _restore_vocab_slice_config()

        original_init = LogitsProcessor.__init__

        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if not is_vocab_slice_enabled():
                _restore_vocab_slice_config()

        LogitsProcessor.__init__ = patched_init

        original_get_logits = LogitsProcessor._get_logits

        def patched_get_logits(self, hidden_states, lm_head, embedding_bias):
            if is_vocab_slice_enabled():
                return compute_sliced_logits(
                    hidden_states,
                    lm_head,
                    embedding_bias,
                    logits_processor=self,
                    orig_get_logits=original_get_logits,
                )
            return original_get_logits(self, hidden_states, lm_head, embedding_bias)

        LogitsProcessor._get_logits = patched_get_logits

        try:
            original_gather = Sampler.gather_logprobs

            @staticmethod
            def patched_gather_logprobs(logprobs, num_logprobs, token_ids, *args, **kwargs):
                if logprobs is not None and logprobs.numel() > 0:
                    num_logprobs = min(num_logprobs, logprobs.size(-1))
                return original_gather(logprobs, num_logprobs, token_ids, *args, **kwargs)

            Sampler.gather_logprobs = patched_gather_logprobs
        except Exception as exc:
            logger.debug("vllm_gr: Could not patch Sampler.gather_logprobs: %s", exc)

        _SLICED_LOGITS_PATCHED = True
        logger.info("vllm_gr: Patched LogitsProcessor._get_logits for sliced vocabulary projection")
        return True
    except Exception as exc:
        logger.debug("vllm_gr: Skipping sliced-logits patch: %s", exc)
        return False
