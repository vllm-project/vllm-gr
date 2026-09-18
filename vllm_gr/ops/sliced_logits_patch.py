# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime installer for sliced-vocabulary logits projection."""

import torch
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
            if hasattr(self, "_gr_vocab_slice"):
                window = self._gr_vocab_slice
                if window is None:
                    return original_get_logits(self, hidden_states, lm_head, embedding_bias)
                return compute_sliced_logits(
                    hidden_states,
                    lm_head,
                    embedding_bias,
                    offset=window[0],
                    slice_size=window[1],
                    logits_processor=self,
                    orig_get_logits=original_get_logits,
                )
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

        # V2 ModelRunner: sampling lives in the new worker-local module, whose
        # ``compute_topk_logprobs`` does the same torch.topk the V1
        # ``gather_logprobs`` clamp above protects. With a narrow catalog
        # vocabulary slice the logits have far fewer columns than the request's
        # num_logprobs (e.g. an impossible 3-token catalog slices to a single
        # column), and the unclamped topk raises "selected index k out of
        # range". The callers bind the function into their own module
        # namespace (``from ... import compute_topk_logprobs``), so the clamp
        # must be installed on every importer's binding, not just the defining
        # module.
        def _make_clamped_v2_topk(original_v2_topk):
            def patched_compute_topk_logprobs(logits, num_logprobs, *args, **kwargs):
                if logits is not None and torch.is_tensor(logits) and logits.numel() > 0:
                    num_logprobs = min(num_logprobs, logits.size(-1))
                return original_v2_topk(logits, num_logprobs, *args, **kwargs)

            return patched_compute_topk_logprobs

        try:
            from vllm.v1.worker.gpu.sample import logprob as v2_logprob

            v2_logprob.compute_topk_logprobs = _make_clamped_v2_topk(
                v2_logprob.compute_topk_logprobs
            )
        except Exception as exc:
            logger.debug("vllm_gr: Could not patch V2 compute_topk_logprobs: %s", exc)

        for _v2_module_name in (
            "vllm.v1.worker.gpu.sample.sampler",
            "vllm.v1.worker.gpu.spec_decode.rejection_sampler",
        ):
            try:
                import importlib

                _v2_module = importlib.import_module(_v2_module_name)
                _bound = getattr(_v2_module, "compute_topk_logprobs", None)
                if callable(_bound):
                    _v2_module.compute_topk_logprobs = _make_clamped_v2_topk(_bound)
            except Exception as exc:
                logger.debug(
                    "vllm_gr: Could not patch %s.compute_topk_logprobs: %s",
                    _v2_module_name,
                    exc,
                )

        _SLICED_LOGITS_PATCHED = True
        logger.info("vllm_gr: Patched LogitsProcessor._get_logits for sliced vocabulary projection")
        return True
    except Exception as exc:
        logger.debug("vllm_gr: Skipping sliced-logits patch: %s", exc)
        return False
