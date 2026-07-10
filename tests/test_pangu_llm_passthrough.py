# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"

"""Unit tests for the Pangu LLM passthrough path.

After removing the GR adapter / recommend runtime, ``GRLLM`` is a thin
subclass of vLLM's :class:`~vllm.entrypoints.llm.LLM` that always delegates
construction to the standard vLLM initializer. These tests verify that
Pangu-style checkpoints go through that passthrough path.
"""

import gc
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _has_vllm_engine_runtime() -> bool:
    try:
        import vllm.engine.arg_utils  # noqa: F401
        import vllm.platforms  # noqa: F401
    except Exception:
        return False
    return True


_HAS_VLLM = _has_vllm_engine_runtime()
_SKIP = "vLLM engine/runtime submodules are unavailable in this environment"


@unittest.skipUnless(_HAS_VLLM, _SKIP)
class TestPanguGrllmPassthrough(unittest.TestCase):
    def test_grllm_init_delegates_to_vllm_llm(self):
        """GRLLM should delegate construction to vLLM's LLM.__init__."""
        from vllm_gr.entrypoints.gr import GRLLM

        with mock.patch(
            "vllm.entrypoints.llm.LLM.__init__",
            return_value=None,
        ) as mock_llm_init:
            GRLLM(model="/fake/pangu")

        mock_llm_init.assert_called_once_with(model="/fake/pangu", async_scheduling=False)

    def test_grllm_rejects_async_scheduling(self):
        from vllm_gr.entrypoints.gr import GRLLM

        with self.assertRaisesRegex(ValueError, "async_scheduling=True"):
            GRLLM(model="/fake/pangu", async_scheduling=True)


class TestPanguPluginInitializeRuntime(unittest.TestCase):
    def test_initialize_runtime_does_not_raise(self):
        from vllm_gr import plugin

        with mock.patch("vllm_gr.patch.re_apply_patches", return_value=0):
            state = plugin.initialize_runtime()
        self.assertIn("patch_count", state)


@unittest.skipUnless(_HAS_VLLM, _SKIP)
class TestPanguGrllmShutdown(unittest.TestCase):
    def test_grllm_shutdown_releases_engine(self):
        from vllm_gr.entrypoints.gr import GRLLM

        llm = GRLLM.__new__(GRLLM)
        llm._shutdown_complete = False
        engine = mock.Mock()
        engine.shutdown = mock.Mock()
        llm.llm_engine = engine

        with mock.patch(
            "vllm_gr.utils.npu_adapter.release_accelerator_memory",
            create=True,
        ):
            with mock.patch.object(gc, "collect") as mock_gc:
                GRLLM.shutdown(llm)

        engine.shutdown.assert_called_once_with()
        mock_gc.assert_called_once_with()
        self.assertIsNone(llm.llm_engine)
        self.assertTrue(llm._shutdown_complete)


@unittest.skipUnless(_HAS_VLLM, _SKIP)
class TestGrllmOfflineBeamSearch(unittest.TestCase):
    def test_step_engine_waits_for_logprobs_after_v2_placeholder(self):
        """A V2 chunked-prefill placeholder is not a beam-search result."""
        from vllm_gr.entrypoints.gr import _step_engine_and_collect_outputs

        placeholder = SimpleNamespace(
            request_id="session-0",
            outputs=[
                SimpleNamespace(
                    finish_reason=None,
                    logprobs=SimpleNamespace(token_ids=[]),
                )
            ],
        )
        usable_output = SimpleNamespace(
            request_id="session-0",
            outputs=[
                SimpleNamespace(
                    finish_reason=None,
                    logprobs=SimpleNamespace(token_ids=[10, 11]),
                )
            ],
        )
        step_results = iter([[placeholder], [usable_output]])
        llm_engine = SimpleNamespace(step=lambda: next(step_results))
        instances = [SimpleNamespace(beams=[object()])]

        outputs = _step_engine_and_collect_outputs(
            llm_engine=llm_engine,
            instances_batch=instances,
            token=1,
            prompt_start=0,
            instance_prefill_id={0: "session-0"},
        )

        self.assertEqual(outputs, {"session-0": usable_output})

    def test_step_engine_detects_no_progress_after_v2_placeholder(self):
        from vllm_gr.entrypoints.gr import _step_engine_and_collect_outputs

        placeholder = SimpleNamespace(
            request_id="session-0",
            outputs=[
                SimpleNamespace(
                    finish_reason=None,
                    logprobs=SimpleNamespace(token_ids=[]),
                )
            ],
        )
        llm_engine = SimpleNamespace(step=lambda: [placeholder])
        instances = [SimpleNamespace(beams=[object()])]

        with self.assertRaisesRegex(RuntimeError, "made no progress"):
            _step_engine_and_collect_outputs(
                llm_engine=llm_engine,
                instances_batch=instances,
                token=1,
                prompt_start=0,
                instance_prefill_id={0: "session-0"},
                max_steps_without_usable_output=3,
            )

    def test_step_engine_rejects_terminal_output_without_logprobs(self):
        from vllm_gr.entrypoints.gr import _step_engine_and_collect_outputs

        terminal_output = SimpleNamespace(
            request_id="session-0",
            outputs=[
                SimpleNamespace(
                    finish_reason="length",
                    logprobs=SimpleNamespace(token_ids=[]),
                )
            ],
        )
        llm_engine = SimpleNamespace(step=lambda: [terminal_output])
        instances = [SimpleNamespace(beams=[object()])]

        with self.assertRaisesRegex(RuntimeError, "finished before producing candidate logprobs"):
            _step_engine_and_collect_outputs(
                llm_engine=llm_engine,
                instances_batch=instances,
                token=1,
                prompt_start=0,
                instance_prefill_id={0: "session-0"},
            )

    def test_custom_beam_search_cleans_session_when_engine_step_fails(self):
        from vllm.sampling_params import SamplingParams

        from vllm_gr.entrypoints.gr import _custom_beam_search_batch

        engine_core = mock.Mock()
        llm_engine = SimpleNamespace(engine_core=engine_core)
        instances = [SimpleNamespace(beams=[SimpleNamespace()])]
        request = SimpleNamespace(
            external_req_id="offline-beam-0-0",
            request_id="session-0",
        )

        with (
            mock.patch(
                "vllm_gr.entrypoints.gr._prepare_beam_step_requests",
                return_value=[request],
            ),
            mock.patch(
                "vllm_gr.entrypoints.gr._step_engine_and_collect_outputs",
                side_effect=RuntimeError("step failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "step failed"):
                _custom_beam_search_batch(
                    llm_engine=llm_engine,
                    instances_batch=instances,
                    token_iter=range(1),
                    prompt_start=0,
                    tokenizer=SimpleNamespace(eos_token_id=2),
                    beam_width=1,
                    beam_search_params=SamplingParams(max_tokens=1),
                    sid_end_token_id=None,
                    ignore_eos=False,
                    initial_logprobs=mock.Mock(),
                    logprobs_num=1,
                )

        engine_core.beam_request_step_update.assert_called_once()
        cleanup = engine_core.beam_request_step_update.call_args.args[0]
        self.assertEqual(cleanup.session_id, "session-0")
        self.assertEqual(cleanup.beam_width, 0)

    def test_beam_search_uses_upstream_lora_sequence_helper(self):
        from vllm.sampling_params import BeamSearchParams

        from vllm_gr.entrypoints.gr import GRLLM

        llm = GRLLM.__new__(GRLLM)
        llm._preprocess_cmpl = mock.Mock(
            return_value=[{"type": "token", "prompt_token_ids": [1, 2]}]
        )
        llm._lora_request_to_seq = mock.Mock(side_effect=RuntimeError("sentinel helper call"))

        self.assertFalse(hasattr(llm, "_get_beam_search_lora_requests"))

        with self.assertRaisesRegex(RuntimeError, "sentinel helper call"):
            GRLLM.beam_search(
                llm,
                prompts=[{"prompt_token_ids": [1, 2]}],
                params=BeamSearchParams(beam_width=2, max_tokens=1),
            )

        llm._preprocess_cmpl.assert_called_once_with([{"prompt_token_ids": [1, 2]}])
        llm._lora_request_to_seq.assert_called_once_with(None, 1)

    def test_prepare_beam_step_requests_applies_eos_via_sampling_params(self):
        from vllm.sampling_params import SamplingParams

        from vllm_gr.entrypoints.gr import _prepare_beam_step_requests

        llm_engine = SimpleNamespace(
            output_processor=mock.Mock(),
            engine_core=mock.Mock(),
        )
        instance = SimpleNamespace(beams=[SimpleNamespace(tokens=[10, 20, 30], lora_request=None)])

        engine_core_requests = _prepare_beam_step_requests(
            llm_engine=llm_engine,
            instances_batch=[instance],
            token=1,
            prompt_start=0,
            tokenizer=SimpleNamespace(eos_token_id=7),
            beam_search_params=SamplingParams(
                max_tokens=1,
                temperature=0.0,
                detokenize=False,
            ),
            sid_end_token_id=99,
            instance_prefill_id={0: "session-0"},
            instance_prev_beam_internal_ids={0: ["parent-0"]},
            instance_fork_info={0: [(0, 30)]},
            instance_prefix_len={0: 2},
            instance_completed_sessions=set(),
        )

        self.assertEqual(engine_core_requests, [])
        llm_engine.output_processor.add_request.assert_called_once()
        ec_req = llm_engine.output_processor.add_request.call_args[0][0]
        self.assertEqual(ec_req.request_id, "session-0")
        self.assertEqual(ec_req.prompt_token_ids, [10, 20, 30])
        self.assertEqual(ec_req.sampling_params.eos_token_id, 99)

        llm_engine.engine_core.beam_request_step_update.assert_called_once()
        update = llm_engine.engine_core.beam_request_step_update.call_args[0][0]
        self.assertEqual(update.eos_token_id, 99)
        self.assertEqual(update.beam_tokens, [[30]])

    def test_parse_step_logprobs_rejects_unexpanded_beam_output(self):
        """A single V2-runner logits row must not be split across many beams."""
        from vllm_gr.entrypoints.gr import _parse_step_logprobs

        beam_width = 128
        flat = SimpleNamespace(
            token_ids=list(range(beam_width + 1)),
            logprobs=[-1.0] * (beam_width + 1),
            ranks=[1] * (beam_width + 1),
            decoded_tokens=[None] * (beam_width + 1),
        )
        result = SimpleNamespace(outputs=[SimpleNamespace(finish_reason=None, logprobs=flat)])
        instance = SimpleNamespace(
            beams=[SimpleNamespace(cum_logprob=0.0, tokens=[]) for _ in range(beam_width)]
        )

        with self.assertRaisesRegex(ValueError, "invalid logprobs shape"):
            _parse_step_logprobs(
                result=result,
                instance=instance,
                token=1,
                logprobs_num=beam_width,
                fork_info_j=[(idx, idx) for idx in range(beam_width)],
            )

    def test_parse_step_logprobs_accepts_expanded_beam_output(self):
        """A correctly expanded grouped output is split once per parent beam."""
        from vllm_gr.entrypoints.gr import _parse_step_logprobs

        flat = SimpleNamespace(
            token_ids=[10, 10, 11, 20, 20, 21],
            logprobs=[-0.1, -0.1, -0.2, -0.3, -0.3, -0.4],
            ranks=[1, 1, 2, 1, 1, 2],
            decoded_tokens=[None] * 6,
        )
        result = SimpleNamespace(outputs=[SimpleNamespace(finish_reason=None, logprobs=flat)])
        instance = SimpleNamespace(
            beams=[
                SimpleNamespace(cum_logprob=0.0, tokens=[]),
                SimpleNamespace(cum_logprob=-1.0, tokens=[]),
            ]
        )

        is_error, token_ids, logprobs, beam_cache = _parse_step_logprobs(
            result=result,
            instance=instance,
            token=1,
            logprobs_num=2,
            fork_info_j=[(0, 10), (1, 20)],
        )

        self.assertFalse(is_error)
        self.assertEqual(token_ids, [10, 11, 20, 21])
        self.assertEqual(logprobs, [-0.1, -0.2, -1.3, -1.4])
        self.assertEqual(len(beam_cache), 2)


if __name__ == "__main__":
    unittest.main()
