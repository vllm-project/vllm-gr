# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest
from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence
from vllm.inputs import tokens_input
from vllm.logprobs import FlatLogprobs

from vllm_gr.constraints import BeamConstraintCoordinator
from vllm_gr.entrypoints.gr import GRLLM


class _Tokenizer:
    eos_token_id = 2

    def __init__(self) -> None:
        self.decoded: list[list[int]] = []

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|sid_begin|>": 1, "<|sid_end|>": 3}[token]

    def decode(self, tokens: list[int]) -> str:
        self.decoded.append(list(tokens))
        return "".join({1: "<|sid_begin|>", 3: "<|sid_end|>"}.get(t, f"[{t}]") for t in tokens)


class _BatchTokenizer(_Tokenizer):
    def batch_decode(self, tokens: list[list[int]]) -> list[str]:
        return [self.decode(row) for row in tokens]


@pytest.mark.parametrize(  # type: ignore[misc]
    "output_mode,batch_decode,include_prompt",
    [("full", batch, prompt) for batch in (False, True) for prompt in (None, False, True)]
    # Compact output never decodes text; one tokenizer/text configuration is enough.
    + [("compact", False, None)],
)
@pytest.mark.parametrize("begin", [None, "<|sid_begin|>"])  # type: ignore[misc]
@pytest.mark.parametrize("end", [None, "<|sid_end|>"])  # type: ignore[misc]
@pytest.mark.parametrize("stop_at_eos", [False, True])  # type: ignore[misc]
def test_offline_output_uses_original_prompt_boundary(
    batch_decode: bool,
    include_prompt: bool | None,
    begin: str | None,
    end: str | None,
    stop_at_eos: bool,
    output_mode: str,
) -> None:
    tokenizer = _BatchTokenizer() if batch_decode else _Tokenizer()
    original_tokens = [10, 11]
    prepared = tokens_input(original_tokens)
    next_token = tokenizer.eos_token_id if stop_at_eos else 20

    def generate(prompts: Any, **kwargs: Any) -> list[SimpleNamespace]:
        flat = FlatLogprobs(
            start_indices=[0],
            end_indices=[2],
            token_ids=[next_token, next_token],
            logprobs=[-0.2, -0.2],
            ranks=[1, 1],
            decoded_tokens=[f"[{next_token}]", f"[{next_token}]"],
        )
        return [SimpleNamespace(outputs=[SimpleNamespace(logprobs=flat)]) for _ in prompts]

    runner: Any = SimpleNamespace(
        get_tokenizer=lambda: tokenizer,
        _preprocess_cmpl=lambda prompts: [prepared],
        _lora_request_to_seq=lambda request, count: [None] * count,
        constraint_strategy=None,
        constraint_coordinator=BeamConstraintCoordinator(None),
        llm_engine=SimpleNamespace(vllm_config=None),
        generate=generate,
    )
    params: Any = SimpleNamespace(
        beam_width=1,
        max_tokens=1 + int(begin is not None) + int(end is not None),
        temperature=1.0,
        ignore_eos=False,
        begin_token=begin,
        end_token=end,
    )
    kwargs: dict[str, Any] = (
        {} if include_prompt is None else {"decode_prompt_in_text": include_prompt}
    )
    kwargs["output_mode"] = output_mode
    outputs = GRLLM.beam_search(runner, [{"prompt_token_ids": original_tokens}], params, **kwargs)
    beam = outputs[0].sequences[0]
    generated = ([1] if begin else []) + [next_token] + ([3] if end and not stop_at_eos else [])
    if output_mode == "compact":
        assert beam.token_ids == tuple(generated)
        assert beam.cumulative_logprob == pytest.approx(-0.2)
        assert tokenizer.decoded == []
        assert prepared["prompt_token_ids"] == [10, 11]
        return
    expected_tail = (begin or "") + f"[{next_token}]" + ((end or "") if not stop_at_eos else "")
    assert beam.tokens == original_tokens + generated
    # The output assembly restores full sequences in place and must drop the
    # shared-prefix / logprob-reconstruction bookkeeping attributes that
    # dataclasses.replace used to shed by rebuilding instances.
    assert not any(
        hasattr(beam, attr) for attr in ("_prompt_token_ids", "_lp_parent", "_lp_step_data")
    )
    assert beam.text == ("[10][11]" if include_prompt is not False else "") + expected_tail
    assert generated in tokenizer.decoded
    assert (original_tokens in tokenizer.decoded) == (include_prompt is not False)
    assert list(beam.logprobs.token_ids) == generated
    assert beam.cum_logprob == pytest.approx(-0.2)
    assert prepared["prompt_token_ids"] == [10, 11]


def _lazy_runner(tokenizer: _Tokenizer, original_tokens: list[int]) -> Any:
    prepared = tokens_input(original_tokens)

    def generate(prompts: Any, **kwargs: Any) -> list[SimpleNamespace]:
        flat = FlatLogprobs(
            start_indices=[0],
            end_indices=[2],
            token_ids=[20, 21],
            logprobs=[-0.1, -0.2],
            ranks=[1, 2],
            decoded_tokens=["[20]", "[21]"],
        )
        return [SimpleNamespace(outputs=[SimpleNamespace(logprobs=flat)]) for _ in prompts]

    return SimpleNamespace(
        get_tokenizer=lambda: tokenizer,
        _preprocess_cmpl=lambda prompts: [prepared],
        _lora_request_to_seq=lambda request, count: [None] * count,
        constraint_strategy=None,
        constraint_coordinator=BeamConstraintCoordinator(None),
        llm_engine=SimpleNamespace(vllm_config=None),
        generate=generate,
    )


def _lazy_params(beam_width: int = 2) -> Any:
    return SimpleNamespace(
        beam_width=beam_width,
        max_tokens=1,
        temperature=1.0,
        ignore_eos=False,
        begin_token=None,
        end_token=None,
    )


def test_lazy_beams_materialize_only_when_read() -> None:
    tokenizer = _Tokenizer()
    original_tokens = [10, 11]
    runner = _lazy_runner(tokenizer, original_tokens)

    outputs = GRLLM.beam_search(runner, [{"prompt_token_ids": original_tokens}], _lazy_params())
    seqs = outputs[0].sequences
    assert len(seqs) == 2

    # Returning from beam_search only decodes the prompt once; neither beam's
    # tail has been detokenized or reconstructed yet.
    assert tokenizer.decoded == [original_tokens]
    assert all(getattr(seq, "_lazy_materializer", None) is not None for seq in seqs)

    # Reading a non-deferred field (cum_logprob) must not materialize.
    _ = seqs[0].cum_logprob
    assert tokenizer.decoded == [original_tokens]

    # First read of sequences[0] materializes exactly that beam: full token
    # restore + one tail decode, then the plain dataclass fast path returns.
    assert seqs[0].tokens == [10, 11, 20]
    assert tokenizer.decoded == [original_tokens, [20]]
    assert seqs[0].__class__ is BeamSearchSequence
    # Reconstructed logprobs = empty init (no sid_begin) + the per-step
    # candidate segment [20, 21] kept for downstream rescoring.
    assert list(seqs[0].logprobs.token_ids) == [20, 21]

    # sequences[1] is still lazy until read.
    assert getattr(seqs[1], "_lazy_materializer", None) is not None
    assert tokenizer.decoded == [original_tokens, [20]]

    assert seqs[1].text == "[10][11][21]"
    assert tokenizer.decoded == [original_tokens, [20], [21]]
    assert seqs[1].__class__ is BeamSearchSequence
    assert seqs[1].tokens == [10, 11, 21]


def test_prompt_text_is_decoded_once_per_tokenizer() -> None:
    tokenizer = _Tokenizer()
    original_tokens = [10, 11]
    runner = _lazy_runner(tokenizer, original_tokens)
    prompt = [{"prompt_token_ids": original_tokens}]

    GRLLM.beam_search(runner, prompt, _lazy_params(beam_width=1))
    assert tokenizer.decoded.count(original_tokens) == 1
    assert "_gr_prompt_text_cache" in tokenizer.__dict__

    # Second request with the same prompt hits the per-tokenizer LRU; with
    # lazy beams and nothing read yet, the second call decodes NOTHING.
    GRLLM.beam_search(runner, prompt, _lazy_params(beam_width=1))
    assert tokenizer.decoded.count(original_tokens) == 1
    assert len(tokenizer.decoded) == 1

    # Reading the beam still detokenizes its tail (outside the cache).
    beam = GRLLM.beam_search(runner, prompt, _lazy_params(beam_width=1))[0].sequences[0]
    assert tokenizer.decoded.count(original_tokens) == 1  # still cached
    assert beam.tokens == [10, 11, 20]
    assert tokenizer.decoded[-1] == [20]


def test_compact_output_matches_full_without_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_gr.entrypoints.gr as gr

    tokenizer = _Tokenizer()
    # More requested beams than distinct candidates also checks no padding.
    params = _lazy_params(beam_width=4)
    prompt = [{"prompt_token_ids": [10, 11]}]
    full = GRLLM.beam_search(_lazy_runner(tokenizer, [10, 11]), prompt, params)
    expected = [(tuple(seq.tokens[2:]), seq.cum_logprob) for seq in full[0].sequences]

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("compact output must not reconstruct logprobs or decode text")

    monkeypatch.setattr(gr, "reconstruct_beam_logprobs_batch", forbidden)
    monkeypatch.setattr(gr, "fast_decode", forbidden)
    monkeypatch.setattr(gr, "_prompt_text_cached", forbidden)
    compact = GRLLM.beam_search(
        _lazy_runner(_Tokenizer(), [10, 11]), prompt, params, output_mode="compact"
    )
    assert len(compact[0].sequences) == 2
    assert [(seq.token_ids, seq.cumulative_logprob) for seq in compact[0].sequences] == expected


def test_invalid_output_mode_fails_before_inference() -> None:
    with pytest.raises(ValueError, match="output_mode"):
        GRLLM.beam_search(SimpleNamespace(), [], _lazy_params(), output_mode="invalid")


def _p2d_root(initial_logprobs: FlatLogprobs) -> BeamSearchSequence:
    root = BeamSearchSequence(
        orig_prompt=tokens_input([10, 11]),
        tokens=[],
        logprobs=initial_logprobs,
        lora_request=None,
        cum_logprob=0.0,
    )
    root._prompt_token_ids = [10, 11, 1]
    root._lp_parent = None
    root._lp_step_data = None
    return root


def test_materialize_p2d_final_beams_offline() -> None:
    from vllm_gr.entrypoints.gr import _materialize_p2d_final_beams
    from vllm_gr.logprobs import reconstruct_beam_logprobs_batch
    from vllm_gr.v1.worker.prefill_graph_runner import P2D_SENTINEL_RANK

    # begin token id 1 seeds the shared initial logprobs; beams carry three
    # generated tokens each.  The third beam is a worker-marked dead beam.
    initial_logprobs = FlatLogprobs()
    initial_logprobs.append_fast([1], [0.0], [None], [None])
    flat = FlatLogprobs(
        start_indices=[0],
        end_indices=[9],
        token_ids=[20, 21, 22, 30, 31, 32, -1, -1, -1],
        logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -10.0, -10.0, -10.0],
        ranks=[P2D_SENTINEL_RANK] * 9,
        decoded_tokens=[None] * 9,
    )
    root = _p2d_root(initial_logprobs)

    beams = _materialize_p2d_final_beams(
        flat=flat,
        root=root,
        beam_width=3,
        num_steps=3,
        ignore_eos=False,
        eos_token_id=2,
        sid_end_token_id=3,
    )

    # Dead beam skipped; live beams carry the full tail including sid_end.
    assert [beam.tokens for beam in beams] == [[20, 21, 22, 3], [30, 31, 32, 3]]
    assert [beam.cum_logprob for beam in beams] == [pytest.approx(-0.6), pytest.approx(-1.5)]
    assert [beam.finish_reason for beam in beams] == [None, None]

    # Lazy logprob reconstruction replays the same per-step structure as the
    # engine-driven fork loop: begin + one segment per generated token + end.
    reconstruct_beam_logprobs_batch(beams, initial_logprobs, sid_end_token_id=3)
    assert list(beams[0].logprobs.token_ids) == [1, 20, 21, 22, 3]
    assert list(beams[0].logprobs.logprobs) == [
        pytest.approx(v) for v in (0.0, -0.1, -0.2, -0.3, 0.0)
    ]

    # EOS pruning: a beam whose second token is EOS terminates there, drops the
    # post-EOS tokens, and never appends the session-end marker.
    eos_flat = FlatLogprobs(
        start_indices=[0],
        end_indices=[3],
        token_ids=[20, 2, 22],
        logprobs=[-0.1, -0.2, -0.3],
        ranks=[P2D_SENTINEL_RANK] * 3,
        decoded_tokens=[None] * 3,
    )
    eos_beams = _materialize_p2d_final_beams(
        flat=eos_flat,
        root=root,
        beam_width=1,
        num_steps=3,
        ignore_eos=False,
        eos_token_id=2,
        sid_end_token_id=3,
    )
    assert eos_beams[0].tokens == [20, 2]
    assert eos_beams[0].cum_logprob == pytest.approx(-0.3)
    assert eos_beams[0].finish_reason == "stop"
    reconstruct_beam_logprobs_batch(eos_beams, initial_logprobs, sid_end_token_id=3)
    assert list(eos_beams[0].logprobs.token_ids) == [1, 20, 2]

    # Nested first-row unwrap (serving-engine wire shape) is accepted.
    nested = FlatLogprobs(
        start_indices=[0],
        end_indices=[3],
        token_ids=[[20, 21, 22]],
        logprobs=[[-0.1, -0.2, -0.3]],
        ranks=[P2D_SENTINEL_RANK],
        decoded_tokens=[None],
    )
    nested_beams = _materialize_p2d_final_beams(
        flat=nested,
        root=root,
        beam_width=1,
        num_steps=3,
        ignore_eos=False,
        eos_token_id=2,
        sid_end_token_id=None,
    )
    assert nested_beams[0].tokens == [20, 21, 22]
    assert nested_beams[0].cum_logprob == pytest.approx(-0.6)
