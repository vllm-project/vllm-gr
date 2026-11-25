# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Offline CUDA beam search over explicit per-stage token IDs."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.helpers.paths import REPO_ROOT

pytestmark = [pytest.mark.gpu_test, pytest.mark.l3, pytest.mark.model_test, pytest.mark.slow]

_WIDTH = 500
_STAGE_IDS = [list(range(10, 1010, 2)), [1018, 1013], [1020]]


def _prepare_model(path: Path) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    vocab_size = 1024
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=False,
        )
    )
    # Zero attention/MLP updates and an embedding of [8, 0, ...] make
    # the final normalized first feature exactly 8 in FP16 for every input.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.model.embed_tokens.weight[:, 0] = 8
        model.model.norm.weight.fill_(1)
        model.lm_head.weight[_STAGE_IDS[0], 0] = torch.arange(1, _WIDTH + 1) / 256
        model.lm_head.weight[_STAGE_IDS[1], 0] = torch.tensor([33 / 1024, 32 / 1024])
        model.lm_head.weight[0, 0] = 100
    model.eval().save_pretrained(path)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({f"tok{token}": token for token in range(vocab_size)}, unk_token="tok0")
        ),
        unk_token="tok0",
        bos_token="tok1",
        eos_token="tok2",
        pad_token="tok0",
    )
    tokenizer.save_pretrained(path)


def _check_offline(model: str) -> None:
    from vllm_gr import GRLLM
    from vllm_gr.sampling_params import BeamSearchParams

    with GRLLM(
        model=model,
        attention_config={"backend": "CUSTOM"},
        dtype="float16",
        enforce_eager=True,
        async_scheduling=True,
        max_model_len=64,
        max_num_batched_tokens=1024,
        max_num_seqs=4,
        gpu_memory_utilization=0.1,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        max_logprobs=_WIDTH,
        gr={
            "beam_width": [_WIDTH] * 3,
            "candidate_width": [_WIDTH] * 3,
            "max_decode_steps": 3,
            "constraint_mode": "none",
            "worker_decision": True,
            "beam_graph_enabled": False,
            "prefill_graph_enabled": False,
            "sid_stage_token_ids": _STAGE_IDS,
        },
    ) as llm:
        (output,) = llm.beam_search(
            [{"prompt_token_ids": [4, 5, 7]}],
            BeamSearchParams(beam_width=_WIDTH, max_tokens=3, temperature=0.0, ignore_eos=True),
            output_mode="compact",
            use_tqdm=False,
        )

    a_scores = (torch.arange(1, _WIDTH + 1) / 32).log_softmax(0)
    b_scores = torch.tensor([33 / 128, 32 / 128]).log_softmax(0)
    raw = a_scores[:, None] + b_scores
    order = raw.flatten().argsort(descending=True)[:_WIDTH]
    expected = [
        (_STAGE_IDS[0][index // 2], _STAGE_IDS[1][index % 2], _STAGE_IDS[2][0])
        for index in order.tolist()
    ]
    assert [beam.token_ids for beam in output.sequences] == expected
    assert [beam.cumulative_logprob for beam in output.sequences] == pytest.approx(
        raw.flatten()[order].tolist(), abs=2e-5
    )


@pytest.mark.timeout(420)  # type: ignore[misc]
def test_explicit_sparse_stage_domains_support_width_500(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    _prepare_model(tmp_path)
    env = dict(
        os.environ,
        VLLM_USE_V2_MODEL_RUNNER="0",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        HF_HUB_OFFLINE="1",
        PYTHONPATH=os.pathsep.join([str(REPO_ROOT), os.environ.get("PYTHONPATH", "")]),
    )
    subprocess.run(
        [sys.executable, __file__, str(tmp_path)],
        env=env,
        check=True,
        timeout=390,
    )


if __name__ == "__main__":
    _check_offline(sys.argv[1])
