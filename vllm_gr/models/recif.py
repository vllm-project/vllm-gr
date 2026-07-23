# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""RECIF Qwen3-MoE model integration for vLLM.

The training checkpoint contains:

* ``_model_rank0.pt``: a headless Megatron Qwen3-MoE backbone.
* ``external_rank0.pt``: three independent SID heads under ``lm_heads``.

Each item is represented by four tokens::

    [ctx=0, sid0, sid1 + 8192, sid2 + 16384]

For vLLM, the three ``Linear(hidden_size, 8192)`` heads are concatenated into
one ``lm_head.weight`` of shape ``[24576, hidden_size]``. During hierarchical
beam search, ``allowed_token_ids`` restricts each generation step to the
corresponding 8192-token range. This is mathematically equivalent to applying
one of the original three heads and normalizing over its 8192 classes.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

VOCAB = 8192
NUM_SID_LEVELS = 3
FULL_VOCAB = NUM_SID_LEVELS * VOCAB
RECIF_ARCHITECTURE = "RecifForCausalLM"

_ARCHITECTURE_KEYS = (
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "hidden_size",
    "num_experts",
)


# vLLM is optional at import time so that checkpoint conversion helpers can be
# unit-tested in a lightweight environment. Instantiating the model still
# requires vLLM.
try:
    from vllm.model_executor.models.qwen3_moe import (
        Qwen3MoeForCausalLM as _VllmQwen3MoeForCausalLM,
    )
except ImportError:  # pragma: no cover - only used by lightweight unit tests

    class _VllmQwen3MoeForCausalLM(nn.Module):
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            super().__init__()
            raise RuntimeError(
                "vLLM is required to instantiate RecifForCausalLM"
            )


class RecifForCausalLM(_VllmQwen3MoeForCausalLM):
    """Qwen3-MoE RECIF model backed by vLLM's native implementation.

    The backbone, paged attention, KV cache, tensor parallelism and fused MoE
    execution are inherited from vLLM. This class only converts the original
    Megatron checkpoint and merges the three external SID heads.
    """

    # The model directory contains two PyTorch checkpoint files instead of the
    # normal Hugging Face safetensors layout.
    allow_patterns_overrides = ["_model_rank0.pt", "external_rank0.pt"]

    # Kept explicitly for model inspection/registry tooling. The base class has
    # the same mapping.
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }

    def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_text_config

        # 保存checkpoint路径，供load_weights加载external_rank0.pt使用
        self.model_path = vllm_config.model_config.model

        # The embedding table and the merged LM head both use three disjoint
        # 8192-token SID ranges. Keep this aligned with training.
        if int(config.vocab_size) != FULL_VOCAB:
            raise ValueError(
                f"RECIF requires vocab_size={FULL_VOCAB}, "
                f"got {config.vocab_size}."
            )
        if bool(getattr(config, "tie_word_embeddings", False)):
            raise ValueError(
                "RECIF requires tie_word_embeddings=false because the SID "
                "heads are independent from the input embedding table."
            )

        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
        )


    def load_weights(
        self,
        weights: Iterable[
            tuple[str, torch.Tensor | Mapping[str, torch.Tensor]]
        ],
    ) -> set[str]:

        arch = architecture_from_config(self.config)

        # vLLM默认只会传入主模型权重(_model_rank0.pt)
        # 手动补充RECIF额外的SID heads
        weights = list(weights)

        external_path = os.path.join(
            self.model_path,
            "external_rank0.pt",
        )

        if os.path.exists(external_path):
            external = torch.load(
                external_path,
                map_location="cpu",
                weights_only=False,
            )

            weights.append(
                (
                    "lm_heads",
                    external["lm_heads"],
                )
            )

        prepared = prepare_vllm_weights(
            weights,
            arch,
        )

        return super().load_weights(prepared)


def architecture_from_config(config: Any) -> dict[str, int]:
    """Extract the architecture fields needed by Megatron weight conversion."""

    architecture: dict[str, int] = {}
    for key in _ARCHITECTURE_KEYS:
        value = getattr(config, key, None)
        if value is None:
            raise ValueError(f"RECIF config is missing required field: {key}")
        architecture[key] = int(value)
    return architecture


def _split_qkv(qkv_w, num_heads, num_kv_heads, head_dim, hidden):
    """Megatron interleaved [nkv, (hpg+2)*hd, hidden] -> q, k, v."""
    hpg = num_heads // num_kv_heads
    group = (hpg + 2) * head_dim
    qs = hpg * head_dim
    qkv = qkv_w.reshape(num_kv_heads, group, hidden)
    q = qkv[:, :qs, :].reshape(-1, hidden).contiguous()
    k = qkv[:, qs:qs + head_dim, :].reshape(-1, hidden).contiguous()
    v = qkv[:, qs + head_dim:, :].reshape(-1, hidden).contiguous()
    return q, k, v

def _split_gated(fc1):
    """Megatron GatedMLP fc1 [2*inter, hidden] -> gate, up."""
    inter = fc1.shape[0] // 2
    return fc1[:inter, :].contiguous(), fc1[inter:, :].contiguous()


def megatron_to_hf(sd, arch):
    """Map a (single-rank) Megatron backbone state_dict to HF Qwen3-MoE keys."""
    L = arch['num_hidden_layers']
    nh, nkv, hd, hs = (arch['num_attention_heads'], arch['num_key_value_heads'],
                       arch['head_dim'], arch['hidden_size'])
    ne = arch['num_experts']
    hf = {}
    hf['model.embed_tokens.weight'] = sd['embedding.word_embeddings.weight']
    hf['model.norm.weight'] = sd['decoder.final_layernorm.weight']
    for i in range(L):
        pm, ph = f'decoder.layers.{i}', f'model.layers.{i}'
        q, k, v = _split_qkv(sd[f'{pm}.self_attention.linear_qkv.weight'], nh, nkv, hd, hs)
        hf[f'{ph}.self_attn.q_proj.weight'] = q
        hf[f'{ph}.self_attn.k_proj.weight'] = k
        hf[f'{ph}.self_attn.v_proj.weight'] = v
        hf[f'{ph}.self_attn.o_proj.weight'] = sd[f'{pm}.self_attention.linear_proj.weight']
        hf[f'{ph}.self_attn.q_norm.weight'] = sd[f'{pm}.self_attention.q_layernorm.weight']
        hf[f'{ph}.self_attn.k_norm.weight'] = sd[f'{pm}.self_attention.k_layernorm.weight']
        hf[f'{ph}.input_layernorm.weight'] = sd[f'{pm}.self_attention.linear_qkv.layer_norm_weight']
        hf[f'{ph}.post_attention_layernorm.weight'] = sd[f'{pm}.pre_mlp_layernorm.weight']
        hf[f'{ph}.mlp.gate.weight'] = sd[f'{pm}.mlp.router.weight']
        for e in range(ne):
            g, u = _split_gated(sd[f'{pm}.mlp.experts.linear_fc1.weight{e}'])
            hf[f'{ph}.mlp.experts.{e}.gate_proj.weight'] = g
            hf[f'{ph}.mlp.experts.{e}.up_proj.weight'] = u
            hf[f'{ph}.mlp.experts.{e}.down_proj.weight'] = \
                sd[f'{pm}.mlp.experts.linear_fc2.weight{e}']
    return hf


def _collect_head_mapping(
    value: Mapping[str, torch.Tensor],
    destination: dict[str, torch.Tensor],
) -> None:
    """Collect head tensors from an ``lm_heads`` checkpoint mapping."""

    for raw_name, head_weight in value.items():
        if not isinstance(head_weight, torch.Tensor):
            raise TypeError(f"RECIF head {raw_name!r} is not a tensor")

        name = str(raw_name)
        if name.startswith("lm_heads."):
            name = name[len("lm_heads.") :]
        destination[name] = head_weight


def prepare_vllm_weights(
    weights: Iterable[
        tuple[str, torch.Tensor | Mapping[str, torch.Tensor]]
    ],
    arch: Mapping[str, int],
) -> list[tuple[str, torch.Tensor]]:

    megatron: dict[str, torch.Tensor] = {}
    passthrough: list[tuple[str, torch.Tensor]] = []
    head_weights: dict[str, torch.Tensor] = {}
    has_merged_lm_head = False

    for raw_name, value in weights:
        name = str(raw_name)

        if name.endswith("_extra_state"):
            continue

        if name == "lm_heads":
            if not isinstance(value, Mapping):
                raise TypeError(
                    "external_rank0.pt['lm_heads'] must be a mapping"
                )
            _collect_head_mapping(value, head_weights)
            continue

        if name.startswith("lm_heads."):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"RECIF head {name!r} is not a tensor")
            head_weights[name[len("lm_heads.") :]] = value
            continue

        if name.startswith("heads."):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"RECIF head {name!r} is not a tensor")
            head_weights[name] = value
            continue

        if name.startswith("embedding.") or name.startswith("decoder."):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Megatron weight {name!r} is not a tensor")
            megatron[name] = value
            continue

        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Weight {name!r} is not a tensor")

        if name == "lm_head.weight":
            has_merged_lm_head = True
        passthrough.append((name, value))

    if megatron:
        passthrough.extend(megatron_to_hf(megatron, arch).items())

    if head_weights:
        required_names = [
            f"heads.{level}.weight" for level in range(NUM_SID_LEVELS)
        ]
        missing = [name for name in required_names if name not in head_weights]
        if missing:
            raise RuntimeError(
                f"Missing RECIF SID head weights: {', '.join(missing)}"
            )
        if has_merged_lm_head:
            raise RuntimeError(
                "Checkpoint contains both lm_head.weight and separate RECIF "
                "SID heads; refusing to load ambiguous output weights."
            )

        hidden_size = int(arch["hidden_size"])
        expected_shape = (VOCAB, hidden_size)
        ordered_heads: list[torch.Tensor] = []
        for name in required_names:
            weight = head_weights[name]
            if tuple(weight.shape) != expected_shape:
                raise ValueError(
                    f"Unexpected {name} shape {tuple(weight.shape)}; "
                    f"expected {expected_shape}"
                )
            ordered_heads.append(weight)

        merged_head = torch.cat(ordered_heads, dim=0).contiguous()
        passthrough.append(("lm_head.weight", merged_head))
        has_merged_lm_head = True

    if not has_merged_lm_head:
        raise RuntimeError(
            "Missing RECIF output weights: expected external lm_heads or "
            "an already merged lm_head.weight"
        )

    return passthrough


def sid_to_bytes(v):
    """int64 sid -> (sa, sb, sc), each in [0, 8191]. sid = sa | sb<<14 | sc<<28."""
    v = int(v)
    return v & 0x1FFF, (v >> 14) & 0x1FFF, (v >> 28) & 0x1FFF


def bytes_to_sid(sa, sb, sc):
    return int(sa) | (int(sb) << 14) | (int(sc) << 28)


def build_prefix(history_sids, device):
    """User history (list of int64 SIDs) -> token tensor [1, 4K+1].

    Per item: [ctx=0, sa, sb+8192, sc+16384]. Trailing ctx=0 = target slot.
    """
    toks = []
    for v in history_sids:
        sa, sb, sc = sid_to_bytes(v)
        toks += [0, sa, sb + VOCAB, sc + 2 * VOCAB]
    toks += [0]  # target item's ctx slot
    return torch.tensor([toks], dtype=torch.long, device=device)

def _logprob_value(value: Any) -> float:
    """Read a log probability from vLLM Logprob objects or numeric stubs."""

    if hasattr(value, "logprob"):
        return float(value.logprob)
    return float(value)


def _topk_logprobs(
    logprobs: Mapping[int, Any] | Sequence[tuple[int, Any]] | None,
    topk: int,
) -> list[tuple[int, float]]:
    """Normalize vLLM logprob output and return descending top-k entries."""

    if logprobs is None:
        raise RuntimeError(
            "vLLM returned no logprobs; SamplingParams.logprobs must be set"
        )

    if isinstance(logprobs, Mapping):
        raw_items = logprobs.items()
    else:
        raw_items = logprobs

    items = [
        (int(token_id), _logprob_value(value))
        for token_id, value in raw_items
    ]
    items.sort(key=lambda item: item[1], reverse=True)
    return items[:topk]


def _first_token_logprobs(request_output: Any) -> Mapping[int, Any]:
    """Extract first generated token logprobs from a vLLM RequestOutput."""

    if not getattr(request_output, "outputs", None):
        raise RuntimeError("vLLM returned no completion output")

    completion = request_output.outputs[0]
    logprobs = getattr(completion, "logprobs", None)
    if not logprobs:
        raise RuntimeError("vLLM returned no generated-token logprobs")
    return logprobs[0]


def _level_token_ids(level: int) -> list[int]:
    if level < 0 or level >= NUM_SID_LEVELS:
        raise ValueError(f"Invalid RECIF SID level: {level}")
    start = level * VOCAB
    return list(range(start, start + VOCAB))


def beam_search(
    llm: Any,
    history_sids: Sequence[int],
    bf: tuple[int, int, int] = (8, 8, 8),
    beam: int = 64,
):
    """Run RECIF's three-level beam search through vLLM scheduling."""

    from vllm.sampling_params import SamplingParams

    prefix = build_prefix(history_sids, "cpu")[0].tolist()
    beams = [(prefix, (), 0.0)]

    for level in range(NUM_SID_LEVELS):
        params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            logprobs=bf[level],
            allowed_token_ids=_level_token_ids(level),
            detokenize=False,
        )
        outputs = llm.generate(
            [{"prompt_token_ids": tokens} for tokens, _, _ in beams],
            sampling_params=params,
            use_tqdm=False,
        )

        candidates = []
        offset = level * VOCAB
        end = offset + VOCAB
        for (tokens, sid, score), output in zip(beams, outputs):
            for token_id, logprob in _topk_logprobs(
                _first_token_logprobs(output), bf[level]
            ):
                if not offset <= token_id < end:
                    raise RuntimeError(
                        "RECIF received a logprob outside the current SID "
                        f"level: level={level}, token_id={token_id}. Construct "
                        "vllm.LLM with logprobs_mode='processed_logprobs' so "
                        "allowed_token_ids is applied before log-softmax."
                    )
                candidates.append(
                    (
                        tokens + [token_id],
                        sid + (token_id - offset,),
                        score + logprob,
                    )
                )

        candidates.sort(key=lambda item: item[2], reverse=True)
        beams = candidates[: bf[0] if level == 0 else beam]

    triples = torch.tensor([sid for _, sid, _ in beams], dtype=torch.long)
    scores = torch.tensor([score for _, _, score in beams], dtype=torch.float32)
    return triples.numpy(), scores.numpy()


def prepare_beam_search_inputs(
    llm: Any,
    prompts: Sequence[Any],
):
    """Convert batched int64 SID histories to tokenizer-free vLLM prompts."""

    from vllm.inputs import TokensPrompt

    from vllm_gr.entrypoints.beam_search_config import PreparedBeamInputs

    token_prompts: list[TokensPrompt] = []
    for index, prompt in enumerate(prompts):
        if not isinstance(prompt, Mapping) or "history_sids" not in prompt:
            raise TypeError(
                "RECIF prompts must contain 'history_sids'; "
                f"invalid prompt at index {index}"
            )
        history_sids = prompt["history_sids"]
        if not isinstance(history_sids, Sequence) or isinstance(
            history_sids, (str, bytes)
        ):
            raise TypeError("RECIF history_sids must be a sequence of integers")
        token_ids = build_prefix(history_sids, "cpu")[0].tolist()
        token_prompts.append(TokensPrompt(prompt_token_ids=token_ids))

    engine_inputs = list(llm._preprocess_cmpl(token_prompts))
    generated_starts = []
    for prompt in engine_inputs:
        decoder_prompt = (
            prompt if prompt["type"] != "enc_dec" else prompt["decoder_prompt"]
        )
        generated_starts.append(len(decoder_prompt["prompt_token_ids"]))
    return PreparedBeamInputs(
        engine_inputs=engine_inputs,
        generated_starts=generated_starts,
        output_starts=list(generated_starts),
        prompt_texts=[""] * len(engine_inputs),
    )


def build_beam_search_steps(params: Any):
    """Build RECIF's three per-level expansion and token-range policies."""

    from vllm_gr.entrypoints.beam_search_config import BeamSearchStep

    branch_factors = getattr(params, "branch_factors", None) or (8, 8, 8)
    if len(branch_factors) != NUM_SID_LEVELS:
        raise ValueError("RECIF branch_factors must contain exactly three values")
    if any(value < 1 or value > VOCAB for value in branch_factors):
        raise ValueError("RECIF branch_factors values must be in [1, 8192]")

    steps: list[BeamSearchStep] = []
    for level, expand_width in enumerate(branch_factors):
        start = level * VOCAB
        keep_width = branch_factors[0] if level == 0 else params.beam_width
        steps.append(
            BeamSearchStep(
                expand_width=expand_width,
                keep_width=keep_width,
                allowed_token_ids=list(range(start, start + VOCAB)),
            )
        )
    return steps


def finalize_beam_search_outputs(
    instances: list[Any],
    prepared: Any,
    initial_logprobs: Any,
    beam_width: int,
):
    """Convert each winning three-token RECIF path back to an int64 SID."""

    from vllm.entrypoints.generate.beam_search.utils import BeamSearchOutput

    from vllm_gr.entrypoints.beam_search_config import select_best_beams
    from vllm_gr.logprobs import reconstruct_beam_logprobs

    outputs = []
    for index, instance in enumerate(instances):
        best_beams = select_best_beams(instance, beam_width)
        prompt_len = prepared.output_starts[index]
        for result in best_beams:
            reconstruct_beam_logprobs(result, initial_logprobs)
            generated_tokens = result.tokens[prompt_len:]
            if len(generated_tokens) != NUM_SID_LEVELS:
                raise RuntimeError(
                    "RECIF beam search must generate three SID levels"
                )
            sid = bytes_to_sid(
                generated_tokens[0],
                generated_tokens[1] - VOCAB,
                generated_tokens[2] - 2 * VOCAB,
            )
            result.text = str(sid)
            setattr(result, "sid", sid)
        outputs.append(BeamSearchOutput(sequences=best_beams))
    return outputs


def register_recif_model() -> None:
    """Register the custom architecture before constructing ``vllm.LLM``."""

    try:
        from vllm import ModelRegistry
    except ImportError:  # pragma: no cover - optional in helper-only tests
        return

    ModelRegistry.register_model(
        RECIF_ARCHITECTURE,
        "vllm_gr.models.recif:RecifForCausalLM",
    )


__all__ = [
    "FULL_VOCAB",
    "NUM_SID_LEVELS",
    "RECIF_ARCHITECTURE",
    "RecifForCausalLM",
    "VOCAB",
    "architecture_from_config",
    "beam_search",
    "build_prefix",
    "build_beam_search_steps",
    "bytes_to_sid",
    "megatron_to_hf",
    "finalize_beam_search_outputs",
    "prepare_beam_search_inputs",
    "prepare_vllm_weights",
    "register_recif_model",
    "sid_to_bytes",
]
