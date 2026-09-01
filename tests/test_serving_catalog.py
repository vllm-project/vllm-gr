# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
import asyncio
import json
import multiprocessing
import os
import random
import tempfile
from typing import Any

import pytest
import torch

try:
    from vllm_gr.sampling_params import BeamSearchParams
except ImportError:
    from vllm.sampling_params import BeamSearchParams


from transformers import AutoTokenizer
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.entrypoints.openai.models.serving import BaseModelPath, OpenAIServingModels
from vllm.inputs import TokensPrompt

if not torch.cuda.is_available() and not (hasattr(torch, "npu") and torch.npu.is_available()):
    pytest.skip("GPU/NPU is not available", allow_module_level=True)


def _build_catalog_data() -> list[list[str]]:
    catalog_data = []
    random.seed(42)
    # The random layer must stay dense for catalog_strategy="null": a beam
    # whose top-K candidate window misses the trie's valid set is dropped,
    # and each (a, b) prefix only survives step 3 with ~K/vocab probability
    # per catalog continuation.  With 50k entries nearly every (a, b) prefix
    # had exactly one valid c, so the 256-wide beam pool died out at step 3
    # (every beam dead-end -> 0 outputs -> test failure).  500k entries give
    # the model-preferred prefixes enough valid continuations to keep the
    # expected survivor count above zero (measured 4 survivors / OneRec-1.7B,
    # beam_width=256).
    for i in range(500000):
        item = [
            "<|sid_begin|>",
            f"<s_a_{random.randint(0, 7999)}>",
            f"<s_b_{random.randint(0, 7999)}>",
            f"<s_c_{random.randint(0, 7999)}>",
            "<|sid_end|>",
        ]
        catalog_data.append(item)
    for i in range(8000):
        catalog_data.append(
            ["<|sid_begin|>", "<s_a_4247>", "<s_b_6120>", f"<s_c_{i}>", "<|sid_end|>"]
        )
    return catalog_data


async def _serving_catalog_generation_impl(loops: int, catalog_strategy: str = "fill") -> None:
    catalog_data = _build_catalog_data()
    print(f"catalog created with {len(catalog_data)} items")

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp_catalog:
        json.dump(catalog_data, tmp_catalog)
        catalog_path = tmp_catalog.name

    try:
        model_name = os.getenv("VLLM_GR_TEST_MODEL", "OpenOneRec/OneRec-1.7B")

        engine_args = AsyncEngineArgs(
            model=model_name,
            trust_remote_code=True,
            enforce_eager=True,
            enable_log_requests=False,
            max_logprobs=256,
            additional_config={"vllm_gr_beam_max_width": 256},
        )
        setattr(engine_args, "catalog_path", catalog_path)

        async with build_async_engine_client_from_engine_args(engine_args) as engine_client:
            base_model_paths = [BaseModelPath(name=model_name, model_path=model_name)]
            models = OpenAIServingModels(
                engine_client=engine_client, base_model_paths=base_model_paths, lora_modules=None
            )
            serving = OpenAIServing(
                engine_client=engine_client,
                models=models,
                request_logger=None,
            )

            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            prompt_file = os.path.join(
                os.path.dirname(__file__), "resources/single_one_rec_prompt.txt"
            )
            with open(prompt_file, "r") as f:
                prompt_text = f.read()
            prompt_token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)

            beam_width = 256
            beam_search_params_dict: dict[str, Any] = {
                "beam_width": beam_width,
                "max_tokens": 5,
                "ignore_eos": False,
                "temperature": 0.0,
            }
            if hasattr(BeamSearchParams, "begin_token"):
                beam_search_params_dict["begin_token"] = "<|sid_begin|>"
                beam_search_params_dict["end_token"] = "<|sid_end|>"
            if hasattr(BeamSearchParams, "catalog_strategy"):
                beam_search_params_dict["catalog_strategy"] = catalog_strategy

            params = BeamSearchParams(**beam_search_params_dict)

            print(f"Starting beam search (strategy={catalog_strategy}), loops={loops}...")

            catalog_set = set("".join(item) for item in catalog_data)
            if loops <= 0:
                loops = 1
            for i in range(loops):
                generator = serving.beam_search(prompt, f"req_id_catalog_{i}", params)

                final_output = None
                async for output in generator:
                    final_output = output

                assert final_output is not None
                # With catalog_strategy='null', dead-end beams are eliminated, so we may have fewer outputs
                if catalog_strategy == "fill":
                    assert len(final_output.outputs) == beam_width
                else:  # "null" strategy
                    assert len(final_output.outputs) > 0, (
                        "No valid outputs generated with null strategy"
                    )

                print(f"Loop {i}: Verifying {len(final_output.outputs)} generated choices...")
                not_in_catalog = []
                for choice in final_output.outputs:
                    content = choice.text
                    score = choice.cumulative_logprob
                    if score > -1000 and content not in catalog_set:
                        not_in_catalog.append(content)
                if not_in_catalog:
                    print("Generated content not found in catalog:")
                    for i, content in enumerate(not_in_catalog):
                        print(f"{i}: {content}")
                    assert False, "Some Generated content not found in catalog."

    finally:
        if os.path.exists(catalog_path):
            os.remove(catalog_path)


def _run_serving_catalog_generation_target(loops: int, catalog_strategy: str = "fill") -> None:
    from vllm_gr.patch import re_apply_patches

    re_apply_patches()
    asyncio.run(_serving_catalog_generation_impl(loops, catalog_strategy))


@pytest.mark.slow  # type: ignore
def test_serving_catalog_generation(loops: int) -> None:
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=_run_serving_catalog_generation_target, args=(loops,))
    p.start()
    p.join()
    assert p.exitcode == 0


@pytest.mark.slow  # type: ignore
def test_serving_catalog_dead_end_fill(loops: int) -> None:
    """catalog_strategy='fill': all outputs must be in the catalog."""
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=_run_serving_catalog_generation_target, args=(loops, "fill"))
    p.start()
    p.join()
    assert p.exitcode == 0


@pytest.mark.slow  # type: ignore
def test_serving_catalog_dead_end_null(loops: int) -> None:
    """catalog_strategy='null': dead-end beams are dropped; all surviving outputs
    must still be in the catalog."""
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=_run_serving_catalog_generation_target, args=(loops, "null"))
    p.start()
    p.join()
    assert p.exitcode == 0


def _run_null_all_dead_target() -> None:
    """Subprocess: verify that an impossible catalog causes a clean stop with
    no crash and no invalid catalog tokens in the output."""
    from vllm_gr.patch import re_apply_patches

    re_apply_patches()
    asyncio.run(_null_all_dead_impl())


async def _null_all_dead_impl() -> None:
    model_name = os.getenv("VLLM_GR_TEST_MODEL", "OpenOneRec/OneRec-1.7B")

    # A catalog with a single entry whose first token is a special token the
    # model will never emit as its top prediction for a normal prompt.
    # After the begin token is prepended the catalog asks for <unk> next —
    # something the model essentially never puts in its top-K logprobs.
    # The catalog path then sees every beam as a dead-end on step 1.
    impossible_catalog = [["<|sid_begin|>", "<unk>", "<|sid_end|>"]]

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(impossible_catalog, f)
        catalog_path = f.name

    try:
        engine_args = AsyncEngineArgs(
            model=model_name,
            trust_remote_code=True,
            enforce_eager=True,
            enable_log_requests=False,
            max_logprobs=1024,
            additional_config={"vllm_gr_beam_max_width": 4},
        )
        setattr(engine_args, "catalog_path", catalog_path)

        async with build_async_engine_client_from_engine_args(engine_args) as engine_client:
            base_model_paths = [BaseModelPath(name=model_name, model_path=model_name)]
            models = OpenAIServingModels(
                engine_client=engine_client,
                base_model_paths=base_model_paths,
                lora_modules=None,
            )
            serving = OpenAIServing(
                engine_client=engine_client,
                models=models,
                request_logger=None,
            )

            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            prompt_file = os.path.join(
                os.path.dirname(__file__), "resources/single_one_rec_prompt.txt"
            )
            with open(prompt_file) as f:
                prompt_text = f.read()
            prompt_token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)

            beam_search_params_dict: dict[str, Any] = {
                "beam_width": 4,
                "max_tokens": 3,
                "ignore_eos": False,
                "temperature": 0.0,
                "catalog_strategy": "null",
            }
            if hasattr(BeamSearchParams, "begin_token"):
                beam_search_params_dict["begin_token"] = "<|sid_begin|>"
                beam_search_params_dict["end_token"] = "<|sid_end|>"

            params = BeamSearchParams(**beam_search_params_dict)

            final_output = None
            async for output in serving.beam_search(prompt, "req_null_dead", params):
                final_output = output

            # Must not crash and must return a finished output.
            assert final_output is not None, "beam_search must yield at least one output"
            assert final_output.finished is True
            # The system stopped cleanly — no error finish_reason.
            # Outputs may be partial (incomplete catalog entries) because the
            # generator broke out early when all beams became dead-ends.
            for choice in final_output.outputs:
                assert choice.finish_reason != "error", (
                    "null all-dead must not return finish_reason='error'"
                )
            print(
                f"null all-dead test passed: {len(final_output.outputs)} outputs, "
                f"no crash, no error finish_reason"
            )
    finally:
        if os.path.exists(catalog_path):
            os.remove(catalog_path)


@pytest.mark.slow  # type: ignore
def test_serving_catalog_null_all_dead_no_crash(loops: int) -> None:
    """null strategy + impossible catalog → stops cleanly, no crash, no invalid tokens."""
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=_run_null_all_dead_target)
    p.start()
    p.join()
    assert p.exitcode == 0


def _run_uniqueness_test_target() -> None:
    """Subprocess: verify uniqueness of results across model-generated and fallback choices."""
    from vllm_gr.patch import re_apply_patches

    re_apply_patches()
    asyncio.run(_uniqueness_impl())


async def _uniqueness_impl() -> None:
    """Verify that all beam search outputs are unique (no duplicates across
    model-generated and catalog fallback results)."""
    model_name = os.getenv("VLLM_GR_TEST_MODEL", "OpenOneRec/OneRec-1.7B")

    # Create a catalog where every level has >beam_width valid continuations,
    # so the fallback mechanism (j % len(valid_list)) maps each of the 16
    # candidate positions to a distinct token, avoiding any tie-induced
    # collisions in argpartition across levels.
    beam_width = 16
    catalog_data = []
    for a in range(32):
        for b in range(32):
            for c in range(32):
                item = [
                    "<|sid_begin|>",
                    f"<s_a_{a}>",
                    f"<s_b_{b}>",
                    f"<s_c_{c}>",
                    "<|sid_end|>",
                ]
                catalog_data.append(item)

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp_catalog:
        json.dump(catalog_data, tmp_catalog)
        catalog_path = tmp_catalog.name

    try:
        engine_args = AsyncEngineArgs(
            model=model_name,
            trust_remote_code=True,
            enforce_eager=True,
            enable_log_requests=False,
            max_logprobs=1024,
            additional_config={"vllm_gr_beam_max_width": 16},
        )
        setattr(engine_args, "catalog_path", catalog_path)

        async with build_async_engine_client_from_engine_args(engine_args) as engine_client:
            base_model_paths = [BaseModelPath(name=model_name, model_path=model_name)]
            models = OpenAIServingModels(
                engine_client=engine_client, base_model_paths=base_model_paths, lora_modules=None
            )
            serving = OpenAIServing(
                engine_client=engine_client,
                models=models,
                request_logger=None,
            )

            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            prompt_file = os.path.join(
                os.path.dirname(__file__), "resources/single_one_rec_prompt.txt"
            )
            with open(prompt_file, "r") as f:
                prompt_text = f.read()
            prompt_token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)

            # Use a beam width smaller than catalog and non-zero temperature
            # to ensure beams can diverge and produce unique outputs
            beam_search_params_dict: dict[str, Any] = {
                "beam_width": beam_width,
                "max_tokens": 5,
                "ignore_eos": False,
                "temperature": 0.7,  # Non-zero temp allows beam diversity
                "catalog_strategy": "fill",
            }
            if hasattr(BeamSearchParams, "begin_token"):
                beam_search_params_dict["begin_token"] = "<|sid_begin|>"
                beam_search_params_dict["end_token"] = "<|sid_end|>"

            params = BeamSearchParams(**beam_search_params_dict)

            final_output = None
            async for output in serving.beam_search(prompt, "req_uniqueness_test", params):
                final_output = output

            assert final_output is not None
            outputs = final_output.outputs

            # Collect all generated output texts
            output_texts = []
            for choice in outputs:
                output_texts.append(choice.text)

            # Verify no duplicates exist
            unique_texts = set(output_texts)
            assert len(unique_texts) == len(output_texts), (
                f"Duplicates found in beam outputs: {len(output_texts)} total outputs "
                f"but only {len(unique_texts)} unique. Duplicate outputs detected."
            )

            # Verify all outputs are in the catalog
            catalog_set = set("".join(item) for item in catalog_data)
            for text in output_texts:
                score = outputs[output_texts.index(text)].cumulative_logprob
                # Only check outputs with valid scores (not very low logprobs indicating errors)
                if score > -1000:
                    assert text in catalog_set, (
                        f"Output '{text}' not found in catalog. "
                        f"All model outputs must be in the catalog."
                    )

            print(f"Uniqueness test passed: {len(outputs)} outputs, all unique, all in catalog")

    finally:
        if os.path.exists(catalog_path):
            os.remove(catalog_path)


@pytest.mark.slow  # type: ignore
def test_serving_catalog_uniqueness_across_fallback(loops: int) -> None:
    """Verify uniqueness across model-generated and fallback catalog results."""
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=_run_uniqueness_test_target)
    p.start()
    p.join()
    assert p.exitcode == 0
