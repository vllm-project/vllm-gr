# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
import asyncio
import json
import multiprocessing
import os
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
    # A large random catalog is both expensive and statistically fragile: a
    # sparse random (a, b) prefix can still lose every continuation under the
    # ``null`` strategy.  These values are the leading deterministic
    # continuations for this file's fixed OneRec prompt.  Their Cartesian
    # product keeps 32 valid branches at every trie level, including each
    # observed high-probability path, using 32**3 entries instead of 508k.
    a_values = (
        271,
        2912,
        4362,
        4785,
        4005,
        169,
        3285,
        2939,
        8165,
        2315,
        4392,
        7121,
        3398,
        978,
        6121,
        3326,
        5621,
        526,
        5557,
        2025,
        2811,
        7538,
        4462,
        2429,
        4866,
        1629,
        7592,
        5908,
        7414,
        2206,
        5183,
        2967,
    )
    b_values = (
        5945,
        1346,
        3417,
        6020,
        7295,
        5491,
        7746,
        2961,
        518,
        7504,
        386,
        2282,
        7069,
        4518,
        1460,
        5812,
        1544,
        2381,
        8139,
        6612,
        7916,
        3036,
        5681,
        3305,
        454,
        1135,
        4125,
        5421,
        7624,
        7397,
        7125,
        7717,
    )
    c_values = (
        2133,
        4476,
        5861,
        4618,
        2655,
        4790,
        4222,
        2858,
        1529,
        7344,
        1011,
        7656,
        4783,
        4241,
        1377,
        3052,
        2167,
        6924,
        856,
        992,
        7856,
        6422,
        13,
        6023,
        2149,
        5762,
        7612,
        155,
        2262,
        6907,
        1370,
        5305,
    )
    return [
        [
            "<|sid_begin|>",
            f"<s_a_{a}>",
            f"<s_b_{b}>",
            f"<s_c_{c}>",
            "<|sid_end|>",
        ]
        for a in a_values
        for b in b_values
        for c in c_values
    ]


def test_compact_catalog_fixture_geometry() -> None:
    catalog = _build_catalog_data()
    assert len(catalog) == 32**3
    assert len({tuple(item) for item in catalog}) == len(catalog)


async def _serving_catalog_generation_impl(loops: int) -> None:
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

            catalog_set = set("".join(item) for item in catalog_data)
            # Both request strategies share the same model, catalog and engine.
            for catalog_strategy in ("fill", "null"):
                params_dict = beam_search_params_dict.copy()
                if hasattr(BeamSearchParams, "catalog_strategy"):
                    params_dict["catalog_strategy"] = catalog_strategy
                params = BeamSearchParams(**params_dict)
                print(f"Starting beam search (strategy={catalog_strategy}), loops={loops}...")

                for i in range(max(1, loops)):
                    generator = serving.beam_search(
                        prompt, f"req_id_catalog_{catalog_strategy}_{i}", params
                    )

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

                    print(
                        f"{catalog_strategy} loop {i}: Verifying {len(final_output.outputs)} generated choices..."
                    )
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


def _run_serving_catalog_generation_target(loops: int) -> None:
    from vllm_gr.patch import re_apply_patches

    re_apply_patches()
    asyncio.run(_serving_catalog_generation_impl(loops))


@pytest.mark.slow  # type: ignore
def test_serving_catalog_dead_end_strategies(loops: int) -> None:
    """Check fill and null outputs against the catalog using one engine."""
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=_run_serving_catalog_generation_target, args=(loops,))
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
