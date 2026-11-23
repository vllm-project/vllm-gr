# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

import random

from vllm.benchmarks.datasets import BenchmarkDataset, SampleRequest
from vllm.tokenizers import TokenizerLike

from .open_one_rec_loader import TASK_REGISTRY, DataLoaderWrapper

# -----------------------------------------------------------------------------
# OpenOneRec Dataset Implementation
# -----------------------------------------------------------------------------


class OneRecDataset(BenchmarkDataset):
    """
    Custom dataset for OneRec benchmark with multiple task types.
    Loads data using data_loader.load_data() with task_name, split, and sample_size.
    """

    def __init__(
        self,
        # OneRec-specific configuration
        task_types: list[str] | None = None,
        custom_input_len: int | None = None,
        sample_size: int | None = None,
        enable_thinking: bool | None = None,
        tokenizer: TokenizerLike | None = None,
        # BenchmarkDataset parent params, declared explicitly so they are
        # discoverable on the signature instead of hidden in **kwargs.
        dataset_path: str | None = None,
        random_seed: int = BenchmarkDataset.DEFAULT_SEED,
        disable_shuffle: bool = False,
        **kwargs,
    ) -> None:
        # Fail-fast validation for the tokenizer. Without it the
        # DataLoaderWrapper would fail deep in the call chain with an opaque
        # error like "NoneType has no attribute 'apply_chat_template'".
        if tokenizer is None:
            raise ValueError(
                "tokenizer is required. Pass it explicitly, e.g. OneRecDataset(tokenizer=...)."
            )

        # OneRec-owned state
        self.task_types = task_types or [
            "rec_reason",
            "item_understand",
            "ad",
            "product",
            "label_cond",
            "video",
            "interactive",
            "label_pred",
        ]
        self.tokenizer = tokenizer
        self.custom_input_len = custom_input_len
        self.sample_size = sample_size
        self.enable_thinking = enable_thinking

        # Explicitly forward parent params. tokenizer is NOT in kwargs anymore,
        # so BenchmarkDataset.__init__ never receives an unknown field.
        super().__init__(
            dataset_path=dataset_path,
            random_seed=random_seed,
            disable_shuffle=disable_shuffle,
            **kwargs,
        )

        self.data_loader = DataLoaderWrapper(
            tokenizer=self.tokenizer,
            data_dir=self.dataset_path,
            enable_thinking=self.enable_thinking,
            custom_input_len=self.custom_input_len,
        )

        self.load_data()

    def load_data(self) -> None:
        self.data = []

        for task_type in self.task_types:
            # Look up the per-task generation config to get the correct
            # max_new_tokens. Previously this was hardcoded to 5, which
            # silently truncated generation for tasks like rec_reason
            # (max_new_tokens=2000) and item_understand (=128).
            if task_type not in TASK_REGISTRY:
                raise ValueError(
                    f"Unknown task type {task_type!r}. Available: {list(TASK_REGISTRY.keys())}"
                )
            task_config = TASK_REGISTRY[task_type].config
            gen_config = task_config.get("generation_config", {})
            expected_output_len = gen_config.get("max_new_tokens")
            if expected_output_len is None:
                raise ValueError(
                    f"Task {task_type!r} config is missing generation_config.max_new_tokens"
                )

            test_data = self.data_loader.load_data(
                task_name=task_type,
                split="test",
                sample_size=self.sample_size,
            )

            for id, data in test_data.items():
                self.data.append(
                    {
                        "prompt": data["prompt"],
                        "chat_messages": data.get("chat_messages"),
                        "chat_template_kwargs": data.get("chat_template_kwargs"),
                        "expected_output_len": expected_output_len,
                        "ground_truth": data["ground_truth"],
                        "task_type": task_type,
                        "id": id,
                    }
                )

        random.seed(self.random_seed)
        if not self.disable_shuffle:
            random.shuffle(self.data)

    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs,
    ) -> list[SampleRequest]:
        """Generate sample requests from the loaded OneRec data.

        Aligns with the parent ``BenchmarkDataset.sample`` signature so the
        ``request_id_prefix`` and ``no_oversample`` flags are honored when
        passed by the caller. Note: OneRec callers intentionally pass
        ``no_oversample=True`` because duplicating requests would inflate
        throughput numbers via spurious KV-cache hits — see the patched
        get_samples / get_requests entrypoints.
        """
        runtime_template_kwargs = kwargs.get("chat_template_kwargs")
        sampled_requests: list[SampleRequest] = []
        for item in self.data[:num_requests]:
            prompt = item["prompt"]
            prompt_len = len(tokenizer(prompt).input_ids)
            expected_output_len = item["expected_output_len"]
            template_kwargs = item.get("chat_template_kwargs")
            request_overrides = None
            if template_kwargs is not None or runtime_template_kwargs is not None:
                request_overrides = {
                    "chat_template_kwargs": {
                        **(template_kwargs or {}),
                        **(runtime_template_kwargs or {}),
                    }
                }
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=expected_output_len,
                    request_id=f"{request_id_prefix}{item['task_type']}_{item['id']}",
                    chat_messages=item.get("chat_messages"),
                    request_overrides=request_overrides,
                )
            )

        # Reuse the parent's oversample + request_id uniqueness check.
        # No-op when the caller passes no_oversample=True (the OneRec default).
        self.maybe_oversample_requests(
            sampled_requests,
            num_requests,
            request_id_prefix=request_id_prefix,
            no_oversample=no_oversample,
        )
        return sampled_requests
