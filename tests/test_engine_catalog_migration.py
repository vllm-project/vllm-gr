# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import ClassVar

from vllm_gr.config.parser import BEAM_CONTROL_MODE_KEY, CATALOG_PATH_KEY
from vllm_gr.constraints.trie_resource import trie_resource_from_wire
from vllm_gr.entrypoints.openai import serving_models


class _Tokenizer:
    ids: ClassVar = {"<begin>": 1, "a": 2, "<end>": 3}

    def convert_tokens_to_ids(self, tokens):
        return [self.ids.get(token, -1) for token in tokens]


def test_engine_mode_compiles_catalog_without_retaining_serving_catalog(
    tmp_path,
    monkeypatch,
) -> None:
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps([["<begin>", "a", "<end>"]]))
    tokenizer = _Tokenizer()

    def fake_original_init(instance, engine_client, base_model_paths, *, lora_modules=None):
        instance.input_processor = SimpleNamespace(tokenizer=tokenizer)

    monkeypatch.setattr(serving_models, "_original_init", fake_original_init)
    engine_client = SimpleNamespace(
        vllm_config=SimpleNamespace(
            additional_config={
                CATALOG_PATH_KEY: str(catalog_path),
                BEAM_CONTROL_MODE_KEY: "engine_per_step_request",
            }
        )
    )
    models = SimpleNamespace()

    serving_models.init_gr(models, engine_client, [])

    assert models.catalog is None
    assert models.constraint_strategy is None
    assert models.constraint_coordinator.enabled is False
    resource = models.beam_constraint_resource
    assert resource.valid_next_token_ids(resource.root_state()) == {1}
    rebuilt = trie_resource_from_wire(models.beam_constraint_resource_wire)
    assert rebuilt.digest == resource.digest
