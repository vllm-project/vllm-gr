# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Create a tiny random Llama checkpoint and GR config without downloads."""

import argparse
import json
from pathlib import Path


def prepare_model(output: Path, *, context_length: int = 128, width: int = 4) -> tuple[Path, Path]:
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "model"
    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=context_length,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = LlamaForCausalLM(config).eval()
    model.save_pretrained(model_path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel(
                {f"tok{i}": i for i in range(128)},
                unk_token="tok0",
            )
        ),
        unk_token="tok0",
        bos_token="tok1",
        eos_token="tok2",
        pad_token="tok0",
    )
    tokenizer.save_pretrained(model_path)
    gr_path = output / "gr-config.json"
    gr_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "attention_backend": "CUSTOM",
                "constrained_generation": False,
                "constraint_table": {"enabled": False, "backend": "cuda", "max_top_k": width},
                "beam": {
                    "execution_mode": "v1",
                    "graph_enabled": False,
                    "worker_decision": True,
                    "max_width": width,
                    "max_decode_steps": 3,
                },
            },
            indent=2,
        )
        + "\n"
    )
    return model_path, gr_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model, config = prepare_model(args.output.resolve())
    print(f"model={model}\nconfig={config}")
