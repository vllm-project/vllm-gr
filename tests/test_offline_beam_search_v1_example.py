# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-free regression coverage for the offline NPU graph example."""

from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from examples.offline_inference.beam_search import offline_beam_search_v1 as example


@pytest.mark.parametrize(  # type: ignore[misc]
    "steps,captured,dispatches,fails",
    [
        (1, True, 0, False),
        (1, False, 0, False),
        (2, True, 1, False),
        (3, True, 2, False),
        (2, True, 0, True),
        (2, False, 1, True),
    ],
)
def test_npu_graph_example_requires_replay_only_for_decode(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    steps: int,
    captured: bool,
    dispatches: int,
    fails: bool,
) -> None:
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "example",
            "--model",
            str(tmp_path),
            "--catalog",
            str(tmp_path / "catalog.json"),
            "--prompt",
            str(tmp_path / "prompt.txt"),
            "--constraint-backend",
            "npu",
            "--execution",
            "graph",
            "--batch-size",
            "2",
            "--generated-steps",
            str(steps),
        ],
    )
    args = example.parse_args()
    sequence = SimpleNamespace(
        tokens=[1, 2, 3, 4, 5], cum_logprob=-1.0, finish_reason="length", stop_reason=None
    )
    outputs = [SimpleNamespace(sequences=[sequence]) for _ in range(args.batch_size)]
    stats = {"captured_shapes": [[2, 128]] if captured else [], "full_dispatches": dispatches}
    llm = SimpleNamespace(
        get_tokenizer=lambda: SimpleNamespace(convert_ids_to_tokens=lambda ids: ids),
        beam_search_v1=lambda prompts, params: outputs,
        collective_rpc=lambda name: [stats],
    )
    modules: dict[str, dict[str, Any]] = {
        "vllm.inputs": {"tokens_input": lambda ids, **kw: dict(prompt_token_ids=ids, **kw)},
        "vllm_gr.entrypoints.gr": {"GRLLM": lambda **kw: nullcontext(llm)},
        "vllm_gr.sampling_params": {"BeamSearchParams": lambda **kw: SimpleNamespace(**kw)},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(example, "build_artifact", lambda *args: ("artifact", "tokenizer"))
    monkeypatch.setattr(example, "make_prompt", lambda *args: [11, 12])
    if fails:
        with pytest.raises(RuntimeError, match="capture/replay was not observed"):
            example.run(args, tmp_path / "unused.safetensors")
    else:
        example.run(args, tmp_path / "unused.safetensors")
        output = capsys.readouterr().out
        assert '"npu_beam_graph"' in output
        assert "request=0 returned_beams=1" in output and "request=1 returned_beams=1" in output
