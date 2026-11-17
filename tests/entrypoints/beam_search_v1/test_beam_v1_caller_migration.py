# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Check V1 caller batching, suffix decoding and E2E measurement boundaries."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self

import pytest

from tools.AB_Tests.run_offline_benchmark import measure_e2e


class _Engine:
    def __init__(self, wrong_items: bool = False, wrong_beams: bool = False) -> None:
        self.batches: list[list[str]] = []
        self.wrong_items = wrong_items
        self.wrong_beams = wrong_beams

    def get_tokenizer(self) -> Any:
        return SimpleNamespace(decode=lambda tokens, **kw: ":".join(map(str, tokens)))

    def beam_search_v1(
        self, prompts: list[dict[str, str]], params: Any, *, output_mode: str
    ) -> list[Any]:
        assert output_mode == "compact"
        self.batches.append([p["prompt"] for p in prompts])
        return [
            SimpleNamespace(
                sequences=[
                    SimpleNamespace(token_ids=(10, 20, 30, 40, 50), cumulative_logprob=-float(i))
                    for i in range(params.beam_width - int(self.wrong_beams))
                ]
            )
            for _ in prompts[: len(prompts) - int(self.wrong_items)]
        ]


def test_accuracy_preserves_order_and_handles_partial_batches() -> None:
    from benchmarks.open_one_rec.one_rec_acc_test import run_offline_v1_inference

    prompts = [f"prompt {i} with normalized whitespace" for i in range(7)]
    for batch_size, expected in ((1, [1] * 7), (2, [2, 2, 2, 1]), (4, [4, 2, 1])):
        engine = _Engine()
        outputs = run_offline_v1_inference(
            engine, prompts, SimpleNamespace(beam_width=2, max_tokens=5), batch_size
        )
        assert [len(batch) for batch in engine.batches] == expected
        assert [output.prompt for output in outputs] == prompts
        assert all(len(output.outputs) == 2 for output in outputs)
        assert all(output.outputs[0].text == "10:20:30:40:50" for output in outputs)
        assert all(output.outputs[1].logprobs == -1.0 for output in outputs)
    empty = _Engine()
    assert run_offline_v1_inference(empty, [], object(), 4) == []
    assert empty.batches == []


def test_accuracy_rejects_mismatched_output_counts() -> None:
    from benchmarks.open_one_rec.one_rec_acc_test import run_offline_v1_inference

    for engine in (_Engine(wrong_items=True), _Engine(wrong_beams=True)):
        with pytest.raises(RuntimeError, match="unexpected number"):
            run_offline_v1_inference(
                engine, ["prompt"], SimpleNamespace(beam_width=2, max_tokens=5), 1
            )


@pytest.mark.parametrize("eager", [False, True])  # type: ignore[misc]
def test_accuracy_cli_configures_beam_capacity(
    monkeypatch: pytest.MonkeyPatch, eager: bool
) -> None:
    import transformers

    import benchmarks.open_one_rec.one_rec_acc_test as benchmark

    class EngineReached(Exception):
        pass

    def engine(**kwargs: Any) -> None:
        assert kwargs["enforce_eager"] is eager
        assert kwargs["async_scheduling"] is True
        raise EngineReached

    monkeypatch.setattr(
        "vllm_gr.config.parser.load_gr_config",
        lambda _: SimpleNamespace(beam_max_width=128),
    )
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: object())
    monkeypatch.setattr(
        benchmark,
        "DataLoaderWrapper",
        lambda **kw: SimpleNamespace(
            load_data=lambda **kw: {"0": {"prompt": "test", "ground_truth": "a SID"}}
        ),
    )
    monkeypatch.setattr(benchmark, "GRLLM", engine)
    monkeypatch.setattr(
        sys,
        "argv",
        ["accuracy", "--task-type", "video", "--config", "config.json"]
        + (["--enforce-eager"] if eager else []),
    )
    with pytest.raises(EngineReached):
        benchmark.main()


@pytest.mark.parametrize("template_kwargs", [None, {"enable_thinking": False}])  # type: ignore[misc]
def test_accuracy_http_forwards_template_options(
    monkeypatch: pytest.MonkeyPatch, template_kwargs: dict[str, Any] | None
) -> None:
    import benchmarks.open_one_rec.one_rec_acc_test as benchmark

    payloads = []

    def post(url: str, **kwargs: Any) -> Any:
        payloads.append(kwargs["json"])
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": "result"}}]},
        )

    monkeypatch.setattr(benchmark.requests, "post", post)
    content = "a question with enough content to satisfy the transcript coverage guard " * 2
    prompt = f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
    outputs = benchmark.run_api_inference(
        [prompt],
        "model",
        "http://localhost/unused",
        "EMPTY",
        {},
        max_concurrency=1,
        chat_template_kwargs=template_kwargs,
    )
    assert outputs[0].outputs[0].text == "result"
    assert payloads[0]["messages"] == [{"role": "user", "content": content}]
    if template_kwargs is None:
        assert "chat_template_kwargs" not in payloads[0]
    else:
        assert payloads[0]["chat_template_kwargs"] == template_kwargs


@pytest.mark.parametrize(  # type: ignore[misc]
    "options,message",
    [
        (["--endpoint", "/v1/chat/completions", "--enforce-eager"], "on the server"),
        (["--endpoint", "/v1/chat/completions", "--chat-template-kwargs", "[]"], "JSON object"),
        (["--chat-template-kwargs", "{}"], "only to online"),
    ],
)
def test_accuracy_cli_rejects_misleading_execution_options(
    monkeypatch: pytest.MonkeyPatch, capsys: Any, options: list[str], message: str
) -> None:
    import benchmarks.open_one_rec.one_rec_acc_test as benchmark

    monkeypatch.setattr(sys, "argv", ["accuracy", "--task-type", "video", *options])
    with pytest.raises(SystemExit, match="2"):
        benchmark.main()
    assert message in capsys.readouterr().err


def test_e2e_includes_lazy_text_and_uses_only_v1(monkeypatch: Any) -> None:
    events: list[str] = []

    class Beam:
        @property
        def text(self) -> str:
            events.append("decode")
            return "result"

    def run(prompts: Any, params: Any, **kwargs: Any) -> list[Any]:
        events.append("v1")
        return [SimpleNamespace(sequences=[Beam(), Beam()])]

    def clock() -> int:
        events.append("clock")
        return len(events) * 1_000_000

    monkeypatch.setattr("tools.AB_Tests.run_offline_benchmark.time.perf_counter_ns", clock)
    result = measure_e2e(SimpleNamespace(beam_search_v1=run), [1, 2], SimpleNamespace(max_tokens=5))
    assert events == ["clock", "v1", "decode", "decode", "clock"]
    assert result == {"e2e_ms": 4.0, "returned_beams": 2, "aggregate_output_tokens": 10}


def test_benchmark_keeps_artifact_until_engine_shutdown(monkeypatch: Any) -> None:
    from argparse import Namespace
    from pathlib import Path

    import examples.offline_inference.beam_search.offline_beam_search_v1 as example
    from tools.AB_Tests.run_offline_benchmark import create_v1_llm
    from vllm_gr.entrypoints import gr

    artifacts: list[Path] = []
    events: list[str] = []

    def build(model: Path, catalog: Path, path: Path) -> tuple[str, str]:
        path.write_bytes(b"artifact")
        artifacts.append(path)
        return "artifact-digest", "tokenizer-digest"

    class Engine:
        def __init__(self, **kwargs: Any) -> None:
            config = kwargs["gr"]
            assert "execution_mode" not in config
            assert config["beam_width"] == [128, 128, 128]
            assert config["max_decode_steps"] == 3
            assert config["constraint_table"]["digests"]["artifact"] == "artifact-digest"
            assert kwargs["async_scheduling"] is True
            assert "enable_chunked_prefill" not in kwargs
            assert "enable_prefix_caching" not in kwargs
            assert artifacts[-1].exists()

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *exc: object) -> None:
            assert artifacts[-1].exists()
            events.append("shutdown")

    monkeypatch.setattr(example, "build_artifact", build)
    monkeypatch.setattr(gr, "GRLLM", Engine)
    args = Namespace(model="model", catalog=Path("catalog"), beam_width=128, input_length=1024)
    with pytest.raises(RuntimeError, match="request failed"), create_v1_llm(args, 5):
        raise RuntimeError("request failed")
    assert events == ["shutdown"]
    assert not artifacts[0].exists()


@pytest.mark.parametrize("failure_stage", [None, "inference", "text"])  # type: ignore[misc]
def test_onerec_benchmark_closes_engine_before_deleting_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure_stage: str | None
) -> None:
    import examples.offline_inference.beam_search.offline_beam_search_v1 as example
    from benchmarks.open_one_rec.one_rec_v1_benchmark import run_v1_benchmark
    from vllm_gr.entrypoints import gr

    artifacts: list[Path] = []
    events: list[str] = []
    failure = RuntimeError(f"{failure_stage} failed")

    def build(model: Path, catalog: Path, path: Path) -> tuple[str, str]:
        path.write_bytes(b"artifact")
        artifacts.append(path)
        return "artifact-digest", "tokenizer-digest"

    class Beam:
        tokens = [1, 2, 10, 20, 30, 40, 50]
        cum_logprob = -1.0

        @property
        def text(self) -> str:
            assert artifacts[0].exists()
            events.append("text")
            if failure_stage == "text":
                raise failure
            return "result"

    class Engine:
        def __init__(self, **kwargs: Any) -> None:
            path = Path(kwargs["gr"]["constraint_table"]["path"])
            assert path == artifacts[0] and path.is_file()

        def __enter__(self) -> Self:
            events.append("enter")
            return self

        def __exit__(self, *exc: object) -> None:
            assert artifacts[0].is_file()
            assert exc[1] is (failure if failure_stage is not None else None)
            events.append("shutdown")

        def reset_prefix_cache(self) -> bool:
            return True

        def beam_search_v1(self, prompts: Any, params: Any, *, output_mode: str) -> list[Any]:
            assert output_mode == "full"
            assert artifacts[0].is_file()
            events.append("inference")
            if failure_stage == "inference":
                raise failure
            return [SimpleNamespace(sequences=[Beam(), Beam()])]

    monkeypatch.setattr(example, "build_artifact", build)
    monkeypatch.setattr(gr, "GRLLM", Engine)
    output = tmp_path / "result.json"
    args = SimpleNamespace(
        num_prompts=1,
        warmup_requests=0,
        n=2,
        constraint_triples="catalog",
        output_json=str(output),
        output_len=5,
        custom_input_len=0,
        max_model_len=16,
        max_num_batched_tokens=16,
        model="model",
        dataset_path="dataset",
    )
    tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [1, 2])
    requests = [SimpleNamespace(prompt="prompt", expected_output_len=5)]
    if failure_stage is None:
        run_v1_benchmark(args, tokenizer, requests)
        assert len(json.loads(output.read_text())["samples"]) == 1
        assert events == ["enter", "inference", "text", "text", "shutdown"]
    else:
        with pytest.raises(RuntimeError) as caught:
            run_v1_benchmark(args, tokenizer, requests)
        assert caught.value is failure
        assert not output.exists()
        expected = ["enter", "inference"]
        if failure_stage == "text":
            expected.append("text")
        assert events == [*expected, "shutdown"]
    assert len(artifacts) == 1
    assert not artifacts[0].parent.exists()


pytestmark = pytest.mark.cpu_test
