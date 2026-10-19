# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side regressions runnable without vLLM, torch, or an NPU.

Execute selected real methods via AST extraction to avoid importing the NPU
runtime. Tensor doubles test ownership/routing, not kernel correctness.
Run: python tests/test_npu_prefill_bsnd.py
"""

import ast
import unittest
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RUNNER = "vllm_gr/v1/worker/npu_prefill_graph_runner.py"
PATCH = "vllm_gr/v1/worker/npu_model_runner_patch.py"
ATTENTION = "vllm_gr/v1/attention/backends/beam_attn_npu.py"
BACKEND = "vllm_gr/v1/attention/backends/beam_attn.py"


def extract(path: str, name: str, env: dict[str, Any]) -> Callable[..., Any]:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8-sig"))
    fn = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name
    )
    fn.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            fn,
        ],
        type_ignores=[],
    )
    # Execute only a selected function from the local repository under test.
    exec(compile(ast.fix_missing_locations(module), path, "exec"), env)  # noqa: S102
    return cast(Callable[..., Any], env[name])


class Tensor:
    def __init__(
        self,
        shape: Sequence[int],
        dtype: str = "int64",
        device: str | SimpleNamespace = "cpu",
        contiguous: bool = True,
        owner: "Tensor | None" = None,
    ) -> None:
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self._contiguous = contiguous
        self.owner = owner or self
        self.values: int | tuple[int, ...] | None = None

    def data_ptr(self) -> int:
        return id(self.owner)

    def stride(self) -> tuple[int, ...]:
        strides: list[int] = []
        size = 1
        for dim in reversed(self.shape):
            strides.insert(0, size)
            size *= dim
        return tuple(strides)

    def copy_(self, other: "Tensor") -> "Tensor":
        self.values = other.values
        return self

    def fill_(self, value: int) -> "Tensor":
        self.values = value
        return self

    def reshape(self, *shape: int) -> "Tensor":
        return Tensor(shape, self.dtype, self.device, self._contiguous, self.owner)

    def contiguous(self) -> "Tensor":
        if self._contiguous:
            return self
        return Tensor(self.shape, self.dtype, self.device)

    def is_contiguous(self) -> bool:
        return self._contiguous

    def view(self, *shape: int) -> "Tensor":
        if not self._contiguous:
            raise ValueError("noncontiguous view")
        return self.reshape(*shape)


class FakeTorch:
    int64 = "int64"
    int32 = "int32"

    @staticmethod
    def empty(shape: int | Sequence[int], dtype: str, device: str | SimpleNamespace) -> Tensor:
        return Tensor((shape,) if isinstance(shape, int) else shape, dtype, device)

    zeros = empty

    @staticmethod
    def arange(size: int, dtype: str, device: str | SimpleNamespace) -> Tensor:
        result = Tensor((size,), dtype, device)
        result.values = tuple(range(size))
        return result


class BSNDRegressions(unittest.TestCase):
    def test_compaction_uses_only_slices_and_preserves_source(self) -> None:
        compact = extract(RUNNER, "_compact_bsnd_output", {})

        class Rows:
            def __init__(self, values: list[int], start: int = 0, stop: int | None = None) -> None:
                self.values, self.start = values, start
                self.stop = len(values) if stop is None else stop

            def __getitem__(self, index: object) -> "Rows":
                # A list/tensor index here would reintroduce the hidden H2D.
                if not isinstance(index, slice):
                    raise TypeError("only basic slicing is allowed")
                return Rows(self.values, 0 if index.start is None else index.start, index.stop)

            def copy_(self, source: "Rows", non_blocking: bool = False) -> None:
                if not non_blocking or self.values is source.values:
                    raise AssertionError("require nonblocking independent destination")
                self.values[self.start : self.stop] = source.values[source.start : source.stop]

        for counts in ([90, 100, 110, 120], [101] * 4, [128] * 4, [1, 128, 2, 127]):
            with self.subTest(counts=counts):
                source = list(range(512))
                destination = [-1] * 512
                compact(Rows(source), Rows(destination), 128, counts)
                expected = [i * 128 + j for i, n in enumerate(counts) for j in range(n)]
                self.assertEqual(destination[: sum(counts)], expected)
                self.assertEqual(source, list(range(512)))
                self.assertEqual(destination[sum(counts) :], [-1] * (512 - sum(counts)))

    def backend_fixture(
        self,
    ) -> tuple[Callable[..., Any], tuple[Any, ...], list[tuple[str, Any]]]:
        calls: list[tuple[str, Any]] = []
        platform = SimpleNamespace(reshape_and_cache=lambda *args: calls.append(("cache", args)))
        fn = extract(
            BACKEND,
            "prefill_forward",
            {
                "NPU_DEVICE": True,
                "CUDA_DEVICE": False,
                "_get_platform_execution_module": lambda: platform,
            },
        )
        owner = SimpleNamespace(
            kv_sharing_target_layer_name=None,
            kv_cache_dtype="auto",
            scale=0.25,
            sliding_window=(-1, -1),
            logits_soft_cap=0,
            sinks=None,
            alibi_slopes=None,
            vllm_flash_attn_version=None,
            prefill_attention=lambda **kw: calls.append(("tnd", kw)),
        )
        meta = SimpleNamespace(
            prefill_fia_layout="BSND",
            num_reqs=4,
            seq_lens_cpu=[128] * 4,
            slot_mapping=object(),
            causal=True,
            npu_causal_atten_mask=object(),
        )
        args = (
            owner,
            SimpleNamespace(),
            Tensor((512, 8, 16)),
            Tensor((512, 2, 16)),
            Tensor((512, 2, 16)),
            (object(), object()),
            meta,
            Tensor((512, 8, 16)),
        )
        return fn, args, calls

    def test_backend_writes_cache_but_fia_reads_current_layer_kv(self) -> None:
        fn, args, calls = self.backend_fixture()
        helper = SimpleNamespace(prefill_attention_bsnd=lambda **kw: calls.append(("bsnd", kw)))
        with patch.dict(
            "sys.modules", {"vllm_gr.v1.attention.backends": SimpleNamespace(beam_attn_npu=helper)}
        ):
            for _ in range(2):  # same dispatch in warmup and capture
                self.assertIs(fn(*args), args[-1])
        self.assertEqual([kind for kind, _ in calls], ["cache", "bsnd"] * 2)
        self.assertIs(calls[0][1][2], args[5][0])
        self.assertIs(calls[0][1][4], args[6].slot_mapping)
        self.assertIs(calls[1][1]["key"], args[3])
        self.assertIs(calls[1][1]["value"], args[4])
        self.assertNotIn("key_cache", calls[1][1])

    def test_invalid_bsnd_rejected_before_cache_write_and_tnd_still_routes(self) -> None:
        for case in ("length", "sharing", "mask", "window", "missing_key"):
            with self.subTest(case=case):
                fn, args, calls = self.backend_fixture()
                owner, meta = args[0], args[6]
                if case == "length":
                    meta.seq_lens_cpu = [90, 100, 110, 120]
                elif case == "sharing":
                    owner.kv_sharing_target_layer_name = "layer0"
                elif case == "mask":
                    meta.npu_causal_atten_mask = None
                elif case == "window":
                    owner.sliding_window = (127, 0)
                else:
                    args = (*args[:3], None, *args[4:])
                with self.assertRaises(ValueError):
                    fn(*args)
                self.assertEqual(calls, [])
        fn, args, calls = self.backend_fixture()
        args[6].prefill_fia_layout = "TND"
        # Exercise the real NPU guard without importing the accelerator runtime.
        # This is ordinary eager TND, not standard FULL graph capture.
        modes = SimpleNamespace(NONE="NONE", FULL="FULL")
        context = SimpleNamespace(cudagraph_runtime_mode=modes.NONE, capturing=False)
        helper = SimpleNamespace(_GRAPH_CAPTURE_STATE={"capturing": False})
        with patch.dict(
            "sys.modules",
            {
                "vllm.config": SimpleNamespace(CUDAGraphMode=modes),
                "vllm.forward_context": SimpleNamespace(get_forward_context=lambda: context),
                "vllm_gr.v1.attention.backends": SimpleNamespace(beam_attn_npu=helper),
                "vllm_gr.v1.beam.context": SimpleNamespace(
                    BEAM_DATA_VAR=SimpleNamespace(get=lambda: None)
                ),
            },
        ):
            self.assertIs(fn(*args), args[-1])
        self.assertEqual([kind for kind, _ in calls], ["cache", "tnd"])
        self.assertIs(calls[1][1]["key_cache"], args[5][0])
        self.assertIs(calls[1][1]["value_cache"], args[5][1])

    def test_capture_preparation_reuses_input_storage_and_refreshes_contents(self) -> None:
        prepare = extract(
            RUNNER,
            "_prepare_capture_buffers",
            {"torch": FakeTorch, "SUPPORTED_NUM_REQS": list(range(1, 9))},
        )
        configurations: list[tuple[dict[str, bool | int], tuple[int, ...]]] = [
            ({}, (512,)),
            ({"uses_mrope": True}, (3, 512)),
            ({"uses_xdrope_dim": 2}, (2, 512)),
        ]
        for options, shape in configurations:
            with self.subTest(shape=shape):
                owner = SimpleNamespace(device="cpu", _get_max_kvlen=lambda _: 8192)
                buf = SimpleNamespace(
                    input_ids_buf=Tensor((512,)), pad_token_id=0, block_table_buf=None
                )
                model = SimpleNamespace(**options)
                prepare(owner, buf, 512, model)  # BSND capture preparation
                positions, slots = buf.positions_buf, buf.slot_mapping_buf
                positions.fill_(-9)
                slots.fill_(-1)
                prepare(owner, buf, 512, model)  # subsequent TND / second B capture
                self.assertIs(buf.positions_buf, positions)
                self.assertIs(buf.slot_mapping_buf, slots)
                self.assertEqual(positions.shape, shape)
                self.assertEqual(slots.dtype, "int32")
                self.assertEqual(positions.values, tuple(range(512)))
                self.assertEqual(slots.values, tuple(range(512)))

    def test_replay_guard_rejects_replaced_capture_input(self) -> None:
        descriptor = extract(RUNNER, "_bsnd_tensor_descriptor", {})
        validate = extract(RUNNER, "_validate_bsnd_capture_inputs", {})
        original = Tensor((512,))
        buf = SimpleNamespace(positions_buf=original)
        owner = SimpleNamespace(
            _bsnd_tensor_descriptor=descriptor,
            bsnd_entries={
                (512, 4): {
                    "capture_inputs": {"positions_buf": original},
                    "capture_descriptors": {"positions_buf": descriptor(original)},
                }
            },
        )
        validate(owner, buf, 512, 4)
        buf.positions_buf = Tensor((512,))
        with self.assertRaisesRegex(RuntimeError, "positions_buf"):
            validate(owner, buf, 512, 4)

    def test_selection_uses_packed_row_order_not_scheduler_order(self) -> None:
        select = extract(PATCH, "_select_bsnd_spec", {})
        batch = SimpleNamespace(
            req_id_to_index={"a": 0, "b": 1, "c": 2, "d": 3},
            num_computed_tokens_cpu=[0] * 4,
            num_prompt_tokens=[90, 100, 110, 120],
        )
        scheduler = SimpleNamespace(num_scheduled_tokens={"a": 90, "c": 110, "b": 100, "d": 120})
        runner = SimpleNamespace(input_batch=batch)
        graph_runner = SimpleNamespace(
            match_bsnd_bucket=lambda b, length: (512, b), has_bsnd_captured_graph=lambda t, b: True
        )
        self.assertEqual(select(runner, scheduler, graph_runner, 4), (512, 4, [90, 100, 110, 120]))
        batch.num_computed_tokens_cpu[2] = 1
        self.assertIsNone(select(runner, scheduler, graph_runner, 4))
        batch.num_computed_tokens_cpu[2] = 0
        batch.req_id_to_index["d"] = 2
        self.assertIsNone(select(runner, scheduler, graph_runner, 4))

    def test_fia_receives_contiguous_inputs_and_aliasing_output(self) -> None:
        calls: list[dict[str, Any]] = []
        npu = SimpleNamespace(
            _npu_fused_infer_attention_score_v2_get_max_workspace=lambda **kw: object(),
            npu_fused_infer_attention_score_v2=SimpleNamespace(out=lambda **kw: calls.append(kw)),
        )
        fn = extract(
            ATTENTION,
            "prefill_attention_bsnd",
            {
                "torch": FakeTorch,
                "torch_npu": npu,
                "_BSNB_WORKSPACE_CACHE": {},
                "_BSNB_LSE_CACHE": {},
            },
        )
        device = SimpleNamespace(index=0)
        q = Tensor((512, 8, 16), "bf16", device, contiguous=False)
        k = Tensor((512, 2, 16), "bf16", device, contiguous=False)
        out = Tensor((512, 8, 16), "bf16", device)
        meta = SimpleNamespace(
            bsnd_resources={},
            num_reqs=4,
            num_heads=8,
            num_kv_heads=2,
            causal=True,
            npu_causal_atten_mask=object(),
        )
        fn(out, q, k, k, meta, 0.25)
        call = calls[-1]
        for name in ("query", "key", "value"):
            self.assertTrue(call[name].is_contiguous())
        self.assertEqual(call["key"].shape, (4, 128, 2, 16))
        self.assertEqual(call["out"][0].data_ptr(), out.data_ptr())
        self.assertNotIn("actual_seq_qlen", call)
        self.assertNotIn("actual_seq_kvlen", call)
        self.assertNotIn("block_table", call)
        out._contiguous = False
        with self.assertRaisesRegex(ValueError, "output buffer"):
            fn(out, q, k, k, meta, 0.25)
        self.assertEqual(len(calls), 1)


# 20261019: Additional migration contracts against the latest source and bootstrap.
class BSNDContracts(unittest.TestCase):
    def test_specs_disabled_sorted_and_validated(self) -> None:
        parse = extract(RUNNER, "parse_bsnd_capture_specs", {})
        self.assertEqual(parse("", [128, 512, 1024]), ())
        self.assertEqual(parse("1024,4;512,4;512,4", [512, 1024]), ((512, 4), (1024, 4)))
        for raw in ("512", "512,x", "0,4", "512,3", "513,4", "256,4", "512,0"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse(raw, [512, 1024])

    def test_selection_uses_smallest_ready_graph(self) -> None:
        match = extract(RUNNER, "match_bsnd_bucket", {})
        runner = SimpleNamespace(
            _BSND_CAPTURE_SPECS=((512, 4), (1024, 4)),
            has_bsnd_captured_graph=lambda t, b: t == 1024,
        )
        self.assertEqual(match(runner, 4, 100), (1024, 4))
        self.assertIsNone(match(runner, 2, 100))
        self.assertIsNone(match(runner, 4, 257))

    def test_total_tokens_are_not_per_request_capacity(self) -> None:
        select = extract(PATCH, "_select_bsnd_spec", {})
        runner = SimpleNamespace(
            input_batch=SimpleNamespace(
                req_id_to_index={str(i): i for i in range(4)},
                num_computed_tokens_cpu=[0] * 4,
                num_prompt_tokens=[1, 1, 509, 1],
            )
        )
        scheduled = SimpleNamespace(
            num_scheduled_tokens=dict(zip(map(str, range(4)), [1, 1, 509, 1]))
        )
        graph = SimpleNamespace(
            match_bsnd_bucket=lambda b, m: (512, 4) if m <= 128 else None,
            has_bsnd_captured_graph=lambda *args: True,
        )
        self.assertIsNone(select(runner, scheduled, graph, 4))


# 20261019: Exercise the actual latest model-forward wrapper with host doubles.
class ForwardRouting(unittest.TestCase):
    def exercise(
        self,
        spec: tuple[int, int, list[int]] | None,
        *,
        fail: bool = False,
        model_kwargs: dict[str, object] | None = None,
    ) -> tuple[list[str], str]:
        import sys
        from functools import wraps

        calls: list[str] = []

        def bsnd(**kwargs: Any) -> str:
            calls.append("bsnd")
            if fail:
                raise RuntimeError("device replay failed")
            return "bsnd-hidden"

        def tnd(**kwargs: Any) -> str:
            calls.append("tnd")
            return "tnd-hidden"

        def native(*args: Any, **kwargs: Any) -> str:
            calls.append("native")
            return "native-hidden"

        graph = SimpleNamespace(
            has_bsnd_captured_graph=lambda *args: True,
            execute_model_prefill_bsnd=bsnd,
            log_replay_bsnd=lambda *args: None,
            match_bucket=lambda n: 512,
            has_captured_graph=lambda *args: True,
            execute_model_prefill=tnd,
            log_replay=lambda *args: None,
        )
        env: dict[str, Any] = {
            "wraps": wraps,
            "_WORKER_MODEL_FORWARD_PATCH_MARKER": "_test_patched",
            "is_prefill_graph_enabled": lambda: True,
            "is_prefill_phase_batch": lambda *args: True,
            "get_prefill_graph_runner": lambda runner: graph,
            "SUPPORTED_NUM_REQS": (1, 2, 4, 8),
            "_select_bsnd_spec": lambda *args: spec,
            "logger": SimpleNamespace(debug=lambda *args: None),
        }
        extract(PATCH, "_record_npu_prefill_dispatch", env)
        factory = extract(PATCH, "make_patched_model_forward_npu", env)
        forward = factory(native)
        runner = SimpleNamespace(
            _current_scheduler_output=SimpleNamespace(
                num_scheduled_tokens={str(i): 100 for i in range(4)}, total_num_scheduled_tokens=400
            )
        )
        context = SimpleNamespace(
            get_forward_context=lambda: SimpleNamespace(
                attn_metadata={"layer": SimpleNamespace(slot_mapping=object())}
            )
        )
        with patch.dict(sys.modules, {"vllm.forward_context": context}):
            result = forward(runner, 400, object(), object(), **(model_kwargs or {}))
        if not model_kwargs:
            self.assertEqual(runner._gr_prefill_graph_dispatches, {"replay:4": 1})
        return calls, cast(str, result)

    def test_bsnd_hit_returns_compact_hidden_without_tnd(self) -> None:
        self.assertEqual(self.exercise((512, 4, [100] * 4)), (["bsnd"], "bsnd-hidden"))

    def test_bsnd_miss_preserves_latest_tnd(self) -> None:
        self.assertEqual(self.exercise(None), (["tnd"], "tnd-hidden"))

    def test_replay_error_is_not_swallowed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "device replay failed"):
            self.exercise((512, 4, [100] * 4), fail=True)

    def test_extra_model_payload_uses_native_forward(self) -> None:
        self.assertEqual(
            self.exercise((512, 4, [100] * 4), model_kwargs={"extra": 1}),
            (["native"], "native-hidden"),
        )


if __name__ == "__main__":
    unittest.main()
