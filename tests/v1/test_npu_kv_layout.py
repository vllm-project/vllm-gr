# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""CPU policy controls; these do not claim native allocation or NPU execution."""

from __future__ import annotations

import importlib.util
import sys
from enum import Enum
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from tests.helpers.paths import REPO_ROOT


class _AttentionBackend(Enum):
    CUSTOM = "custom"
    OTHER = "other"


class _Decode(Enum):
    XLLM = "xllm"
    CASCADE = "cascade"


class _Spec:
    pass


class _BeamImpl:
    pass


@pytest.fixture  # type: ignore[misc]
def policy(monkeypatch: pytest.MonkeyPatch) -> Any:
    path = REPO_ROOT / "vllm_gr/v1/worker/npu/kv_layout_patch.py"
    spec = importlib.util.spec_from_file_location("_test_npu_kv_layout", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def native(layer: Any, vllm_config: Any, spec: Any) -> bool:
        calls.append((layer, vllm_config, spec))
        return getattr(vllm_config, "native_result", False)

    beam = SimpleNamespace(
        NPU_DEVICE=True,
        CUDA_DEVICE=False,
        BeamAttentionImpl=_BeamImpl,
        AttentionType=SimpleNamespace(DECODER="decoder"),
        _get_npu_decode_execution=lambda: "xllm",
    )
    modules = {
        "vllm.platforms": SimpleNamespace(current_platform=SimpleNamespace(device_type="npu")),
        "vllm_ascend.attention.utils": SimpleNamespace(requires_contiguous_pa_kv_cache=native),
        "vllm_ascend.worker.model_runner_v1": SimpleNamespace(
            requires_contiguous_pa_kv_cache=native
        ),
        "vllm_gr.config": SimpleNamespace(
            GRConfig=SimpleNamespace(from_vllm_config=lambda config: config.gr),
            GRAttentionBackend=_AttentionBackend,
            NPUDecodeExecution=_Decode,
        ),
        "vllm_gr.v1.attention.backends.beam_attn": beam,
        "vllm_gr.ops.npu.beam_ops": SimpleNamespace(is_available=lambda: True),
        "vllm.v1.kv_cache_interface": SimpleNamespace(FullAttentionSpec=_Spec),
        "torch": SimpleNamespace(bfloat16="bf16", float16="fp16"),
    }
    imports = []

    def load(name: str) -> Any:
        imports.append(name)
        return modules[name]

    monkeypatch.setattr(module, "import_module", load)
    monkeypatch.setattr(module, "version", lambda name: "0.30.0rc2")
    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    return SimpleNamespace(
        module=module, modules=modules, calls=calls, native=native, imports=imports
    )


def _case() -> tuple[Any, Any, Any]:
    impl = _BeamImpl()
    vars(impl).update(
        attn_type="decoder",
        sliding_window=(-1, -1),
        kv_sharing_target_layer_name=None,
        alibi_slopes=None,
        sinks=None,
        logits_soft_cap=0,
        kv_cache_dtype="auto",
        head_size=128,
        num_kv_heads=8,
    )
    layer = SimpleNamespace(impl=impl)
    config = SimpleNamespace(
        use_v2_model_runner=False,
        gr=SimpleNamespace(
            attention_backend=_AttentionBackend.CUSTOM,
            beam_max_width=128,
            npu_decode_execution=_Decode.XLLM,
        ),
        model_config=SimpleNamespace(
            runner_type="generate", is_hybrid=False, quantization=None, dtype="bf16"
        ),
        cache_config=SimpleNamespace(cache_dtype="auto", kv_sharing_fast_prefill=False),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        quant_config=None,
        speculative_config=None,
        kv_transfer_config=None,
        additional_config={"vllm_gr": {}},
        compilation_config=SimpleNamespace(static_forward_context={"layer": layer}),
    )
    spec = _Spec()
    vars(spec).update(
        dtype="bf16",
        page_size_bytes=524288,
        real_page_size_bytes=524288,
        head_size=128,
        num_kv_heads=8,
    )
    return layer, config, spec


def _installed(policy: Any) -> Any:
    assert policy.module.apply_npu_contiguous_kv_cache_patch() == 1
    utils = policy.modules["vllm_ascend.attention.utils"]
    runner = policy.modules["vllm_ascend.worker.model_runner_v1"]
    assert utils.requires_contiguous_pa_kv_cache is runner.requires_contiguous_pa_kv_cache
    return utils.requires_contiguous_pa_kv_cache


def test_both_aliases_enable_exact_dense_beam(policy: Any) -> None:
    wrapper = _installed(policy)
    case = _case()
    assert wrapper(*case) is True
    assert policy.calls == [case]


@pytest.mark.l1  # type: ignore[misc]
def test_rc1_updates_only_already_imported_factory_runner(
    policy: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(policy.module, "version", lambda _: "0.31.0rc1+local")
    patch = policy.module.apply_npu_contiguous_kv_cache_patch
    assert patch() == 0
    assert policy.imports == ["vllm.platforms"]

    runner = ModuleType("vllm_ascend.worker.v1.model_runner")
    setattr(runner, "requires_contiguous_pa_kv_cache", policy.native)
    assert patch(runner_module=runner) == 1
    assert patch(runner_module=runner) == 0
    bound_policy = getattr(runner, "requires_contiguous_pa_kv_cache")
    assert bound_policy is (
        policy.modules["vllm_ascend.attention.utils"].requires_contiguous_pa_kv_cache
    )
    assert bound_policy(*_case()) is True


def _observed_compiler_defaults() -> dict[str, bool]:
    return {
        "enable_npugraph_ex": False,
        "enable_static_kernel": False,
        "enable_super_kernel": False,
        "fuse_norm_quant": True,
        "fuse_qknorm_rope": True,
        "fuse_muls_add": True,
    }


@pytest.mark.parametrize("complete", [False, True])  # type: ignore[misc]
def test_observed_native_compiler_defaults_and_missing_spec_string(
    policy: Any, complete: bool
) -> None:
    layer, config, spec = _case()
    assert not hasattr(spec, "cache_dtype_str")
    assert spec.page_size_bytes == spec.real_page_size_bytes == 524288
    config.additional_config["ascend_compilation_config"] = (
        _observed_compiler_defaults() if complete else {}
    )
    assert _installed(policy)(layer, config, spec) is True


@pytest.mark.parametrize("name", list(_observed_compiler_defaults()))  # type: ignore[misc]
@pytest.mark.parametrize("invalid_kind", ["nondefault", "nonbool"])  # type: ignore[misc]
def test_compiler_nondefaults_and_nonboolean_values_do_not_opt_in(
    policy: Any, name: str, invalid_kind: str
) -> None:
    layer, config, spec = _case()
    defaults: dict[str, Any] = _observed_compiler_defaults()
    defaults[name] = not defaults[name] if invalid_kind == "nondefault" else int(defaults[name])
    config.additional_config["ascend_compilation_config"] = defaults
    assert _installed(policy)(layer, config, spec) is False


@pytest.mark.parametrize("value", [None, False, [], {"unqualified_compiler_option": False}])  # type: ignore[misc]
def test_unknown_compiler_mapping_is_not_accepted(policy: Any, value: Any) -> None:
    layer, config, spec = _case()
    config.additional_config["ascend_compilation_config"] = value
    assert _installed(policy)(layer, config, spec) is False


def test_real_full_attention_spec_030_contract(policy: Any) -> None:
    """Use native upstream fields on runtime hosts; Windows may lack vLLM."""
    if sys.platform == "win32":
        torch = pytest.importorskip("torch")
        native = pytest.importorskip("vllm.v1.kv_cache_interface")
    else:
        import torch
        import vllm.v1.kv_cache_interface as native

    spec = native.FullAttentionSpec(
        block_size=128, num_kv_heads=8, head_size=128, dtype=torch.bfloat16
    )
    assert spec.page_size_bytes == spec.real_page_size_bytes == 524288
    layer, config, _ = _case()
    config.model_config.dtype = torch.bfloat16
    config.additional_config["ascend_compilation_config"] = _observed_compiler_defaults()
    policy.modules["torch"] = torch
    policy.modules["vllm.v1.kv_cache_interface"] = native
    assert _installed(policy)(layer, config, spec) is True


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])  # type: ignore[misc]
@pytest.mark.parametrize("layout", [None, "NHD", "LBNHC"])  # type: ignore[misc]
def test_supported_dtype_and_explicit_token_major_layout(
    policy: Any, monkeypatch: pytest.MonkeyPatch, dtype: str, layout: str | None
) -> None:
    if layout is not None:
        monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", layout)
    layer, config, spec = _case()
    config.model_config.dtype = spec.dtype = dtype
    assert _installed(policy)(layer, config, spec) is True


@pytest.mark.parametrize(  # type: ignore[misc]
    "field,value",
    [
        ("use_v2_model_runner", True),
        ("gr.attention_backend", _AttentionBackend.OTHER),
        ("gr.beam_max_width", None),
        ("gr.beam_max_width", None),
        ("gr.beam_max_width", 0),
        ("gr.npu_decode_execution", _Decode.CASCADE),
        ("model_config.runner_type", "pooling"),
        ("model_config.is_hybrid", True),
        ("model_config.quantization", "fp8"),
        ("quant_config", object()),
        ("speculative_config", object()),
        ("kv_transfer_config", object()),
        ("cache_config.cache_dtype", "fp8"),
        ("cache_config.kv_sharing_fast_prefill", True),
        ("parallel_config.tensor_parallel_size", 2),
        ("parallel_config.pipeline_parallel_size", 2),
        ("parallel_config.decode_context_parallel_size", 2),
        ("parallel_config.prefill_context_parallel_size", 2),
        ("additional_config", {"xlite_graph_config": {"enabled": True}}),
    ],
)
def test_ineligible_engine_delegates_to_native(policy: Any, field: str, value: Any) -> None:
    wrapper = _installed(policy)
    layer, config, spec = _case()
    target = config
    *parts, name = field.split(".")
    for part in parts:
        target = getattr(target, part)
    setattr(target, name, value)
    assert wrapper(layer, config, spec) is False
    assert policy.calls == [(layer, config, spec)]


@pytest.mark.parametrize(  # type: ignore[misc]
    "field,value",
    [
        ("sliding_window", (127, 0)),
        ("kv_sharing_target_layer_name", "other"),
        ("alibi_slopes", object()),
        ("sinks", object()),
        ("logits_soft_cap", 1.0),
        ("kv_cache_dtype", "fp8"),
        ("fa_quant_layer", True),
        ("attn_type", "encoder"),
    ],
)
def test_ineligible_layer_preserves_native_layout(policy: Any, field: str, value: Any) -> None:
    layer, config, spec = _case()
    setattr(layer.impl, field, value)
    assert _installed(policy)(layer, config, spec) is False


@pytest.mark.parametrize(  # type: ignore[misc]
    "field,value",
    [
        ("dtype", "fp8"),
        ("dtype", "fp16"),
        ("page_size_bytes", 8192),
        ("real_page_size_bytes", 0),
        ("cache_dtype_str", "fp8"),
        ("head_size", 64),
        ("num_kv_heads", 4),
    ],
)
def test_ineligible_spec_preserves_native_layout(policy: Any, field: str, value: Any) -> None:
    layer, config, spec = _case()
    setattr(spec, field, value)
    assert _installed(policy)(layer, config, spec) is False


@pytest.mark.parametrize("which", ["layer", "spec", "mixed", "unknown_context", "not_registered"])  # type: ignore[misc]
def test_exact_types_and_homogeneous_registered_layers_required(policy: Any, which: str) -> None:
    layer, config, spec = _case()
    if which == "layer":

        class Derived(_BeamImpl):
            pass

        changed_impl = Derived()
        vars(changed_impl).update(vars(layer.impl))
        layer.impl = changed_impl
    elif which == "spec":

        class DerivedSpec(_Spec):
            pass

        changed_spec = DerivedSpec()
        vars(changed_spec).update(vars(spec))
        spec = changed_spec
    elif which == "mixed":
        config.compilation_config.static_forward_context["other"] = SimpleNamespace(impl=object())
    elif which == "unknown_context":
        config.compilation_config.static_forward_context = None
    else:
        config.compilation_config.static_forward_context = {"different": _case()[0]}
    assert _installed(policy)(layer, config, spec) is False


@pytest.mark.parametrize("layout", ["HND", "LBHNC", "BLHNC", "unknown", ""])  # type: ignore[misc]
def test_other_layouts_do_not_opt_in(
    policy: Any, monkeypatch: pytest.MonkeyPatch, layout: str
) -> None:
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", layout)
    assert _installed(policy)(*_case()) is False


def test_actual_execution_and_operator_presence_required(policy: Any) -> None:
    wrapper = _installed(policy)
    beam = policy.modules["vllm_gr.v1.attention.backends.beam_attn"]
    beam._get_npu_decode_execution = lambda: "cascade"
    assert wrapper(*_case()) is False
    beam._get_npu_decode_execution = lambda: "xllm"
    policy.modules["vllm_gr.ops.npu.beam_ops"].is_available = lambda: False
    assert wrapper(*_case()) is False


def test_config_not_cached_and_unknown_attributes_do_not_opt_in(policy: Any) -> None:
    wrapper = _installed(policy)
    layer, config, spec = _case()
    assert wrapper(layer, config, spec) is True
    config.gr.npu_decode_execution = _Decode.CASCADE
    assert wrapper(layer, config, spec) is False
    config.gr.npu_decode_execution = _Decode.XLLM
    del config.model_config.is_hybrid
    assert wrapper(layer, config, spec) is False
    assert wrapper(*_case()) is True


def test_native_result_and_exception_are_not_replaced(policy: Any) -> None:
    wrapper = _installed(policy)
    case = _case()
    case[1].native_result = True
    case[1].gr = None
    assert wrapper(*case) is True
    assert "vllm_gr.v1.attention.backends.beam_attn" not in policy.imports

    def failing(layer: Any, vllm_config: Any, spec: Any) -> bool:
        raise RuntimeError("native failure")

    wrapper = policy.module._wrap_policy(failing)
    with pytest.raises(RuntimeError, match="native failure"):
        wrapper(*case)


def test_ordinary_config_never_imports_beam_backend(policy: Any) -> None:
    case = _case()
    case[1].gr.beam_max_width = None
    assert _installed(policy)(*case) is False
    assert "vllm_gr.v1.attention.backends.beam_attn" not in policy.imports


@pytest.mark.parametrize("device", ["cuda", "cpu"])  # type: ignore[misc]
def test_other_platform_install_does_not_import_ascend(policy: Any, device: str) -> None:
    policy.modules["vllm.platforms"].current_platform.device_type = device
    assert policy.module.apply_npu_contiguous_kv_cache_patch() == 0
    assert policy.imports == ["vllm.platforms"]


@pytest.mark.parametrize("revision", ["0.22.1rc1", "0.30.0", "0.30.0rc2+local", "0.31.0"])  # type: ignore[misc]
def test_unknown_vendor_version_does_not_import_allocator(
    policy: Any, monkeypatch: pytest.MonkeyPatch, revision: str
) -> None:
    monkeypatch.setattr(policy.module, "version", lambda name: revision)
    assert policy.module.apply_npu_contiguous_kv_cache_patch() == 0
    assert policy.imports == ["vllm.platforms"]


@pytest.mark.parametrize("mismatch", ["missing", "different", "signature"])  # type: ignore[misc]
def test_unknown_binding_contract_is_not_overwritten(policy: Any, mismatch: str) -> None:
    runner = policy.modules["vllm_ascend.worker.model_runner_v1"]
    if mismatch == "missing":
        del runner.requires_contiguous_pa_kv_cache
    elif mismatch == "different":
        runner.requires_contiguous_pa_kv_cache = lambda layer, vllm_config, spec: False
    else:

        def changed(layer: Any, vllm_config: Any, spec: Any, extra: Any = None) -> bool:
            return False

        for name in (
            "vllm_ascend.worker.model_runner_v1",
            "vllm_ascend.attention.utils",
        ):
            policy.modules[name].requires_contiguous_pa_kv_cache = changed
    before = getattr(runner, "requires_contiguous_pa_kv_cache", None)
    assert policy.module.apply_npu_contiguous_kv_cache_patch() == 0
    assert getattr(runner, "requires_contiguous_pa_kv_cache", None) is before


@pytest.mark.parametrize(  # type: ignore[misc]
    "alias", ["vllm_ascend.attention.utils", "vllm_ascend.worker.model_runner_v1"]
)
def test_idempotence_and_known_original_alias_repair(policy: Any, alias: str) -> None:
    wrapper = _installed(policy)
    assert policy.module.apply_npu_contiguous_kv_cache_patch() == 0
    setattr(policy.modules[alias], "requires_contiguous_pa_kv_cache", policy.native)
    assert policy.module.apply_npu_contiguous_kv_cache_patch() == 1
    assert policy.modules[alias].requires_contiguous_pa_kv_cache is wrapper
    assert policy.module.apply_npu_contiguous_kv_cache_patch() == 0
    assert wrapper(*_case()) is True
    assert len(policy.calls) == 1


def test_foreign_rebinding_after_install_is_preserved(policy: Any) -> None:
    wrapper = _installed(policy)
    runner = policy.modules["vllm_ascend.worker.model_runner_v1"]

    def foreign(layer: Any, vllm_config: Any, spec: Any) -> bool:
        return False

    runner.requires_contiguous_pa_kv_cache = foreign
    assert policy.module.apply_npu_contiguous_kv_cache_patch() == 0
    assert runner.requires_contiguous_pa_kv_cache is foreign
    assert policy.modules["vllm_ascend.attention.utils"].requires_contiguous_pa_kv_cache is wrapper


def test_module_import_is_stdlib_only() -> None:
    path = REPO_ROOT / "vllm_gr/v1/worker/npu/kv_layout_patch.py"
    spec = importlib.util.spec_from_file_location("_npu_kv_layout_no_runtime", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert isinstance(module, ModuleType)
    assert callable(module.apply_npu_contiguous_kv_cache_patch)


pytestmark = pytest.mark.cpu_test
