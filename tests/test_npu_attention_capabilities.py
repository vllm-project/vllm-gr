# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disallow-untyped-decorators=False

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from vllm_gr.platforms.ascend_services import AscendBeamAttentionService
from vllm_gr.v1.attention.backends import beam_attn

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


@pytest.fixture
def attention_config(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    import vllm.config

    config = SimpleNamespace(
        additional_config={
            "vllm_gr_npu_decode_execution": "cascade",
            "vllm_gr_enable_short_suffix_attention": False,
            "vllm_gr_prefill_flash_attn_version": None,
        }
    )
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setattr(beam_attn, "_attention_service", lambda: AscendBeamAttentionService())
    monkeypatch.setattr(beam_attn.envs, "VLLM_BATCH_INVARIANT", False)
    return config


def _options(**features: Any) -> dict[str, Any]:
    options = dict(
        num_heads=2,
        head_size=128,
        scale=128**-0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )
    options.update(features)
    return options


def _assert_rejected_before_initialization(
    monkeypatch: pytest.MonkeyPatch, features: dict[str, Any]
) -> None:
    resolve = Mock(side_effect=AssertionError("must reject before resolving execution module"))
    tensor = Mock(side_effect=AssertionError("must reject before allocating ALiBi tensor"))
    monkeypatch.setattr(beam_attn, "_get_platform_execution_module", resolve)
    monkeypatch.setattr(torch, "tensor", tensor)

    # Call the real initializer on an observable empty instance so a rejected
    # configuration cannot leave partially initialized attention state behind.
    impl = object.__new__(beam_attn.BeamAttentionImpl)
    with pytest.raises(NotImplementedError) as error:
        beam_attn.BeamAttentionImpl.__init__(impl, **_options(**features))

    for name in features:
        assert name in str(error.value)
    assert vars(impl) == {}
    resolve.assert_not_called()
    tensor.assert_not_called()


@pytest.mark.parametrize(
    "features",
    [
        pytest.param({"sliding_window": 1}, id="window-one"),
        pytest.param({"sliding_window": 128}, id="window-128"),
        pytest.param({"sliding_window": 0}, id="window-zero-is-not-none"),
        pytest.param({"sliding_window": -1}, id="raw-minus-one-is-not-sentinel"),
        pytest.param({"alibi_slopes": [0.5, 0.25]}, id="alibi-nonzero"),
        pytest.param({"alibi_slopes": [0.0, 0.0]}, id="alibi-all-zero"),
        pytest.param({"alibi_slopes": []}, id="alibi-empty-is-not-none"),
        pytest.param({"logits_soft_cap": 1.0}, id="softcap-one"),
        pytest.param({"logits_soft_cap": 50.0}, id="softcap-positive"),
        pytest.param({"logits_soft_cap": -1.0}, id="softcap-negative"),
        pytest.param({"logits_soft_cap": float("nan")}, id="softcap-nan"),
    ],
)
def test_npu_rejects_unsupported_attention_features_before_initialization(
    monkeypatch: pytest.MonkeyPatch,
    attention_config: SimpleNamespace,
    features: dict[str, Any],
) -> None:
    _assert_rejected_before_initialization(monkeypatch, features)


@pytest.mark.parametrize(
    "features",
    [
        {"sliding_window": 128, "alibi_slopes": [0.0, 0.0]},
        {"alibi_slopes": [], "logits_soft_cap": 50.0},
        {"sliding_window": 128, "alibi_slopes": [0.5, 0.25], "logits_soft_cap": 50.0},
    ],
)
def test_npu_reports_all_unsupported_features_together(
    monkeypatch: pytest.MonkeyPatch,
    attention_config: SimpleNamespace,
    features: dict[str, Any],
) -> None:
    _assert_rejected_before_initialization(monkeypatch, features)


@pytest.mark.parametrize("mode", ["cascade", "xllm"])
def test_npu_rejection_does_not_require_loading_selected_execution_module(
    monkeypatch: pytest.MonkeyPatch,
    attention_config: SimpleNamespace,
    mode: str,
) -> None:
    attention_config.additional_config["vllm_gr_npu_decode_execution"] = mode
    # The resolver would import xllm when requested. Rejecting before it is
    # called also works on hosts where that optional extension is unavailable.
    _assert_rejected_before_initialization(monkeypatch, {"sliding_window": 128})


@pytest.mark.parametrize("soft_cap", [None, 0.0, -0.0])
def test_npu_default_features_keep_existing_normalization(
    monkeypatch: pytest.MonkeyPatch,
    attention_config: SimpleNamespace,
    soft_cap: float | None,
) -> None:
    flash_version = Mock(return_value=None)
    execution = SimpleNamespace(get_flash_attn_version=flash_version)
    resolve = Mock(return_value=execution)
    # Substitute only the execution module: the shipped Ascend service carries
    # the real admissibility answers, but resolving its module needs the vendor
    # runtime, which this host does not have.
    service = AscendBeamAttentionService()
    monkeypatch.setattr(service, "execution_module", resolve)
    monkeypatch.setattr(beam_attn, "_attention_service", lambda: service)

    impl = beam_attn.BeamAttentionImpl(**_options(logits_soft_cap=soft_cap))

    assert impl.sliding_window == (-1, -1)
    assert impl.alibi_slopes is None
    assert impl.logits_soft_cap == 0.0
    assert impl.prefill_flash_attn_version is None
    resolve.assert_called_once_with()
    flash_version.assert_called_once_with(requires_alibi=False, head_size=128)


@pytest.mark.parametrize(
    "features",
    [
        {"sliding_window": 128},
        {"alibi_slopes": [0.5, 0.25]},
        {"logits_soft_cap": 50.0},
        {"sliding_window": 128, "alibi_slopes": [0.5, 0.25], "logits_soft_cap": 50.0},
    ],
)
def test_cuda_features_are_not_rejected_by_npu_capability_guard(
    monkeypatch: pytest.MonkeyPatch,
    attention_config: SimpleNamespace,
    features: dict[str, Any],
) -> None:
    flash_version = Mock(return_value=2)
    execution = SimpleNamespace(get_flash_attn_version=flash_version)
    from .attention_service_utils import attention_service

    monkeypatch.setattr(beam_attn, "_attention_service", lambda: attention_service(execution))

    impl = beam_attn.BeamAttentionImpl(**_options(**features))

    expected_window = (
        (features["sliding_window"] - 1, 0) if "sliding_window" in features else (-1, -1)
    )
    assert impl.sliding_window == expected_window
    if "alibi_slopes" in features:
        assert isinstance(impl.alibi_slopes, torch.Tensor)
        assert impl.alibi_slopes.dtype == torch.float32
        assert impl.alibi_slopes.tolist() == features["alibi_slopes"]
    else:
        assert impl.alibi_slopes is None
    assert impl.logits_soft_cap == features.get("logits_soft_cap", 0.0)
    assert impl.vllm_flash_attn_version == 2
    assert impl.prefill_flash_attn_version == 2
    flash_version.assert_called_once_with(requires_alibi="alibi_slopes" in features, head_size=128)
