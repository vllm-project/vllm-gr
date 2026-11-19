# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Production plugin routing with synthetic platform identities and native hooks."""

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_gr import _env_check, npu_spawn_patch, patch, plugin
from vllm_gr.platforms import loader
from vllm_gr.platforms.registry import (
    _reset_backend_registry_for_tests,
    get_active_backend,
)
from vllm_gr.v1 import ascend_compat

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


@pytest.fixture(autouse=True)  # type: ignore[misc]
def isolated_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    _reset_backend_registry_for_tests()
    monkeypatch.setattr(loader, "entry_points", lambda **kwargs: ())
    yield
    _reset_backend_registry_for_tests()


class _SyntheticAscendPlatform:
    """Ascend identity built from the vendor class lineage, without the vendor."""

    _enum = "OOT"
    device_type = "npu"
    device_name = "npu"


_SyntheticAscendPlatform.__module__ = "vllm_ascend.platform"
_SyntheticAscendPlatform.__qualname__ = "NPUPlatform"


def _synthetic_platform(device_type: str) -> object:
    """Return a synthetic native platform for each supported backend."""
    if device_type == "npu":
        return _SyntheticAscendPlatform()
    return SimpleNamespace(_enum="CUDA", device_type="cuda", device_name="cuda")


@pytest.mark.parametrize("device_type", ["cuda", "npu"])  # type: ignore[misc]
def test_plugin_routes_validation_and_patches_through_selected_provider(
    monkeypatch: pytest.MonkeyPatch, device_type: str
) -> None:
    import vllm.platforms

    monkeypatch.setattr(vllm.platforms, "current_platform", _synthetic_platform(device_type))
    events: list[str] = []

    def hook(name: str, result: object) -> Mock:
        def invoke(*args: object, **kwargs: object) -> object:
            events.append(name)
            return result

        return Mock(side_effect=invoke)

    monkeypatch.setattr(_env_check, "configure_model_runner_default", hook("defaults", None))
    cuda_check = hook("cuda_environment", {})
    npu_check = hook("npu_environment", {})
    monkeypatch.setattr(_env_check, "check_runtime_versions", cuda_check)
    monkeypatch.setattr(_env_check, "check_ascend_environment", npu_check)
    monkeypatch.setattr(
        patch.pangu_compat, "apply_pangu_tied_embeddings_patch", hook("pangu", True)
    )
    attention = hook("attention", 1)
    monkeypatch.setattr(patch.selector_patch, "apply_custom_attention_backend_patch", attention)
    spawn = hook("spawn", True)
    worker = hook("worker", 1)
    monkeypatch.setattr(npu_spawn_patch, "apply_npu_spawn_patch", spawn)
    monkeypatch.setattr(ascend_compat, "patch_npu_worker_init_device", worker)
    monkeypatch.setattr(npu_spawn_patch, "is_npu_spawn_patched", lambda: spawn.called)
    monkeypatch.setattr(ascend_compat, "is_npu_worker_patched", lambda: worker.called)
    beam = hook("beam", 1)
    monkeypatch.setattr(patch, "_apply_gr_beam_patches", beam)
    monkeypatch.setattr(patch, "get_last_beam_patch_report", lambda: None)

    for _ in range(2):
        events.clear()
        state = plugin.initialize_runtime()
        expected_backend_id = "ascend" if device_type == "npu" else "cuda"
        assert get_active_backend().backend_id == expected_backend_id
        backend_events = ["spawn", "worker"] if device_type == "npu" else []
        assert events == [
            "defaults",
            f"{device_type}_environment",
            "pangu",
            "attention",
            *backend_events,
            "beam",
        ]
        assert state == {"patch_count": 5 if device_type == "npu" else 3, "patches": {}}
        attention.assert_called_with(force=True)
        beam.assert_called_with(force=True)

    if device_type == "npu":
        npu_check.assert_called_with(require_npu_platform=True)
        cuda_check.assert_not_called()
    else:
        cuda_check.assert_called_with()
        npu_check.assert_not_called()
        spawn.assert_not_called()
        worker.assert_not_called()


def test_environment_failure_prevents_runtime_patch_installation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.platforms

    monkeypatch.setattr(vllm.platforms, "current_platform", _synthetic_platform("cuda"))
    monkeypatch.setattr(_env_check, "configure_model_runner_default", lambda: None)
    monkeypatch.setattr(
        _env_check, "check_runtime_versions", Mock(side_effect=RuntimeError("version"))
    )
    install = Mock()
    monkeypatch.setattr(patch, "re_apply_patches", install)
    with pytest.raises(RuntimeError, match="version"):
        plugin.initialize_runtime()
    install.assert_not_called()


def test_partial_provider_failure_stops_common_plan_and_retains_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.platforms

    # Exercise the direct entry used by a fresh worker, before any env dispatch.
    monkeypatch.setattr(vllm.platforms, "current_platform", _synthetic_platform("npu"))
    monkeypatch.setattr(patch.pangu_compat, "apply_pangu_tied_embeddings_patch", lambda: False)
    monkeypatch.setattr(
        patch.selector_patch, "apply_custom_attention_backend_patch", lambda **kw: 0
    )
    spawn = Mock(return_value=True)
    monkeypatch.setattr(npu_spawn_patch, "apply_npu_spawn_patch", spawn)
    failure = RuntimeError("native worker hook failed")
    monkeypatch.setattr(ascend_compat, "patch_npu_worker_init_device", Mock(side_effect=failure))
    beam = Mock()
    monkeypatch.setattr(patch, "_apply_gr_beam_patches", beam)

    with pytest.raises(RuntimeError, match="backend 'ascend'.*partial.*restart") as raised:
        patch.re_apply_patches()
    assert raised.value.__cause__ is failure
    assert get_active_backend().backend_id == "ascend"
    spawn.assert_called_once_with()
    beam.assert_not_called()
