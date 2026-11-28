# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Production plugin routing with synthetic platform identities and native hooks."""

from unittest.mock import Mock

import pytest

import vllm_gr.platforms as table
from tests.helpers.ascend_platform import AscendGRPlatform
from tests.helpers.native_platform import native_platform
from vllm_gr import patch, plugin
from vllm_gr.engine import startup
from vllm_gr.platforms import compat
from vllm_gr.platforms.ascend import config_patch, environment, spawn_patch

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


class _SyntheticAscendPlatform(AscendGRPlatform.native_platform_type):  # type: ignore[misc]
    pass


def _synthetic_platform(device_type: str) -> object:
    """Return a synthetic native platform for each supported backend."""
    if device_type == "npu":
        return _SyntheticAscendPlatform()
    return native_platform("CUDA", "cuda")


@pytest.mark.parametrize("device_type", ["cuda", "npu"])  # type: ignore[misc]
def test_plugin_routes_validation_and_patches_through_selected_integration(
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

    monkeypatch.setattr(startup, "configure_model_runner_default", hook("defaults", None))
    cuda_check = hook("cuda_environment", {})
    npu_check = hook("npu_environment", {})
    monkeypatch.setattr(compat, "check_vllm_runtime", cuda_check)
    monkeypatch.setattr(environment, "check_ascend_environment", npu_check)
    spawn = hook("spawn", True)
    worker = hook("sanitize", None)
    monkeypatch.setattr(spawn_patch, "apply_npu_spawn_patch", spawn)
    monkeypatch.setattr(config_patch, "apply_engine_core_config_sanitizer", worker)
    beam = hook("beam", 1)
    monkeypatch.setattr(patch, "_apply_gr_beam_patches", beam)
    monkeypatch.setattr(patch, "get_last_beam_patch_report", lambda: None)

    activate = Mock(wraps=table.get_current_gr_platform)
    monkeypatch.setattr(table, "get_current_gr_platform", activate)
    for first_install in (True, False):
        events.clear()
        activate.reset_mock()
        state = plugin.initialize_runtime()
        activate.assert_called_once_with()
        backend_events = ["spawn", "sanitize"] if device_type == "npu" else []
        assert events == [
            "defaults",
            f"{device_type}_environment",
            *backend_events,
            "beam",
        ]
        assert state == {
            "patch_count": 2 if device_type == "npu" else 1,
            "patches": {},
            "attention_selector": True,
        }
        beam.assert_called_with()

    if device_type == "npu":
        npu_check.assert_called_with(require_npu_platform=True)
        cuda_check.assert_not_called()
    else:
        cuda_check.assert_called_with()
        npu_check.assert_not_called()
        spawn.assert_not_called()
        worker.assert_not_called()


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "platform_name,placement", [("ROCM", "cuda"), ("CPU", "cpu"), ("UNSPECIFIED", "")]
)
def test_ordinary_vllm_startup_on_platform_without_gr(
    monkeypatch: pytest.MonkeyPatch, platform_name: str, placement: str
) -> None:
    import vllm.platforms

    from vllm_gr.platforms.errors import UnsupportedBackendError

    monkeypatch.setattr(
        vllm.platforms,
        "current_platform",
        native_platform(platform_name, placement),
    )
    defaults = Mock()
    monkeypatch.setattr(startup, "configure_model_runner_default", defaults)
    plugin.register()
    with pytest.raises(UnsupportedBackendError):
        plugin.initialize_runtime()
    defaults.assert_not_called()


def test_environment_failure_prevents_runtime_patch_installation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.platforms

    monkeypatch.setattr(vllm.platforms, "current_platform", _synthetic_platform("cuda"))
    monkeypatch.setattr(startup, "configure_model_runner_default", lambda: None)
    monkeypatch.setattr(compat, "check_vllm_runtime", Mock(side_effect=RuntimeError("version")))
    install = Mock()
    monkeypatch.setattr(patch, "re_apply_patches", install)
    with pytest.raises(RuntimeError, match="version"):
        plugin.initialize_runtime()
    install.assert_not_called()


def test_partial_platform_failure_stops_common_plan_and_retains_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.platforms

    # Exercise the direct entry used by a fresh worker, before any env dispatch.
    monkeypatch.setattr(vllm.platforms, "current_platform", _synthetic_platform("npu"))
    spawn = Mock(return_value=True)
    monkeypatch.setattr(spawn_patch, "apply_npu_spawn_patch", spawn)
    failure = RuntimeError("native worker hook failed")
    monkeypatch.setattr(
        config_patch, "apply_engine_core_config_sanitizer", Mock(side_effect=failure)
    )
    beam = Mock()
    monkeypatch.setattr(patch, "_apply_gr_beam_patches", beam)

    with pytest.raises(RuntimeError, match="backend 'ascend'.*partial.*restart") as raised:
        patch.re_apply_patches()
    assert raised.value.__cause__ is failure
    assert table.get_current_gr_platform().backend_id == "ascend"
    spawn.assert_called_once_with()
    beam.assert_not_called()
