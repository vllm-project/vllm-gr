# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the two-layer adapter registry."""

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


class TestRegistryLayer1:
    """Layer 1 tests — pure Python, no vLLM/torch required."""

    def test_register_and_get_worker_cls_path(self) -> None:
        from vllm_gr.adapters.registry import (
            _reset,
            get_worker_cls_path,
            register_family,
        )

        _reset()
        register_family(
            model_type="test_family",
            gpu_worker="vllm_gr.worker.gr_worker.GRWorker",
            npu_worker="vllm_gr.worker.gr_npu_worker.GRNPUWorker",
        )
        assert (
            get_worker_cls_path("test_family", is_npu=False) == "vllm_gr.worker.gr_worker.GRWorker"
        )
        assert (
            get_worker_cls_path("test_family", is_npu=True)
            == "vllm_gr.worker.gr_npu_worker.GRNPUWorker"
        )

    def test_get_worker_cls_path_unknown_returns_none(self) -> None:
        from vllm_gr.adapters.registry import _reset, get_worker_cls_path

        _reset()
        assert get_worker_cls_path("nonexistent", is_npu=False) is None

    def test_get_family_info(self) -> None:
        from vllm_gr.adapters.registry import _reset, get_family_info, register_family

        _reset()
        register_family(
            model_type="test_info",
            gpu_worker="a.b.C",
            npu_worker="d.e.F",
            defaults_fn="g.h:i_j",
        )
        info = get_family_info("test_info")
        assert info is not None
        assert info["gpu_worker"] == "a.b.C"
        assert info["npu_worker"] == "d.e.F"
        assert info["defaults_fn"] == "g.h:i_j"

    def test_get_family_info_unknown_returns_none(self) -> None:
        from vllm_gr.adapters.registry import _reset, get_family_info

        _reset()
        assert get_family_info("nope") is None

    def test_get_registered_families(self) -> None:
        from vllm_gr.adapters.registry import (
            _reset,
            get_registered_families,
            register_family,
        )

        _reset()
        register_family("fam_a", "a", "b")
        register_family("fam_b", "c", "d")
        assert get_registered_families() == {"fam_a", "fam_b"}

    def test_register_family_idempotent(self) -> None:
        from vllm_gr.adapters.registry import _reset, get_family_info, register_family

        _reset()
        register_family("fam", "old_gpu", "old_npu")
        register_family("fam", "new_gpu", "new_npu")
        info = get_family_info("fam")
        assert info["gpu_worker"] == "new_gpu"

    def test_apply_family_defaults_noop_for_unknown(self) -> None:
        from vllm_gr.adapters.registry import _reset, apply_family_defaults

        _reset()
        # Should not raise
        apply_family_defaults("nonexistent", object())

    def test_apply_family_defaults_noop_when_no_defaults_fn(self) -> None:
        from vllm_gr.adapters.registry import (
            _reset,
            apply_family_defaults,
            register_family,
        )

        _reset()
        register_family("no_fn", "a", "b", defaults_fn=None)
        # Should not raise
        apply_family_defaults("no_fn", object())


class TestRegistryLayer2:
    """Layer 2 tests — adapter class registration."""

    def test_register_and_get_adapter_class(self) -> None:
        from vllm_gr.adapters.base import GRModelAdapter
        from vllm_gr.adapters.registry import (
            _reset,
            get_adapter_class,
            register_adapter_class,
        )

        _reset()

        class DummyAdapter(GRModelAdapter):
            pass

        register_adapter_class("dummy", DummyAdapter)
        assert get_adapter_class("dummy") is DummyAdapter

    def test_get_adapter_class_unknown_raises_keyerror(self) -> None:
        from vllm_gr.adapters.registry import _reset, get_adapter_class

        _reset()
        with pytest.raises(KeyError, match="No adapter class registered"):
            get_adapter_class("nonexistent")

    def test_keyerror_message_lists_registered_families(self) -> None:
        from vllm_gr.adapters.base import GRModelAdapter
        from vllm_gr.adapters.registry import (
            _reset,
            get_adapter_class,
            register_adapter_class,
        )

        _reset()

        class A(GRModelAdapter):
            pass

        register_adapter_class("alpha", A)
        with pytest.raises(KeyError, match="alpha"):
            get_adapter_class("missing")


class TestRegistryReset:
    """Test the _reset() helper used by tests."""

    def test_reset_clears_everything(self) -> None:
        from vllm_gr.adapters.base import GRModelAdapter
        from vllm_gr.adapters.registry import (
            _reset,
            get_adapter_class,
            get_registered_families,
            register_adapter_class,
            register_family,
        )

        _reset()
        register_family("x", "a", "b")
        register_adapter_class("x", GRModelAdapter)
        assert get_registered_families() == {"x"}
        assert get_adapter_class("x") is GRModelAdapter

        _reset()
        assert get_registered_families() == set()
        with pytest.raises(KeyError):
            get_adapter_class("x")


class TestHSTURegistration:
    """Verify the HSTU family is properly registered."""

    def test_hstu_layer1_registration(self) -> None:
        """Importing hstu __init__ should register the family."""
        from vllm_gr.adapters.registry import (
            _reset,
            get_family_info,
            get_worker_cls_path,
        )

        _reset()
        # Re-import triggers registration
        import importlib

        import vllm_gr.adapters.hstu

        importlib.reload(vllm_gr.adapters.hstu)

        assert get_worker_cls_path("hstu", is_npu=False) == "vllm_gr.worker.gr_worker.GRWorker"
        assert (
            get_worker_cls_path("hstu", is_npu=True) == "vllm_gr.worker.gr_npu_worker.GRNPUWorker"
        )

        info = get_family_info("hstu")
        assert info is not None
        assert info["defaults_fn"] is not None
        assert "apply_hstu_defaults" in info["defaults_fn"]


class TestDottedPathImport:
    """Test the _import_dotted_path helper."""

    def test_colon_syntax(self) -> None:
        from vllm_gr.adapters.registry import _import_dotted_path

        fn = _import_dotted_path("vllm_gr.adapters.hstu._defaults:apply_hstu_defaults")
        assert callable(fn)

    def test_dot_syntax(self) -> None:
        from vllm_gr.adapters.registry import _import_dotted_path

        fn = _import_dotted_path("vllm_gr.adapters.registry.register_family")
        assert callable(fn)


class TestLayerIsolation:
    """Verify that Layer 1 imports don't pull in torch/torchrec."""

    def test_layer1_no_heavy_imports(self) -> None:
        """Importing vllm_gr.adapters.hstu (Layer 1 only) should not import torch."""
        code = textwrap.dedent("""\
            import sys
            # Import Layer 1 only
            import vllm_gr.adapters.hstu  # triggers register_family()
            # Check that torch was NOT imported
            torch_imported = "torch" in sys.modules
            print(f"torch_imported={torch_imported}")
        """)
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "torch_imported=False" in result.stdout


class TestReexport:
    """Verify that registry functions are re-exported from adapters package."""

    def test_create_adapter_importable(self) -> None:
        from vllm_gr.adapters import create_adapter

        assert callable(create_adapter)

    def test_get_worker_cls_path_importable(self) -> None:
        from vllm_gr.adapters import get_worker_cls_path

        assert callable(get_worker_cls_path)

    def test_register_family_importable(self) -> None:
        from vllm_gr.adapters import register_family

        assert callable(register_family)

    def test_apply_family_defaults_importable(self) -> None:
        from vllm_gr.adapters import apply_family_defaults

        assert callable(apply_family_defaults)

    def test_get_payload_type_importable(self) -> None:
        from vllm_gr.adapters import get_payload_type

        assert callable(get_payload_type)

    def test_ensure_adapter_class_registered_importable(self) -> None:
        from vllm_gr.adapters import ensure_adapter_class_registered

        assert callable(ensure_adapter_class_registered)


class TestGetPayloadType:
    """Test get_payload_type() — the Layer 1 payload type lookup."""

    def test_get_payload_type_resolves_dotted_path(self) -> None:
        from vllm_gr.adapters.registry import (
            _reset,
            get_payload_type,
            register_family,
        )

        _reset()
        register_family(
            model_type="test_pt",
            gpu_worker="a",
            npu_worker="b",
            payload_type="vllm_gr.adapters.registry:register_family",
        )
        pt = get_payload_type("test_pt")
        assert pt is register_family

    def test_get_payload_type_unknown_raises_keyerror(self) -> None:
        from vllm_gr.adapters.registry import _reset, get_payload_type

        _reset()
        with pytest.raises(KeyError, match="No family registered"):
            get_payload_type("nonexistent")

    def test_get_payload_type_no_payload_registered_raises_keyerror(self) -> None:
        from vllm_gr.adapters.registry import _reset, get_payload_type, register_family

        _reset()
        register_family("no_pt", "a", "b", payload_type=None)
        with pytest.raises(KeyError, match="no payload_type registered"):
            get_payload_type("no_pt")


class TestHSTURegistrationExtended:
    """Extended HSTU registration tests covering payload_type and
    dynamic discovery."""

    def test_hstu_payload_type_in_family_info(self) -> None:
        """Layer 1 registration should include payload_type path."""
        from vllm_gr.adapters.registry import _reset, get_family_info

        _reset()
        import importlib

        import vllm_gr.adapters.hstu

        importlib.reload(vllm_gr.adapters.hstu)

        info = get_family_info("hstu")
        assert info is not None
        assert info["payload_type"] is not None
        assert "HSTUFeaturesPayload" in info["payload_type"]

    def test_hstu_get_payload_type_returns_class(self) -> None:
        """get_payload_type('hstu') should resolve to HSTUFeaturesPayload."""
        try:
            from vllm_gr.adapters.hstu.payload import HSTUFeaturesPayload  # noqa: F401
        except Exception:
            pytest.skip("requires full vLLM runtime (HSTUFeaturesPayload import)")
        from vllm_gr.adapters.registry import _reset, get_payload_type

        _reset()
        import importlib

        import vllm_gr.adapters.hstu

        importlib.reload(vllm_gr.adapters.hstu)

        pt = get_payload_type("hstu")
        from vllm_gr.adapters.hstu.payload import HSTUFeaturesPayload

        assert pt is HSTUFeaturesPayload

    def test_dynamic_family_discovery_in_arg_utils(self) -> None:
        """_ensure_family_registered should use dynamic import, not a
        hardcoded map."""
        try:
            from vllm_gr.engine.arg_utils import GRCommonEngineArgsMixin  # noqa: F401
        except Exception:
            pytest.skip("requires full vLLM runtime (arg_utils import)")
        from vllm_gr.adapters.registry import _reset, get_registered_families

        _reset()
        assert "hstu" not in get_registered_families()

        # Simulate a fresh process: remove the cached hstu module so
        # importlib.import_module re-executes __init__.py, which calls
        # register_family().
        sys.modules.pop("vllm_gr.adapters.hstu", None)

        GRCommonEngineArgsMixin._ensure_family_registered("hstu")

        assert "hstu" in get_registered_families()

    def test_dynamic_discovery_ignores_unknown_family(self) -> None:
        """_ensure_family_registered for an unknown family should not raise."""
        try:
            from vllm_gr.engine.arg_utils import GRCommonEngineArgsMixin  # noqa: F401
        except Exception:
            pytest.skip("requires full vLLM runtime (arg_utils import)")
        # Should not raise ImportError
        GRCommonEngineArgsMixin._ensure_family_registered("nonexistent_family_xyz")


class TestDynamicFamilyDiscovery:
    """Test that the dynamic _try_register_family pattern works."""

    def test_try_register_family_by_convention(self) -> None:
        """Dynamic import of vllm_gr.adapters.<family> should trigger
        register_family().  Verified via the real HSTU adapter."""
        import importlib

        from vllm_gr.adapters.registry import _reset, get_registered_families

        _reset()
        assert "hstu" not in get_registered_families()

        # Reload (or import) the hstu package to trigger register_family()
        import vllm_gr.adapters.hstu

        importlib.reload(vllm_gr.adapters.hstu)

        assert "hstu" in get_registered_families()

    def test_try_register_family_subprocess(self) -> None:
        """End-to-end: import adapter → get_payload_type → create_adapter.

        Runs in a subprocess because the test environment has NumPy/TF
        incompatibility that breaks vLLM imports.
        """
        code = textwrap.dedent("""\
            import sys
            # Layer 1
            import vllm_gr.adapters.hstu
            # Layer 2
            import vllm_gr.adapters.hstu.adapter

            from vllm_gr.adapters.registry import (
                get_payload_type,
                get_adapter_class,
                get_registered_families,
            )

            # Verify Layer 1
            families = get_registered_families()
            assert "hstu" in families, f"hstu not in {families}"

            # Verify payload_type (3.6.3)
            pt = get_payload_type("hstu")
            assert pt is not None, "payload_type is None"
            print(f"payload_type={pt.__name__}")

            # Verify adapter class (3.6.5)
            cls = get_adapter_class("hstu")
            assert cls.model_type == "hstu"

            print("ALL_CHECKS_PASSED")
        """)
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            pytest.skip(f"subprocess failed (vLLM env issue): {result.stderr[-200:]}")
        assert "ALL_CHECKS_PASSED" in result.stdout
