# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Offline ``GRLLM`` backend wiring into the EngineCore config.

GRLLM forwards canonical configuration and native arguments to vLLM, rejects
retired frontend constraint arguments before initialization, and leaves artifact
loading to the workers. The retained EngineArgs compatibility layer is tested
separately from the public GRLLM constructor.

The modules under test import real vLLM symbols at import time, so the vLLM
module graph is faked here — in every environment, including CI, where the real
vLLM modules are shadowed for the duration of these tests only.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Iterator
from contextlib import ExitStack
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from unittest import mock

import msgspec
import pytest

# ``arg_utils`` / ``gr`` are cached in sys.modules once a process imports
# them (possibly against the real vLLM in CI).  Each test re-imports them
# against its own fakes so patch state cannot leak across tests.
_VLLM_GR_REIMPORT = (
    "vllm_gr.engine.arg_utils",
    "vllm_gr.entrypoints.gr",
    "vllm_gr.entrypoints.llm",
)


def _install_fake_vllm_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide every vLLM module the offline GRLLM import chain needs."""

    # ``arg_utils`` patches this class (add_cli_args / create_engine_config /
    # from_cli_args / __init__), so it must carry the four originals first.
    class EngineArgs:
        def __init__(self, model: str = "", **kwargs: Any) -> None:
            self.model = model
            for key, value in kwargs.items():
                setattr(self, key, value)

        def add_cli_args(self, parser: Any) -> Any:  # replaced by the patch
            raise NotImplementedError

        def create_engine_config(  # replaced by the patch
            self, usage_context: Any = None, headless: bool = False
        ) -> Any:
            raise NotImplementedError

        @classmethod
        def from_cli_args(cls, args: Any) -> Any:  # replaced by the patch
            raise NotImplementedError

    class AsyncEngineArgs(EngineArgs):
        pass

    class LLM:
        """Stand-in for vLLM's LLM base; records the super().__init__ kwargs.

        Models the contract ``GRLLM.__init__`` relies on after the runtime
        wiring landed: ``llm_engine.vllm_config.additional_config`` carries
        the GR values the (patched) ``EngineArgs.create_engine_config``
        threaded through ``super().__init__`` kwargs — the same round-trip
        the real flow produces, so GRLLM's post-super resolution reads back
        exactly what it passed down.
        """

        def __init__(self, model: str = "", **kwargs: Any) -> None:
            self.model = model
            self._last_kwargs = kwargs
            additional_config: dict[str, Any] = {}
            catalog_path = kwargs.get("catalog_path")
            constraint_backend = kwargs.get("constraint_backend")
            if catalog_path is not None:
                additional_config["vllm_gr_catalog_path"] = catalog_path
            if constraint_backend is not None:
                additional_config["vllm_gr_constraint_backend"] = str(constraint_backend)
            self.llm_engine = SimpleNamespace(
                # Post-wiring GRLLM calls model_config.get_vocab_size() for
                # the Constraint Table install; model the method form.
                model_config=SimpleNamespace(vocab_size=100, get_vocab_size=lambda: 100),
                vllm_config=SimpleNamespace(additional_config=additional_config),
            )

        def get_tokenizer(self) -> Any:
            return SimpleNamespace()

    class BeamSearchSequence:
        pass

    class BeamSearchInstance:
        pass

    class BeamSearchOutput:
        pass

    class SamplingParams:
        def __init__(self, n: int = 1, **kwargs: Any) -> None:
            self.n = n

    # vllm_gr.sampling_params subclasses the upstream BeamSearchParams, which is
    # a msgspec Struct (``dict=True, omit_defaults=True`` subclass kwargs).
    class BeamSearchParams(
        msgspec.Struct,
        dict=True,
        omit_defaults=True,
    ):
        def __post_init__(self) -> None:
            pass

    class FlatLogprobs:
        pass

    class TokensPrompt:
        pass

    class LoRARequest:
        pass

    class VLLMValidationError(ValueError):
        pass

    class EngineCoreRequest:
        pass

    class FlexibleArgumentParser:
        pass

    class AttentionBackendEnum(str, Enum):
        CUSTOM = "CUSTOM"

    def _module(name: str, **symbols: Any) -> ModuleType:
        module = ModuleType(name)
        for key, value in symbols.items():
            setattr(module, key, value)
        return module

    modules: dict[str, ModuleType] = {
        "vllm.engine": ModuleType("vllm.engine"),
        "vllm.engine.arg_utils": _module(
            "vllm.engine.arg_utils", EngineArgs=EngineArgs, AsyncEngineArgs=AsyncEngineArgs
        ),
        "vllm.utils": ModuleType("vllm.utils"),
        # ``GRLLM.__init__`` calls ``vllm_gr.plugin.initialize_runtime()`` →
        # ``check_environment()`` → ``from vllm.platforms import
        # current_platform``.  The real ``vllm.platforms`` would import
        # ``vllm.utils.import_utils`` through the fake non-package
        # ``vllm.utils`` above, so it must be faked as well.  A non-npu
        # device_type keeps the check on the version-only path.
        "vllm.platforms": _module(
            "vllm.platforms",
            current_platform=SimpleNamespace(device_type="cpu"),
        ),
        # The entrypoint imports tokenizer types at runtime; the
        # real ``vllm.tokenizers`` package pulls in ``vllm.utils.import_utils``,
        # which the fake ``vllm.utils`` above shadows.  Faking it here keeps
        # this file self-contained when run in isolation.
        "vllm.tokenizers": _module("vllm.tokenizers", TokenizerLike=Any),
        "vllm.utils.argparse_utils": _module(
            "vllm.utils.argparse_utils",
            FlexibleArgumentParser=FlexibleArgumentParser,
        ),
        "vllm.exceptions": _module("vllm.exceptions", VLLMValidationError=VLLMValidationError),
        "vllm.inputs": _module("vllm.inputs", TokensPrompt=TokensPrompt),
        "vllm.logprobs": _module("vllm.logprobs", FlatLogprobs=FlatLogprobs),
        "vllm.sampling_params": _module(
            "vllm.sampling_params",
            SamplingParams=SamplingParams,
            BeamSearchParams=BeamSearchParams,
        ),
        "vllm.lora": ModuleType("vllm.lora"),
        "vllm.lora.request": _module("vllm.lora.request", LoRARequest=LoRARequest),
        "vllm.v1": ModuleType("vllm.v1"),
        "vllm.v1.engine": _module("vllm.v1.engine", EngineCoreRequest=EngineCoreRequest),
        "vllm.v1.attention": ModuleType("vllm.v1.attention"),
        "vllm.v1.attention.backends": ModuleType("vllm.v1.attention.backends"),
        "vllm.v1.attention.backends.registry": _module(
            "vllm.v1.attention.backends.registry",
            AttentionBackendEnum=AttentionBackendEnum,
        ),
        "vllm.entrypoints": ModuleType("vllm.entrypoints"),
        "vllm.entrypoints.llm": _module("vllm.entrypoints.llm", LLM=LLM),
        "vllm.entrypoints.generate": ModuleType("vllm.entrypoints.generate"),
        "vllm.entrypoints.generate.beam_search": ModuleType(
            "vllm.entrypoints.generate.beam_search"
        ),
        "vllm.entrypoints.generate.beam_search.utils": _module(
            "vllm.entrypoints.generate.beam_search.utils",
            BeamSearchSequence=BeamSearchSequence,
            BeamSearchInstance=BeamSearchInstance,
            BeamSearchOutput=BeamSearchOutput,
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture(autouse=True)  # type: ignore[misc]
def _isolated_vllm_imports(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fake vLLM per test and re-import arg_utils / gr against the fakes.

    Both vllm-gr modules are cached process-wide (in CI they may already be
    imported against the real vLLM).  Re-importing them per test keeps the
    patch state deterministic without mutating the real vLLM classes.
    """
    from vllm_gr.engine import startup

    # These fakes exercise typed argument forwarding. Native CLI conversion is
    # covered in tests/config; do not rely on a parser cached by another test.
    parser = argparse.ArgumentParser(add_help=False)
    with ExitStack() as restore:
        # Preserve the original module and parent bindings. A later re-import
        # could otherwise capture live, already-patched EngineArgs methods.
        for name in _VLLM_GR_REIMPORT:
            parent_name, _, child_name = name.rpartition(".")
            parent = importlib.import_module(parent_name)
            for namespace, key in ((sys.modules, name), (vars(parent), child_name)):
                if key in namespace:
                    restore.callback(namespace.__setitem__, key, namespace.pop(key))
                else:
                    restore.callback(namespace.pop, key, None)
        with monkeypatch.context() as isolated:
            isolated.setattr(startup, "_native_parser", lambda: parser)
            _install_fake_vllm_modules(isolated)
            yield


def _fake_config() -> SimpleNamespace:
    """VllmConfig stand-in: additional_config is a plain mapping."""
    return SimpleNamespace(
        additional_config={}, scheduler_config=SimpleNamespace(async_scheduling=True)
    )


class TestEngineArgsGrInitPatch:
    """The offline path reaches EngineCore only through this patch."""

    def test_patch_mounts_and_stashes_gr_values(self) -> None:
        from vllm_gr.engine.arg_utils import patch_add_cli_args

        patch_add_cli_args()

        from vllm.engine.arg_utils import EngineArgs

        engine_args = EngineArgs(
            model="Qwen/Qwen3-1.7B",
            gr={"beam_width": 8},
        )
        assert engine_args.gr == {"beam_width": 8}
        # Native LLM.from_engine_args forwards vars(engine_args) to its constructor.
        assert EngineArgs(**vars(engine_args)).gr == engine_args.gr

    def test_patch_applies_engine_defaults_and_preserves_overrides(self) -> None:
        from vllm_gr.engine.arg_utils import patch_add_cli_args

        patch_add_cli_args()

        from vllm.engine.arg_utils import EngineArgs

        native = EngineArgs(model="Qwen/Qwen3-1.7B")
        assert native._gr_selected is False
        assert not hasattr(native, "max_model_len")

        defaults = EngineArgs(model="Qwen/Qwen3-1.7B", gr={})
        assert defaults._gr_selected is True
        assert defaults.max_model_len == 8192
        assert defaults.max_num_batched_tokens == 8192
        assert defaults.enable_chunked_prefill is False
        assert defaults.enable_prefix_caching is True

        overridden = EngineArgs(
            model="Qwen/Qwen3-1.7B",
            gr={},
            max_model_len=4096,
            max_num_batched_tokens=4096,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
        )
        assert overridden.max_model_len == 4096
        assert overridden.max_num_batched_tokens == 4096
        assert overridden.enable_chunked_prefill is True
        assert overridden.enable_prefix_caching is False

        long_context = EngineArgs(
            model="Qwen/Qwen3-1.7B",
            gr={},
            max_model_len=32768,
        )
        assert long_context.max_model_len == 32768
        assert long_context.max_num_batched_tokens == 32768

        chunked_long_context = EngineArgs(
            model="Qwen/Qwen3-1.7B",
            gr={},
            max_model_len=32768,
            enable_chunked_prefill=True,
        )
        assert chunked_long_context.max_num_batched_tokens == 8192

        explicit_batched_limit = EngineArgs(
            model="Qwen/Qwen3-1.7B",
            gr={},
            max_model_len=32768,
            max_num_batched_tokens=16384,
        )
        assert explicit_batched_limit.max_num_batched_tokens == 16384

    @pytest.mark.l1  # type: ignore[misc]
    def test_unconstrained_default_needs_no_artifact(self) -> None:
        """The default Beam engine must bind a model-specific constraint table."""
        from vllm_gr.engine import arg_utils
        from vllm_gr.engine.arg_utils import create_engine_config, patch_add_cli_args

        patch_add_cli_args()

        from vllm.engine.arg_utils import EngineArgs

        engine_args = EngineArgs(model=None)
        vllm_config = _fake_config()
        with mock.patch.object(
            arg_utils,
            "origin_create_engine_config",
            return_value=vllm_config,
        ):
            result = create_engine_config(engine_args)
        from vllm_gr.config import GRConfig

        assert GRConfig.from_vllm_config(result).constraint_table is None


class TestGrllmCanonicalConstruction:
    def _import_gr(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        import logging

        # These setup hooks touch native libraries. The constructor tests
        # exercise argument forwarding using the lightweight LLM above.
        for name, symbols in {
            "vllm.logger": {"init_logger": logging.getLogger},
            "vllm_gr.utils.npu_adapter": {"release_accelerator_memory": mock.Mock()},
            "vllm_gr.plugin": {"initialize_runtime": mock.Mock()},
        }.items():
            module = ModuleType(name)
            vars(module).update(cast(dict[str, Any], symbols))
            monkeypatch.setitem(sys.modules, name, module)

        from vllm_gr.entrypoints import gr

        monkeypatch.setattr(gr, "_install_renderer_eos_token_id_cache", lambda: None)
        return gr

    def test_plain_construction_preserves_native_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gr = self._import_gr(monkeypatch)
        native = {"max_model_len": 4096, "dtype": "float16"}
        llm = gr.GRLLM(model="model", **native)
        assert {
            key: value for key, value in llm._last_kwargs.items() if not key.startswith("_gr_")
        } == native
        assert llm._last_kwargs["_gr_mode"] is True
        assert set(native) <= llm._last_kwargs["_gr_explicit_native_fields"]
        assert not hasattr(llm, "constraint_strategy")

    @pytest.mark.parametrize(
        "kwargs",
        [  # type: ignore[misc]
            {"catalog_path": "/missing/catalog.json"},
            {"constrained_generation": False},
            {"constraint_backend": "trie"},
            {"constraint_backend": "constraint_table"},
        ],
    )
    def test_legacy_arguments_rejected_before_initialization(
        self, monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, str]
    ) -> None:
        gr = self._import_gr(monkeypatch)
        initialize = mock.Mock(side_effect=AssertionError("must reject before runtime startup"))
        monkeypatch.setattr(sys.modules["vllm_gr.plugin"], "initialize_runtime", initialize)
        with pytest.raises(ValueError, match="Removed.*config"):
            gr.GRLLM(model="model", **kwargs)
        initialize.assert_not_called()

    def test_canonical_config_passed_without_legacy_injection(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        gr = self._import_gr(monkeypatch)
        config = tmp_path / "scenario.yaml"
        config.write_text("max-model-len: 4096\ngr: {}\n")
        llm = gr.GRLLM(model="model", config=config, async_scheduling=True)
        assert llm._last_kwargs["config"] == config
        assert "max_model_len" not in llm._last_kwargs
        assert llm._last_kwargs["async_scheduling"] is True
        assert llm._last_kwargs["_gr_startup_document"]["max-model-len"] == 4096
        assert "async_scheduling" in llm._last_kwargs["_gr_explicit_native_fields"]
        assert "max_model_len" not in llm._last_kwargs["_gr_explicit_native_fields"]
        assert not hasattr(llm, "constraint_coordinator")


pytestmark = pytest.mark.cpu_test
