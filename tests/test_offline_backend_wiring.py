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

import sys
from enum import Enum
from types import ModuleType, SimpleNamespace
from typing import Any, Iterator
from unittest import mock

import msgspec
import pytest

from vllm_gr.config import ConstraintBackend, GRConfig

# ``arg_utils_gr`` / ``gr`` are cached in sys.modules once a process imports
# them (possibly against the real vLLM in CI).  Each test re-imports them
# against its own fakes so patch state cannot leak across tests.
_VLLM_GR_REIMPORT = (
    "vllm_gr.engine.arg_utils_gr",
    "vllm_gr.entrypoints.gr",
)


def _drop_reimported_modules() -> None:
    for name in list(sys.modules):
        if name in _VLLM_GR_REIMPORT:
            del sys.modules[name]
    # ``from vllm_gr.engine import arg_utils_gr`` resolves through the cached
    # parent package's ``__dict__`` before falling back to sys.modules, so
    # dropping the sys.modules entry alone leaves the stale submodule attribute
    # on the package.  The next test would then import the previous test's
    # module (whose ``EngineArgs`` / ``origin_*`` globals point at the old fake
    # class) instead of re-executing against its own fakes.
    for parent_name, submodule_name in (
        ("vllm_gr.engine", "arg_utils_gr"),
        ("vllm_gr.entrypoints", "gr"),
    ):
        parent = sys.modules.get(parent_name)
        if parent is not None and hasattr(parent, submodule_name):
            delattr(parent, submodule_name)


def _install_fake_vllm_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide every vLLM module the offline GRLLM import chain needs."""

    # ``arg_utils_gr`` patches this class (add_cli_args / create_engine_config /
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
    class BeamSearchParams(  # type: ignore[call-arg]
        msgspec.Struct,  # type: ignore[misc]
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
        "vllm.engine.arg_utils": _module("vllm.engine.arg_utils", EngineArgs=EngineArgs),
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
        # ``vllm_gr.utils.catalog`` imports ``TokenizerLike`` at runtime; the
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
    """Fake vLLM per test and re-import arg_utils_gr / gr against the fakes.

    Both vllm-gr modules are cached process-wide (in CI they may already be
    imported against the real vLLM).  Re-importing them per test keeps the
    patch state deterministic without mutating the real vLLM classes.
    """
    _install_fake_vllm_modules(monkeypatch)
    _drop_reimported_modules()
    yield
    _drop_reimported_modules()


def _fake_config() -> SimpleNamespace:
    """VllmConfig stand-in: additional_config is a plain mapping."""
    return SimpleNamespace(additional_config={})


class TestEngineArgsGrInitPatch:
    """The offline path reaches EngineCore only through this patch."""

    def test_patch_mounts_and_stashes_gr_values(self) -> None:
        from vllm_gr.engine.arg_utils_gr import patch_add_cli_args

        patch_add_cli_args()

        from vllm.engine.arg_utils import EngineArgs

        engine_args = EngineArgs(
            model="Qwen/Qwen3-1.7B",
            catalog_path="/models/triples.json",
            constraint_backend="constraint_table",
        )
        assert engine_args.catalog_path == "/models/triples.json"
        assert engine_args.constraint_backend == "constraint_table"

    def test_patch_applies_engine_defaults_and_preserves_overrides(self) -> None:
        from vllm_gr.engine.arg_utils_gr import patch_add_cli_args

        patch_add_cli_args()

        from vllm.engine.arg_utils import EngineArgs

        defaults = EngineArgs(model="Qwen/Qwen3-1.7B")
        assert defaults.max_model_len == 8192
        assert defaults.max_num_batched_tokens == 8192
        assert defaults.enable_chunked_prefill is False
        assert defaults.enable_prefix_caching is True

        overridden = EngineArgs(
            model="Qwen/Qwen3-1.7B",
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
            max_model_len=32768,
        )
        assert long_context.max_model_len == 32768
        assert long_context.max_num_batched_tokens == 32768

        chunked_long_context = EngineArgs(
            model="Qwen/Qwen3-1.7B",
            max_model_len=32768,
            enable_chunked_prefill=True,
        )
        assert chunked_long_context.max_num_batched_tokens == 8192

        explicit_batched_limit = EngineArgs(
            model="Qwen/Qwen3-1.7B",
            max_model_len=32768,
            max_num_batched_tokens=16384,
        )
        assert explicit_batched_limit.max_num_batched_tokens == 16384

    def test_create_engine_config_writes_constraint_backend(self) -> None:
        """The threaded values reach VllmConfig via the patched factory."""
        from vllm_gr.engine import arg_utils_gr
        from vllm_gr.engine.arg_utils_gr import create_engine_config, patch_add_cli_args

        patch_add_cli_args()

        from vllm.engine.arg_utils import EngineArgs

        engine_args = EngineArgs(
            model=None,  # model=None skips the vocab-slice resolution block
            catalog_path="/models/triples.json",
            constraint_backend="constraint_table",
        )
        vllm_config = _fake_config()
        with mock.patch.object(
            arg_utils_gr,
            "origin_create_engine_config",
            return_value=vllm_config,
        ):
            result = create_engine_config(engine_args)

        parsed = GRConfig.from_vllm_config(result)
        assert parsed.constraint_backend is ConstraintBackend.CONSTRAINT_TABLE
        assert parsed.catalog_path == "/models/triples.json"

    def test_plain_construction_writes_nothing(self) -> None:
        """A plain EngineArgs must not change the (default) GR config."""
        from vllm_gr.engine import arg_utils_gr
        from vllm_gr.engine.arg_utils_gr import create_engine_config, patch_add_cli_args

        patch_add_cli_args()

        from vllm.engine.arg_utils import EngineArgs

        engine_args = EngineArgs(model=None)
        vllm_config = _fake_config()
        with mock.patch.object(
            arg_utils_gr,
            "origin_create_engine_config",
            return_value=vllm_config,
        ):
            result = create_engine_config(engine_args)

        assert GRConfig.from_vllm_config(result) == GRConfig()


class TestGrllmCanonicalConstruction:
    def _import_gr(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        import vllm_gr.entrypoints.gr as gr
        import vllm_gr.entrypoints.openai.beam_search_patch as beam_search_patch
        import vllm_gr.plugin as plugin

        monkeypatch.setattr(plugin, "initialize_runtime", lambda: {"patch_count": 0})
        monkeypatch.setattr(beam_search_patch, "patch_engine_core_process_entrypoint", lambda: None)
        monkeypatch.setattr(gr, "_install_renderer_eos_token_id_cache", lambda: None)
        return gr

    def test_plain_construction_preserves_native_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gr = self._import_gr(monkeypatch)
        llm = gr.GRLLM(model="model")
        assert llm._last_kwargs == {}
        assert not hasattr(llm, "constraint_strategy")

    @pytest.mark.parametrize("constrained_generation", [False, True])  # type: ignore[misc]
    def test_explicit_constrained_generation_reaches_native_constructor(
        self, monkeypatch: pytest.MonkeyPatch, constrained_generation: bool
    ) -> None:
        gr = self._import_gr(monkeypatch)
        llm = gr.GRLLM(model="model", constrained_generation=constrained_generation)

        assert llm._last_kwargs == {"constrained_generation": constrained_generation}

    @pytest.mark.parametrize(
        "kwargs",
        [  # type: ignore[misc]
            {"catalog_path": "/missing/catalog.json"},
            {"constraint_backend": "trie"},
            {"constraint_backend": "constraint_table"},
        ],
    )
    def test_legacy_arguments_rejected_before_initialization(
        self, monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, str]
    ) -> None:
        gr = self._import_gr(monkeypatch)
        initialize = mock.Mock(side_effect=AssertionError("must reject before runtime startup"))
        monkeypatch.setattr("vllm_gr.plugin.initialize_runtime", initialize)
        with pytest.raises(ValueError, match="removed.*vllm_gr_config"):
            gr.GRLLM(model="model", **kwargs)
        initialize.assert_not_called()

    def test_canonical_config_passed_without_legacy_injection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gr = self._import_gr(monkeypatch)
        config = {"schema_version": 1, "beam": {"execution_mode": "v1"}}
        llm = gr.GRLLM(model="model", vllm_gr_config=config, async_scheduling=True)
        assert llm._last_kwargs == {"vllm_gr_config": config, "async_scheduling": True}
        assert llm._last_kwargs["vllm_gr_config"] is config
        assert not hasattr(llm, "constraint_coordinator")


pytestmark = pytest.mark.cpu_test
