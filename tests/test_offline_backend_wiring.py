# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline ``GRLLM`` backend wiring into the EngineCore config.

Regression for the review finding: ``GRLLM(model, catalog_path=...,
constraint_backend="constraint_table")`` must write the constraint backend into
the VllmConfig before EngineCore is constructed, otherwise EngineCore never
marks requests ``is_constraint_prefill`` / ``is_constrained`` and the Worker
raw-logits hook never runs.

vLLM 0.22.1's ``LLM.__init__`` builds its ``EngineArgs`` internally
(``EngineArgs(model=..., **kwargs)``), so the fix threads ``catalog_path`` /
``constraint_backend`` through the ``super().__init__`` kwargs and the
``arg_utils_gr`` patch stashes them on ``EngineArgs`` (mirroring what
``from_cli_args`` does on the CLI path) so ``create_engine_config`` merges them
into VllmConfig.

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
        """Stand-in for vLLM's LLM base; records the super().__init__ kwargs."""

        def __init__(self, model: str = "", **kwargs: Any) -> None:
            self.model = model
            self._last_kwargs = kwargs
            self.llm_engine = SimpleNamespace(model_config=SimpleNamespace(vocab_size=100))

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

    def test_patch_is_inactive_by_default(self) -> None:
        from vllm_gr.engine import arg_utils_gr

        assert arg_utils_gr.engine_args_accepts_gr_values() is False

    def test_patch_mounts_and_stashes_gr_values(self) -> None:
        from vllm_gr.engine import arg_utils_gr
        from vllm_gr.engine.arg_utils_gr import patch_add_cli_args

        patch_add_cli_args()
        assert arg_utils_gr.engine_args_accepts_gr_values() is True

        from vllm.engine.arg_utils import EngineArgs

        engine_args = EngineArgs(
            model="Qwen/Qwen3-1.7B",
            catalog_path="/models/triples.json",
            constraint_backend="constraint_table",
        )
        assert engine_args.catalog_path == "/models/triples.json"
        assert engine_args.constraint_backend == "constraint_table"

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


class TestGrllmThreadsBackendIntoSuper:
    """GRLLM.__init__ must forward the GR values into the EngineArgs kwargs."""

    def _import_gr_and_patch_strategy(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        """Import GRLLM with the strategy/install call sites stubbed out."""
        import vllm_gr.entrypoints.gr as gr
        from vllm_gr.constraints.strategies import ConstraintTableStrategy

        monkeypatch.setattr(
            gr,
            "create_constraint_strategy",
            lambda backend, tokenizer, catalog_path: ConstraintTableStrategy(catalog_path),
        )
        monkeypatch.setattr(
            gr,
            "install_constraint_table",
            lambda engine, *, catalog_path, tokenizer, vocab_size: SimpleNamespace(digest="d-123"),
        )
        return gr

    def test_grllm_forwards_constraint_table_to_super(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from vllm_gr.engine.arg_utils_gr import patch_add_cli_args

        patch_add_cli_args()
        self._import_gr_and_patch_strategy(monkeypatch)
        from vllm_gr.entrypoints.gr import GRLLM

        llm = GRLLM(
            model="Qwen/Qwen3-1.7B",
            catalog_path="/models/triples.json",
            constraint_backend="constraint_table",
        )

        # super().__init__ received the values for EngineArgs(model=..., **kwargs).
        assert llm._last_kwargs["catalog_path"] == "/models/triples.json"
        assert llm._last_kwargs["constraint_backend"] == "constraint_table"
        # The offline strategy wiring still completes (compile + inject + digest).
        assert llm.constraint_strategy.digest == "d-123"

    def test_grllm_default_construction_threads_none_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from vllm_gr.engine.arg_utils_gr import patch_add_cli_args

        patch_add_cli_args()
        self._import_gr_and_patch_strategy(monkeypatch)
        from vllm_gr.entrypoints.gr import GRLLM

        llm = GRLLM(model="Qwen/Qwen3-1.7B")

        assert llm._last_kwargs["catalog_path"] is None
        assert llm._last_kwargs["constraint_backend"] is ConstraintBackend.TRIE
        assert llm.constraint_strategy is None
        assert llm.catalog is None

    def test_grllm_raises_when_gr_patch_not_active(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without the arg_utils patch, a constrained GRLLM must fail loudly."""
        self._import_gr_and_patch_strategy(monkeypatch)
        from vllm_gr.entrypoints.gr import GRLLM

        with pytest.raises(RuntimeError, match="runtime patches are not active"):
            GRLLM(
                model="Qwen/Qwen3-1.7B",
                catalog_path="/models/triples.json",
                constraint_backend="constraint_table",
            )

    def test_grllm_plain_construction_without_patch_is_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A default GRLLM (no catalog) must work without the patch."""
        self._import_gr_and_patch_strategy(monkeypatch)
        from vllm_gr.entrypoints.gr import GRLLM

        llm = GRLLM(model="Qwen/Qwen3-1.7B")

        assert llm._last_kwargs == {}
        assert llm.constraint_strategy is None

    def test_grllm_trie_catalog_without_patch_is_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A Trie catalog (default backend) must not demand the arg_utils patch.

        Regression: the Pangu OneRec offline benchmark constructs
        ``GRLLM(model, catalog_path=...)`` with the default Trie backend.  The
        Trie catalog is handled driver-side, so this must not raise even when
        the vllm-gr plugin has not run yet (vLLM 0.22 loads general plugins
        lazily from inside ``EngineArgs.__init__``).
        """
        self._import_gr_and_patch_strategy(monkeypatch)
        from vllm_gr.entrypoints.gr import GRLLM

        llm = GRLLM(model="Qwen/Qwen3-1.7B", catalog_path="/models/catalog.json")

        assert llm._last_kwargs == {}  # no values threaded (patch inactive)
        assert llm.constraint_strategy is not None
