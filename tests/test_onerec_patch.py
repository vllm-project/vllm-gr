# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for benchmarks.open_one_rec.one_rec_patch.

Verifies the patch-triggering contract after the refactor that decoupled
package import from side effects:

1. Importing ``benchmarks.open_one_rec`` (the package) MUST NOT trigger the
   monkey-patches against ``vllm.benchmarks.*``. This is what lets unit tests
   and other tooling import the package without polluting global vllm state.

2. Importing ``benchmarks.open_one_rec.one_rec_patch`` explicitly MUST apply
   all 7 patches. This is what the entrypoints (``one_rec_main.py``,
   ``one_rec_acc_test.py``) rely on.

The tests purge ``sys.modules`` between cases so each case observes a clean
import state, mirroring the approach in ``test_yuanrong_monkey_patch.py``.

When real ``vllm`` is unavailable (e.g. sandboxed CI without GPU deps), the
``TestPatchTriggerContractWithRealVllm`` cases auto-skip and the
``TestPatchTriggerContractWithStubVllm`` cases provide equivalent coverage by
installing stub ``vllm.benchmarks.*`` modules.
"""

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _purge_modules(*prefixes: str) -> dict[str, types.ModuleType]:
    """Remove modules matching any prefix from sys.modules, returning the saved snapshot."""
    saved: dict[str, types.ModuleType] = {}
    for name, module in list(sys.modules.items()):
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            saved[name] = module
            sys.modules.pop(name, None)
    return saved


def _restore_modules(saved: dict[str, types.ModuleType]) -> None:
    for name, module in saved.items():
        sys.modules[name] = module


def _drop_modules_created_during(
    prefixes: tuple[str, ...],
    saved: dict[str, types.ModuleType],
) -> None:
    """Remove modules under `prefixes` that were created after the purge snapshot.

    `_restore_modules` only overwrites entries present in the snapshot; modules
    imported for the first time during the test (e.g. a stub-era
    ``benchmarks.open_one_rec.one_rec_patch``) would otherwise leak into
    ``sys.modules`` and shadow the real modules for later test classes.
    """
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            if name not in saved:
                sys.modules.pop(name, None)


def _rebind_parent_attrs(prefixes: tuple[str, ...]) -> None:
    """Re-sync parent-package attribute bindings with ``sys.modules``.

    Importing a submodule binds it as an attribute on its parent package, but
    purge/restore cycles do not undo that binding: after a test that imported
    a fresh ``vllm.benchmarks`` against stubs, ``vllm.benchmarks`` (the
    attribute) can point at a discarded module object while ``sys.modules``
    holds a different one.  Re-bind every module under `prefixes` onto its
    parent so attribute access and ``sys.modules`` agree again.
    """
    for name, module in list(sys.modules.items()):
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            parent_name, _, child = name.rpartition(".")
            if not parent_name:
                continue
            parent = sys.modules.get(parent_name)
            if parent is not None:
                try:
                    setattr(parent, child, module)
                except (AttributeError, TypeError):
                    pass


def _install_stub_third_party() -> dict[str, types.ModuleType]:
    """Install stub modules for third-party libs the patch chain imports.

    The patch module + its transitive imports (open_one_rec_dataset ->
    open_one_rec_loader -> tasks/*) pull in pandas / aiohttp / tqdm /
    transformers / sklearn / numpy / regex. We don't exercise any of their
    functionality in these tests — we only verify patch *registration* — so
    MagicMock stubs (which auto-answer any attribute access) suffice.
    """
    stubs: dict[str, types.ModuleType] = {}
    for name in (
        "pandas",
        "aiohttp",
        "tqdm",
        "transformers",
        "sklearn",
        "sklearn.metrics",
        "numpy",
        "regex",
    ):
        if name not in sys.modules:
            mod = mock.MagicMock(name=name)
            # Mark as package so `from X.Y import Z` resolves to submodule.
            if name in ("sklearn", "transformers", "regex", "numpy"):
                mod.__path__ = []
            sys.modules[name] = mod
            stubs[name] = mod
    return stubs


def _install_stub_vllm_benchmarks() -> dict[str, types.ModuleType]:
    """Install minimal fake vllm.benchmarks modules that one_rec_patch patches.

    Reproduces the surface the patch module touches:
    - vllm                                  (parent package)
    - vllm.benchmarks                       (parent)
    - vllm.benchmarks.datasets              (get_samples, add_dataset_parser)
    - vllm.benchmarks.throughput            (add_cli_args, get_requests, validate_args, main)
    - vllm.benchmarks.lib.endpoint_request_func (ASYNC_REQUEST_FUNCS dict)

    Also satisfies the side-effect imports the patch module performs
    (for example, ``from vllm import LLM``) by leaving them as bare
    ModuleType attributes — the patch module only needs them importable,
    not functional.
    """
    modules: dict[str, types.ModuleType] = {}

    def _get_or_create(name: str) -> types.ModuleType:
        if name in sys.modules:
            return sys.modules[name]
        mod = types.ModuleType(name)
        mod.__path__ = []  # mark as package so submodule imports succeed
        sys.modules[name] = mod
        modules[name] = mod
        # Bind to parent's attribute namespace so code paths that do
        # `vllm.benchmarks.datasets.get_samples = ...` (attribute access, not
        # import statement) resolve correctly. Bare ModuleType stubs in
        # sys.modules do NOT auto-bind to parent attributes.
        parts = name.split(".")
        if len(parts) > 1:
            parent_name = ".".join(parts[:-1])
            parent = sys.modules.get(parent_name)
            if parent is not None:
                setattr(parent, parts[-1], mod)
        return mod

    vllm = _get_or_create("vllm")
    # Patch module does `from vllm import LLM` and `import vllm` (line-level).
    vllm.LLM = object  # type: ignore[attr-defined]

    _get_or_create("vllm.benchmarks")  # create package so submodule imports resolve

    # Pre-create vllm.benchmarks submodules that one_rec_patch may import
    # depending on the vllm version (not all versions have all of these;
    # creating them as empty packages is harmless and avoids ModuleNotFoundError
    # when the stub environment doesn't match the CI vllm version exactly).
    for _sub in ("serve", "latency", "startup"):
        _serve_mod = _get_or_create(f"vllm.benchmarks.{_sub}")
        # Populate symbols that one_rec_patch imports from these modules
        # (different vllm versions surface different names; stub them all).
        for _sym in (
            "add_cli_args",
            "main",
            "main_async",
            "get_requests",
            "get_samples",
            "validate_args",
        ):
            if not hasattr(_serve_mod, _sym):
                setattr(_serve_mod, _sym, lambda *a, **k: None)

    datasets = _get_or_create("vllm.benchmarks.datasets")
    # Original (unpatched) functions — names must NOT start with "patched_".
    datasets.get_samples = lambda *a, **k: None  # type: ignore[attr-defined]
    datasets.add_dataset_parser = lambda *a, **k: None  # type: ignore[attr-defined]

    # The patch module also imports BenchmarkDataset/SampleRequest from this
    # module — give it dummy symbols. BenchmarkDataset needs
    # DEFAULT_SEED and an __init__ that accepts the parent params
    # (dataset_path/random_seed/disable_shuffle) like the real one.
    class _StubBenchmarkDataset:  # mimics vllm.benchmarks.datasets.BenchmarkDataset
        DEFAULT_SEED = 0

        def __init__(self, dataset_path=None, random_seed=None, disable_shuffle=False, **kwargs):
            self.dataset_path = dataset_path
            self.random_seed = random_seed if random_seed is not None else self.DEFAULT_SEED
            self.disable_shuffle = disable_shuffle
            self.data = None

        def maybe_oversample_requests(
            self, requests, num_requests, request_id_prefix="", no_oversample=False
        ):
            """Mirrors the parent class oversample + uniqueness check."""
            import random
            from dataclasses import replace

            if no_oversample:
                return
            if len(requests) < num_requests:
                random.seed(self.random_seed)
                needed = num_requests - len(requests)
                for i in range(needed):
                    req = replace(
                        random.choice(requests),
                        request_id=request_id_prefix + str(len(requests) + i),
                    )
                    requests.append(req)
            ids = [req.request_id for req in requests]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate request_id found in sampled requests.")

    datasets.BenchmarkDataset = _StubBenchmarkDataset  # type: ignore[attr-defined]

    # SampleRequest is a dataclass in vllm; stub it as a dataclass too so
    # dataclasses.replace() in maybe_oversample_requests works.
    from dataclasses import dataclass
    from typing import Any

    @dataclass
    class _StubSampleRequest:
        prompt: Any = None
        prompt_len: int = 0
        expected_output_len: Any = None
        multi_modal_data: Any = None
        lora_request: Any = None
        request_id: str | None = None

    datasets.SampleRequest = _StubSampleRequest  # type: ignore[attr-defined]

    # TokenizerLike is exported from vllm.tokenizers in vLLM 0.22+ rather than
    # transitively through vllm.benchmarks.datasets.
    tokenizers = _get_or_create("vllm.tokenizers")
    tokenizers.TokenizerLike = type("TokenizerLike", (), {})  # type: ignore[attr-defined]

    throughput = _get_or_create("vllm.benchmarks.throughput")
    throughput.add_cli_args = lambda *a, **k: None  # type: ignore[attr-defined]
    throughput.get_requests = lambda *a, **k: None  # type: ignore[attr-defined]
    throughput.validate_args = lambda *a, **k: None  # type: ignore[attr-defined]
    throughput.main = lambda *a, **k: None  # type: ignore[attr-defined]

    # endpoint_request_func: patch module imports a batch of names from it
    erf = _get_or_create("vllm.benchmarks.lib.endpoint_request_func")
    erf.ASYNC_REQUEST_FUNCS = {}  # type: ignore[attr-defined]
    for sym in (
        "RequestFuncInput",
        "RequestFuncOutput",
        "StreamedResponseHandler",
    ):
        setattr(erf, sym, type(sym, (), {}))
    for sym in (
        "_get_chat_content",
        "_update_headers_common",
        "_update_payload_common",
        "_validate_api_url",
    ):
        setattr(erf, sym, lambda *a, **k: None)
    erf.RequestFuncInput.__init__ = lambda self, *a, **k: None

    return modules


def _is_real_vllm_available() -> bool:
    """Probe whether real vllm.benchmarks.datasets imports cleanly."""
    saved = _purge_modules("vllm", "benchmarks.open_one_rec")
    try:
        try:
            import vllm.benchmarks.datasets  # noqa: F401

            return True
        except Exception:  # noqa: BLE001
            return False
    finally:
        _restore_modules(saved)


# ---------------------------------------------------------------------------
# Stub-based tests (run in any environment, no real vllm required)
# ---------------------------------------------------------------------------


class TestPatchTriggerContractWithStubVllm(unittest.TestCase):
    """Verifies the contract using stub vllm.benchmarks modules.

    This runs in any environment (even without GPU / numpy deps) because we
    install fake vllm.benchmarks modules before importing one_rec_patch.
    """

    _STUB_PREFIXES = (
        "vllm",
        # setUp replaces the parent package itself, so save and restore the
        # complete tree.  Saving only sampling_params leaves already-imported
        # children in sys.modules without rebinding them on the restored
        # vllm_gr package, making later string-based monkeypatch resolution
        # depend on test order.
        "vllm_gr",
        "benchmarks.open_one_rec",
    )

    def setUp(self) -> None:
        self._saved = _purge_modules(*self._STUB_PREFIXES)
        self._stub_modules = _install_stub_vllm_benchmarks()
        self._stub_modules.update(_install_stub_third_party())
        # Also stub vllm_gr.sampling_params so the fallback import in the
        # accuracy-test entrypoint can be skipped; patch module imports it
        # indirectly via the BeamSearchParams fallback in one_rec_acc_test.py
        # but not in one_rec_patch.py itself. We stub the parent so any
        # `from vllm.sampling_params import BeamSearchParams` in patch module
        # resolves.
        vllm_gr = types.ModuleType("vllm_gr")
        vllm_gr.__path__ = []
        sys.modules["vllm_gr"] = vllm_gr
        self._stub_modules["vllm_gr"] = vllm_gr

    def tearDown(self) -> None:
        # Drop our stubs first so they don't leak into other tests, then
        # restore whatever was there before.  Stub-era repo-local modules
        # imported against the fake vllm are dropped as well (never drop
        # real "vllm" modules: re-importing them re-runs module-level side
        # effects such as torch opaque-type registration).
        for name in list(self._stub_modules):
            sys.modules.pop(name, None)
        _restore_modules(self._saved)
        _drop_modules_created_during(("vllm_gr", "benchmarks.open_one_rec"), self._saved)
        _rebind_parent_attrs(("vllm", "vllm_gr", "benchmarks"))

    def test_package_import_does_not_patch_vllm(self) -> None:
        """Importing the package alone MUST NOT replace any vllm.benchmarks symbol."""
        import benchmarks.open_one_rec as pkg

        # The package itself must not re-export one_rec_patch (clean __init__).
        self.assertFalse(
            hasattr(pkg, "one_rec_patch"),
            "package __init__ should not bind one_rec_patch as an attribute",
        )

        import vllm.benchmarks.datasets as datasets_mod
        import vllm.benchmarks.throughput as throughput_mod

        self.assertNotEqual(
            datasets_mod.get_samples.__name__,
            "patched_get_samples",
            "get_samples was patched without an explicit one_rec_patch import",
        )
        self.assertNotEqual(
            datasets_mod.add_dataset_parser.__name__,
            "patched_add_parser",
            "add_dataset_parser was patched without an explicit one_rec_patch import",
        )
        for attr, expected_name in (
            ("add_cli_args", "patched_add_cli_args"),
            ("get_requests", "patched_get_requests"),
            ("validate_args", "patched_validate_args"),
            ("main", "patched_main"),
        ):
            actual = getattr(throughput_mod, attr)
            self.assertNotEqual(
                actual.__name__,
                expected_name,
                f"throughput.{attr} was patched without an explicit one_rec_patch import",
            )

    def test_explicit_patch_import_applies_all_seven_patches(self) -> None:
        """Importing one_rec_patch explicitly MUST apply all 7 replacements."""
        # isort: off
        # Import order matters: patch must load before we observe vllm state.
        import benchmarks.open_one_rec.one_rec_patch as patch_mod

        import vllm.benchmarks.datasets as datasets_mod
        import vllm.benchmarks.throughput as throughput_mod
        from vllm.benchmarks.lib.endpoint_request_func import ASYNC_REQUEST_FUNCS
        # isort: on

        self.assertIs(
            datasets_mod.get_samples,
            patch_mod.patched_get_samples,
            "datasets.get_samples not patched",
        )
        self.assertIs(
            datasets_mod.add_dataset_parser,
            patch_mod.patched_add_parser,
            "datasets.add_dataset_parser not patched",
        )
        self.assertIs(
            throughput_mod.add_cli_args,
            patch_mod.patched_add_cli_args,
            "throughput.add_cli_args not patched",
        )
        self.assertIs(
            throughput_mod.get_requests,
            patch_mod.patched_get_requests,
            "throughput.get_requests not patched",
        )
        self.assertIs(
            throughput_mod.validate_args,
            patch_mod.patched_validate_args,
            "throughput.validate_args not patched",
        )
        self.assertIs(
            throughput_mod.main,
            patch_mod.patched_main,
            "throughput.main not patched",
        )
        self.assertIs(
            ASYNC_REQUEST_FUNCS["openai-chat"],
            patch_mod.patched_async_request_openai_chat_completions,
            "ASYNC_REQUEST_FUNCS['openai-chat'] not patched",
        )

    def test_explicit_patch_import_does_not_register_atexit_cleanup(self) -> None:
        """Online benchmark clients must not run engine cleanup at process exit."""
        with mock.patch("atexit.register") as register:
            import benchmarks.open_one_rec.one_rec_patch  # noqa: F401

        register.assert_not_called()

    def test_offline_beam_search_scopes_engine_cleanup(self) -> None:
        """The offline path still closes the GRLLM instance that it owns."""
        import benchmarks.open_one_rec.one_rec_patch as patch_mod
        from benchmarks.open_one_rec import open_one_rec_dataset

        class _StubBeamSearchParams:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        sampling_params = types.ModuleType("vllm_gr.sampling_params")
        sampling_params.BeamSearchParams = _StubBeamSearchParams  # type: ignore[attr-defined]
        sys.modules[sampling_params.__name__] = sampling_params
        self._stub_modules[sampling_params.__name__] = sampling_params

        entrypoints = types.ModuleType("vllm_gr.entrypoints")
        entrypoints.__path__ = []
        gr_module = types.ModuleType("vllm_gr.entrypoints.gr")
        engine = mock.MagicMock()
        engine.__enter__.return_value = engine
        engine.beam_search.return_value = []
        gr_module.GRLLM = mock.Mock(return_value=engine)  # type: ignore[attr-defined]
        entrypoints.gr = gr_module  # type: ignore[attr-defined]
        sys.modules[entrypoints.__name__] = entrypoints
        sys.modules[gr_module.__name__] = gr_module
        self._stub_modules[entrypoints.__name__] = entrypoints
        self._stub_modules[gr_module.__name__] = gr_module

        request = types.SimpleNamespace(prompt="prompt", expected_output_len=2)
        dataset = mock.MagicMock()
        dataset.sample.return_value = [request]
        dataset_class = mock.Mock(return_value=dataset)

        transformers = sys.modules["transformers"]
        args = types.SimpleNamespace(
            model="fake-model",
            dataset_path="./data",
            task_types="ad",
            num_prompts=1,
            n=2,
            output_len=None,
            output_json=None,
        )

        with (
            mock.patch.object(open_one_rec_dataset, "OneRecDataset", dataset_class),
            mock.patch.object(transformers, "AutoTokenizer") as auto_tokenizer,
        ):
            auto_tokenizer.from_pretrained.return_value = object()
            patch_mod._run_offline_beam_search_benchmark(args)

        engine.__enter__.assert_called_once_with()
        engine.__exit__.assert_called_once()

    def test_idempotent_reimport_keeps_patches(self) -> None:
        """Re-importing the patch module (sys.modules cache hit) MUST NOT undo patches."""
        # isort: off
        # Import order matters: patch must load before we observe vllm state.
        import benchmarks.open_one_rec.one_rec_patch as patch_mod

        # Second import resolves from sys.modules cache, no re-execution.
        import benchmarks.open_one_rec.one_rec_patch as patch_mod_again

        self.assertIs(patch_mod, patch_mod_again)

        import vllm.benchmarks.datasets as datasets_mod

        # isort: on
        self.assertIs(
            datasets_mod.get_samples,
            patch_mod.patched_get_samples,
            "re-import dropped the patch on get_samples",
        )

    def test_seven_patches_count_matches(self) -> None:
        """Guard against silent drift: the patch module should register exactly 7 hooks.

        If a future commit adds or removes a hook, this test forces the author
        to update the count and review the contract here.
        """
        import benchmarks.open_one_rec.one_rec_patch as patch_mod

        # Names of patched symbols exported by the module (functions only,
        # not the ASYNC_REQUEST_FUNCS dict entry).
        expected = {
            "patched_get_samples",
            "patched_add_parser",
            "patched_add_cli_args",
            "patched_get_requests",
            "patched_validate_args",
            "patched_main",
            "patched_async_request_openai_chat_completions",
        }
        for name in expected:
            self.assertTrue(
                hasattr(patch_mod, name),
                f"patch module is missing expected symbol {name!r}; has the patch surface changed?",
            )


# ---------------------------------------------------------------------------
# OneRecDataset construction contract (problem 4 regression tests)
# ---------------------------------------------------------------------------


class TestOneRecDatasetInitContract(unittest.TestCase):
    """Verifies the OneRecDataset.__init__ contract after problem 4 fix.

    Covers:
    - 4.1: tokenizer is a real parameter, not stolen from kwargs
    - 4.2: tokenizer is required (fail-fast ValueError)
    - 4.3 / 4.5 / 4.6: parent params (disable_shuffle, random_seed,
      dataset_path) are explicit and forwarded correctly
    - 4.4: enable_thinking is a real parameter, propagated to DataLoaderWrapper
    """

    def setUp(self) -> None:
        # Stub both vllm.benchmarks (for BenchmarkDataset base class) and
        # the third-party libs the loader imports transitively, so this test
        # class runs in any environment without real vllm / pandas / etc.
        self._saved = _purge_modules(
            "vllm",
            "vllm_gr",
            "benchmarks.open_one_rec",
        )
        self._stub_vllm = _install_stub_vllm_benchmarks()
        self._stub_vllm.update(_install_stub_third_party())
        vllm_gr = types.ModuleType("vllm_gr")
        vllm_gr.__path__ = []
        sys.modules["vllm_gr"] = vllm_gr
        self._stub_vllm["vllm_gr"] = vllm_gr

        # Import the module fresh so the `from .open_one_rec_loader import ...`
        # resolves against the stubbed vllm.benchmarks.
        from benchmarks.open_one_rec import open_one_rec_dataset as ds_mod

        self._ds_mod = ds_mod
        # Replace DataLoaderWrapper with a recording stub so we don't need
        # real data files / tokenizer.
        self._loader_calls: list[dict[str, object]] = []

        test_self = self

        class _RecordingLoader:
            def __init__(inner_self, **kwargs):  # noqa: N805
                test_self._loader_calls.append(kwargs)
                inner_self.kwargs = kwargs

            def load_data(inner_self, **kwargs):  # noqa: N805
                return {}

        self._orig_loader = ds_mod.DataLoaderWrapper
        ds_mod.DataLoaderWrapper = _RecordingLoader

    def tearDown(self) -> None:
        self._ds_mod.DataLoaderWrapper = self._orig_loader
        # Drop stubs first so they don't leak, then restore saved modules.
        for name in list(self._stub_vllm):
            sys.modules.pop(name, None)
        _restore_modules(self._saved)
        _drop_modules_created_during(("vllm_gr", "benchmarks.open_one_rec"), self._saved)
        _rebind_parent_attrs(("vllm", "vllm_gr", "benchmarks"))

    def _make(self, **overrides):
        """Build a OneRecDataset with defaults that satisfy validation."""
        defaults: dict[str, object] = {
            "tokenizer": object(),  # any non-None sentinel
            "dataset_path": "/fake/data",
            "task_types": ["video"],
            "sample_size": 1,
        }
        defaults.update(overrides)
        # Import here so the stub is already in place.
        OneRecDataset = self._ds_mod.OneRecDataset
        return OneRecDataset(**defaults)

    # -- 4.2: required param validation -------------------------------------

    def test_missing_tokenizer_raises_with_clear_message(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._make(tokenizer=None)
        self.assertIn("tokenizer", str(ctx.exception))

    # -- 4.1: tokenizer is explicit, not stolen from kwargs -----------------

    def test_tokenizer_in_kwargs_does_not_double_pass_to_parent(self) -> None:
        """Passing tokenizer via kwargs (legacy style) must still work and
        must not leak into BenchmarkDataset.__init__ as an unknown kwarg."""
        # Using the explicit `tokenizer=` param (the supported path).
        ds = self._make(tokenizer="my-tok")
        self.assertIs(ds.tokenizer, "my-tok")

    # -- 4.4: enable_thinking is propagated ---------------------------------

    def test_enable_thinking_propagated_to_loader(self) -> None:
        self._make(enable_thinking=True)
        self.assertEqual(self._loader_calls[-1]["enable_thinking"], True)

    def test_enable_thinking_none_propagated_to_loader(self) -> None:
        # Default: None means "use task config's enable_thinking".
        self._make(enable_thinking=None)
        self.assertIsNone(self._loader_calls[-1]["enable_thinking"])

    # -- 4.3 / 4.5 / 4.6: explicit parent params ----------------------------

    def test_disable_shuffle_forwarded_to_parent(self) -> None:
        ds = self._make(disable_shuffle=True)
        self.assertTrue(ds.disable_shuffle)

    def test_random_seed_forwarded_to_parent(self) -> None:
        ds = self._make(random_seed=42)
        self.assertEqual(ds.random_seed, 42)

    def test_dataset_path_forwarded_to_parent(self) -> None:
        ds = self._make(dataset_path="/custom/path")
        self.assertEqual(ds.dataset_path, "/custom/path")
        # And it's what DataLoaderWrapper receives as data_dir.
        self.assertEqual(self._loader_calls[-1]["data_dir"], "/custom/path")


# ---------------------------------------------------------------------------
# OneRecDataset.sample contract (problem 5 regression tests)
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal tokenizer stub: returns deterministic input_ids by prompt length."""

    def __call__(self, prompt: str):
        return type("Encoded", (), {"input_ids": list(range(len(prompt)))})()


class TestOneRecDatasetSampleContract(unittest.TestCase):
    """Verifies the OneRecDataset.sample contract after problem 5 fix.

    Covers:
    - 5.1 / 5.2: signature aligns with parent (request_id_prefix, no_oversample, return type)
    - 5.3: request_id_prefix is actually used in the generated request_id
    - 5.4: oversample behavior — no_oversample=True returns as-is; False pads to num_requests
    - 5.5: both call paths (get_samples / get_requests) pass consistent params
    """

    def setUp(self) -> None:
        self._saved = _purge_modules(
            "vllm",
            "vllm_gr",
            "benchmarks.open_one_rec",
        )
        self._stub_vllm = _install_stub_vllm_benchmarks()
        self._stub_vllm.update(_install_stub_third_party())
        vllm_gr = types.ModuleType("vllm_gr")
        vllm_gr.__path__ = []
        sys.modules["vllm_gr"] = vllm_gr
        self._stub_vllm["vllm_gr"] = vllm_gr

        from benchmarks.open_one_rec import open_one_rec_dataset as ds_mod

        self._ds_mod = ds_mod

        # Replace DataLoaderWrapper with a stub that injects controlled data.
        class _StubLoader:
            def __init__(inner_self, **kwargs):  # noqa: N805
                inner_self.kwargs = kwargs

            def load_data(inner_self, **kwargs):  # noqa: N805
                # Return a controlled payload: 3 samples for a single task.
                return {
                    "0": {"prompt": "aaa", "ground_truth": "gt0"},
                    "1": {"prompt": "bb", "ground_truth": "gt1"},
                    "2": {"prompt": "c", "ground_truth": "gt2"},
                }

        self._orig_loader = ds_mod.DataLoaderWrapper
        ds_mod.DataLoaderWrapper = _StubLoader

    def tearDown(self) -> None:
        self._ds_mod.DataLoaderWrapper = self._orig_loader
        for name in list(self._stub_vllm):
            sys.modules.pop(name, None)
        _restore_modules(self._saved)
        _drop_modules_created_during(("vllm_gr", "benchmarks.open_one_rec"), self._saved)
        _rebind_parent_attrs(("vllm", "vllm_gr", "benchmarks"))

    def _make_dataset(self, **overrides):
        defaults: dict[str, object] = {
            "tokenizer": _FakeTokenizer(),
            "dataset_path": "/fake/data",
            "task_types": ["video"],
            "sample_size": None,
        }
        defaults.update(overrides)
        return self._ds_mod.OneRecDataset(**defaults)

    # -- 5.3: request_id_prefix is honored ----------------------------------

    def test_request_id_prefix_appears_in_request_id(self) -> None:
        ds = self._make_dataset()
        requests = ds.sample(
            tokenizer=_FakeTokenizer(),
            num_requests=3,
            request_id_prefix="warmup_",
        )
        for req in requests:
            self.assertTrue(req.request_id.startswith("warmup_"))

    def test_empty_prefix_preserves_task_type_id_format(self) -> None:
        ds = self._make_dataset()
        requests = ds.sample(
            tokenizer=_FakeTokenizer(),
            num_requests=3,
            request_id_prefix="",
        )
        # Without prefix, format is "{task_type}_{id}" (legacy behavior).
        self.assertTrue(all(req.request_id.startswith("video_") for req in requests))

    # -- 5.4: oversample behavior -------------------------------------------

    def test_no_oversample_returns_at_most_available(self) -> None:
        ds = self._make_dataset()  # 3 samples available
        requests = ds.sample(
            tokenizer=_FakeTokenizer(),
            num_requests=10,  # request more than available
            no_oversample=True,
        )
        self.assertEqual(len(requests), 3)

    def test_oversample_pads_to_num_requests(self) -> None:
        ds = self._make_dataset()  # 3 samples available
        requests = ds.sample(
            tokenizer=_FakeTokenizer(),
            num_requests=5,
            no_oversample=False,
        )
        self.assertEqual(len(requests), 5)

    def test_oversample_emits_unique_request_ids(self) -> None:
        # Parent's maybe_oversample_requests raises ValueError on duplicate ids.
        ds = self._make_dataset()
        requests = ds.sample(
            tokenizer=_FakeTokenizer(),
            num_requests=5,
            no_oversample=False,
        )
        ids = [req.request_id for req in requests]
        self.assertEqual(len(ids), len(set(ids)))

    # -- 5.1 / 5.2: signature alignment -------------------------------------

    def test_sample_accepts_parent_signature_kwargs(self) -> None:
        """sample() must accept request_id_prefix and no_oversample without
        raising — they are part of the parent's abstract signature."""
        ds = self._make_dataset()
        # Should not raise.
        ds.sample(
            tokenizer=_FakeTokenizer(),
            num_requests=2,
            request_id_prefix="t_",
            no_oversample=True,
        )


# ---------------------------------------------------------------------------
# OneRecDataset.expected_output_len from task config (problem 6 regression)
# ---------------------------------------------------------------------------


class TestOneRecDatasetExpectedOutputLen(unittest.TestCase):
    """Verifies that expected_output_len comes from task_config.generation_config
    instead of a hardcoded default.

    Previously every sample got expected_output_len=5, which silently truncated
    rec_reason (max_new_tokens=2000) and item_understand (=128) to 5 tokens.
    """

    def setUp(self) -> None:
        self._saved = _purge_modules(
            "vllm",
            "vllm_gr",
            "benchmarks.open_one_rec",
        )
        self._stub_vllm = _install_stub_vllm_benchmarks()
        self._stub_vllm.update(_install_stub_third_party())
        vllm_gr = types.ModuleType("vllm_gr")
        vllm_gr.__path__ = []
        sys.modules["vllm_gr"] = vllm_gr
        self._stub_vllm["vllm_gr"] = vllm_gr

        from benchmarks.open_one_rec import open_one_rec_dataset as ds_mod

        self._ds_mod = ds_mod

        class _StubLoader:
            def __init__(inner_self, **kwargs):  # noqa: N805
                pass

            def load_data(inner_self, **kwargs):  # noqa: N805
                # Single sample, keyed by row index.
                return {"0": {"prompt": "p", "ground_truth": "gt"}}

        self._orig_loader = ds_mod.DataLoaderWrapper
        ds_mod.DataLoaderWrapper = _StubLoader

    def tearDown(self) -> None:
        self._ds_mod.DataLoaderWrapper = self._orig_loader
        for name in list(self._stub_vllm):
            sys.modules.pop(name, None)
        _restore_modules(self._saved)
        _drop_modules_created_during(("vllm_gr", "benchmarks.open_one_rec"), self._saved)
        _rebind_parent_attrs(("vllm", "vllm_gr", "benchmarks"))

    def _build_for_task(self, task_type: str):
        return self._ds_mod.OneRecDataset(
            tokenizer=_FakeTokenizer(),
            dataset_path="/fake/data",
            task_types=[task_type],
            sample_size=None,
        )

    def _expected_output_len(self, ds) -> int:
        return ds.data[0]["expected_output_len"]

    def test_recommendation_tasks_use_max_new_tokens_5(self) -> None:
        """ad / product / video / label_cond / interactive all have max_new_tokens=5."""
        for task in ("ad", "product", "video", "label_cond", "interactive"):
            ds = self._build_for_task(task)
            self.assertEqual(
                self._expected_output_len(ds),
                5,
                f"{task} should have expected_output_len=5 from its config",
            )

    def test_rec_reason_uses_max_new_tokens_2000(self) -> None:
        """rec_reason config has max_new_tokens=2000; previously truncated to 5."""
        ds = self._build_for_task("rec_reason")
        self.assertEqual(self._expected_output_len(ds), 2000)

    def test_item_understand_uses_max_new_tokens_128(self) -> None:
        """item_understand config has max_new_tokens=128; previously truncated to 5."""
        ds = self._build_for_task("item_understand")
        self.assertEqual(self._expected_output_len(ds), 128)

    def test_label_pred_uses_max_new_tokens_1(self) -> None:
        """label_pred config has max_new_tokens=1; previously defaulted to 5."""
        ds = self._build_for_task("label_pred")
        self.assertEqual(self._expected_output_len(ds), 1)

    def test_expected_output_len_propagates_to_sample_request(self) -> None:
        """The corrected value must flow through to SampleRequest.expected_output_len."""
        ds = self._build_for_task("rec_reason")
        requests = ds.sample(
            tokenizer=_FakeTokenizer(),
            num_requests=1,
            no_oversample=True,
        )
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].expected_output_len, 2000)

    def test_unknown_task_type_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._build_for_task("not_a_real_task")
        self.assertIn("not_a_real_task", str(ctx.exception))


# ---------------------------------------------------------------------------
# Integration tests against real vllm (auto-skip if unavailable)
# ---------------------------------------------------------------------------


@unittest.skipUnless(_is_real_vllm_available(), "real vllm.benchmarks not importable")
class TestPatchTriggerContractWithRealVllm(unittest.TestCase):
    """Same contract as above, but against the real installed vllm.

    Runs in environments where vllm imports cleanly (dev box, GPU CI).
    Auto-skips in bare/minimal environments (e.g. CPU-only sandboxes).
    """

    _PURGE_PREFIXES = (
        "vllm.benchmarks",
        "benchmarks.open_one_rec",
    )

    def setUp(self) -> None:
        self._saved = _purge_modules(*self._PURGE_PREFIXES)

    def tearDown(self) -> None:
        _restore_modules(self._saved)
        # The fresh real vllm.benchmarks submodules imported during the test
        # were patched by a patch-module instance that we also drop below;
        # removing them keeps the restored snapshot consistent (the patch
        # module binds submodules onto whatever vllm.benchmarks package is
        # in sys.modules at import time).
        _drop_modules_created_during(("vllm.benchmarks", "benchmarks.open_one_rec"), self._saved)
        _rebind_parent_attrs(("vllm", "benchmarks"))

    def test_package_import_does_not_patch_real_vllm(self) -> None:
        # isort: off
        # Import order matters: must observe state after package import.
        import benchmarks.open_one_rec as pkg  # noqa: F401

        import vllm.benchmarks.datasets as datasets_mod
        # isort: on

        self.assertNotEqual(
            datasets_mod.get_samples.__name__,
            "patched_get_samples",
            "importing the package must not patch vllm",
        )

    def test_explicit_patch_import_patches_real_vllm(self) -> None:
        try:
            import benchmarks.open_one_rec.one_rec_patch as patch_mod
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"could not import one_rec_patch against real vllm: {exc}")

        import vllm.benchmarks.datasets as datasets_mod

        self.assertIs(
            datasets_mod.get_samples,
            patch_mod.patched_get_samples,
            "real-vllm datasets.get_samples not patched",
        )


# ---------------------------------------------------------------------------
# BaseLoader._make_metadata_serializable contract (problem 9 regression)
# ---------------------------------------------------------------------------


class TestMakeMetadataSerializable(unittest.TestCase):
    """Verifies the _make_metadata_serializable contract after problem 9 fix.

    Covers:
    - 9.1: no KeyError when 'answer' key is missing (defensive)
    - 9.2: does NOT mutate the input metadata_dict in place
    - 9.3: answer is dropped from the output, row_index is injected
    """

    def setUp(self) -> None:
        self._saved = _purge_modules(
            "vllm",
            "vllm_gr",
            "benchmarks.open_one_rec",
        )
        self._stub_vllm = _install_stub_vllm_benchmarks()
        self._stub_vllm.update(_install_stub_third_party())
        vllm_gr = types.ModuleType("vllm_gr")
        vllm_gr.__path__ = []
        sys.modules["vllm_gr"] = vllm_gr
        self._stub_vllm["vllm_gr"] = vllm_gr

        from benchmarks.open_one_rec import open_one_rec_loader as loader_mod

        self._loader_mod = loader_mod
        # BaseLoader.__init__ requires a non-None tokenizer; bypass __init__
        # since _make_metadata_serializable doesn't touch self state.
        self._loader = loader_mod.BaseLoader.__new__(loader_mod.BaseLoader)

    def tearDown(self) -> None:
        for name in list(self._stub_vllm):
            sys.modules.pop(name, None)
        _restore_modules(self._saved)
        _drop_modules_created_during(("vllm_gr", "benchmarks.open_one_rec"), self._saved)
        _rebind_parent_attrs(("vllm", "vllm_gr", "benchmarks"))

    def test_drops_answer_and_injects_row_index(self) -> None:
        meta = {"answer": "gt_value", "answer_pid": [1, 2], "extra": "keep"}
        result = self._loader._make_metadata_serializable(42, meta)

        self.assertEqual(result["row_index"], 42)
        self.assertNotIn("answer", result)
        self.assertEqual(result["answer_pid"], [1, 2])
        self.assertEqual(result["extra"], "keep")

    def test_does_not_mutate_input_dict(self) -> None:
        """The input metadata_dict must retain its 'answer' key after the call.

        Regression for the old `del metadata_dict['answer']` behavior which
        mutated the caller's dict in place.
        """
        meta = {"answer": "gt_value", "other": 1}
        original = dict(meta)
        self._loader._make_metadata_serializable(0, meta)
        # Input must be unchanged.
        self.assertEqual(meta, original)
        self.assertIn("answer", meta)

    def test_missing_answer_key_does_not_raise(self) -> None:
        """Defensive: if 'answer' is absent, the function should not raise.

        The call site pre-checks for 'answer', but pop/del would still KeyError
        if a future refactor skips that check. The dict-comprehension filter
        is safe by construction.
        """
        meta = {"answer_pid": [1, 2]}
        # Should not raise.
        result = self._loader._make_metadata_serializable(0, meta)
        self.assertNotIn("answer", result)
        self.assertEqual(result["answer_pid"], [1, 2])

    def test_row_index_non_int_falls_back_to_str(self) -> None:
        """When idx has no __int__, row_index becomes str(idx)."""
        meta = {"answer": "x"}
        result = self._loader._make_metadata_serializable("abc", meta)
        self.assertEqual(result["row_index"], "abc")


# ---------------------------------------------------------------------------
# patched_validate_args contract (problem 10 regression)
# ---------------------------------------------------------------------------


class TestPatchedValidateArgs(unittest.TestCase):
    """Verifies the patched_validate_args contract after problem 10 fix.

    Covers:
    - 10.1: onerec path sets tokenizer default from model
    - 10.2: onerec path raises on missing dataset_path with clear message
    - 10.3: non-onerec path delegates to original validator
    """

    def setUp(self) -> None:
        self._saved = _purge_modules(
            "vllm",
            "vllm_gr",
            "benchmarks.open_one_rec",
        )
        self._stub_vllm = _install_stub_vllm_benchmarks()
        self._stub_vllm.update(_install_stub_third_party())
        vllm_gr = types.ModuleType("vllm_gr")
        vllm_gr.__path__ = []
        sys.modules["vllm_gr"] = vllm_gr
        self._stub_vllm["vllm_gr"] = vllm_gr

        from benchmarks.open_one_rec import one_rec_patch as patch_mod

        self._patch_mod = patch_mod

    def tearDown(self) -> None:
        for name in list(self._stub_vllm):
            sys.modules.pop(name, None)
        _restore_modules(self._saved)
        _drop_modules_created_during(("vllm_gr", "benchmarks.open_one_rec"), self._saved)
        _rebind_parent_attrs(("vllm", "vllm_gr", "benchmarks"))

    def _args(self, **overrides):
        from argparse import Namespace

        defaults = {
            "dataset_name": "onerec",
            "model": "fake-model",
            "tokenizer": None,
            "dataset_path": "./data",
        }
        defaults.update(overrides)
        return Namespace(**defaults)

    def test_onerec_sets_tokenizer_default_from_model(self) -> None:
        args = self._args(tokenizer=None)
        self._patch_mod.patched_validate_args(args)
        self.assertEqual(args.tokenizer, "fake-model")

    def test_onerec_keeps_explicit_tokenizer(self) -> None:
        args = self._args(tokenizer="explicit-tok")
        self._patch_mod.patched_validate_args(args)
        self.assertEqual(args.tokenizer, "explicit-tok")

    def test_onerec_missing_dataset_path_raises(self) -> None:
        args = self._args(dataset_path=None)
        with self.assertRaises(ValueError) as ctx:
            self._patch_mod.patched_validate_args(args)
        self.assertIn("--dataset-path", str(ctx.exception))

    def test_onerec_empty_dataset_path_raises(self) -> None:
        args = self._args(dataset_path="")
        with self.assertRaises(ValueError):
            self._patch_mod.patched_validate_args(args)

    def test_non_onerec_delegates_to_original(self) -> None:
        """For non-onerec datasets, the original vllm validator must be called.

        We stub original_validate_args with a sentinel to detect delegation.
        """
        calls: list = []
        sentinel = lambda a: calls.append(a)  # noqa: E731
        original = self._patch_mod.original_validate_args
        self._patch_mod.original_validate_args = sentinel
        try:
            args = self._args(dataset_name="random")
            self._patch_mod.patched_validate_args(args)
            self.assertEqual(len(calls), 1)
            self.assertIs(calls[0], args)
        finally:
            self._patch_mod.original_validate_args = original


class TestP2DModeOption(unittest.TestCase):
    """Unit tests for --p2d-mode CLI option parsing and vllm_xargs handling."""

    _STUB_PREFIXES = (
        "vllm",
        # setUp replaces the parent package itself, so save and restore the
        # complete tree.  Saving only sampling_params leaves already-imported
        # children in sys.modules without rebinding them on the restored
        # vllm_gr package, making later string-based monkeypatch resolution
        # depend on test order.
        "vllm_gr",
        "vllm_gr.sampling_params",
        "benchmarks.open_one_rec",
    )

    def setUp(self) -> None:
        self._saved = _purge_modules(*self._STUB_PREFIXES)
        self._stubs = _install_stub_vllm_benchmarks()
        self._stubs.update(_install_stub_third_party())
        vllm_gr = types.ModuleType("vllm_gr")
        vllm_gr.__path__ = []
        sys.modules["vllm_gr"] = vllm_gr
        self._stubs["vllm_gr"] = vllm_gr
        import benchmarks.open_one_rec.one_rec_patch as patch_mod

        self._patch_mod = patch_mod

    def tearDown(self) -> None:
        for name in list(self._stubs.keys()):
            sys.modules.pop(name, None)
        _restore_modules(self._saved)
        _drop_modules_created_during(("vllm_gr", "benchmarks.open_one_rec"), self._saved)
        _rebind_parent_attrs(("vllm", "vllm_gr", "benchmarks"))

    def test_get_p2d_mode_from_argv(self) -> None:
        with mock.patch.object(sys, "argv", ["script.py", "--p2d-mode", "eager_p2d"]):
            val = self._patch_mod._get_p2d_mode_from_argv()
            self.assertEqual(val, "eager_p2d")

        with mock.patch.object(sys, "argv", ["script.py"]):
            val = self._patch_mod._get_p2d_mode_from_argv()
            self.assertIsNone(val)

    def test_add_beams_arguments_registers_p2d_mode(self) -> None:
        import argparse

        parser = argparse.ArgumentParser()
        self._patch_mod._add_beams_arguments(parser)
        args = parser.parse_args(["--p2d-mode", "monolithic"])
        self.assertEqual(args.p2d_mode, "monolithic")


class TestStepBeamWidthsOption(unittest.TestCase):
    """Verifies --step-beam-widths plumbing in one_rec_patch."""

    def setUp(self) -> None:
        self._saved = _purge_modules(
            "vllm",
            "vllm_gr",
            "benchmarks.open_one_rec",
        )
        self._stub_vllm = _install_stub_vllm_benchmarks()
        self._stub_vllm.update(_install_stub_third_party())
        vllm_gr = types.ModuleType("vllm_gr")
        vllm_gr.__path__ = []
        sys.modules["vllm_gr"] = vllm_gr
        self._stub_vllm["vllm_gr"] = vllm_gr

        from benchmarks.open_one_rec import one_rec_patch as patch_mod

        self._patch_mod = patch_mod

    def tearDown(self) -> None:
        for name in list(self._stub_vllm):
            sys.modules.pop(name, None)
        _restore_modules(self._saved)
        _drop_modules_created_during(("vllm_gr", "benchmarks.open_one_rec"), self._saved)
        _rebind_parent_attrs(("vllm", "vllm_gr", "benchmarks"))

    def test_get_step_beam_widths_from_argv_tokens(self) -> None:
        with mock.patch.object(sys, "argv", ["script.py", "--step-beam-widths", "128,64,32"]):
            self.assertEqual(self._patch_mod._get_step_beam_widths_from_argv(), [128, 64, 32])

    def test_get_step_beam_widths_from_argv_equals(self) -> None:
        with mock.patch.object(sys, "argv", ["script.py", "--step-beam-widths=128,64,32"]):
            self.assertEqual(self._patch_mod._get_step_beam_widths_from_argv(), [128, 64, 32])

    def test_get_step_beam_widths_from_argv_malformed(self) -> None:
        for bad in ("128,64", "128,-1,32", "abc,1,2", "128,64,32,16"):
            with mock.patch.object(sys, "argv", ["script.py", "--step-beam-widths", bad]):
                self.assertIsNone(self._patch_mod._get_step_beam_widths_from_argv())

    def test_get_step_beam_widths_from_argv_absent(self) -> None:
        with mock.patch.object(sys, "argv", ["script.py"]):
            self.assertIsNone(self._patch_mod._get_step_beam_widths_from_argv())

    def test_parse_step_beam_widths_value(self) -> None:
        self.assertEqual(self._patch_mod._parse_step_beam_widths_value("128,64,32"), [128, 64, 32])
        self.assertIsNone(self._patch_mod._parse_step_beam_widths_value("128,64"))
        self.assertIsNone(self._patch_mod._parse_step_beam_widths_value("abc"))
        self.assertIsNone(self._patch_mod._parse_step_beam_widths_value(None))

    def test_add_beams_arguments_registers_step_beam_widths(self) -> None:
        import argparse

        parser = argparse.ArgumentParser()
        self._patch_mod._add_beams_arguments(parser)
        args = parser.parse_args(["--step-beam-widths", "32,128,32"])
        self.assertEqual(args.step_beam_widths, "32,128,32")

    def test_request_func_input_carries_step_beam_widths(self) -> None:
        from benchmarks.open_one_rec.one_rec_patch import RequestFuncInput

        req = RequestFuncInput(model="m", prompt="p", step_beam_widths=[128, 64, 32])
        self.assertEqual(req.step_beam_widths, [128, 64, 32])
        req_default = RequestFuncInput(model="m", prompt="p")
        self.assertEqual(req_default.step_beam_widths, self._patch_mod.ARGS_STEP_BEAM_WIDTHS)

    def test_patched_async_request_openai_chat_completions_includes_step_beam_widths(
        self,
    ) -> None:
        import asyncio

        from benchmarks.open_one_rec.one_rec_patch import RequestFuncInput

        patch = self._patch_mod
        patch._validate_api_url = lambda *a, **k: None
        patch._update_payload_common = lambda *a, **k: None
        patch._update_headers_common = lambda *a, **k: None
        patch._get_chat_content = lambda *a, **k: "hello"
        patch.RequestFuncOutput = type("RequestFuncOutput", (), {})
        patch.StreamedResponseHandler = type("StreamedResponseHandler", (), {})

        async def _run():
            fake_resp = mock.MagicMock()
            fake_resp.status = 400
            fake_resp.__aenter__ = mock.AsyncMock(return_value=fake_resp)
            fake_resp.__aexit__ = mock.AsyncMock(return_value=False)
            fake_session = mock.MagicMock()
            fake_session.post.return_value = fake_resp
            req = RequestFuncInput(model="m", prompt="p", step_beam_widths=[128, 64, 32])
            req.api_url = "http://localhost:8000/v1/chat/completions"
            req.model_name = None
            req.model = "m"
            req.output_len = 8
            req.prompt_len = 10
            await patch.patched_async_request_openai_chat_completions(req, fake_session)
            return fake_session.post.call_args.kwargs["json"]

        payload = asyncio.run(_run())
        self.assertEqual(payload["vllm_xargs"]["step_beam_widths"], [128, 64, 32])
        self.assertTrue(payload["use_beam_search"])
        self.assertEqual(payload["n"], 32)

        async def _run_absent():
            fake_resp = mock.MagicMock()
            fake_resp.status = 400
            fake_resp.__aenter__ = mock.AsyncMock(return_value=fake_resp)
            fake_resp.__aexit__ = mock.AsyncMock(return_value=False)
            fake_session = mock.MagicMock()
            fake_session.post.return_value = fake_resp
            req = RequestFuncInput(model="m", prompt="p")
            req.api_url = "http://localhost:8000/v1/chat/completions"
            req.model_name = None
            req.model = "m"
            req.output_len = 8
            req.prompt_len = 10
            await patch.patched_async_request_openai_chat_completions(req, fake_session)
            return fake_session.post.call_args.kwargs["json"]

        payload_absent = asyncio.run(_run_absent())
        self.assertNotIn("step_beam_widths", payload_absent["vllm_xargs"])


class TestPrintBeamSearchDelta(unittest.TestCase):
    def test_print_beam_search_delta_format(self):
        import io
        from unittest.mock import patch

        from benchmarks.open_one_rec.one_rec_patch import _print_beam_search_delta

        before = {
            "requests": 0,
            "p2d_requests": 0,
            "execution_time": 0.0,
            "prefill_time": 0.0,
            "decode_time": 0.0,
            "sort_time": 0.0,
        }
        after = {
            "requests": 10,
            "p2d_requests": 10,
            "execution_time": 1.0,
            "prefill_time": 0.0,
            "decode_time": 0.0,
            "sort_time": 0.0,
            "_seen": {"requests", "p2d_requests"},
        }

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            _print_beam_search_delta(before, after)

        output = buf.getvalue()
        self.assertIn("P2D Requests:                            10", output)
        self.assertIn("Beam Search Requests (total):            10", output)
        self.assertNotIn("Tier 1 Requests", output)
        self.assertNotIn("Tier 2 Requests", output)
        self.assertNotIn("Tier 3 Requests", output)
        self.assertNotIn("Eager Requests", output)
        self.assertNotIn("Graph Requests", output)


class TestPatchedAsyncRequestOpenAIChatCompletionsStreaming(unittest.TestCase):
    """Verifies that patched_async_request_openai_chat_completions handles

    empty SSE lines, keepalive comments, malformed chunks, and chunk boundaries
    robustly without crashing with json.decoder.JSONDecodeError.
    """

    def setUp(self):
        from vllm.benchmarks.lib.endpoint_request_func import (
            RequestFuncOutput,
            StreamedResponseHandler,
        )

        from benchmarks.open_one_rec import one_rec_patch as patch

        self.patch = patch
        self._orig_validate = getattr(patch, "_validate_api_url", None)
        self._orig_update_headers = getattr(patch, "_update_headers_common", None)
        self._orig_update_payload = getattr(patch, "_update_payload_common", None)
        self._orig_get_chat = getattr(patch, "_get_chat_content", None)
        self._orig_handler = getattr(patch, "StreamedResponseHandler", None)
        self._orig_output = getattr(patch, "RequestFuncOutput", None)

        patch._validate_api_url = lambda *a, **k: None
        patch._update_headers_common = lambda *a, **k: None
        patch._update_payload_common = lambda *a, **k: None
        patch._get_chat_content = lambda *a, **k: [{"type": "text", "text": "p"}]
        patch.StreamedResponseHandler = StreamedResponseHandler
        patch.RequestFuncOutput = RequestFuncOutput

    def tearDown(self):
        patch = self.patch
        if self._orig_validate is not None:
            patch._validate_api_url = self._orig_validate
        if self._orig_update_headers is not None:
            patch._update_headers_common = self._orig_update_headers
        if self._orig_update_payload is not None:
            patch._update_payload_common = self._orig_update_payload
        if self._orig_get_chat is not None:
            patch._get_chat_content = self._orig_get_chat
        if self._orig_handler is not None:
            patch.StreamedResponseHandler = self._orig_handler
        if self._orig_output is not None:
            patch.RequestFuncOutput = self._orig_output

    def _create_mock_response(self, chunks: list[bytes]):
        class MockContent:
            def __init__(self, data):
                self._data = data

            async def iter_any(self):
                for c in self._data:
                    yield c

        fake_resp = mock.MagicMock()
        fake_resp.status = 200
        fake_resp.reason = "OK"
        fake_resp.content = MockContent(chunks)
        fake_resp.__aenter__ = mock.AsyncMock(return_value=fake_resp)
        fake_resp.__aexit__ = mock.AsyncMock(return_value=False)
        return fake_resp

    def test_handles_empty_lines_and_sse_pings_without_json_decode_error(self):
        import asyncio

        from vllm.benchmarks.lib.endpoint_request_func import RequestFuncInput

        from benchmarks.open_one_rec import one_rec_patch as patch

        chunks = [
            b": keepalive\n\n",
            b"data:\n\n",
            b"data: \n\n",
            b"event: ping\n\n",
            b"id: 12345\n\n",
            b'data: {"choices": [{"delta": {"content": "hello"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        fake_resp = self._create_mock_response(chunks)
        fake_session = mock.MagicMock()
        fake_session.post.return_value = fake_resp

        req = RequestFuncInput(
            prompt="p",
            api_url="http://localhost:8000/v1/chat/completions",
            prompt_len=10,
            output_len=8,
            model="m",
        )

        output = asyncio.run(patch.patched_async_request_openai_chat_completions(req, fake_session))
        self.assertTrue(output.success)
        self.assertEqual(output.generated_text, "hello")
        self.assertFalse(output.error)

    def test_handles_split_chunks_across_tcp_packets(self):
        import asyncio

        from vllm.benchmarks.lib.endpoint_request_func import RequestFuncInput

        from benchmarks.open_one_rec import one_rec_patch as patch

        chunks = [
            b'data: {"choices": [{"delta": {"content": "wo',
            b'rld"}}]}\n',
            b"\ndata: [DONE]\n\n",
        ]
        fake_resp = self._create_mock_response(chunks)
        fake_session = mock.MagicMock()
        fake_session.post.return_value = fake_resp

        req = RequestFuncInput(
            prompt="p",
            api_url="http://localhost:8000/v1/chat/completions",
            prompt_len=10,
            output_len=8,
            model="m",
        )

        output = asyncio.run(patch.patched_async_request_openai_chat_completions(req, fake_session))
        self.assertTrue(output.success)
        self.assertEqual(output.generated_text, "world")
        self.assertFalse(output.error)

    def test_handles_malformed_chunk_gracefully(self):
        import asyncio

        from vllm.benchmarks.lib.endpoint_request_func import RequestFuncInput

        from benchmarks.open_one_rec import one_rec_patch as patch

        chunks = [
            b"data: {bad json\n\n",
            b'data: {"choices": [{"delta": {"content": "good"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        fake_resp = self._create_mock_response(chunks)
        fake_session = mock.MagicMock()
        fake_session.post.return_value = fake_resp

        req = RequestFuncInput(
            prompt="p",
            api_url="http://localhost:8000/v1/chat/completions",
            prompt_len=10,
            output_len=8,
            model="m",
        )

        output = asyncio.run(patch.patched_async_request_openai_chat_completions(req, fake_session))
        self.assertTrue(output.success)
        self.assertEqual(output.generated_text, "good")
        self.assertFalse(output.error)


if __name__ == "__main__":
    unittest.main()
