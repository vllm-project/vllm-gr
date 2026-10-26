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

When ``vllm`` is not installed, the
``TestPatchTriggerContractWithRealVllm`` cases auto-skip and the
``TestPatchTriggerContractWithStubVllm`` cases provide equivalent coverage by
installing stub ``vllm.benchmarks.*`` modules.
"""

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
from __future__ import annotations

import subprocess
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


def _legacy_phase_scrape(rows: list[tuple[str, float, float]]) -> str:
    return "\n".join(
        f"vllm_gr_beam_search_{name}{{{labels}}} {value}"
        for labels, requests, prefill in rows
        for name, value in (
            ("requests_total", requests),
            ("prefill_time_seconds_total", prefill),
            ("decode_time_seconds_total", prefill * 2),
            ("sort_time_seconds_total", prefill * 0.5),
        )
    )


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
        "_get_chat_content",
        "_update_headers_common",
        "_update_payload_common",
        "_validate_api_url",
    ):
        setattr(erf, sym, type(sym, (), {}))
    erf.RequestFuncInput.__init__ = lambda self, *a, **k: None

    return modules


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
        # restore whatever was there before.
        for name in list(self._stub_modules):
            sys.modules.pop(name, None)
        _restore_modules(self._saved)

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

    def test_cached_modules_without_parent_bindings_are_patched(self) -> None:
        """Cached submodules remain usable after a parent package is restored."""
        from importlib import import_module

        datasets = sys.modules["vllm.benchmarks.datasets"]
        throughput = sys.modules["vllm.benchmarks.throughput"]
        serve = sys.modules["vllm.benchmarks.serve"]
        original_get_samples = datasets.get_samples
        original_main = throughput.main

        # Import hooks or test isolation can retain sys.modules entries
        # without the corresponding attributes on their parent packages.
        delattr(sys.modules["vllm"], "benchmarks")
        for name in ("datasets", "throughput", "serve"):
            delattr(sys.modules["vllm.benchmarks"], name)

        patch_mod = import_module("benchmarks.open_one_rec.one_rec_patch")
        self.assertIs(patch_mod.original_get_samples, original_get_samples)
        self.assertIs(patch_mod.original_main, original_main)
        for module, name, replacement in (
            (datasets, "get_samples", "patched_get_samples"),
            (datasets, "add_dataset_parser", "patched_add_parser"),
            (throughput, "add_cli_args", "patched_add_cli_args"),
            (throughput, "get_requests", "patched_get_requests"),
            (throughput, "validate_args", "patched_validate_args"),
            (throughput, "main", "patched_main"),
            (serve, "add_cli_args", "patched_serve_add_cli_args"),
            (serve, "get_samples", "patched_get_samples"),
            (serve, "main", "patched_serve_main"),
        ):
            with self.subTest(module=module.__name__, name=name):
                self.assertIs(getattr(module, name), getattr(patch_mod, replacement))

    def test_legacy_metrics_preserve_absence_and_match_exact_names(self) -> None:
        import benchmarks.open_one_rec.one_rec_patch as patch_mod

        cases = (
            ("", {}),
            ("vllm:num_requests_running 1", {}),
            ("vllm_gr_beam_search_requests_total_created 100", {}),
            ("vllm_gr_beam_search_requests_total NaN", {}),
            ("vllm_gr_beam_search_requests_total inf", {}),
            ("vllm_gr_beam_search_requests_total invalid", {}),
            ("vllm_gr_beam_search_requests_total -1", {}),
            ("vllm_gr_beam_search_requests_total 2", {"requests": {(): 2.0}}),
            ("vllm_gr_beam_search_requests_total 2 1234567", {"requests": {(): 2.0}}),
            (
                'vllm_gr_beam_search_requests_total{engine="0"} 2\n'
                'vllm_gr_beam_search_requests_total{engine="1"} 3',
                {"requests": {(("engine", "0"),): 2.0, (("engine", "1"),): 3.0}},
            ),
            (
                'vllm_gr_beam_search_requests_total{model_name="model A",engine="0"} 2',
                {"requests": {(("engine", "0"), ("model_name", "model A")): 2.0}},
            ),
            (
                'vllm_gr_beam_search_requests_total{engine="0"} 2\n'
                'vllm_gr_beam_search_requests_total{engine="1"} NaN',
                {},
            ),
            (
                'vllm_gr_beam_search_requests_total{engine="0"} 2\n'
                'vllm_gr_beam_search_requests_total{engine="0"} 3',
                {},
            ),
        )
        for text, expected in cases:
            with self.subTest(text=text), mock.patch.object(patch_mod.requests, "get") as get:
                get.return_value = types.SimpleNamespace(status_code=200, text=text)
                self.assertEqual(patch_mod._fetch_prometheus_metrics("http://server"), expected)
                get.assert_called_once_with("http://server/metrics", timeout=5)

    def test_legacy_metrics_http_failure_is_unavailable(self) -> None:
        import benchmarks.open_one_rec.one_rec_patch as patch_mod

        with mock.patch.object(patch_mod.requests, "get") as get:
            get.return_value = types.SimpleNamespace(status_code=404)
            self.assertEqual(patch_mod._fetch_prometheus_metrics("http://server"), {})
            get.side_effect = OSError("connection failed")
            self.assertEqual(patch_mod._fetch_prometheus_metrics("http://server"), {})

    def test_legacy_phase_report_rejects_reset_masked_by_another_engine(self) -> None:
        before = _legacy_phase_scrape([('engine="0"', 100, 10), ('engine="1"', 10, 1)])
        after = _legacy_phase_scrape([('engine="0"', 120, 14), ('engine="1"', 0, 0)])
        self.assertEqual(self._legacy_phase_report(before, after), "")

    def _legacy_phase_report(self, before: str, after: str) -> str:
        import benchmarks.open_one_rec.one_rec_patch as patch_mod

        with (
            mock.patch.object(patch_mod.requests, "get") as get,
            mock.patch("builtins.print") as output,
        ):
            get.side_effect = [
                types.SimpleNamespace(status_code=200, text=text) for text in (before, after)
            ]
            patch_mod._print_beam_search_delta(
                patch_mod._fetch_prometheus_metrics("http://server"),
                patch_mod._fetch_prometheus_metrics("http://server"),
            )
        return "\n".join(str(call.args[0]) for call in output.call_args_list)

    def test_legacy_phase_report_aggregates_only_valid_series_deltas(self) -> None:
        before = _legacy_phase_scrape(
            [
                ('engine="0",model_name="model A"', 100, 10),
                ('engine="1",model_name="model B"', 10, 1),
            ]
        )
        for requests, prefill in ((15, 2), (10, 1)):
            # Label order and sample order are irrelevant. An idle engine is valid.
            after = _legacy_phase_scrape(
                [
                    ('model_name="model B",engine="1"', requests, prefill),
                    ('model_name="model A",engine="0"', 120, 14),
                ]
            )
            with self.subTest(requests=requests):
                text = self._legacy_phase_report(before, after)
                self.assertIn("Avg Prefill Time (ms):                   200.00", text)
                self.assertIn("Avg Decode Time (ms):                    400.00", text)
                self.assertIn("Avg Sort Time (ms):                      100.000", text)
                self.assertIn(f"Beam Search Requests:                    {requests + 10}", text)

    def test_legacy_phase_report_preserves_model_label_identity(self) -> None:
        before = _legacy_phase_scrape(
            [
                ('engine="0",model_name="model A"', 100, 10),
                ('engine="0",model_name="model B"', 10, 1),
            ]
        )
        after = _legacy_phase_scrape(
            [
                ('engine="0",model_name="model A"', 120, 14),
                ('engine="0",model_name="model B"', 0, 0),
            ]
        )
        self.assertEqual(self._legacy_phase_report(before, after), "")

    def test_legacy_phase_report_rejects_changed_or_incomplete_series(self) -> None:
        before = _legacy_phase_scrape([('engine="0"', 100, 10), ('engine="1"', 10, 1)])
        after = _legacy_phase_scrape([('engine="0"', 120, 14), ('engine="1"', 15, 2)])
        phase = 'vllm_gr_beam_search_decode_time_seconds_total{engine="1"}'
        incomplete = "\n".join(line for line in after.splitlines() if not line.startswith(phase))
        invalid_snapshots = {
            "disappearing": _legacy_phase_scrape([('engine="0"', 120, 14)]),
            "new": after + "\n" + _legacy_phase_scrape([('engine="2"', 1, 0.2)]),
            "relabelled": after.replace('engine="1"', 'engine="2"'),
            "incomplete": incomplete,
            "phase_reset_masked_by_other_engine": incomplete + f"\n{phase} 0",
            "nonfinite": incomplete + f"\n{phase} NaN",
        }
        for name, invalid in invalid_snapshots.items():
            with self.subTest(name=name):
                self.assertEqual(self._legacy_phase_report(before, invalid), "")
        self.assertEqual(self._legacy_phase_report(incomplete, after), "")
        self.assertEqual(self._legacy_phase_report(incomplete, incomplete), "")

    def test_legacy_phase_report_requires_complete_monotonic_counters(self) -> None:
        import benchmarks.open_one_rec.one_rec_patch as patch_mod

        before = {
            "requests": {(): 2.0},
            "prefill_time": {(): 1.0},
            "decode_time": {(): 2.0},
            "sort_time": {(): 0.5},
        }
        after = {
            "requests": {(): 4.0},
            "prefill_time": {(): 1.4},
            "decode_time": {(): 2.6},
            "sort_time": {(): 0.6},
        }
        with mock.patch("builtins.print") as output:
            patch_mod._print_beam_search_delta(before, after)
        text = "\n".join(str(call.args[0]) for call in output.call_args_list)
        self.assertIn("200.00", text)
        self.assertIn("300.00", text)
        self.assertIn("50.000", text)
        for invalid in (
            None,
            {},
            {"requests": {(): 4.0}},
            before,
            {**after, "decode_time": {(): 0.0}},
        ):
            with self.subTest(after=invalid), mock.patch("builtins.print") as output:
                patch_mod._print_beam_search_delta(before, invalid)
                output.assert_not_called()

    def test_legacy_phase_report_rejects_inconsistent_beam_counters(self) -> None:
        before = _legacy_phase_scrape([('engine="0"', 100, 10), ('engine="1"', 10, 1)])
        after = _legacy_phase_scrape([('engine="0"', 120, 14), ('engine="1"', 15, 2)])
        metric = "vllm_gr_beam_search_total_beams_generated_total"
        initial = f'\n{metric}{{engine="0"}} 1000\n{metric}{{engine="1"}} 100'
        healthy = f'\n{metric}{{engine="0"}} 1200\n{metric}{{engine="1"}} 150'
        invalid = {
            "masked_reset": f'\n{metric}{{engine="0"}} 1200\n{metric}{{engine="1"}} 0',
            "new_series": healthy + f'\n{metric}{{engine="2"}} 10',
            "disappearing_series": f'\n{metric}{{engine="0"}} 1200',
            "relabelled_series": healthy.replace('engine="1"', 'engine="2"'),
            "disappearing_counter": "",
        }
        for name, counters in invalid.items():
            with self.subTest(name=name):
                self.assertEqual(self._legacy_phase_report(before + initial, after + counters), "")
        self.assertEqual(self._legacy_phase_report(before, after + healthy), "")
        self.assertEqual(
            self._legacy_phase_report(before + f'\n{metric}{{engine="0"}} 1000', after + healthy),
            "",
        )

    def test_legacy_phase_report_accepts_monotonic_optional_beam_counters(self) -> None:
        before = _legacy_phase_scrape([('engine="0"', 100, 10), ('engine="1"', 10, 1)])
        after = _legacy_phase_scrape([('engine="0"', 120, 14), ('engine="1"', 15, 2)])
        metric = "vllm_gr_beam_search_total_beams_generated_total"
        expected = self._legacy_phase_report(before, after)
        self.assertIn("Avg Prefill Time (ms):                   200.00", expected)
        self.assertEqual(
            self._legacy_phase_report(
                before + f'\n{metric}{{engine="0"}} 1000\n{metric}{{engine="1"}} 100',
                after + f'\n{metric}{{engine="1"}} 150\n{metric}{{engine="0"}} 1200',
            ),
            expected,
        )

    def test_missing_legacy_metrics_does_not_change_benchmark_result(self) -> None:
        import benchmarks.open_one_rec.one_rec_patch as patch_mod

        args = types.SimpleNamespace(base_url="http://server", n=2)
        with (
            mock.patch.object(patch_mod.requests, "get", side_effect=OSError("unavailable")),
            mock.patch.object(patch_mod, "original_serve_main", return_value=17) as serve,
        ):
            self.assertEqual(patch_mod.patched_serve_main(args), 17)
        serve.assert_called_once_with(args)

    def test_offline_beam_search_always_delegates_to_v1(self) -> None:
        """The former default path uses the same V1 owner as the explicit flag."""
        import benchmarks.open_one_rec.one_rec_patch as patch_mod
        from benchmarks.open_one_rec import one_rec_v1_benchmark, open_one_rec_dataset

        request = types.SimpleNamespace(prompt="prompt", expected_output_len=5)
        dataset = mock.MagicMock()
        dataset.sample.return_value = [request]
        args = types.SimpleNamespace(
            model="fake-model",
            dataset_path="./data",
            task_types="ad",
            num_prompts=1,
            warmup_requests=2,
            constraint_triples="triples.json",
            output_json="result.json",
            beam_search_v1=False,
        )
        transformers = sys.modules["transformers"]
        with (
            mock.patch.object(open_one_rec_dataset, "OneRecDataset", return_value=dataset),
            mock.patch.object(transformers, "AutoTokenizer") as auto_tokenizer,
            mock.patch.object(one_rec_v1_benchmark, "run_v1_benchmark") as run_v1,
        ):
            tokenizer = auto_tokenizer.from_pretrained.return_value
            patch_mod._run_offline_beam_search_benchmark(args)
            dataset.sample.assert_called_once_with(
                num_requests=2,
                no_oversample=True,
                tokenizer=tokenizer,
            )
            run_v1.assert_called_once_with(args, tokenizer, [request])

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
            def __init__(inner_self, **kwargs):
                test_self._loader_calls.append(kwargs)
                inner_self.kwargs = kwargs

            def load_data(inner_self, **kwargs):
                return {}

        self._orig_loader = ds_mod.DataLoaderWrapper
        ds_mod.DataLoaderWrapper = _RecordingLoader

    def tearDown(self) -> None:
        self._ds_mod.DataLoaderWrapper = self._orig_loader
        # Drop stubs first so they don't leak, then restore saved modules.
        for name in list(self._stub_vllm):
            sys.modules.pop(name, None)
        _restore_modules(self._saved)

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
            def __init__(inner_self, **kwargs):
                inner_self.kwargs = kwargs

            def load_data(inner_self, **kwargs):
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
            def __init__(inner_self, **kwargs):
                pass

            def load_data(inner_self, **kwargs):
                # Single sample, keyed by row index.
                return {"0": {"prompt": "p", "ground_truth": "gt"}}

        self._orig_loader = ds_mod.DataLoaderWrapper
        ds_mod.DataLoaderWrapper = _StubLoader

    def tearDown(self) -> None:
        self._ds_mod.DataLoaderWrapper = self._orig_loader
        for name in list(self._stub_vllm):
            sys.modules.pop(name, None)
        _restore_modules(self._saved)

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


class TestPatchTriggerContractWithRealVllm(unittest.TestCase):
    """Check real imports in fresh processes, without re-registering Torch types.

    Purging vllm from sys.modules cannot undo Torch's opaque type registrations.
    In a combined test run, importing it again fails before the contract runs.
    Only an absent vllm installation is a skip; broken imports are failures.
    """

    def _run_contract(self, explicit_patch: bool) -> None:
        script = """
import importlib.util
import sys

if importlib.util.find_spec("vllm") is None:
    raise SystemExit(77)

import vllm.benchmarks.datasets as datasets_mod
original_get_samples = datasets_mod.get_samples
import benchmarks.open_one_rec
assert datasets_mod.get_samples is original_get_samples

if sys.argv[1] == "patch":
    import benchmarks.open_one_rec.one_rec_patch as patch_mod
    assert datasets_mod.get_samples is patch_mod.patched_get_samples
"""
        result = subprocess.run(
            [sys.executable, "-c", script, "patch" if explicit_patch else "package"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 77:
            self.skipTest("vllm is not installed")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_package_import_does_not_patch_real_vllm(self) -> None:
        self._run_contract(explicit_patch=False)

    def test_explicit_patch_import_patches_real_vllm(self) -> None:
        self._run_contract(explicit_patch=True)


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
        calls: list[object] = []

        def sentinel(a: object) -> None:
            calls.append(a)

        original = self._patch_mod.original_validate_args
        self._patch_mod.original_validate_args = sentinel
        try:
            args = self._args(dataset_name="random")
            self._patch_mod.patched_validate_args(args)
            self.assertEqual(len(calls), 1)
            self.assertIs(calls[0], args)
        finally:
            self._patch_mod.original_validate_args = original


if __name__ == "__main__":
    unittest.main()
