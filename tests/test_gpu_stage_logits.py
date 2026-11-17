# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""CPU contract checks for runner-local CUDA stage projection routing."""

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_gr.ops import gpu_stage_logits as module

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


class Processor:
    logits_as_input = False

    def _get_logits(self, hidden, head, bias, skip_gather=False):
        return ("original", hidden, head, bias, skip_gather, self._gr_vocab_slice)


def test_beam_scope_routes_only_its_processor_and_restores_on_error(monkeypatch):
    calls = []

    def sliced(
        hidden, head, bias, *, offset, slice_size, logits_processor, orig_get_logits, skip_gather
    ):
        calls.append((offset, slice_size, logits_processor, skip_gather))
        return ("sliced", hidden, head, bias)

    monkeypatch.setattr(module, "compute_sliced_logits", sliced)
    processor = Processor()
    other = Processor()
    other._gr_vocab_slice = None
    projection = module.install_gpu_stage_projection((processor,), (10, 4))
    monkeypatch.setattr(projection, "_cache_usable", lambda *_: True)
    assert processor._get_logits(1, 2, 3)[0] == "original"
    with pytest.raises(ValueError, match="test exception"):
        with projection.beam_scope():
            assert processor._get_logits(1, 2, 3)[0] == "sliced"
            assert other._get_logits(1, 2, 3)[0] == "original"
            raise ValueError("test exception")
    assert processor._get_logits(1, 2, 3)[0] == "original"
    assert calls == [(10, 4, processor, False)]
    assert projection.stats() == {"full": 2, "beam_slice": 1, "beam_cached": 1, "beam_fallback": 0}
    projection.close()
    assert "_get_logits" not in processor.__dict__
    assert "_gr_vocab_slice" not in processor.__dict__


def test_nested_scope_fails_without_corrupting_outer_scope(monkeypatch):
    processor = Processor()
    monkeypatch.setattr(module, "compute_sliced_logits", lambda *args, **kwargs: "slice")
    projection = module.install_gpu_stage_projection((processor,), (10, 4))
    monkeypatch.setattr(projection, "_cache_usable", lambda *_: True)
    with projection.beam_scope():
        with pytest.raises(RuntimeError, match="Nested"):
            with projection.beam_scope():
                pass
        assert processor._get_logits(1, 2, 3) == "slice"
    assert processor._get_logits(1, 2, 3)[0] == "original"
    projection.close()


def test_scope_does_not_leak_to_other_thread(monkeypatch):
    processor = Processor()
    monkeypatch.setattr(module, "compute_sliced_logits", lambda *args, **kwargs: "slice")
    projection = module.install_gpu_stage_projection((processor,), (10, 4))
    monkeypatch.setattr(projection, "_cache_usable", lambda *_: True)
    with ThreadPoolExecutor(max_workers=1) as executor:
        with projection.beam_scope():
            assert processor._get_logits(1, 2, 3) == "slice"
            assert executor.submit(processor._get_logits, 1, 2, 3).result()[0] == "original"
    projection.close()


def test_install_failure_restores_prior_processor_state():
    first = Processor()
    first._gr_vocab_slice = (3, 7)
    original = lambda *args: "original"  # noqa: E731
    first._get_logits = original
    second = Processor()
    second.logits_as_input = True
    with pytest.raises(ValueError, match="LM-head"):
        module.install_gpu_stage_projection((first, second), (10, 4))
    assert first._get_logits is original
    assert first._gr_vocab_slice == (3, 7)
    assert "_get_logits" not in second.__dict__


def test_empty_processor_inventory_fails_at_initialization():
    with pytest.raises(ValueError, match="logits processor"):
        module.install_gpu_stage_projection((), (10, 4))


def test_close_restores_preexisting_instance_hook_and_window():
    processor = Processor()
    prior = lambda *args: "prior"  # noqa: E731
    processor._get_logits = prior
    processor._gr_vocab_slice = (2, 4)
    projection = module.install_gpu_stage_projection((processor,), (10, 4))
    projection.close()
    projection.close()
    assert processor._get_logits is prior
    assert processor._gr_vocab_slice == (2, 4)
    with pytest.raises(RuntimeError, match="closed"):
        with projection.beam_scope():
            pass


def test_uncached_head_and_mutated_weight_use_current_full_window(monkeypatch):
    class TensorProcessor:
        logits_as_input = False

        def _get_logits(self, hidden, head, bias, skip_gather=False):
            return F.linear(hidden, head.weight, bias)

    head = SimpleNamespace(weight=torch.arange(40, dtype=torch.float32).reshape(20, 2))
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    processor = TensorProcessor()
    projection = module.install_gpu_stage_projection((processor,), (10, 4))
    monkeypatch.setattr(module, "compute_sliced_logits", lambda *a, **kw: "cached")
    assert torch.equal(processor._get_logits(hidden, head, None), F.linear(hidden, head.weight))
    with projection.beam_scope():
        actual = processor._get_logits(hidden, head, None)
    assert torch.equal(actual, F.linear(hidden, head.weight)[:, 10:14])
    assert actual.is_contiguous()

    projection._cached_head = head
    projection._source_weight = head.weight
    projection._source_bias = None
    projection._weight_version = projection._version(head.weight)
    projection._bias_version = None
    with projection.beam_scope():
        assert processor._get_logits(hidden, head, None) == "cached"
    head.weight.add_(1)
    with projection.beam_scope():
        actual = processor._get_logits(hidden, head, None)
    assert torch.equal(actual, F.linear(hidden, head.weight)[:, 10:14])
    assert projection.stats()["beam_cached"] == 1
    assert projection.stats()["beam_fallback"] == 2
    projection.close()


def test_real_sliced_projection_uses_correct_callback_and_window():
    class TensorProcessor:
        logits_as_input = False

        def _get_logits(self, hidden, head, bias, skip_gather=False):
            return F.linear(hidden, head.weight, bias)

    head = torch.nn.Linear(2, 20, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.arange(40, dtype=torch.float32).reshape(20, 2))
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    processor = TensorProcessor()
    projection = module.install_gpu_stage_projection((processor,), (10, 4))
    projection._cached_head = head
    projection._source_weight = head.weight
    projection._source_bias = None
    projection._weight_version = projection._version(head.weight)
    projection._bias_version = None
    full = processor._get_logits(hidden, head, None)
    with projection.beam_scope():
        sliced = processor._get_logits(hidden, head, None)
    assert sliced.shape == (2, 4)
    torch.testing.assert_close(sliced, full[:, 10:14])
    assert projection.stats()["beam_cached"] == 1
    projection.close()


def test_weight_update_scope_blocks_projections_and_recovers_or_fails():
    processor = Processor()
    projection = module.install_gpu_stage_projection((processor,), (10, 4))
    projection.begin_weight_update()
    with pytest.raises(RuntimeError, match="weight update"):
        processor._get_logits(1, 2, 3)
    projection.finish_weight_update()
    assert processor._get_logits(1, 2, 3)[0] == "original"
    projection.begin_weight_update()
    projection.fail_weight_update()
    with pytest.raises(RuntimeError, match="weight update"):
        processor._get_logits(1, 2, 3)
    with pytest.raises(RuntimeError, match="weight update"):
        projection.begin_weight_update()
    projection.close()


@pytest.mark.parametrize("action", ["update", "close"])
def test_projection_releases_old_head_weights(action):
    processor = Processor()
    projection = module.install_gpu_stage_projection((processor,), (10, 4))
    head = SimpleNamespace(weight=torch.ones((20, 2)), bias=torch.ones(20))
    old_weight = weakref.ref(head.weight)
    old_bias = weakref.ref(head.bias)
    projection._cached_head = head
    projection._source_weight = head.weight
    projection._source_bias = head.bias
    projection._weight_version = projection._version(head.weight)
    projection._bias_version = projection._version(head.bias)
    projection.cached_bytes = 100
    projection._prime_bytes = 100
    if action == "update":
        projection.begin_weight_update()
    else:
        projection.close()
    head.weight = torch.zeros((20, 2))
    head.bias = torch.zeros(20)
    gc.collect()
    assert old_weight() is None
    assert old_bias() is None
    assert projection._weight_version is None
    assert projection._bias_version is None
    assert projection.cached_bytes == 0
    if action == "update":
        projection.finish_weight_update()
        projection.close()


@pytest.mark.parametrize("tp_size,rank_start", [(2, 0), (1, 8)])
def test_fallback_rejects_unassembled_rank_local_logits(tp_size, rank_start):
    class TensorProcessor:
        logits_as_input = False

        def _get_logits(self, hidden, head, bias, skip_gather=False):
            return F.linear(hidden, head.weight, bias)

    head = SimpleNamespace(
        weight=torch.ones((20, 2)),
        tp_size=tp_size,
        vocab_start_index=rank_start,
        vocab_end_index=rank_start + 20,
    )
    processor = TensorProcessor()
    projection = module.install_gpu_stage_projection((processor,), (10, 4))
    with projection.beam_scope(), pytest.raises(RuntimeError, match="full vocabulary"):
        processor._get_logits(torch.ones((1, 2)), head, None, True)
    projection.close()


def test_prime_does_not_run_projection_for_ineligible_head(monkeypatch):
    processor = Processor()
    projection = module.install_gpu_stage_projection((processor,), (10, 4))
    monkeypatch.setattr(
        module, "compute_sliced_logits", lambda *args, **kwargs: pytest.fail("unexpected prime")
    )
    head = SimpleNamespace(weight=torch.ones((20, 2)))  # CPU is ineligible.
    projection.prime(SimpleNamespace(model=SimpleNamespace(lm_head=head)))
    assert projection._cached_head is None
    assert projection.cached_bytes == 0
    assert projection.stats()["beam_cached"] == 0
    projection.close()


@pytest.mark.parametrize("fail", [False, True])
def test_gr_dispatch_scopes_projection_and_cleans_up(monkeypatch, fail):
    from vllm_gr.v1.worker import gpu_beam_stage_runner, gr_execution

    processor = Processor()
    projection = module.install_gpu_stage_projection((processor,), (10, 4))
    monkeypatch.setattr(projection, "_cache_usable", lambda *_: True)
    monkeypatch.setattr(module, "compute_sliced_logits", lambda *args, **kwargs: "slice")
    worker = SimpleNamespace(device="cuda:0", _gr_gpu_stage_projection=projection)
    scheduled = object()
    monkeypatch.setattr(gr_execution, "worker_input", lambda item: item)

    class StageRunner:
        execution_error = None

        def execute(self, local, original, *args, **kwargs):
            assert local is scheduled
            assert processor._get_logits(1, 2, 3) == "slice"
            if fail:
                raise ValueError("stage failed")
            return "executed"

    stage = StageRunner()
    monkeypatch.setattr(gpu_beam_stage_runner, "get_stage_runner", lambda *a, **kw: stage)
    if fail:
        with pytest.raises(ValueError, match="stage failed"):
            gr_execution.execute_gr_stage(worker, scheduled, None)
        assert isinstance(stage.execution_error, ValueError)
    else:
        assert gr_execution.execute_gr_stage(worker, scheduled, None) == "executed"
    assert processor._get_logits(1, 2, 3)[0] == "original"
    projection.close()


@pytest.mark.parametrize("start", ["start_weight_update", "start_draft_weight_update"])
def test_worker_weight_transfer_retires_cache(start):
    class Worker:
        def start_weight_update(self):
            assert self.model_runner._gr_gpu_stage_projection.update_active
            assert not hasattr(self.head, "_sliced_weight")

        def start_draft_weight_update(self):
            return self.start_weight_update_impl()

        def start_weight_update_impl(self):
            assert self.model_runner._gr_gpu_stage_projection.update_active
            assert not hasattr(self.head, "_sliced_weight")

        def update_weights(self):
            assert self.model_runner._gr_gpu_stage_projection.update_active

        def finish_weight_update(self):
            assert self.model_runner._gr_gpu_stage_projection.update_active

    module.install_gpu_weight_update_hooks(Worker)
    first_wrapper = Worker.start_weight_update
    module.install_gpu_weight_update_hooks(Worker)
    assert Worker.start_weight_update is first_wrapper
    processor = Processor()
    projection = module.install_gpu_stage_projection((processor,), (10, 4))
    head = SimpleNamespace(_sliced_weight=object(), _sliced_proxy=object())
    projection._cached_head = head
    worker = Worker()
    worker.model_runner = SimpleNamespace(_gr_gpu_stage_projection=projection)
    worker.head = head
    getattr(worker, start)()
    worker.update_weights()
    worker.finish_weight_update()
    assert not projection.update_active
    assert not projection.update_failed
    assert projection._cached_head is None
    assert not hasattr(head, "_sliced_weight")
    assert not hasattr(head, "_sliced_proxy")
    projection.close()


@pytest.mark.parametrize("kind", ["start", "update", "finish"])
def test_worker_weight_transfer_failure_blocks_generation(kind):
    class Worker:
        def start_weight_update(self):
            if kind == "start":
                raise ValueError("start failed")

        def update_weights(self):
            if kind == "update":
                raise ValueError("update failed")

        def finish_weight_update(self):
            if kind == "finish":
                raise ValueError("finish failed")

    module.install_gpu_weight_update_hooks(Worker)
    processor = Processor()
    projection = module.install_gpu_stage_projection((processor,), (10, 4))
    worker = Worker()
    worker.model_runner = SimpleNamespace(_gr_gpu_stage_projection=projection)
    if kind == "start":
        with pytest.raises(ValueError, match="failed"):
            worker.start_weight_update()
    else:
        worker.start_weight_update()
        if kind == "update":
            with pytest.raises(ValueError, match="failed"):
                worker.update_weights()
        else:
            worker.update_weights()
            with pytest.raises(ValueError, match="failed"):
                worker.finish_weight_update()
    assert projection.update_failed
    with pytest.raises(RuntimeError, match="weight update"):
        processor._get_logits(1, 2, 3)
    projection.close()


@pytest.mark.parametrize("runner_kind", ["v1", "v2"])
@pytest.mark.parametrize("via_worker", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_runner_reload_lifecycle(runner_kind, via_worker, failure):
    class V1:
        def reload_weights(self):
            assert self._gr_gpu_stage_projection.update_active
            assert not hasattr(self.head, "_sliced_weight")
            self.calls += 1
            if failure:
                raise ValueError("reload failed")

    V1.reload_weights = module.make_gpu_stage_reload_weights(V1.reload_weights)
    assert module.make_gpu_stage_reload_weights(V1.reload_weights) is V1.reload_weights

    class V2:
        def reload_weights(self):
            return V1.reload_weights(self)

    V2.reload_weights = module.make_gpu_stage_reload_weights(V2.reload_weights)
    runner = V1() if runner_kind == "v1" else V2()
    runner.head = SimpleNamespace(_sliced_weight=object(), _sliced_proxy=object())
    runner.calls = 0
    processor = Processor()
    projection = module.install_gpu_stage_projection((processor,), (10, 4))
    runner._gr_gpu_stage_projection = projection
    projection._cached_head = runner.head

    class Worker:
        model_runner = runner

        def reload_weights(self):
            return self.model_runner.reload_weights()

    call = Worker().reload_weights if via_worker else runner.reload_weights
    if failure:
        with pytest.raises(ValueError, match="reload failed"):
            call()
        assert projection.update_failed
        with pytest.raises(RuntimeError, match="weight update"):
            processor._get_logits(1, 2, 3)
    else:
        call()
        assert not projection.update_failed
        assert not projection.update_active
        assert projection._cached_head is None
        assert processor._get_logits(1, 2, 3)[0] == "original"
    assert runner.calls == 1
    assert not hasattr(runner.head, "_sliced_proxy")
    projection.close()


def test_direct_runner_reload_rejected_during_transfer():
    class Runner:
        def reload_weights(self):
            raise AssertionError("native reload should not run")

    Runner.reload_weights = module.make_gpu_stage_reload_weights(Runner.reload_weights)
    projection = module.install_gpu_stage_projection((Processor(),), (10, 4))
    runner = Runner()
    runner._gr_gpu_stage_projection = projection
    projection.begin_weight_update()
    with pytest.raises(RuntimeError, match="weight update"):
        runner.reload_weights()
    assert not projection.update_failed
    projection.finish_weight_update()
    projection.close()
