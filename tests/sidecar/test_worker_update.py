# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc,no-untyped-call"
"""Heads refresh their own derived weights; Trie reload switches in one call."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

from vllm_gr.ops.lm_head_cache import install_lm_head_cache_reload_hooks
from vllm_gr.ops.sliced_logits import compute_sliced_logits


def _head(input_size: int, output_size: int, *, bias: bool = False, device: str = "cpu") -> Any:
    # A native head with TP disabled exercises reload without distributed setup.
    with torch.device(device):
        head = ParallelLMHead(
            output_size,
            input_size,
            bias=bias,
            params_dtype=torch.float32,
            padding_size=1,
            disable_tp=True,
        )
    with torch.no_grad():
        head.weight.normal_()
        if head.bias is not None:
            head.bias.zero_()
    return head


def _install_head_hooks(head: Any, name: str = "lm_head") -> None:
    model = torch.nn.Module()
    if name == "alias":
        # named_modules() visits output first and deduplicates its lm_head alias.
        model.output = head
        model.lm_head = head
    else:
        setattr(model, name, head)
    install_lm_head_cache_reload_hooks(model)


@pytest.mark.l1
def test_head_refreshes_its_own_slice_during_native_weight_processing() -> None:
    class QuantMethod:
        def process_weights_after_loading(self, layer: Any) -> None:
            pass

        def apply(self, layer: Any, hidden: torch.Tensor, bias: Any = None) -> torch.Tensor:
            return torch.nn.functional.linear(hidden, layer.weight, bias)

    head = _head(4, 8)
    head.quant_method = QuantMethod()
    _install_head_hooks(head)
    # Match the unquantized head contract used by the sliced projection.
    from unittest.mock import patch

    with patch("vllm_gr.ops.sliced_logits.is_supported_unquantized_lm_head", return_value=True):
        compute_sliced_logits(torch.ones(1, 4), head, offset=2, slice_size=2, quant_group_size=1)
    pointer = head._sliced_weight.data_ptr()
    with torch.no_grad():
        head.weight.copy_(torch.arange(32).reshape(8, 4).float())
    head.quant_method.process_weights_after_loading(head)
    torch.testing.assert_close(head._sliced_weight, head.weight[2:4])
    assert head._sliced_weight.data_ptr() == pointer


@pytest.mark.l1
def test_reload_trie_switches_then_releases_old_table_on_expiry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np

    from tests.constraints.test_cpu_backend import _select
    from tests.constraints.test_loader import _config
    from tests.constraints.test_table import make_table
    from vllm_gr.config import GRConfig, write_canonical_gr_config
    from vllm_gr.constraints import versioned_table
    from vllm_gr.constraints.artifact import save_constraint_table_artifact
    from vllm_gr.constraints.loader import load_constraint_backend
    from vllm_gr.constraints.versioned_table import INITIAL_CONSTRAINT_VERSION
    from vllm_gr.v1.worker.hot_update import reload_trie, versioned_constraint_table

    now = [0.0]
    monkeypatch.setattr(versioned_table, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setenv("VLLM_GR_TRIE_RETENTION_SECONDS", "5")
    old_path, new_path = tmp_path / "old.safetensors", tmp_path / "new.safetensors"
    old_digest = save_constraint_table_artifact(make_table(), old_path)
    save_constraint_table_artifact(
        make_table(prefix2_values=np.asarray([0, 2, 3, 1, 4], dtype=np.int32)), new_path
    )
    config = _config(old_path, old_digest)
    old = load_constraint_backend(
        config, device=torch.device("cpu"), vocab_size=6, dtype=torch.float32, max_rows=2
    )
    assert old is not None
    native = SimpleNamespace(additional_config={})
    write_canonical_gr_config(native, GRConfig(beam_max_width=4, constraint_table=config))
    runner = SimpleNamespace(
        device="cpu",
        constraint_backend=old,
        model_config=SimpleNamespace(get_vocab_size=lambda: 6, dtype=torch.float32),
        vllm_config=native,
    )
    runner.constraint_backend = versioned_constraint_table(runner, old)
    table = runner.constraint_backend
    worker = SimpleNamespace(model_runner=runner)
    assert reload_trie(worker, str(new_path), "v2") == {"version": "v2", "ready": True}
    assert runner.constraint_backend is table
    logits = torch.arange(6, dtype=torch.float32).reshape(1, 6)
    old_output = _select(
        table.get(INITIAL_CONSTRAINT_VERSION), logits, [[1, 2, 0]], step=2, top_k=2
    )
    new_output = _select(table.get(), logits, [[1, 2, 0]], step=2, top_k=2)
    assert set(old_output.token_ids.flatten().tolist()) == {0, 1}
    assert set(new_output.token_ids.flatten().tolist()) == {0, 2}
    with pytest.raises(ValueError, match="different contents"):
        reload_trie(worker, str(old_path), "v2")
    now[0] = 5.0
    assert runner.constraint_backend is table
    with pytest.raises(KeyError):
        table.get(INITIAL_CONSTRAINT_VERSION)
    table.close()


@pytest.mark.l2
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("head_name", ["lm_head", "output", "alias"])
def test_sliced_cache_survives_native_layerwise_reload(
    bias: bool, inference: bool, head_name: str
) -> None:
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )
    from vllm.model_executor.model_loader.reload import (
        finalize_layerwise_reload,
        initialize_layerwise_reload,
        record_metadata_for_reloading,
    )

    head = _head(4, 16, bias=bias)
    head.quant_method = UnquantizedEmbeddingMethod()
    head.quant_method.process_weights_after_loading = lambda layer: None
    # The real model loader records metadata before lazily built GR caches exist.
    record_metadata_for_reloading(head)
    _install_head_hooks(head, head_name)
    hidden = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    # Use F.linear through the processor to avoid platform-specific GEMM dispatch.
    def project(processor: Any, h: Any, layer: Any, b: Any, skip_gather: bool) -> Any:
        return torch.nn.functional.linear(h, layer.weight, b)

    def logits() -> Any:
        return compute_sliced_logits(
            hidden,
            head,
            head.bias,
            offset=3,
            slice_size=5,
            quant_group_size=4,
            logits_processor=SimpleNamespace(),
            orig_get_logits=project,
        )

    with torch.inference_mode(inference):
        logits()
    cached = head._sliced_weight
    cached_bias = head._sliced_bias
    pointers = (head.weight.data_ptr(), cached.data_ptr())
    proxy = head._sliced_proxy
    for i in range(3):
        weights = torch.arange(64, dtype=torch.float32).reshape(16, 4) + i
        bias_values = torch.arange(16, dtype=torch.float32) + i if bias else None
        initialize_layerwise_reload(head)
        head.weight.weight_loader(head.weight, weights)
        if bias:
            head.bias.weight_loader(head.bias, bias_values)
        finalize_layerwise_reload(head, SimpleNamespace(dtype=torch.float32))
        expected = torch.nn.functional.linear(
            hidden, weights[3:8], bias_values[3:8] if bias else None
        )
        torch.testing.assert_close(logits(), expected)
        assert head._sliced_weight is cached and head._sliced_proxy is proxy
        assert head._sliced_bias is cached_bias
        assert (head.weight.data_ptr(), cached.data_ptr()) == pointers
        assert set(dict(head.named_parameters())) == ({"weight", "bias"} if bias else {"weight"})


@pytest.mark.l1
@pytest.mark.parametrize("bias", [False, True])
def test_sliced_reload_hook_tracks_window_changes_without_rewrapping(bias: bool) -> None:
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    head = _head(4, 16, bias=bias)
    head.quant_method = UnquantizedEmbeddingMethod()
    processed: list[Any] = []

    def original_process(layer: Any) -> None:
        processed.append(layer)

    head.quant_method.process_weights_after_loading = original_process
    _install_head_hooks(head)
    hidden = torch.ones(2, 4)

    def logits(offset: int) -> Any:
        return compute_sliced_logits(
            hidden, head, head.bias, offset=offset, slice_size=3, quant_group_size=4
        )

    logits(1)
    hook = head.quant_method.process_weights_after_loading
    first_proxy = head._sliced_proxy
    logits(1)
    assert head._sliced_proxy is first_proxy
    logits(5)
    assert head._sliced_proxy is not first_proxy
    assert head.quant_method.process_weights_after_loading is hook
    assert processed == []

    cached = head._sliced_weight
    with torch.no_grad():
        head.weight.add_(2)
        if bias:
            head.bias.add_(3)
    hook(head)
    assert processed == [head]
    torch.testing.assert_close(
        logits(5),
        torch.nn.functional.linear(hidden, head.weight[5:8], head.bias[5:8] if bias else None),
    )
    assert head._sliced_weight is cached
    assert head.quant_method.process_weights_after_loading is hook
    assert processed == [head]


@pytest.mark.l2
@pytest.mark.parametrize("loaded_owner", ["embed_tokens", "lm_head"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu_test)])
def test_tied_head_caches_follow_native_layerwise_reload(loaded_owner: str, device: str) -> None:
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )
    from vllm.model_executor.model_loader.reload import (
        finalize_layerwise_reload,
        initialize_layerwise_reload,
        record_metadata_for_reloading,
    )

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for the transposed head cache")
    model = torch.nn.Module()
    model.embed_tokens = torch.nn.Embedding(8, 4, device=device)
    model.lm_head = _head(4, 8, device=device)
    embed, head = model.embed_tokens, model.lm_head
    embed.quant_method = UnquantizedEmbeddingMethod()
    head.quant_method = UnquantizedEmbeddingMethod()
    head.quant_method.tie_weights(head, embed)
    with torch.no_grad():
        embed.weight.fill_(1)
    record_metadata_for_reloading(model)
    install_lm_head_cache_reload_hooks(model)
    hook = head.quant_method.process_weights_after_loading
    install_lm_head_cache_reload_hooks(model)
    assert head.quant_method.process_weights_after_loading is hook
    hidden = torch.ones(1, 4, device=device)

    def project(processor: Any, h: Any, layer: Any, b: Any, skip_gather: bool) -> Any:
        return torch.nn.functional.linear(h, layer.weight, b)

    def logits() -> Any:
        return compute_sliced_logits(
            hidden,
            head,
            offset=2,
            slice_size=3,
            quant_group_size=1,
            logits_processor=object(),
            orig_get_logits=project,
        )

    with torch.inference_mode():
        logits()
        if device == "cuda":
            head._gr_unconstrained_head_layout = (
                head.weight.detach().T.contiguous(),
                torch.Event(),
            )
    sliced = head._sliced_weight
    pointer = head.weight.data_ptr()
    for value in (3.0, 5.0, 2.0):
        initialize_layerwise_reload(model)
        owner = getattr(model, loaded_owner)
        owner.weight.weight_loader(owner.weight, torch.full((8, 4), value, device=device))
        finalize_layerwise_reload(model, SimpleNamespace(dtype=torch.float32))
        assert head.weight is embed.weight and head.weight.data_ptr() == pointer
        assert head._sliced_weight is sliced
        torch.testing.assert_close(logits(), torch.full((1, 3), value * 4, device=device))
        if device == "cuda":
            transposed, ready = head._gr_unconstrained_head_layout
            ready.synchronize()
            torch.testing.assert_close(transposed, head.weight.T)


pytestmark = pytest.mark.cpu_test
