# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static lane classification for the test suite.

Tests are split into two lanes so CPU-only hosts (and GitHub-hosted CI
runners) can execute everything that does not need a CUDA device:

* the **GPU lane** (``pytest -m requires_gpu``) owns every test that needs a
  CUDA device;
* the **CPU lane** (``pytest -m "not requires_gpu"``) owns the rest.

A file lands on the GPU lane when it lives under one of ``GPU_DIRECTORIES``
(``kernels`` and ``st``: kernel tests and system tests default to needing the
card) or is listed in ``GPU_TEST_FILES``.  Files under a GPU directory that
are verifiably device-free (pure-torch reference implementations) opt back
into the CPU lane via ``CPU_OVERRIDES``.

Curation notes for reviewers of ``GPU_TEST_FILES``:

* files that only *mock* CUDA (``monkeypatch``/``mock.patch`` of platform
  probes, ``torch.cuda`` streams, or ``device=torch.device("cuda")`` objects
  handed to mocked runners) stay on the CPU lane — the device object never
  allocates;
* files that parametrise over ``cpu`` **and** ``cuda`` devices with
  availability guards stay on the **GPU** lane, so their CUDA cases keep
  running somewhere (a CPU host would skip them and the coverage would be
  lost);
* ambiguous files default to the GPU lane; reclassify after observing the
  CPU lane run.

``tests/test_gpu_lanes.py`` guards the lists against renames and deletions.
"""

from __future__ import annotations

import os
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parent

#: Test directories whose files need a CUDA device unless overridden below.
GPU_DIRECTORIES = frozenset({"kernels", "st"})

#: GPU-lane files outside :data:`GPU_DIRECTORIES`.
GPU_TEST_FILES = frozenset(
    {
        # Real CUDA execution or CUDA/CPU device parametrisation.
        "constraints/test_cuda_backend.py",
        "constraints/test_multi_trie.py",
        "test_beam_attention_pool.py",
        "test_beam_prefix_policy.py",
        "test_beam_session_ops_integration.py",
        "test_dynamic_beam.py",
        "test_short_suffix_config.py",
        "test_sliced_logits.py",
        # End-to-end engines.
        "test_offline_beam_search.py",
        "test_serving_beam_search.py",
        # CUDA graph managers / GPU runner plumbing (ambiguous: kept on card).
        "test_prefill_graph_runner.py",
        "test_gpu_model_runner_patch.py",
        "test_gpu_prefill_graph_runner_v2.py",
        "test_constraint_resource_inject.py",
    }
)

#: Device-free files inside :data:`GPU_DIRECTORIES`.
CPU_OVERRIDES = frozenset(
    {
        "kernels/test_cache_unshared_kv.py",
        "kernels/test_rec_constrained_topk.py",
        "kernels/attention/test_merge_tnd_interleaved_pool.py",
    }
)


def requires_gpu(path: str | os.PathLike) -> bool:
    """Return whether the test file at ``path`` belongs to the GPU lane."""

    try:
        rel = Path(path).resolve().relative_to(TESTS_ROOT)
    except ValueError:  # outside tests/: not ours to classify
        return False
    parts = rel.parts
    if parts and parts[0] in GPU_DIRECTORIES:
        return rel.as_posix() not in CPU_OVERRIDES
    return rel.as_posix() in GPU_TEST_FILES


def classify_all() -> dict[str, list[str]]:
    """Return ``{"gpu": [...], "cpu": [...]}`` for every test file on disk."""

    lanes = {"gpu": [], "cpu": []}
    for path in sorted(TESTS_ROOT.rglob("test_*.py")):
        rel = path.relative_to(TESTS_ROOT).as_posix()
        lanes["gpu" if requires_gpu(path) else "cpu"].append(rel)
    return lanes
