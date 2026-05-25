# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest
import importlib.util
import sys
import types
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _install_stub_package(name: str) -> None:
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_positional_encoder():
    _install_stub_package("vllm_gr.models")
    _install_stub_package("vllm_gr.models.dlrm_v3")
    _install_stub_package("vllm_gr.models.dlrm_v3.modules")
    _install_stub_package("vllm_gr.ops")
    _install_stub_package("vllm_gr.ops.pytorch")

    _load_module(
        "vllm_gr.models.dlrm_v3.common",
        ROOT / "vllm_gr" / "models" / "dlrm_v3" / "common.py",
    )
    _load_module(
        "vllm_gr.ops.pytorch.pt_position",
        ROOT / "vllm_gr" / "ops" / "pytorch" / "pt_position.py",
    )
    _load_module(
        "vllm_gr.ops.position",
        ROOT / "vllm_gr" / "ops" / "position.py",
    )
    module = _load_module(
        "vllm_gr.models.dlrm_v3.modules.positional_encoder",
        ROOT / "vllm_gr" / "models" / "dlrm_v3" / "modules" / "positional_encoder.py",
    )
    return module.HSTUPositionalEncoder


class TestPositionalEncoderParity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.HSTUPositionalEncoder = _load_positional_encoder()

    def test_uses_query_time_relative_buckets(self):
        encoder = self.HSTUPositionalEncoder(
            num_position_buckets=8,
            num_time_buckets=4,
            embedding_dim=2,
            contextual_seq_len=0,
        )
        with torch.no_grad():
            encoder._position_embeddings_weight.copy_(
                torch.tensor(
                    [
                        [0.0, 0.0],
                        [1.0, 10.0],
                        [2.0, 20.0],
                        [3.0, 30.0],
                        [4.0, 40.0],
                        [5.0, 50.0],
                        [6.0, 60.0],
                        [7.0, 70.0],
                    ]
                )
            )
            encoder._timestamp_embeddings_weight.copy_(
                torch.tensor(
                    [
                        [100.0, 1000.0],
                        [200.0, 2000.0],
                        [300.0, 3000.0],
                        [400.0, 4000.0],
                        [500.0, 5000.0],
                    ]
                )
            )

        seq_embeddings = torch.tensor(
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
            ]
        )
        output = encoder(
            max_seq_len=3,
            seq_lengths=torch.tensor([3]),
            seq_offsets=torch.tensor([0, 3]),
            seq_timestamps=torch.tensor([10, 70, 130]),
            seq_embeddings=seq_embeddings,
            num_targets=None,
        )

        expected = torch.tensor(
            [
                [1.0 * (2**0.5) + 3.0 + 200.0, 1.0 * (2**0.5) + 30.0 + 2000.0],
                [2.0 * (2**0.5) + 2.0 + 200.0, 2.0 * (2**0.5) + 20.0 + 2000.0],
                [3.0 * (2**0.5) + 1.0 + 100.0, 3.0 * (2**0.5) + 10.0 + 1000.0],
            ]
        )
        torch.testing.assert_close(output, expected)

    def test_timestamp_bucket_clamp_matches_golden_reference(self):
        encoder = self.HSTUPositionalEncoder(
            num_position_buckets=8,
            num_time_buckets=4,
            embedding_dim=2,
            contextual_seq_len=0,
        )
        with torch.no_grad():
            encoder._position_embeddings_weight.zero_()
            encoder._timestamp_embeddings_weight.copy_(
                torch.tensor(
                    [
                        [100.0, 1000.0],
                        [200.0, 2000.0],
                        [300.0, 3000.0],
                        [400.0, 4000.0],
                        [500.0, 5000.0],
                    ]
                )
            )

        output = encoder(
            max_seq_len=3,
            seq_lengths=torch.tensor([3]),
            seq_offsets=torch.tensor([0, 3]),
            seq_timestamps=torch.tensor([0, 540, 600]),
            seq_embeddings=torch.zeros(3, 2),
            num_targets=None,
        )

        expected = torch.tensor(
            [
                [200.0, 2000.0],
                [200.0, 2000.0],
                [100.0, 1000.0],
            ]
        )
        torch.testing.assert_close(output, expected)

    def test_contextual_prefix_keeps_front_positions_fixed(self):
        encoder = self.HSTUPositionalEncoder(
            num_position_buckets=8,
            num_time_buckets=4,
            embedding_dim=1,
            contextual_seq_len=1,
        )
        with torch.no_grad():
            encoder._position_embeddings_weight.copy_(
                torch.tensor([[0.0], [10.0], [20.0], [30.0], [40.0], [50.0], [60.0], [70.0]])
            )
            encoder._timestamp_embeddings_weight.zero_()

        output = encoder(
            max_seq_len=4,
            seq_lengths=torch.tensor([4]),
            seq_offsets=torch.tensor([0, 4]),
            seq_timestamps=torch.tensor([0, 1, 2, 3]),
            seq_embeddings=torch.zeros(4, 1),
            num_targets=torch.tensor([1]),
        )

        expected = torch.tensor([[0.0], [30.0], [20.0], [10.0]])
        torch.testing.assert_close(output, expected)


if __name__ == "__main__":
    unittest.main()
