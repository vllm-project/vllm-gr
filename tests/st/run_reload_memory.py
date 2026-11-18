# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Test-only Worker memory sampling for repeated weight and Trie reloads."""

import argparse
import json
import os
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Any, cast


def install_probes() -> None:
    import torch

    from vllm_gr.constraints.versioned_table import VersionedConstraintTable
    from vllm_gr.v1.worker import model_runner_patch

    def sample(event: str, table: Any = None, **extra: Any) -> None:
        torch.accelerator.synchronize()
        free, total = torch.accelerator.get_memory_info()
        record = dict(
            event=event,
            pid=os.getpid(),
            allocated=torch.accelerator.memory_allocated(),
            reserved=torch.accelerator.memory_reserved(),
            peak_allocated=torch.accelerator.max_memory_allocated(),
            device_used=total - free,
            **extra,
        )
        if table is not None:
            record.update(
                version=table.current,
                versions=len(table.tables),
                table_bytes=sum(b.info.device_memory_bytes for b in table.tables.values()),
            )
        with Path(os.environ["GR_RELOAD_MEMORY_LOG"]).open("a") as log:
            log.write(json.dumps(record) + "\n")

    original_install = model_runner_patch.install_worker_utility_hooks

    def install(worker_cls: Any) -> None:
        original_install(worker_cls)
        native = worker_cls.reload_weights
        if getattr(native, "_test_memory_probe", False):
            return

        @wraps(native)
        def reload_weights(self: Any, *args: Any, **kwargs: Any) -> Any:
            torch.accelerator.synchronize()
            torch.accelerator.reset_peak_memory_stats()
            cache_mode = os.environ.get("GR_RELOAD_HEAD_CACHE")
            head = self.model_runner.get_model().lm_head if cache_mode else None
            if cache_mode == "sliced":
                cache = head._sliced_weight
            elif cache_mode == "transposed":
                cache = head._gr_unconstrained_head_layout[0]
            sample("weight_before")
            result = native(self, *args, **kwargs)
            extra = {}
            if cache_mode:
                if cache_mode == "sliced":
                    assert head._sliced_weight is cache
                    offset, _, rank_start, _ = head._sliced_cache_key
                    start = max(offset - rank_start, 0) - head._sliced_rel_start
                    expected = head.weight[start : start + cache.shape[0]]
                else:
                    from vllm_gr.v1.worker.cuda.gpu_beam_stage_runner import (
                        _get_unconstrained_head_layout,
                    )

                    assert _get_unconstrained_head_layout(self.model_runner)[0] is cache
                    expected = head.weight.T
                torch.testing.assert_close(cache, expected, rtol=0, atol=0)
                assert not any("_sliced" in name for name, _ in head.named_parameters())
                extra["cache"] = dict(
                    mode=cache_mode, pointer=cache.data_ptr(), matches_weight=True
                )
            sample("weight_after", **extra)
            return result

        setattr(reload_weights, "_test_memory_probe", True)
        worker_cls.reload_weights = reload_weights

    model_runner_patch.install_worker_utility_hooks = install
    from vllm_gr.v1.worker.cuda.beam_decode_graph import BeamDecodeGraph

    original_replay = BeamDecodeGraph.replay

    @wraps(original_replay)
    def replay(self: Any) -> Any:
        result = original_replay(self)
        sample("decode_graph_replay")
        return result

    BeamDecodeGraph.replay = replay

    from vllm_gr.v1.worker.cuda.v1.gpu_prefill_graph_runner import GPUPrefillGraphRunner

    original_prefill = GPUPrefillGraphRunner.execute_model_prefill

    @wraps(original_prefill)
    def prefill(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_prefill(self, *args, **kwargs)
        if result is not None:
            sample("prefill_graph_replay")
        return result

    GPUPrefillGraphRunner.execute_model_prefill = prefill
    original_reload, original_get = VersionedConstraintTable.reload, VersionedConstraintTable.get

    def reload(self: Any, path: str, version: str) -> Any:
        sample("trie_before", self)
        result = original_reload(self, path, version)
        sample("trie_after", self)
        return result

    def get(self: Any, version: str | None = None) -> Any:
        count = len(self.tables)
        result = original_get(self, version)
        if len(self.tables) < count:
            sample("trie_expired", self)
            # One final diagnostic flush; normal cycles never empty the allocator.
            if self.current == f"cycle-{os.environ['GR_RELOAD_MEMORY_CYCLES']}":
                torch.accelerator.empty_cache()
                sample("trie_allocator_flush", self)
        return result

    VersionedConstraintTable.reload = reload
    VersionedConstraintTable.get = get


# Spawned Engine Core processes execute this module as __mp_main__ too.
if os.environ.get("GR_RELOAD_MEMORY_LOG"):
    install_probes()


def large_table(path: Path, odd: bool) -> str:
    import numpy as np
    from transformers import AutoTokenizer

    from vllm_gr.constraints.artifact import save_constraint_table_artifact
    from vllm_gr.constraints.table import ConstraintTable, ConstraintTableMetadata
    from vllm_gr.constraints.tokenizer_digest import compute_tokenizer_digest

    vocab = 128
    first = np.arange(vocab, dtype=np.int32)
    offsets = np.arange(vocab + 1, dtype=np.int32) * vocab
    second = np.tile(first, vocab)
    keys = np.arange(vocab * vocab, dtype=np.int64)
    final_offsets = np.arange(vocab * vocab + 1, dtype=np.int32) * 64
    final = np.tile(np.arange(int(odd), vocab, 2, dtype=np.int32), vocab * vocab)
    tensors = (first, offsets, second, keys, final_offsets, final)
    tokenizer = AutoTokenizer.from_pretrained(path.parents[1] / "model")
    table = ConstraintTable(
        *tensors,
        ConstraintTableMetadata(
            format="constraint_table_v1",
            artifact_digest=None,
            tokenizer_digest=compute_tokenizer_digest(tokenizer),
            vocab_size=vocab,
            max_prefix1_degree=vocab,
            max_prefix2_degree=64,
            total_bytes=sum(t.nbytes for t in tensors),
        ),
    )
    return cast(str, save_constraint_table_artifact(table, path))


def main() -> None:
    from transformers import AutoTokenizer

    from examples.online_serving.grpc.prepare_model import prepare_model
    from tests.st.run_checkpoint_reload import prepare, request, server
    from vllm_gr.constraints.tokenizer_digest import compute_tokenizer_digest

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph", action="store_true", help="Enable compile and CUDA Graph")
    parser.add_argument("--cycles", type=int, default=10)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["GR_RELOAD_MEMORY_CYCLES"] = str(args.cycles)
    report: dict[str, Any] = {"cycles": args.cycles, "graph": args.graph}
    first, second, config = prepare(root / "weights")
    weight_log = root / "weight-memory.jsonl"
    weight_log.write_text("")
    os.environ["GR_RELOAD_MEMORY_LOG"] = str(weight_log)
    with server(root, first, config, "weights", Path(__file__).resolve(), graph=args.graph) as (
        stub,
        control,
    ):
        stub.GenerateBatch(request(), timeout=120)
        for i in range(args.cycles + 1):
            control.post("/pause?mode=wait", json={}).raise_for_status()
            control.post(
                "/reload_weights",
                json={"weights_path": str(second if i % 2 else first), "version": f"cycle-{i}"},
            ).raise_for_status()
            control.post("/resume", json={}).raise_for_status()
            stub.GenerateBatch(request(), timeout=120)
    entries = [json.loads(x) for x in weight_log.read_text().splitlines()]
    after = [e for e in entries if e["event"] == "weight_after"][1:]
    assert len(after) == args.cycles
    assert max(e["allocated"] for e in after) == min(e["allocated"] for e in after), after
    report["weights"] = {
        "samples": after,
        "allocated_growth": after[-1]["allocated"] - after[0]["allocated"],
    }

    model, config = prepare_model(root / "tries", context_length=128)
    paths = [root / "tries" / v / "table.safetensors" for v in ("a", "b")]
    for p in paths:
        p.parent.mkdir(exist_ok=True)
    digests = [large_table(p, bool(i)) for i, p in enumerate(paths)]
    conf = json.loads(config.read_text())
    conf["constrained_generation"] = True
    conf["constraint_table"].update(
        enabled=True,
        path=str(paths[0]),
        artifact_digest=digests[0],
        tokenizer_digest=compute_tokenizer_digest(AutoTokenizer.from_pretrained(model)),
    )
    config.write_text(json.dumps(conf))
    trie_log = root / "trie-memory.jsonl"
    trie_log.write_text("")
    os.environ.update(GR_RELOAD_MEMORY_LOG=str(trie_log), VLLM_GR_TRIE_RETENTION_SECONDS="0.5")
    with server(root, model, config, "tries", Path(__file__).resolve(), graph=args.graph) as (
        stub,
        control,
    ):
        stub.GenerateBatch(request(), timeout=120)
        for i in range(args.cycles + 1):
            control.post(
                "/reload_trie", json={"trie_path": str(paths[(i + 1) % 2]), "version": f"cycle-{i}"}
            ).raise_for_status()
            time.sleep(0.6)
            response = stub.GenerateBatch(request(), timeout=120)
            assert all(
                token % 2 == (i + 1) % 2
                for item in response.items
                for token in list(item.beams.output_ids)[2::3]
            )
    entries = [json.loads(x) for x in trie_log.read_text().splitlines()]
    samples = []
    for i in range(1, args.cycles + 1):
        loaded = next(
            e for e in entries if e["event"] == "trie_after" and e["version"] == f"cycle-{i}"
        )
        expired = next(
            e for e in entries if e["event"] == "trie_expired" and e["version"] == f"cycle-{i}"
        )
        assert loaded["versions"] == 2 and expired["versions"] == 1
        released = loaded["allocated"] - expired["allocated"]
        logical_bytes = loaded["table_bytes"] - expired["table_bytes"]
        # Six device tensors; CUDA rounds each allocation to a 512-byte boundary.
        assert 0 <= released - logical_bytes <= 6 * 511, (loaded, expired)
        samples.append({"loaded": loaded, "expired": expired})
    allocated = [s["expired"]["allocated"] for s in samples]
    assert max(allocated) == min(allocated), samples
    report["trie"] = {
        "samples": samples,
        "allocated_growth": allocated[-1] - allocated[0],
        "allocator_flush": [e for e in entries if e["event"] == "trie_allocator_flush"],
    }
    if args.graph:
        for path in (weight_log, trie_log):
            events = [json.loads(line)["event"] for line in path.read_text().splitlines()]
            assert events.count("decode_graph_replay") >= (args.cycles + 1) * 2, path
            assert events.count("prefill_graph_replay") >= args.cycles + 1, path
            report.setdefault("graph_replays", {})[path.stem] = {
                "decode": events.count("decode_graph_replay"),
                "prefill": events.count("prefill_graph_replay"),
            }
    report["passed"] = True
    (root / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--serve":
        sys.argv.pop(1)
        from vllm_gr.entrypoints.cli.main import dispatch_main

        dispatch_main()
    else:
        main()
