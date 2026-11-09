# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify Trie reload through production gRPC serving and the control endpoint."""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from examples.online_serving.grpc.prepare_model import prepare_model
from tests.st.run_checkpoint_reload import outputs, request, server
from vllm_gr.constraints.artifact import save_constraint_table_artifact
from vllm_gr.constraints.artifact_compiler import ConstraintTableArtifactCompiler
from vllm_gr.constraints.json_compiler import compile_relations_json
from vllm_gr.constraints.tokenizer_digest import compute_tokenizer_digest
from vllm_gr.sidecar import default_trie_checker

A: set[tuple[int, ...]] = {(0, 10, 127), (0, 11, 126), (1, 10, 126), (1, 11, 127)}
B: set[tuple[int, ...]] = {(0, 10, 126), (0, 11, 127), (1, 10, 127), (1, 11, 126)}


def prepare(output: Path) -> tuple[Path, Path, Path, Path]:
    from transformers import AutoTokenizer

    model, config_path = prepare_model(output, context_length=128)
    tokenizer = AutoTokenizer.from_pretrained(model)
    digest = compute_tokenizer_digest(tokenizer)
    compiler = ConstraintTableArtifactCompiler(tokenizer, 128, tokenizer_digest=digest)
    paths, digests = [], []
    for version, candidates in [("trie-a", A), ("trie-b", B)]:
        path = output / version / "table.safetensors"
        path.parent.mkdir(exist_ok=True)
        table = compiler.compile(
            compile_relations_json([[f"tok{i}" for i in sid] for sid in sorted(candidates)])
        )
        digests.append(save_constraint_table_artifact(table, path))
        paths.append(path)
    assert A.isdisjoint(B) and default_trie_checker(*paths)
    config = json.loads(config_path.read_text())
    config["constrained_generation"] = True
    config["constraint_table"].update(
        enabled=True, path=str(paths[0]), artifact_digest=digests[0], tokenizer_digest=digest
    )
    config_path.write_text(json.dumps(config, indent=2))
    return model, config_path, paths[0], paths[1]


def candidates(reply: Any) -> list[list[tuple[int, ...]]]:
    decoded = outputs(reply)
    return [
        [tuple(item["ids"][a:b]) for a, b in zip(item["offsets"], item["offsets"][1:])]
        for item in decoded
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph", action="store_true", help="Enable compile and CUDA Graph")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model, config, first, second = prepare(output)
    os.environ["VLLM_GR_TRIE_RETENTION_SECONDS"] = "1"
    result: dict[str, Any] = {
        "cases": [],
        "retention_seconds": 1,
        "batch": 4,
        "decode_steps": 3,
        "graph": args.graph,
    }

    with server(output, model, config, "trie-reload", graph=args.graph) as (stub, control):

        def generate(allowed: set[tuple[int, ...]]) -> list[list[tuple[int, ...]]]:
            actual = candidates(stub.GenerateBatch(request(), timeout=120))
            assert len(actual) == 4
            assert all(len(item) == 4 and set(item) == allowed for item in actual)
            return actual

        def reload(path: Path, version: str) -> None:
            response = control.post(
                "/reload_trie", json={"trie_path": str(path), "version": version}
            )
            response.raise_for_status()
            status = control.get("/service/status").json()
            assert status["service_state"] == "running" and status["trie_version"] == version
            assert status["trie_path"] == str(path)

        result["before"] = generate(A)
        reload(second, "trie-b")
        result["after"] = generate(B)
        result["cases"].append("A -> B: all returned candidates belong to B; no pause/resume")
        time.sleep(1.1)
        assert generate(B) == result["after"]
        result["cases"].append("after retention expires, serving continues on B")
        bad = output / "corrupt.safetensors"
        bad.write_bytes(b"invalid")
        response = control.post("/reload_trie", json={"trie_path": str(bad), "version": "bad"})
        assert response.status_code == 400, response.text
        assert control.get("/service/status").json()["trie_version"] == "trie-b"
        assert generate(B) == result["after"]
        result["cases"].append("invalid artifact rejected; B version and outputs remain intact")
        reload(first, "trie-a")
        assert generate(A) == result["before"]
        result["cases"].append("B -> A: candidates and ordering return to the original A")
        # Same external IDs are deliberately reused throughout the test.
    result["passed"] = True
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
