# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import runpy
import shutil
import subprocess
import sys


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Execute bash commands to get the dynamically calculated gpu and index
    bash_cmd = f'source {script_dir}/get_index.sh && idle_gpu=$(get_gpu 4) && index=$(get_index) && echo "$idle_gpu:$index"'

    try:
        output = subprocess.check_output(["bash", "-c", bash_cmd], text=True).strip()
        idle_gpu, index_str = output.split(":")
        index = int(index_str)
    except Exception as e:
        print(f"Failed to get dynamic configuration from bash: {e}")
        sys.exit(1)

    run_port = str(index + 8000)
    print(f"Dynamically resolved -> CUDA_VISIBLE_DEVICES: {idle_gpu}, Port: {run_port}")

    # Apply to the environment for vLLM
    os.environ["CUDA_VISIBLE_DEVICES"] = idle_gpu

    # Inject arguments natively
    sys.argv = [
        "vllm",
        "serve",
        "--gr",
        "--port",
        run_port,
        "--seed",
        "123",
        "--default-chat-template-kwargs",
        '{"enable_thinking": false}',
        "--max-logprobs",
        "2048",
        "--max_num_seqs",
        "1024",
        "--max_num_batched_tokens",
        "16384",
        "--scheduling-policy",
        "priority",
        "--catalog-path",
        "/home/yoni/OpenOneRec/benchmark_data/catalog.json",
        "OpenOneRec/OneRec-1.7B",
    ]

    vllm_bin = shutil.which("vllm")
    if not vllm_bin:
        print("Failed to find 'vllm' executable in PATH.", file=sys.stderr)
        sys.exit(1)

    # Run vllm executable script, keeping debugger attached
    runpy.run_path(vllm_bin, run_name="__main__")


if __name__ == "__main__":
    main()
