# GPU Installation

## Requirements

- OS: Linux
- Python: 3.11–3.13
- NVIDIA GPU with CUDA driver / runtime
- Compatible versions:
    - `vllm==0.14.1`
    - `torch==2.9.1`

## Install vLLM (base environment)

vllm-gr does **not** install vllm. Prepare the base environment first:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm==0.14.1 --torch-backend=auto
```

## Install vllm-gr

```bash
git clone https://github.com/JiusiServe/vllm-gr.git
cd vllm-gr
uv pip install -e .
```

setup.py detects the CUDA backend and loads `requirements/cuda.txt`, which pulls
in the shared deps from `requirements/common.txt` (no CUDA-only extras).

## Developer tools (optional)

```bash
uv pip install -e ".[dev]"
```

## Verify

```bash
python -c "import vllm_gr; print(vllm_gr.__version__)"
python -c "import msgspec, safetensors; print('cuda deps ok')"
```

## Notes

- `VLLM_GR_TARGET_DEVICE` is optional. Set to `cuda` only when auto-detection is wrong.
- LMCache is temporarily not included in the CUDA requirements path by default.

See [cuda.inc.md](gpu/cuda.inc.md) for command snippets used in documentation includes.
