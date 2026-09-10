
# Installation

## Prerequisites

- Python 3.11+
- Pip 23+
- Rust 1.88.0+ and Cargo (required for source or editable installs; latest stable recommended)
- CUDA 13.0 (for GPU inference)
- 32GB+ RAM (recommended)

Prebuilt vllm-gr wheels already contain the compiled BeamProc executable and
do not require Rust on the target machine. Building from a source checkout or
source distribution compiles BeamProc during installation and therefore
requires both `rustc` and `cargo`. Rust 1.88.0 is the minimum version required
by the locked Rust dependencies; the latest stable toolchain is recommended.

Install the stable toolchain with the official `rustup` installer:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
  sh -s -- -y --profile minimal --default-toolchain stable
source "$HOME/.cargo/env"
rustc --version
cargo --version
```

If `rustup` is already installed, update the toolchain with
`rustup update stable`. See [Install Rust](https://www.rust-lang.org/tools/install)
for platform-specific instructions.

## Setup

1. **Clone the repository**

```bash
git clone https://github.com/JiusiServe/vllm-gr.git
cd vllm-gr
```

2. **Create virtual environment**

This step is not mandatory. You can use a previously installed environment.

```bash
uv venv --python 3.12
source .venv/bin/activate
python -m ensurepip --upgrade
ln -s -r .venv/bin/pip3 .venv/bin/pip
#make sure pip is installed correctly
pip --version
#pip 25.0.1 from /your-path/vllm-gr/.venv/lib/python3.12/site-packages/pip (python 3.12)
```

3. **Install vllm-gr**

For standard installation:

```bash
pip install -e .
```

For developer mode (includes pytest, ruff, etc.):

```bash
pip install -e ".[dev]"
```

Confirm that the resolved environment contains the supported runtime:

```bash
python -c 'import importlib.metadata as m; print("vllm", m.version("vllm")); print("vllm-gr", m.version("vllm-gr"))'
python -m pip check
```

Platform dependencies come from the shared `requirements/common.txt`; there are no CUDA-only extras. See [getting_started/installation/](getting_started/installation/README.md).

> **Note:** Installing using `uv` (e.g. `uv pip install`) may cause race conditions during installation. It is recommended to use `pip` directly to ensure a predictable installation.

4. **Verify installation**

```bash
python -c "import benchmarks.open_one_rec; print('✓ Installation successful')"
```

## Quick Start

### API Server Mode (Production-like)

To run vllm-gr in a server--client configuration, use two terminals

### On the first terminal: Run the server

Use vllm-gr serve (beam search patches are automatically loaded via plugin)

```bash
vllm-gr serve OpenOneRec/OneRec-1.7B \
  --max-logprobs 1024 \
  --max-num-seqs 1024 \
  --max-num-batched-tokens 16384 \
  --scheduling-policy priority \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

### On the second terminal: Run single request against server with CLI

```bash
# CLI Single request Inference
curl -X POST "http://localhost:8000/v1/chat/completions" -H "Content-Type: application/json" \
  --data '{
    "model": "OpenOneRec/OneRec-1.7B",
    "messages": [
      {
        "role": "user",
        "content": "这是一个视频：<|sid_begin|><s_a_346><s_b_6566><s_c_5603><|sid_end|>，帮我总结一下这个视频讲述了什么内容"
      }
    ],
    "use_beam_search": true,
    "n": 5,
    "temperature": 0.5,
    "max_tokens": 5,
    "ignore_eos": false,
    "length_penalty": 1.0,
    "vllm_xargs": { "begin_token": "<|sid_begin|>", "end_token": "<|sid_end|>" }
  }'
```

Or use the [benchmark](benchmarks/README.md) to run a more complete benchmark test.
See [Rust BeamProc Coordinator](rust_beamproc.md) to enable and benchmark the
optional Rust coordinator.

### Data Parallelism

To run `vllm-gr serve` with data parallelism, please see [data_parallelism.md](data_parallelism.md)

### LMCache

To run `vllm-gr serve` with LMCache, please see [lmcache.md](lmcache.md)
