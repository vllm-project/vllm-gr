#!/bin/bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

# Editable installs and cargo clippy do not guarantee a loadable cdylib.
# Build the exact checkout tested below, including the real IPC test peer.
cargo build --release --locked --manifest-path "$repo_root/rust/Cargo.toml" \
    --target-dir "$repo_root/rust/target" \
    --features iceoryx --lib --bin beam-proc --example iceoryx-peer
export VLLM_GR_CE_TRANSPORT_LIBRARY="$repo_root/rust/target/release/libce_transport.so"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

# Run all tests recursively under tests/
python3 -m pytest -x --run-slow --durations=10 -s tests
