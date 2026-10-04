#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
manifest="$repo_root/rust/Cargo.toml"

cargo fmt --manifest-path "$manifest" --all -- --check
cargo clippy --locked --manifest-path "$manifest" \
    --all-targets --features iceoryx -- -D warnings
