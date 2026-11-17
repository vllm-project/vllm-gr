# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Removed startup options fail at every public configuration ingress."""

from pathlib import Path

import pytest

from vllm_gr.config import gr_config_from_mapping, load_gr_config
from vllm_gr.config.parser import load_engine_gr_config

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


@pytest.mark.parametrize(  # type: ignore[misc]
    "key",
    [
        "execution_mode",
        "dynamic_beam_widths",
        "per_parent_beam_widths",
        "engine_driven",
        "catalog_path",
        "constraint_backend",
        "constrained_generation",
        "online_batch_sizes",
        "prefill_beam_batch_sizes",
        "startup_capture_batch_size",
    ],
)
@pytest.mark.parametrize("source", ["mapping", "yaml", "python"])  # type: ignore[misc]
def test_removed_options_fail_without_mutating_shared_defaults(
    tmp_path: Path, key: str, source: str
) -> None:
    with pytest.raises(ValueError, match=key):
        if source == "mapping":
            gr_config_from_mapping({"gr": {key: True}})
        elif source == "yaml":
            path = tmp_path / "scenario.yaml"
            path.write_text("gr:\n  " + key + ": true\n")
            load_gr_config(path)
        else:
            load_engine_gr_config(None, {}, {key: True})
    config = load_engine_gr_config(None, {})
    assert config.beam_max_width == 128
