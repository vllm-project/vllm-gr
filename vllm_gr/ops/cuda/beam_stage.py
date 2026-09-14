# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device preparation and small dispatch snapshots for persistent Beam execution."""

# Triton DSL arguments/decorators are not Python runtime types.
# mypy: disable-error-code="misc,no-untyped-def"

from __future__ import annotations

import triton as tr
import triton.language as tl


@tr.jit
def prepare_decode(
    tokens,
    sequence,
    active,
    generated,
    steps,
    initialized,
    finished,
    error,
    model_tokens,
    positions,
    execution_mask,
    execution_step,
    prefixes,
    suffix_lengths,
    suffix_offsets,
    STATE_ROW: tl.constexpr,
    WORKSPACE_ROW: tl.constexpr,
    ROW_OFFSET: tl.constexpr,
    WIDTH: tl.constexpr,
    PROMPT: tl.constexpr,
    TOKEN_BATCH_STRIDE: tl.constexpr,
    SEQ_BATCH_STRIDE: tl.constexpr,
    SEQ_BEAM_STRIDE: tl.constexpr,
    PREFIX_BATCH_STRIDE: tl.constexpr,
    PREFIX_BEAM_STRIDE: tl.constexpr,
    MAX_STEPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    beam = tl.arange(0, BLOCK)
    step = tl.load(steps + STATE_ROW)
    count = tl.load(generated + STATE_ROW)
    running = (
        tl.load(initialized + STATE_ROW)
        & ~tl.load(finished + STATE_ROW)
        & (tl.load(error + STATE_ROW) == 0)
        & (step >= 0)
        & (step < MAX_STEPS)
        & (count == step + 1)
    )
    live = running & tl.load(active + STATE_ROW * TOKEN_BATCH_STRIDE + beam, beam < WIDTH, other=0)
    token = tl.load(tokens + STATE_ROW * TOKEN_BATCH_STRIDE + beam, beam < WIDTH, other=0)
    tl.store(model_tokens + ROW_OFFSET + beam, tl.where(live, token, 0), beam < WIDTH)
    tl.store(positions + ROW_OFFSET + beam, tl.where(live, PROMPT + step, 0), beam < WIDTH)
    tl.store(execution_mask + beam, live, beam < WIDTH)
    # A scalar is an execution buffer; the persistent counter belongs to STATE_ROW.
    tl.store(execution_step, tl.where(running, step, -1))
    for col in tl.static_range(3):
        value = tl.load(
            sequence + STATE_ROW * SEQ_BATCH_STRIDE + beam * SEQ_BEAM_STRIDE + col,
            (beam < WIDTH) & live & (col < count) & (col < SEQ_BEAM_STRIDE),
            other=-1,
        )
        tl.store(
            prefixes + WORKSPACE_ROW * PREFIX_BATCH_STRIDE + beam * PREFIX_BEAM_STRIDE + col,
            value,
            beam < WIDTH,
        )
    # Fixed packed segments also cover inactive beams. Their suffix is zeroed
    # by the gather kernel, so Attention never consumes stale pool contents.
    length = tl.where(running, step + 1, 1)
    tl.store(suffix_lengths + beam, length, beam < WIDTH)
    tl.store(suffix_offsets + beam, beam * length, beam <= WIDTH)


@tr.jit
def latch_entry(generated, control, STATE_ROW: tl.constexpr):
    tl.store(control, tl.load(generated + STATE_ROW))


@tr.jit
def latch_result(generated, finished, error, control, STATE_ROW: tl.constexpr):
    before = tl.load(control)
    code = tl.load(error + STATE_ROW)
    tl.store(control, tl.load(generated + STATE_ROW) - before)
    tl.store(control + 1, tl.load(finished + STATE_ROW) | (code != 0))
    tl.store(control + 2, code)
