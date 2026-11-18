# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Backend-neutral views, validation and legacy materialization of Beam columns."""

from itertools import repeat

import numpy as np

from vllm_gr.v1.beam.batch_contracts import (
    BeamBatchOutput,
    BeamBatchResult,
    BeamOutputSequence,
    BeamRequestOutput,
    BeamSequenceColumns,
)


def sequence_columns(sequences: BeamSequenceColumns):
    shape = (sequences.count, sequences.steps)
    tokens = np.frombuffer(sequences.token_ids, dtype="<i4").reshape(shape)
    scores = (
        None
        if sequences.cumulative_logprobs is None
        else np.frombuffer(sequences.cumulative_logprobs, dtype="<f4").reshape(sequences.count)
    )
    parents = (
        None
        if sequences.parent_beam_indices is None
        else np.frombuffer(sequences.parent_beam_indices, dtype="<i4").reshape(shape)
    )
    return tokens, scores, parents


def validate_complete_sequences(sequences: BeamSequenceColumns, request=None) -> None:
    """Validate both the owned column shapes and request-dependent semantics."""
    if type(sequences.count) is not int or type(sequences.steps) is not int:
        raise ValueError("Beam column dimensions must be integers")
    if sequences.count < 1 or sequences.steps < 1:
        raise ValueError("Beam column dimensions must be positive")
    for value in (
        sequences.token_ids,
        sequences.cumulative_logprobs,
        sequences.parent_beam_indices,
    ):
        if value is not None and type(value) is not bytes:
            raise ValueError("Beam columns must be immutable bytes")
    tokens, scores, parents = sequence_columns(sequences)
    if np.any(tokens < 0) or (scores is not None and not np.isfinite(scores).all()):
        raise ValueError("Beam columns contain negative tokens or non-finite scores")
    if parents is not None and np.any(parents < 0):
        raise ValueError("Beam columns contain negative parent indices")
    if request is None:
        return
    options, params = request.output_options, request.beam_params
    if sequences.count > options.num_return_sequences or sequences.steps != params.max_tokens:
        raise ValueError("terminal Beam columns violate the result count or effective budget")
    if (scores is not None) != options.include_cumulative_logprob:
        raise ValueError("terminal Beam score presence differs from requested options")
    if (parents is not None) != options.include_parent_beam_indices:
        raise ValueError("terminal Beam parent presence differs from requested options")
    if parents is not None:
        widths = params.dynamic_beam_widths
        bounds = params.beam_width if widths is None else np.asarray((1, *widths[:-1]))
        if np.any(parents >= bounds):
            raise ValueError("terminal Beam result has invalid parent indices")


def materialize_beam_result(result: BeamBatchResult) -> BeamBatchResult:
    """Keep the public tuple-of-sequences API for HTTP and offline consumers."""
    if result.output is None or not any(
        isinstance(item.sequences, BeamSequenceColumns) for item in result.output.items
    ):
        return result
    items = []
    for item in result.output.items:
        sequences = item.sequences
        if isinstance(sequences, BeamSequenceColumns):
            tokens, scores, parents = sequence_columns(sequences)
            sequences = tuple(
                BeamOutputSequence(tuple(t), score, tuple(p), "length", None)
                for t, score, p in zip(
                    tokens.tolist(),
                    repeat(None) if scores is None else scores.tolist(),
                    repeat(()) if parents is None else parents.tolist(),
                )
            )
        items.append(BeamRequestOutput(item.item_index, item.request_id, sequences))
    return BeamBatchResult(
        result.protocol_version, result.batch_id, BeamBatchOutput(tuple(items)), result.failure
    )
