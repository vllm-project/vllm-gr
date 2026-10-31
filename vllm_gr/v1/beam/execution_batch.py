# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-independent execution geometry for a fixed Beam batch session.

This module deliberately contains no tensor, KV-cache, attention, or graph
objects. GPU and NPU adapters consume the same immutable row mapping and own
the device-specific representation themselves.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, NoReturn, TypeGuard, TypeVar, cast

BeamExecutionStage = Literal["prefill", "decode"]

_T = TypeVar("_T")


class BeamExecutionContractError(ValueError):
    """Raised before device execution when batch geometry is inconsistent."""


def _fail(message: str) -> NoReturn:
    raise BeamExecutionContractError(message)


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _non_negative_int(name: str, value: object) -> int:
    if not _is_int(value) or value < 0:
        _fail(f"{name} must be a non-negative integer, got {value!r}")
    return value


def _positive_int(name: str, value: object) -> int:
    if not _is_int(value) or value <= 0:
        _fail(f"{name} must be a positive integer, got {value!r}")
    return value


def _immutable_tuple(name: str, value: object) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        _fail(f"{name} must be an immutable tuple")
    return value


def _attribute(owner: object, name: str) -> object:
    try:
        return getattr(owner, name)
    except AttributeError:
        _fail(f"stage metadata is missing {name!r}")


@dataclass(frozen=True, slots=True)
class BeamBatchExecutionView:
    """Immutable logical-request to physical-row mapping for one GR stage.

    Prefill has one execution row per request. Decode uses request-major
    B * input_width rows. beam_width is the session's immutable storage
    capacity; dynamic stages may consume and retain different active widths.
    """

    session_id: str
    dispatch_id: int
    stage_index: int
    kind: BeamExecutionStage
    decode_step: int | None
    owner_generation: int
    request_ids: tuple[str, ...]
    num_prompt_tokens_by_request: tuple[int, ...]
    beam_width: int
    produces_output: bool
    dynamic_beam_widths: tuple[int, ...] | None = None
    per_parent_beam_widths: tuple[int, ...] | None = None
    # Candidates per beam row (columns), independent of ``beam_width`` (rows).
    # None keeps the pre-decoupling behaviour, K == beam_width.
    candidate_width: int | None = None
    row_request_indices: tuple[int, ...] = field(init=False)
    row_beam_indices: tuple[int, ...] = field(init=False)
    prompt_token_offsets: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            _fail("session_id must be a non-empty string")
        _non_negative_int("dispatch_id", self.dispatch_id)
        _non_negative_int("stage_index", self.stage_index)
        _non_negative_int("owner_generation", self.owner_generation)
        width = _positive_int("beam_width", self.beam_width)
        if self.dynamic_beam_widths is not None:
            widths = _immutable_tuple("dynamic_beam_widths", self.dynamic_beam_widths)
            if len(widths) not in (1, 2, 3) or self.stage_index >= len(widths):
                _fail("dynamic_beam_widths must cover every stage in a 1/2/3-step session")
            for index, active_width in enumerate(widths):
                if _positive_int(f"dynamic_beam_widths[{index}]", active_width) > width:
                    _fail("dynamic_beam_widths entries must not exceed beam_width")
        if not isinstance(self.produces_output, bool):
            _fail(f"produces_output must be a bool, got {self.produces_output!r}")
        if self.candidate_width is not None:
            _positive_int("candidate_width", self.candidate_width)
            # Every schedule step selects ``output_width`` survivors from the
            # candidates its parents supply, so K must cover the widest step.
            # Checked here rather than at the first ``dynamic.select`` so a
            # mismatched K fails before any device work.  A per-stage K
            # schedule overrides the scalar and carries its own bound.
            scalar_widths = (
                () if self.per_parent_beam_widths is not None else (self.dynamic_beam_widths or ())
            )
            for index, active_width in enumerate(scalar_widths):
                if active_width > self.candidate_width:
                    _fail(
                        f"candidate_width {self.candidate_width} is below "
                        f"dynamic_beam_widths[{index}]={active_width}; each parent "
                        "must supply at least as many candidates as the stage retains"
                    )

        if not isinstance(self.request_ids, tuple):
            _fail("request_ids must be an immutable tuple")
        if not self.request_ids:
            _fail("request_ids must contain at least one request")
        seen: set[str] = set()
        for index, request_id in enumerate(self.request_ids):
            if not isinstance(request_id, str) or not request_id.strip():
                _fail(f"request_ids[{index}] must be a non-empty string")
            if request_id in seen:
                _fail("request_ids must be unique and ordered")
            seen.add(request_id)

        if not isinstance(self.num_prompt_tokens_by_request, tuple):
            _fail("num_prompt_tokens_by_request must be an immutable tuple")
        if len(self.num_prompt_tokens_by_request) != len(self.request_ids):
            _fail("num_prompt_tokens_by_request must contain one count per request")
        offsets = [0]
        for index, count in enumerate(self.num_prompt_tokens_by_request):
            offsets.append(
                offsets[-1] + _positive_int(f"num_prompt_tokens_by_request[{index}]", count)
            )

        if self.kind == "prefill":
            if self.stage_index != 0 or self.decode_step is not None:
                _fail("Prefill requires stage_index=0 and decode_step=None")
            rows_per_request = 1
        elif self.kind == "decode":
            expected_step = self.stage_index - 1
            if self.stage_index == 0 or self.decode_step != expected_step:
                _fail("Decode requires stage_index>0 and decode_step=stage_index-1")
            if not self.produces_output:
                _fail("Decode must produce output")
            rows_per_request = self.input_width
        else:
            _fail(f"kind must be 'prefill' or 'decode', got {self.kind!r}")

        request_rows = tuple(
            request_index
            for request_index in range(len(self.request_ids))
            for _ in range(rows_per_request)
        )
        beam_rows = tuple(
            beam_index for _ in self.request_ids for beam_index in range(rows_per_request)
        )
        object.__setattr__(self, "row_request_indices", request_rows)
        object.__setattr__(self, "row_beam_indices", beam_rows)
        object.__setattr__(self, "prompt_token_offsets", tuple(offsets))

    @classmethod
    def from_stage_metadata(cls, metadata: object) -> BeamBatchExecutionView:
        """Build and validate a view from the structural GRStageMetadata API."""

        raw_kind = _attribute(metadata, "kind")
        kind = getattr(raw_kind, "value", raw_kind)
        if kind not in ("prefill", "decode"):
            _fail(f"stage metadata kind is unsupported: {raw_kind!r}")
        normalized_kind = cast(BeamExecutionStage, kind)

        params = _attribute(metadata, "beam_params")
        request_ids = _immutable_tuple("request_ids", _attribute(metadata, "request_ids"))
        prompt_counts = _immutable_tuple(
            "num_prompt_tokens_by_request",
            _attribute(metadata, "num_prompt_tokens_by_request"),
        )
        view = cls(
            session_id=cast(str, _attribute(metadata, "session_id")),
            dispatch_id=cast(int, _attribute(metadata, "dispatch_id")),
            stage_index=cast(int, _attribute(metadata, "stage_index")),
            kind=normalized_kind,
            decode_step=cast(int | None, _attribute(metadata, "decode_step")),
            owner_generation=cast(int, _attribute(metadata, "owner_generation")),
            request_ids=cast(tuple[str, ...], request_ids),
            num_prompt_tokens_by_request=cast(tuple[int, ...], prompt_counts),
            beam_width=cast(int, _attribute(params, "beam_width")),
            produces_output=cast(bool, _attribute(metadata, "produces_output")),
            dynamic_beam_widths=getattr(params, "dynamic_beam_widths", None),
            per_parent_beam_widths=getattr(params, "per_parent_beam_widths", None),
            candidate_width=getattr(params, "candidate_width", None),
        )

        total_prompt_tokens = _positive_int(
            "num_prompt_tokens", _attribute(metadata, "num_prompt_tokens")
        )
        if view.prompt_token_offsets[-1] != total_prompt_tokens:
            _fail("num_prompt_tokens does not equal the per-request prompt-token total")

        capacity = getattr(metadata, "execution_capacity", None)
        if capacity is not None and getattr(capacity, "beam_width", None) != view.beam_width:
            _fail("execution_capacity.beam_width does not match beam_params.beam_width")
        return view

    @property
    def batch_size(self) -> int:
        return len(self.request_ids)

    @property
    def rows_per_request(self) -> int:
        return self.input_width

    @property
    def input_width(self) -> int:
        """Actual parent rows per request consumed by this stage."""
        if self.kind == "prefill":
            return 1
        if self.dynamic_beam_widths is None:
            return self.beam_width
        return self.dynamic_beam_widths[self.stage_index - 1]

    @property
    def output_width(self) -> int:
        """Winners retained after this stage, independently of parent width."""
        if self.dynamic_beam_widths is None:
            return self.beam_width
        return self.dynamic_beam_widths[self.stage_index]

    @property
    def candidate_top_k(self) -> int:
        """Candidates produced per parent row: K, not the survivor count.

        This is the column extent of the candidate buffers the sampler fills.
        A request-level ``per_parent_beam_widths`` schedule gives each stage its
        own K and takes precedence over the configured scalar K.  Without
        either, each stage takes its own ``output_width``: taking fewer than
        the target width per parent can lose global Top-K winners when a
        schedule expands after an earlier pruning step.
        """

        if self.per_parent_beam_widths is not None:
            return self.per_parent_beam_widths[self.stage_index]
        if self.candidate_width is not None:
            return self.candidate_width
        return self.output_width

    @property
    def num_execution_rows(self) -> int:
        return len(self.row_request_indices)

    @property
    def row_request_ids(self) -> tuple[str, ...]:
        return tuple(self.request_ids[index] for index in self.row_request_indices)

    def physical_row(self, request_index: int, beam_index: int = 0) -> int:
        """Return the request-major physical row for a logical Beam row."""

        request_index = self._request_index(request_index)
        if not _is_int(beam_index) or not 0 <= beam_index < self.rows_per_request:
            _fail(f"beam_index must be in [0, {self.rows_per_request}), got {beam_index!r}")
        return request_index * self.rows_per_request + beam_index

    def row_owner(self, physical_row: int) -> tuple[int, int]:
        """Return (request_index, local_beam_index) for a physical row."""

        if not _is_int(physical_row) or not 0 <= physical_row < self.num_execution_rows:
            _fail(f"physical_row must be in [0, {self.num_execution_rows}), got {physical_row!r}")
        return self.row_request_indices[physical_row], self.row_beam_indices[physical_row]

    def rows_for_request(self, request_index: int) -> range:
        """Return the contiguous physical rows owned by one logical request."""

        request_index = self._request_index(request_index)
        start = request_index * self.rows_per_request
        return range(start, start + self.rows_per_request)

    def expand_per_request(self, values: Sequence[_T], *, name: str = "values") -> tuple[_T, ...]:
        """Repeat request-level values onto this stage's physical rows."""

        if len(values) != self.batch_size:
            _fail(f"{name} must contain one value per request")
        return tuple(value for value in values for _ in range(self.rows_per_request))

    def validate_request_order(self, request_ids: Sequence[str]) -> None:
        """Reject a scheduler/runner view that split, mixed, or reordered the batch."""

        if tuple(request_ids) != self.request_ids:
            _fail("request IDs were split, mixed, or reordered")

    def assert_same_logical_batch(self, other: BeamBatchExecutionView) -> None:
        """Validate immutable batch identity across Prefill and Decode stages."""

        if not isinstance(other, BeamBatchExecutionView):
            _fail("other must be BeamBatchExecutionView")
        identity = (
            self.session_id,
            self.owner_generation,
            self.request_ids,
            self.num_prompt_tokens_by_request,
            self.beam_width,
            self.dynamic_beam_widths,
            self.per_parent_beam_widths,
        )
        other_identity = (
            other.session_id,
            other.owner_generation,
            other.request_ids,
            other.num_prompt_tokens_by_request,
            other.beam_width,
            other.dynamic_beam_widths,
            other.per_parent_beam_widths,
        )
        if other_identity != identity:
            _fail("logical Beam batch identity changed across stages")

    def localize_global_parent_rows(self, parent_rows: Sequence[int]) -> tuple[int, ...]:
        """Convert global Decode parents to request-local Beam indices.

        Each child row may select a parent only from the same logical request.
        This catches cross-request contamination before result splitting or KV
        reordering.
        """

        if self.kind != "decode":
            _fail("parent rows are only defined for Decode")
        if len(parent_rows) != self.batch_size * self.output_width:
            _fail("parent_rows must contain one parent per output Beam row")
        local: list[int] = []
        for child_row, parent_row in enumerate(parent_rows):
            if not _is_int(parent_row) or not 0 <= parent_row < self.num_execution_rows:
                _fail(f"parent_rows[{child_row}] is outside the Decode row range")
            child_request = child_row // self.output_width
            parent_request = self.row_request_indices[parent_row]
            if parent_request != child_request:
                _fail(f"parent_rows[{child_row}] crosses a logical request boundary")
            local.append(self.row_beam_indices[parent_row])
        return tuple(local)

    def _request_index(self, request_index: int) -> int:
        if not _is_int(request_index) or not 0 <= request_index < self.batch_size:
            _fail(f"request_index must be in [0, {self.batch_size}), got {request_index!r}")
        return request_index


__all__ = [
    "BeamBatchExecutionView",
    "BeamExecutionContractError",
    "BeamExecutionStage",
]
