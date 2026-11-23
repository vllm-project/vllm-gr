# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Build Constraint Tables directly from a tokenizer's s_a / s_b / s_c tokens.

Scan ``tokenizer.get_added_vocab()`` for the three contiguous special
token groups ``<s_a_*>`` / ``<s_b_*>`` / ``<s_c_*>``, then build the six CSR
tables consumed by ``rec_constrained_topk`` without requiring a catalog JSON
document.  The result is a :class:`~vllm_gr.constraints.compiler.ConstraintTableResource`
(plain NumPy arrays + content digest). Serialize it as a constraint-table
artifact for the V1 worker startup loader.

Two candidate modes (matching the constraint operators):

- **Fully-connected** (default): every valid ``s_a`` maps to every valid
  ``s_b`` maps to every valid ``s_c``.  Table size grows O(N^3); the item
  universe is capped by ``max_valid_items`` (default 500) to keep the tables
  small — but a cap that actually truncates the tokenizer's item groups is
  a hard error, not a silent downgrade (a truncated universe drops most
  real SIDs and makes every request emit the same low-numbered candidates).
  Provide ``valid_item_ids`` / ``prefix1_file`` + ``prefix2_file``, or build
  the triples document from the real catalog
  (``tools/constraints/build_constraint_triples.py``) instead.
- **Sparse** (``prefix1_file`` + ``prefix2_file``): per-key constraints read
  from JSON, avoiding the O(N^3) blow-up for real co-occurrence data.

The FSM slots: step 0 candidates from ``first_token_ids`` (s_a), step 1 from
``prefix1_*`` (s_b), step 2+ from ``prefix2_*`` (s_c).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

from vllm_gr.constraints.compiler import (
    ConstraintTableResource,
    validate_resource_tables,
)

LOGGER = logging.getLogger(__name__)


def _scan_token_ranges(
    tokenizer: Any,
) -> tuple[int | None, int | None, int | None, int, int, int]:
    """Scan added tokens for the contiguous s_a / s_b / s_c ID ranges.

    Returns ``(sa_start, sb_start, sc_start, sa_count, sb_count, sc_count)`` or
    ``(None, None, None, 0, 0, 0)`` when the tokenizer has no such groups
    (caller treats that as "Constraint Tables not applicable").
    """
    sa_start: int | None = None
    sb_start: int | None = None
    sc_start: int | None = None
    sa_count = 0
    sb_count = 0
    sc_count = 0

    try:
        added_tokens = tokenizer.get_added_vocab()
    except AttributeError:  # pragma: no cover - non-HF tokenizer
        return None, None, None, 0, 0, 0

    for token_str, token_id in added_tokens.items():
        if token_str.startswith("<s_a_") and token_str.endswith(">"):
            if sa_start is None or token_id < sa_start:
                sa_start = token_id
            sa_count += 1
        elif token_str.startswith("<s_b_") and token_str.endswith(">"):
            if sb_start is None or token_id < sb_start:
                sb_start = token_id
            sb_count += 1
        elif token_str.startswith("<s_c_") and token_str.endswith(">"):
            if sc_start is None or token_id < sc_start:
                sc_start = token_id
            sc_count += 1

    return sa_start, sb_start, sc_start, sa_count, sb_count, sc_count


def _build_fully_connected(
    *,
    vocab_size: int,
    sa_start: int,
    sb_start: int,
    sc_start: int,
    valid_a_ids: list[int],
    valid_b_ids: list[int],
    valid_c_ids: list[int],
) -> dict[str, Any]:
    """Build the six CSR tables for the fully-connected constraint graph."""
    na, nb, nc = len(valid_a_ids), len(valid_b_ids), len(valid_c_ids)
    if na == 0 or nb == 0 or nc == 0:
        raise ValueError(
            "cannot build fully-connected constraint tables with empty "
            f"s_a/s_b/s_c sets ({na}/{nb}/{nc})"
        )

    # Pre-flight memory estimate; prefix2_values dominates (O(na*nb*nc)).
    est_prefix2_count = na * nb * nc
    est_bytes = (
        est_prefix2_count * 4
        + na * nb * 8
        + na * nb * 4
        + (na * nb + 1) * 4
        + (vocab_size + 1) * 4
        + na * 4
    )
    est_gb = est_bytes / (1024**3)
    if est_gb > 4.0:
        raise ValueError(
            f"constraint tables would use ~{est_gb:.1f} GiB "
            f"({na}x{nb}x{nc}= {est_prefix2_count:,} prefix2 entries); "
            "reduce the valid item universe or provide sparse prefix files"
        )

    first_token_ids = np.asarray([sa_start + n for n in valid_a_ids], dtype=np.int32)
    valid_b_token_ids = np.asarray([sb_start + n for n in valid_b_ids], dtype=np.int32)
    valid_c_token_ids = np.asarray([sc_start + n for n in valid_c_ids], dtype=np.int32)

    counts = np.zeros(vocab_size, dtype=np.int32)
    for a_num in valid_a_ids:
        counts[sa_start + a_num] = nb
    prefix1_offsets = np.zeros(vocab_size + 1, dtype=np.int32)
    np.cumsum(counts, out=prefix1_offsets[1:])

    # tile (not element-wise repeat): each s_a gets the full sorted s_b list,
    # matching the a-major ordering of ``pair_keys`` below. Using np.repeat
    # instead would misalign the CSR bucket.
    prefix1_values = np.tile(valid_b_token_ids, na)

    a_token_ids = np.asarray([sa_start + n for n in valid_a_ids], dtype=np.int64)
    pair_keys = (
        a_token_ids[:, None] * vocab_size + valid_b_token_ids[None, :].astype(np.int64)
    ).reshape(-1)

    n_pairs = na * nb
    prefix2_value_offsets = np.arange(0, (n_pairs + 1) * nc, nc, dtype=np.int32)
    prefix2_values = np.tile(valid_c_token_ids, n_pairs)

    return {
        "first_token_ids": first_token_ids,
        "prefix1_offsets": prefix1_offsets,
        "prefix1_values": prefix1_values,
        "prefix1_pair_keys": pair_keys,
        "prefix2_value_offsets": prefix2_value_offsets,
        "prefix2_values": prefix2_values,
        "max_prefix1_degree": nb,
        "max_prefix2_degree": nc,
    }


def _build_sparse(
    *,
    vocab_size: int,
    sa_start: int,
    sb_start: int,
    sc_start: int,
    prefix1_file: str,
    prefix2_file: str,
) -> dict[str, Any]:
    """Build the six CSR tables from per-key sparse constraint JSON files."""
    with open(prefix1_file, "r", encoding="utf-8") as f:
        prefix1_raw = json.load(f)
    with open(prefix2_file, "r", encoding="utf-8") as f:
        prefix2_raw = json.load(f)

    prefix1 = {int(k_str): [int(x) for x in v] for k_str, v in prefix1_raw.items()}
    prefix2 = {
        tuple(int(p) for p in k_str.split(",")): [int(x) for x in v]
        for k_str, v in prefix2_raw.items()
    }
    valid_a_ids = sorted(prefix1.keys())
    if not valid_a_ids:
        raise ValueError("sparse prefix1 file defines no valid s_a keys")

    first_token_ids = np.asarray([sa_start + n for n in valid_a_ids], dtype=np.int32)

    counts = np.zeros(vocab_size, dtype=np.int32)
    for a_num in valid_a_ids:
        counts[sa_start + a_num] = len(prefix1[a_num])
    prefix1_offsets = np.zeros(vocab_size + 1, dtype=np.int32)
    np.cumsum(counts, out=prefix1_offsets[1:])

    prefix1_values_list: list[int] = []
    pair_keys_list: list[int] = []
    prefix2_values_list: list[int] = []
    prefix2_value_offsets_list = [0]
    offset = 0
    for a_num in valid_a_ids:
        t0 = sa_start + a_num
        for b_num in prefix1[a_num]:
            t1 = sb_start + b_num
            prefix1_values_list.append(t1)
            pair_keys_list.append(t0 * vocab_size + t1)
            sc_list = prefix2.get((a_num, b_num), [])
            for c_num in sc_list:
                prefix2_values_list.append(sc_start + c_num)
            offset += len(sc_list)
            prefix2_value_offsets_list.append(offset)

    prefix1_values = np.asarray(prefix1_values_list, dtype=np.int32)
    pair_keys = np.asarray(pair_keys_list, dtype=np.int64)
    prefix2_value_offsets = np.asarray(prefix2_value_offsets_list, dtype=np.int32)
    prefix2_values = np.asarray(prefix2_values_list, dtype=np.int32)

    max_prefix1_degree = max(len(v) for v in prefix1.values())
    max_prefix2_degree = max((len(v) for v in prefix2.values()), default=0)
    if max_prefix2_degree == 0:
        raise ValueError("sparse prefix2 file defines no valid (s_a,s_b) pairs")

    return {
        "first_token_ids": first_token_ids,
        "prefix1_offsets": prefix1_offsets,
        "prefix1_values": prefix1_values,
        "prefix1_pair_keys": pair_keys,
        "prefix2_value_offsets": prefix2_value_offsets,
        "prefix2_values": prefix2_values,
        "max_prefix1_degree": max_prefix1_degree,
        "max_prefix2_degree": max_prefix2_degree,
    }


def build_constraint_tables_from_tokenizer(
    tokenizer: Any,
    vocab_size: int,
    dtype: str = "int32",
    max_valid_items: int = 500,
    valid_item_ids: list[int] | np.ndarray | None = None,
    prefix1_file: str | None = None,
    prefix2_file: str | None = None,
) -> ConstraintTableResource | None:
    """Build Constraint Tables from a tokenizer's s_a / s_b / s_c token groups.

    Returns a validated :class:`~vllm_gr.constraints.compiler.ConstraintTableResource`
    (ready for worker injection) or ``None`` when the tokenizer has no
    ``<s_a_*>`` / ``<s_b_*>`` / ``<s_c_*>`` groups (Constraint Tables do not
    apply to this model — callers keep their existing path).

    Args:
        tokenizer: the model tokenizer (HF-style, with ``get_added_vocab``).
        vocab_size: ``model_config.vocab_size`` — MUST match ``logits.size(1)``
            (the operator derives the pair-key multiplier from the logits
            shape; a padded embed vocab mismatch breaks prefix2 lookup).
        max_valid_items: cap on the item universe for fully-connected mode.
            A tokenizer whose item groups EXCEED this cap fails with a
            ``ValueError`` (silent truncation produced wrong results);
            raise the cap, pass ``valid_item_ids``, or use sparse files.
        valid_item_ids: explicit ``[0, sa_count)``-indexed item IDs to use in
            fully-connected mode (replaces the first-``max_valid_items`` slice).
            Ignored when ``prefix1_file``/``prefix2_file`` are provided.
        prefix1_file / prefix2_file: sparse per-key constraints (see module
            docstring); when both are provided the sparse tables are built.
    """
    sa_start, sb_start, sc_start, sa_count, sb_count, sc_count = _scan_token_ranges(tokenizer)
    if sa_start is None or sb_start is None or sc_start is None:
        LOGGER.info(
            "vllm-gr: tokenizer has no <s_a_*>/<s_b_*>/<s_c_*> token groups; "
            "skipping tokenizer-based constraint tables"
        )
        return None

    LOGGER.info(
        "vllm-gr: tokenizer constraint ranges s_a=[%s..%s](%d) s_b=[%s..%s](%d) s_c=[%s..%s](%d)",
        sa_start,
        sa_start + sa_count - 1,
        sa_count,
        sb_start,
        sb_start + sb_count - 1,
        sb_count,
        sc_start,
        sc_start + sc_count - 1,
        sc_count,
    )

    del dtype  # tables are always int32/int64 (operator ABI)

    if prefix1_file is not None or prefix2_file is not None:
        if prefix1_file is None or prefix2_file is None:
            raise ValueError("sparse constraint tables require BOTH prefix1_file and prefix2_file")
        tables = _build_sparse(
            vocab_size=vocab_size,
            sa_start=sa_start,
            sb_start=sb_start,
            sc_start=sc_start,
            prefix1_file=prefix1_file,
            prefix2_file=prefix2_file,
        )
    else:
        if valid_item_ids is not None:
            valid_set = set(int(x) for x in valid_item_ids)
            valid_a_ids = sorted(n for n in valid_set if 0 <= n < sa_count)
            valid_b_ids = sorted(n for n in valid_set if 0 <= n < sb_count)
            valid_c_ids = sorted(n for n in valid_set if 0 <= n < sc_count)
        else:
            # A silently truncated universe is worse than no tables: the
            # constrained top-k would only ever emit low-numbered items that
            # real prompts never reference (observed as cross-sample duplicate
            # SIDs and pid=0 lookups).  Truncation must fail fast so the
            # operator notices and supplies the real item universe instead of
            # shipping a benchmark that cannot hit ground truth.
            counts = (sa_count, sb_count, sc_count)
            truncated = [n for n in counts if n > max_valid_items]
            if truncated:
                raise ValueError(
                    f"tokenizer constraint tables would be truncated: item "
                    f"groups s_a/s_b/s_c have {sa_count}/{sb_count}/{sc_count} "
                    f"items but max_valid_items={max_valid_items}. A truncated "
                    "universe drops most real SIDs (every request would emit the "
                    "same low-numbered candidates). Provide --constraint-data "
                    "built from the real catalog via "
                    "tools/constraints/build_constraint_triples.py, or pass "
                    "valid_item_ids/prefix files covering the full universe."
                )
            valid_a_ids = list(range(sa_count))
            valid_b_ids = list(range(sb_count))
            valid_c_ids = list(range(sc_count))
        tables = _build_fully_connected(
            vocab_size=vocab_size,
            sa_start=sa_start,
            sb_start=sb_start,
            sc_start=sc_start,
            valid_a_ids=valid_a_ids,
            valid_b_ids=valid_b_ids,
            valid_c_ids=valid_c_ids,
        )

    resource = ConstraintTableResource(
        first_token_ids=tables["first_token_ids"],
        prefix1_offsets=tables["prefix1_offsets"],
        prefix1_values=tables["prefix1_values"],
        prefix1_pair_keys=tables["prefix1_pair_keys"],
        prefix2_value_offsets=tables["prefix2_value_offsets"],
        prefix2_values=tables["prefix2_values"],
        max_prefix1_degree=tables["max_prefix1_degree"],
        max_prefix2_degree=tables["max_prefix2_degree"],
        vocab_size=vocab_size,
    )
    validate_resource_tables(resource)
    LOGGER.info(
        "vllm-gr: tokenizer constraint tables digest=%s "
        "total_elements=%d total_bytes=%d (items=%d degrees=%d/%d)",
        resource.digest,
        resource.total_elements,
        resource.total_bytes,
        len(tables["first_token_ids"]),
        resource.max_prefix1_degree,
        resource.max_prefix2_degree,
    )
    return resource


__all__ = ["build_constraint_tables_from_tokenizer"]
