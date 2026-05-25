# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Lightweight multitask type definitions shared by config and model code."""

from enum import IntEnum


class MultitaskTaskType(IntEnum):
    BINARY_CLASSIFICATION = 0
    REGRESSION = 1
