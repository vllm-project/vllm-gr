# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

import contextvars

BEAM_DATA_VAR = contextvars.ContextVar("beam_data", default=None)
