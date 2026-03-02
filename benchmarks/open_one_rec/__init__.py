# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
try:
    from . import one_rec_patch

    print("OpenOneRec patch applied.")
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    one_rec_patch = None  # type: ignore
    print("OpenOneRec patch NOT applied.")
