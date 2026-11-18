# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Route asynchronous D2H/conversion exceptions through native Executor failures."""

from functools import wraps


def install_async_output_guard() -> None:
    from vllm.v1.executor.multiproc_executor import WorkerProc
    from vllm.v1.outputs import AsyncModelRunnerOutput

    original = WorkerProc.enqueue_output
    if getattr(original, "_gr_async_output_guard", False):
        return

    @wraps(original)
    def enqueue_output(self, output):
        if isinstance(output, AsyncModelRunnerOutput):
            try:
                output = output.get_output()
            except Exception as error:
                # enqueue_output already serializes exceptions as FAILURE.
                # Keeping conversion inside this guard prevents a dead output
                # thread and a Future that can never become ready.
                output = error
        return original(self, output)

    enqueue_output._gr_async_output_guard = True
    WorkerProc.enqueue_output = enqueue_output
