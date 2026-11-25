# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Scoped GR bindings for the upstream vLLM gRPC server lifecycle."""

from contextlib import ExitStack, contextmanager
from types import SimpleNamespace

from google.protobuf import descriptor_pool

from vllm_gr.entrypoints.beam_search.client import drain_async_cleanup

from .generated import vllm_engine_pb2 as pb
from .generated import vllm_engine_pb2_grpc as rpc
from .serving import BeamVllmEngineServicer


@contextmanager
def patch_grpc_server(*, admin_host: str = "127.0.0.1", admin_port: int | None = None):
    """Replace protocol/servicer references only while upstream serving runs.

    The CLI owns one server per process. Upstream retains engine construction,
    listener options, health, statistics, signals and shutdown ordering.
    """
    from grpc_health.v1 import health_pb2
    from grpc_reflection.v1alpha import reflection, reflection_pb2
    from vllm.entrypoints.launchers import grpc_server as upstream

    # Reflection must describe our extended schema, not the original schema
    # imported into the default pool by the upstream runtime dependencies.
    pool = descriptor_pool.DescriptorPool()

    def add_file(file):
        for dependency in file.dependencies:
            add_file(dependency)
        pool.AddSerializedFile(file.serialized_pb)

    for file in (pb.DESCRIPTOR, health_pb2.DESCRIPTOR, reflection_pb2.DESCRIPTOR):
        add_file(file)

    with ExitStack() as stack:

        def replace(target, name, value):
            original = getattr(target, name)
            stack.callback(setattr, target, name, original)
            setattr(target, name, value)

        def register(servicer, server):
            stop = server.stop
            admin = None
            if admin_port is not None:
                from vllm_gr.entrypoints.serve.management import AdminServer

                admin = AdminServer(
                    servicer.engine,
                    admin_port,
                    admin_host,
                )
                start = server.start

                async def start_with_control():
                    try:
                        await admin.start()
                        await start()
                    except BaseException:
                        await admin.stop()
                        raise

                replace(server, "start", start_with_control)

            async def stop_and_drain(grace):
                try:
                    return await stop(grace)
                finally:
                    if admin is not None:
                        await admin.stop()
                    # Request cancellation may leave retrying cleanup tasks.
                    # Finish them before upstream calls AsyncLLM.shutdown().
                    await drain_async_cleanup()

            replace(server, "stop", stop_and_drain)
            rpc.add_VllmEngineServicer_to_server(servicer, server)

        def enable_reflection(service_names, server):
            reflection.enable_server_reflection(service_names, server, pool=pool)

        replace(upstream, "vllm_engine_pb2", pb)
        replace(
            upstream,
            "vllm_engine_pb2_grpc",
            SimpleNamespace(
                add_VllmEngineServicer_to_server=register,
            ),
        )
        replace(upstream, "VllmEngineServicer", BeamVllmEngineServicer)
        replace(
            upstream,
            "reflection",
            SimpleNamespace(
                SERVICE_NAME=reflection.SERVICE_NAME,
                enable_server_reflection=enable_reflection,
            ),
        )
        yield upstream
