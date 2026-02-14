# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm_gr.entrypoints.openai.serving_engine import beam_search
from vllm_gr.entrypoints.openai.protocol import to_beam_search_params


def patch_beam_search():
    OpenAIServing.beam_search = beam_search


def patch_sampling():
    ChatCompletionRequest.to_beam_search_params = to_beam_search_params


def patch_engine_core_beam_search():
    """Inject engine-core beam search into stock vllm via monkey-patching."""
    import vllm.v1.engine as engine_mod
    from vllm.v1.engine import EngineCoreRequestType
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.engine.core_client import AsyncMPClient
    from vllm.v1.engine.async_llm import AsyncLLM

    from vllm_gr.v1.engine.types import (
        BeamSearchRequest,
        BeamSearchResult,
        BeamSearchSequenceResult,
        PatchedEngineCoreOutputs,
    )
    from vllm_gr.v1.engine.engine_core_patch import (
        make_patched_engine_core_init,
        patched_handle_client_request,
        patched_process_engine_step,
        patched_process_input_sockets,
    )
    from vllm_gr.v1.engine.core_client_patch import (
        beam_search_async,
        make_patched_async_mp_init,
        patched_ensure_output_queue_task,
    )
    from vllm_gr.v1.engine.async_llm_patch import engine_beam_search

    # 1. Inject beam search types into vllm.v1.engine module
    engine_mod.BeamSearchRequest = BeamSearchRequest
    engine_mod.BeamSearchSequenceResult = BeamSearchSequenceResult
    engine_mod.BeamSearchResult = BeamSearchResult

    # 2. Add BEAM_SEARCH to EngineCoreRequestType enum
    beam_val = b"\x06"
    beam_member = object.__new__(EngineCoreRequestType)
    beam_member._name_ = "BEAM_SEARCH"
    beam_member._value_ = beam_val
    EngineCoreRequestType._value2member_map_[beam_val] = beam_member
    EngineCoreRequestType._member_map_["BEAM_SEARCH"] = beam_member
    EngineCoreRequestType._member_names_.append("BEAM_SEARCH")
    EngineCoreRequestType.BEAM_SEARCH = beam_member

    # 3. Replace EngineCoreOutputs with patched version (adds
    #    beam_search_result field).  Also update the module-level name so
    #    that any ``from vllm.v1.engine import EngineCoreOutputs`` done
    #    *after* patching picks up the new class.
    engine_mod.EngineCoreOutputs = PatchedEngineCoreOutputs

    # 4. Wrap EngineCore.__init__ to add active_beam_searches
    EngineCore.__init__ = make_patched_engine_core_init(EngineCore.__init__)

    # 5. Replace EngineCore methods
    EngineCore._process_engine_step = patched_process_engine_step
    EngineCore._handle_client_request = patched_handle_client_request

    # 6. Replace EngineCoreProc.process_input_sockets
    from vllm.v1.engine.core import EngineCoreProc
    EngineCoreProc.process_input_sockets = patched_process_input_sockets

    # 7. Wrap AsyncMPClient.__init__, replace _ensure_output_queue_task,
    #    add beam_search_async
    AsyncMPClient.__init__ = make_patched_async_mp_init(AsyncMPClient.__init__)
    AsyncMPClient._ensure_output_queue_task = patched_ensure_output_queue_task
    AsyncMPClient.beam_search_async = beam_search_async

    # 8. Add AsyncLLM.engine_beam_search
    AsyncLLM.engine_beam_search = engine_beam_search

    # 9. Update the MsgpackDecoder type used in AsyncMPClient to decode
    #    PatchedEngineCoreOutputs instead of the original EngineCoreOutputs.
    #    The decoder is created in MPClient.__init__ as self.decoder, so we
    #    need to update it *after* the client is constructed.  We do this by
    #    further wrapping AsyncMPClient.__init__.
    from vllm.v1.serial_utils import MsgpackDecoder

    _prev_init = AsyncMPClient.__init__

    def _init_with_decoder_fix(self, *args, **kwargs):
        _prev_init(self, *args, **kwargs)
        # Replace the decoder with one that knows about the patched type.
        self.decoder = MsgpackDecoder(PatchedEngineCoreOutputs)

    AsyncMPClient.__init__ = _init_with_decoder_fix
