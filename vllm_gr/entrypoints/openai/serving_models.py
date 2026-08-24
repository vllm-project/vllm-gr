# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.models.serving import (
    BaseModelPath,
    LoRAModulePath,
    OpenAIServingModels,
)

from vllm_gr.config import GRConfig
from vllm_gr.logger import init_logger
from vllm_gr.utils.catalog import Catalog

logger = init_logger(__name__)

_original_init = OpenAIServingModels.__init__

# Catalog versions whose sequences were already transferred to each target
# engine rank (key None = single non-DP engine).  The engine-side catalog
# trie cache lives per EngineCore, so the transfer state must be scoped to
# the same granularity.
_engine_catalog_sent: dict[int | None, set[int]] = {}


def _catalog_transfer_payload(
    catalog: Catalog, rank: int | None
) -> tuple[int | None, list[list[int]] | None]:
    """Return ``(catalog_hash, sequences)`` for a request targeting ``rank``.

    The sequences are transferred only the first time this catalog version is
    seen by the target engine rank; later requests carry just the hash and
    the engine serves the cached trie.
    """
    catalog_hash = catalog._sequences_hash
    sent = _engine_catalog_sent.setdefault(rank, set())
    if catalog_hash in sent:
        return catalog_hash, None
    sequences = catalog.to_sequences()
    sent.add(catalog_hash)
    return catalog_hash, sequences


def _preload_engine_catalog(engine_client: EngineClient, catalog: Catalog) -> None:
    """Warm the engine-side catalog trie cache before any beam-search request.

    The trie build costs ~1s for large catalogs; running it in the engine
    during idle startup keeps it off the request critical path.  In DP mode
    every engine rank holds its own cache, so one preload message is sent per
    rank.  The first beam-search request per rank still carries the sequences
    as a fallback.
    """
    if not hasattr(engine_client, "beam_search_engine"):
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    vllm_config = getattr(engine_client, "vllm_config", None)
    data_parallel_size = None
    if vllm_config is not None and vllm_config.parallel_config is not None:
        data_parallel_size = vllm_config.parallel_config.data_parallel_size
    if data_parallel_size and data_parallel_size > 1:
        ranks: list[int | None] = list(range(data_parallel_size))
    else:
        ranks = [None]

    async def _send() -> None:
        from vllm.sampling_params import SamplingParams

        from vllm_gr.v1.engine.types import BeamSearchRequest

        for rank in ranks:
            try:
                logger.info(
                    "vllm_gr: sending engine catalog preload (hash=%s, rank=%s)",
                    catalog._sequences_hash,
                    rank,
                )
                await engine_client.beam_search_engine(
                    BeamSearchRequest(
                        session_id=f"_catalog_preload_{rank}",
                        prompt_token_ids=[],
                        beam_width=1,
                        max_tokens=1,
                        sampling_params=SamplingParams(),
                        catalog_hash=catalog._sequences_hash,
                        catalog_token_sequences=catalog.to_sequences(),
                        preload_only=True,
                        data_parallel_rank=rank,
                    )
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("vllm_gr: engine catalog preload failed (rank=%s): %s", rank, e)
            else:
                logger.info("vllm_gr: engine catalog preload delivered (rank=%s)", rank)
                # NOTE: the preload warms the engine-side cache but does NOT
                # mark the transfer as done — the send is not an
                # acknowledgment that the engine finished building the trie.
                # The first beam-search request per rank carries the sequences
                # (and marks the set), so no request can ever run with a
                # missing catalog.

    loop.create_task(_send())


def init_gr(
    self,
    engine_client: EngineClient,
    base_model_paths: list[BaseModelPath],
    *,
    lora_modules: list[LoRAModulePath] | None = None,
):
    _original_init(self, engine_client, base_model_paths, lora_modules=lora_modules)
    self.catalog = None
    tokenizer = self.input_processor.tokenizer
    vllm_config = getattr(engine_client, "vllm_config", None)
    catalog_path = GRConfig.from_vllm_config(vllm_config).catalog_path
    if catalog_path:
        self.catalog = Catalog(tokenizer)
        self.catalog.load(catalog_path)
        _preload_engine_catalog(engine_client, self.catalog)


def patch_OpenAIServingModels_init():
    OpenAIServingModels.__init__ = init_gr
