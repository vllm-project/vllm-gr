# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.llm import LLM

from vllm_gr.entrypoints.llm import beam_search
from vllm_gr.entrypoints.openai.serving_models import Catalog


def make_patched_llm_init(original_init):
    def patched_init(self, *args, **kwargs):
        catalog_path = kwargs.pop("catalog_path", None)
        original_init(self, *args, **kwargs)
        self.catalog = None
        if catalog_path is not None:
            self.catalog = Catalog(self.get_tokenizer())
            self.catalog.load(catalog_path)

    return patched_init


def patch_offline_beam_search():
    LLM.beam_search = beam_search
    if not getattr(LLM.__init__, "_is_patched", False):
        LLM.__init__ = make_patched_llm_init(LLM.__init__)
        LLM.__init__._is_patched = True
