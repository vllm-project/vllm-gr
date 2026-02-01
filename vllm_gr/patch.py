from vllm_gr.sampling_params import BeamSearchParams
from vllm_gr.entrypoints.openai.beam_search_patch import (
    patch_beam_search,
    patch_sampeling
)

def run_patch():
    patch_beam_search()
    patch_sampeling()

    