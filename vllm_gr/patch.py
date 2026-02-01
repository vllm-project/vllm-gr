from vllm_gr.entrypoints.openai.beam_search_patch import patch_beam_search
from vllm_gr.entrypoints.openai.beam_search_patch import patch_sampling


def run_patch():
    patch_beam_search()
    patch_sampling()
