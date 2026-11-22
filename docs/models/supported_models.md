# Supported models

vLLM-GR runs OneRec recommendation workloads and provides compatibility for
ordinary text generation with openPangu. Choose the checkpoint and execution
path for your task.

## Model list

| Checkpoint | Task and output | CUDA | Ascend NPU |
| --- | --- | --- | --- |
| [OneRec-1.7B](https://huggingface.co/OpenOneRec/OneRec-1.7B) | Full-vocabulary Beam generation, ranked token sequences | [CUDA quickstart](../getting_started/quickstart.md#cuda) | [Ascend quickstart](../getting_started/quickstart.md#ascend) |
| OneRec-1.7B with a constraint table | Catalog-constrained recommendation, ranked semantic IDs (SIDs) | [Offline and HTTP examples](../examples/README.md) | [Ascend Beam guide](../user_guide/beam_search_npu.md) |
| [openPangu-Embedded-1B-V1.1](https://modelscope.cn/models/FreedomIntelligence/openPangu-Embedded-1B-V1.1) | Native text generation, chat and GSM8K-mini evaluation | [Model guide](pangu.md) | [Model guide and block size](pangu.md#online-serving) |

## Implementation and compatibility

The offline OneRec entry point is `GRLLM.beam_search()`. It uses the GR Beam
worker and scheduler. The [feature matrix](../user_guide/feature_compatibility.md)
covers differences in graph execution, candidate widths and constraint tables
across devices. openPangu uses native vLLM sampling and the
[GR compatibility fix](../../vllm_gr/config/pangu_patch.py) for tied embeddings.

!!! note "openPangu SID example"

    The [SID example](../examples/pangu_sid.md) adds untrained SID embeddings
    to public weights for a catalog-constrained Beam smoke test. Use the
    original public weights for text generation and evaluation.
