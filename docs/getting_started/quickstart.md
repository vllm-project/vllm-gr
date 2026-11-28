# Quickstart

Run a first offline recommendation request with OneRec, then check the result.
For the framework's supported features and device limits, see
[Features and compatibility](../user_guide/feature_compatibility.md). The
[User guide](../user_guide/README.md) links to configuration, serving, catalog
constraints, live updates, and evaluation.

<a id="choose-a-model-and-install"></a>

## Install for your device

Follow the [shared installation steps](installation.md#shared-steps) to
prepare a source checkout, then complete the [CUDA](installation/gpu.md)
or [Ascend NPU](installation/npu.md) installation and environment check.
Keep that environment active and run the remaining commands from the
repository root.

## Configuration used in this quickstart

| Configuration source | How this quickstart uses it | Detailed explanation |
| --- | --- | --- |
| [Packaged defaults](../../vllm_gr/config/default_config.yaml) | Base configuration loaded automatically by `GRLLM` | [Loading and overriding configuration](../configuration/README.md#load-and-override-a-configuration) |
| [OneRec example](../../examples/offline_inference/beam_search/offline_beam_search.py) | Builds `gr` settings from example arguments, on top of the packaged defaults | [Example options](../../examples/offline_inference/beam_search/README.md) and [Beam settings](../configuration/README.md#beam-settings) |

Use the commands below as written for the first run. For your own application,
create a separate startup YAML and pass it with `GRLLM(config=...)`; follow
the configuration guide above for field names and overrides.

<a id="offline-inference"></a>

## OneRec

### Download OneRec

If you have a local OneRec-1.7B snapshot, set `ONEREC_MODEL` to its directory.
Otherwise, install the
[Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
and download the checkpoint.

```bash
export ONEREC_MODEL="$HOME/.cache/vllm-gr/models/OneRec-1.7B"
hf download OpenOneRec/OneRec-1.7B --local-dir "$ONEREC_MODEL"
```

The commands below use the example's built-in recommendation prompt, one
request, Beam width 128, and three generation steps. No catalog is needed
for this first run. Configure the example through its command-line arguments;
for example, save your prompt in a UTF-8 file named `prompt.txt` and add
`--prompt ./prompt.txt` to replace the built-in prompt. See the
[example options](../../examples/offline_inference/beam_search/README.md)
before changing Beam widths or execution settings.

<a id="cuda"></a>

### Run OneRec on CUDA

This command uses the default [ModelRunner V2](../configuration/README.md#modelrunner-selection).

```bash
python examples/offline_inference/beam_search/quickstart.py --model "$ONEREC_MODEL"
```

<a id="ascend"></a>

### Run OneRec on Ascend NPU

The example selects [ModelRunner V1](../configuration/README.md#modelrunner-selection)
and runs in eager mode.

```bash
python examples/offline_inference/beam_search/quickstart.py \
  --model "$ONEREC_MODEL" --constraint-backend npu --execution eager
```

### Expected output

```text
request=0 returned_beams=128
```

The script prints the top five candidates with their token IDs, tokens, and
cumulative log probabilities. A successful run returns 128 beams and exits
without an error. These unconstrained candidates demonstrate inference;
to restrict recommendations to your own items, follow
[constraint tables](../configuration/constraint_tables.md).

## Online serving

To serve OneRec behind an API, see [online serving](../serving/README.md).
