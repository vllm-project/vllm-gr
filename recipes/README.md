# Deployment Recipes

Select a model's YAML with `vllm serve --gr --config` or `GRLLM(config=...)`.
Each recipe overrides the
[common defaults](../vllm_gr/config/default_config.yaml) loaded by vLLM-GR.

- [Pangu Embedded 1B](Pangu/pangu1b.yaml) uses native vLLM sampling and
  attention. Follow the [launch and verification steps](../docs/models/pangu.md#online-serving)
  and pass `--block-size 128` on Ascend.

If you installed a wheel, open `recipes/Pangu/pangu1b.yaml` on GitHub at the
release tag or commit used to build your wheel. Select **Download raw file**
and pass the saved file path to `--config` in the launch command.
