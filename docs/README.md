# vLLM-GR documentation

<a id="vllm-gr-documentation"></a>

## About

vLLM-GR extends vLLM with inference for generative recommendation. Given a
recommendation prompt, a model generates ranked candidate item identifiers.
GR adds Beam execution, optional catalog constraints, and device-specific
execution optimizations while using vLLM for model loading and execution.

- Run recommendation inference offline or expose it through a serving API.
- Restrict generated identifiers to a prepared item catalog and select a
  catalog per request.
- Use the supported CUDA or Ascend NPU execution path for your workload.

See the [OneRec model guide](models/onerec.md) for inputs and outputs. Check
[features and compatibility](user_guide/feature_compatibility.md) for the
supported combinations and their limits.

## Getting started

Follow the [quickstart](getting_started/quickstart.md) to install for your
device and verify your first OneRec recommendation request.

## Find a guide

- [User guide](user_guide/README.md) for tasks such as deploying an API, using
  your item catalog, or checking model and device support.
- [Developer guide](contributing/README.md) for implementation, validation,
  and contributions.

vLLM-GR is distributed under the [Apache License, Version 2.0](../LICENSE).
