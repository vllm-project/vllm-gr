# Security Policy

## Reporting security issues

Do not publish vulnerability details in issues, pull requests, or other public
channels. A private reporting channel for vLLM-GR has not yet been confirmed.
If you already have a private contact with a project maintainer, use it to
coordinate a report.

## Deployment security

vLLM-GR uses the vLLM server and inherits its security boundaries. Review the
[vLLM security guide](https://docs.vllm.ai/en/latest/usage/security.html) for
network isolation and deployment recommendations. Check the installed vLLM
version and the routes enabled in your deployment.

The HTTP API key is optional and does not protect every route. The gRPC server
in the supported vLLM runtime uses an insecure listener without built-in
authentication or authorization. Restrict access to both services and their
internal ports; do not expose them directly to untrusted networks. See the
[HTTP example](examples/online_serving/beam_search/README.md#start-the-server)
and [gRPC guide](docs/serving/README.md#usage-and-tests).

Only enable `--trust-remote-code` for trusted model sources. Treat plugins,
model files, and custom templates as trusted deployment inputs.
