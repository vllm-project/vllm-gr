# vllm serve

`vllm serve ... --gr` starts a GR server. Use HTTP for
[`POST /v1/completions`](../serving/completions.md) or add `--grpc` for
[tokenized gRPC requests](../serving/grpc.md).

## Usage

Enable [GR CLI dispatch](README.md#enable-gr-dispatch), then follow
[online serving](../serving/README.md) for model, device, startup, and readiness
steps. The command form is:

```text
vllm serve MODEL --gr --config ENGINE_YAML [--grpc] [OPTIONS]
```

## Arguments

The model may be positional or supplied with `--model`. Use
`vllm serve --gr --help=all` for the installed vLLM options, including
authentication, TLS, tokenizer, and scheduling.

| Option | Form and purpose |
| --- | --- |
| `--gr` | Boolean switch selecting the GR command |
| `--grpc` | Start the gRPC server |
| `--config` | Read native and GR startup options from YAML |
| `--gr.FIELD` | Override a GR field |
| `--gr-config` | Pass GR options as a JSON object |
| `--host`, `--port` | Listening address and integer port |
| `--served-model-name` | Model name exposed to clients |

See [Beam scheduling requirements](../configuration/README.md#asynchronous-scheduling)
before overriding native scheduler options.

## Configuration

The [configuration guide](../configuration/README.md#load-and-override-a-configuration)
owns YAML layout, precedence, and CLI/Python override examples. Use
`--gr.FIELD` for one GR option or `--gr-config` for a JSON object.
