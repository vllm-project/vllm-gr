# Generated gRPC bindings

The Python modules in this directory are generated from the `.proto` files in
`../proto/`. Do not edit them by hand.

## Regenerating

After changing `../proto/common.proto` or `../proto/vllm_engine.proto`:

```bash
pip install grpcio-tools==1.84.0   # also included in the `dev` extra
python tools/generate_grpc_proto.py
```

`grpcio-tools` is pinned because the bundled protoc version is written into the
output; another release produces a diff even when the schema is unchanged.

The script also rewrites the protoc output so that:

- sibling imports are package-relative (`from . import common_pb2`);
- descriptors are registered in a private `DescriptorPool`, so they do not
  collide with the unmodified `smg-grpc-proto` descriptors when an application
  imports both.
- message classes are built from private descriptors without registering them
  in protobuf's default symbol database, including on the pure-Python backend.

## Drift check

`python tools/generate_grpc_proto.py --check` regenerates into a temporary
directory and fails with a diff if the committed files differ. It runs as the
`check-grpc-generated` pre-commit hook, which CI executes with `--all-files`.

## Provenance

The `.proto` sources are extended copies of
[`smg-grpc-proto`](https://github.com/lightseekorg/smg) 0.4.6 (Apache-2.0); the
extensions are listed at the top of `../proto/vllm_engine.proto`.
