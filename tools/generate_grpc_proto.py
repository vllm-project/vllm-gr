# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# Adapted from https://github.com/vllm-project/vllm
"""Generate the extended SMG bindings with package-relative imports/private pools.

Usage:
    python tools/generate_grpc_proto.py           # rewrite generated/ in place
    python tools/generate_grpc_proto.py --check   # fail if generated/ is stale

Requires ``grpcio-tools==1.84.0`` (declared in the ``dev`` extra). The bundled
protoc version is part of the output, so a different grpcio-tools release can
produce a spurious diff.
"""

import argparse
import difflib
import subprocess
import sys
import tempfile
from pathlib import Path

GRPC_ROOT = Path(__file__).resolve().parents[1] / "vllm_gr/entrypoints/grpc"
GENERATED_FILES = ("common_pb2.py", "vllm_engine_pb2.py", "vllm_engine_pb2_grpc.py")


def proto_notice(proto: Path) -> str:
    """Return the schema's leading comment block as Python comments.

    The .proto header is the single source of truth for SPDX and attribution
    lines; they are copied verbatim, so a binding is never attributed
    differently from the schema it was generated from.

    Scanning steps over blank lines instead of stopping at one, so a blank line
    inside the header cannot truncate the notice without anyone noticing. The
    block ends at the first line that is neither a comment nor blank; trailing
    blanks are dropped, and a file with no comment block yields no notice.
    """
    lines: list[str] = []
    for line in proto.read_text(encoding="utf-8").splitlines():
        if line.startswith("//"):
            lines.append("#" + line[2:])
        elif line.strip():
            break
        else:
            lines.append("#")
    while lines and lines[-1] == "#":
        lines.pop()
    return "".join(line + "\n" for line in lines)


def generate(output: Path) -> None:
    proto_dir = GRPC_ROOT / "proto"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{proto_dir}",
            f"--python_out={output}",
            f"--grpc_python_out={output}",
            str(proto_dir / "common.proto"),
            str(proto_dir / "vllm_engine.proto"),
        ],
        check=True,
    )
    # common.proto defines messages only; protoc emits no service bindings here.
    (output / "common_pb2_grpc.py").unlink(missing_ok=True)
    for path in output.glob("*_pb2*.py"):
        text = path.read_text(encoding="utf-8")
        for module in ("common_pb2", "vllm_engine_pb2"):
            text = text.replace(f"import {module} as ", f"from . import {module} as ")
        if path.name.endswith("_pb2.py"):
            # Preserve the original package/service names without colliding with
            # the unmodified SMG descriptor in applications importing both APIs.
            pool = "_descriptor_pool.DescriptorPool()"
            if path.name == "vllm_engine_pb2.py":
                text = text.replace(
                    "DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(",
                    "_pool = " + pool + "\n"
                    "_pool.AddSerializedFile(common__pb2.DESCRIPTOR.serialized_pb)\n"
                    "DESCRIPTOR = _pool.AddSerializedFile(",
                )
            else:
                text = text.replace("_descriptor_pool.Default()", pool)
            # The stock message builder registers classes in the default
            # symbol database, which leaks private descriptors in pure Python.
            text = text.replace(
                "from google.protobuf.internal import builder as _builder\n",
                "from google.protobuf.internal import builder as _builder\n"
                "from .. import _protobuf\n",
            )
            text = text.replace(
                "_builder.BuildTopDescriptorsAndMessages(",
                "_protobuf.build_top_descriptors_and_messages(",
            )
            module = "vllm_gr.entrypoints.grpc.generated." + path.stem
            text = text.replace(f"'{path.stem}'", repr(module))
        schema = path.name.split("_pb2")[0] + ".proto"
        notice = proto_notice(proto_dir / schema)
        first, newline, rest = text.partition("\n")
        if "coding:" in first:
            text = first + newline + notice + rest
        else:
            text = notice + text
        path.write_text(text, encoding="utf-8", newline="\n")


def check() -> int:
    """Regenerate into a scratch directory and diff against the committed copy."""
    committed = GRPC_ROOT / "generated"
    with tempfile.TemporaryDirectory() as scratch:
        fresh = Path(scratch)
        generate(fresh)
        produced = sorted(path.name for path in fresh.glob("*.py"))
        if produced != sorted(GENERATED_FILES):
            print(f"unexpected generator output: {produced}", file=sys.stderr)
            return 1
        stale = []
        for name in GENERATED_FILES:
            expected = (fresh / name).read_text(encoding="utf-8").splitlines(keepends=True)
            current_path = committed / name
            current = (
                current_path.read_text(encoding="utf-8").splitlines(keepends=True)
                if current_path.exists()
                else []
            )
            if expected != current:
                stale.append(name)
                sys.stderr.writelines(
                    difflib.unified_diff(
                        current, expected, f"committed/{name}", f"regenerated/{name}", n=1
                    )
                )
    if stale:
        print(
            f"\nStale gRPC bindings: {', '.join(stale)}.\n"
            "Run `python tools/generate_grpc_proto.py` and commit the result.",
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify generated/ matches the .proto sources without modifying it",
    )
    args = parser.parse_args()
    if args.check:
        return check()
    generate(GRPC_ROOT / "generated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
