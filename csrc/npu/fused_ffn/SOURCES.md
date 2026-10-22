# Source provenance and retained notices

This directory adapts the user-provided `15.diff.txt` attachment. Its SHA-256 is:

```text
2997385bf884f738fcd6d33a66f99e3b079f289838cacc00d8041976017b4393
```

The attachment contains the `custom_ops` tree listed below. This identifies
the supplied source artifact, rather than asserting that it is a particular
public upstream revision. The operator was extracted, renamed, specialized,
and integrated locally; the attachment itself is not needed to build this
directory.

## Original paths

Paths in the right column are paths inside the attachment. Paths in the left
column are relative to this directory.

| Distributed path | Original attachment path |
| --- | --- |
| `LICENSE` | `custom_ops/LICENSE` |
| `LICENSE.CANN` | `custom_ops/LICENSE.CANN` |
| `CMakeLists.txt` | `custom_ops/CMakeLists.txt` |
| `cmake/config.cmake` | `custom_ops/cmake/config.cmake` |
| `cmake/func.cmake` | `custom_ops/cmake/func.cmake` |
| `cmake/intf.cmake` | `custom_ops/cmake/intf.cmake` |
| `cmake/intf_pub.cmake` | `custom_ops/cmake/intf_pub.cmake` |
| `cmake/scripts/prepare.sh` | `custom_ops/cmake/scripts/prepare.sh` |
| `cmake/util/gen_ops_filter.sh` | `custom_ops/cmake/util/gen_ops_filter.sh` |
| `ops/gr_gate_up_swi_glu/CMakeLists.txt` | `custom_ops/ops/pangu_gate_up_swi_glu/CMakeLists.txt` |
| `ops/gr_gate_up_swi_glu/op_host/host.h` | `custom_ops/ops/pangu_gate_up_swi_glu/op_host/host.h` |
| `ops/gr_gate_up_swi_glu/op_host/gr_gate_up_swi_glu_cpu.cpp` | `custom_ops/ops/pangu_gate_up_swi_glu/op_host/pangu_gate_up_swi_glu_cpu.cpp` |
| `ops/gr_gate_up_swi_glu/op_host/gr_gate_up_swi_glu_tiling.h` | `custom_ops/ops/pangu_gate_up_swi_glu/op_host/pangu_gate_up_swi_glu_tiling.h` |
| `ops/gr_gate_up_swi_glu/op_kernel/gate_up_swi_glu.h` | `custom_ops/ops/pangu_gate_up_swi_glu/op_kernel/gate_up_swi_glu.h` |
| `ops/gr_gate_up_swi_glu/op_kernel/matmul.h` | `custom_ops/ops/pangu_gate_up_swi_glu/op_kernel/matmul.h` |
| `ops/gr_gate_up_swi_glu/op_kernel/gr_gate_up_swi_glu.cpp` | `custom_ops/ops/pangu_gate_up_swi_glu/op_kernel/pangu_gate_up_swi_glu.cpp` |
| `ops/gr_gate_up_swi_glu/op_kernel/small_gate_up.h` | `custom_ops/ops/pangu_gate_up_swi_glu/op_kernel/small_gate_up.h` |
| `ops/gr_gate_up_swi_glu/pytorch/pytorch.h` | `custom_ops/ops/pangu_gate_up_swi_glu/pytorch/pytorch.h` |
| `ops/gr_gate_up_swi_glu/pytorch/pytorch_npu_helper.hpp` | `custom_ops/ops/pangu_gate_up_swi_glu/pytorch/pytorch_npu_helper.hpp` |
| `ops/gr_gate_up_swi_glu/pytorch/bindings.cpp` | `custom_ops/ops/pangu_gate_up_swi_glu/pytorch/bindings.cpp` |

The standalone `build.sh`, `set_env.sh`, `setup.py`, `.gitignore`, README,
and this provenance document were added for this integration. Native C++
and header formatting was normalized with the repository's clang-format
configuration. The supplied files retain their notices and kernel design,
with the specialization and integration changes described below. This table
is not a claim that the files remain byte-identical to the attachment.

## Notices and license files

- [LICENSE](LICENSE) is the supplied Apache License 2.0 text with the
  `Copyright 2025 JD.com` notice. Its contents and notice are retained.
- [LICENSE.CANN](LICENSE.CANN) is the supplied **CANN Open Software License
  Agreement Version 2.0** text. The Huawei 2025 headers in
  `gate_up_swi_glu.h` and `matmul.h` identify that version. Those headers
  remain in their source files.
- The inherited top-level CMake/build-template files instead identify
  **CANN Open Software License Agreement Version 1.0** in their Huawei 2024
  headers. The attachment already combined those Version 1.0 headers with
  the Version 2.0 `LICENSE.CANN` text. This integration preserves that
  distinction; it does not rewrite those headers to Version 2.0 or claim
  that the supplied Version 2.0 text replaces their stated Version 1.0
  terms.
- `pytorch_npu_helper.hpp` retains `Copyright 2025 The xLLM Authors. All
  Rights Reserved.` and its Apache 2.0 header, including the original
  [xLLM license reference](https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE).
- `cmake/util/gen_ops_filter.sh` retains its Huawei 2025 copyright and
  Apache 2.0 notice. Other file-level notices, including SPDX identifiers,
  are retained in their respective source files.

These notices describe the provenance and terms identified by the supplied
files. They do not relicense the CANN sources under the repository's Apache
license. Installed CANN headers, compilers, templates, and generated binaries
are not vendored as source in this directory.

## Integration changes

The operator and vendor names were changed to `GrGateUpSwiGlu` and `gr_ffn`
to keep this build separate from other custom-op packages. Host checks and
kernel geometry specialize the inputs to H=1024 and I=3072. The Gate/Up
weight layout and separate native Down projection are preserved.

The Python extension loads its dedicated OPAPI library by an explicit
absolute path. Its helper uses a private namespace and hidden symbols to
avoid sharing operator-resolution state with another custom-op extension.
The standalone build resolves torch-npu's bundled ACL headers before CANN
headers and links the CANN tiling dependencies required by the tested SDK.
The build entry point and inherited preparation script propagate failures
and select the same Unix Makefiles generator for both build stages.
Inherited build comments were translated into English, and shell variables
were quoted for the repository's lint checks. The unused template version
checker is not distributed: this package has no repository `version.info`,
keeps `CHECK_COMPATIBLE=OFF`, and rejects attempts to enable that check.

The opt-in model adapter, runtime eligibility/fallback policy, compilation
cache marker, and worker hook live in the repository's Python sources.
They are integration code rather than part of the original operator source
tree. Diagnostic kernels, local benchmark artifacts, and generated build
outputs are not part of this source distribution.
