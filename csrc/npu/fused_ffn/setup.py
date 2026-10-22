# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the optional GrGateUpSwiGlu Python extension in place."""

import os
from pathlib import Path

import torch_npu
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch_npu.utils.cpp_extension import NpuExtension

root = Path(__file__).resolve().parent
cann = Path(os.environ["ASCEND_HOME_PATH"]).resolve()
npu = Path(torch_npu.__file__).resolve().parent
extension = NpuExtension(
    name="gr_ffn_ops",
    sources=[str(root / "ops/gr_gate_up_swi_glu/pytorch/bindings.cpp")],
    depends=[
        str(root / "ops/gr_gate_up_swi_glu/pytorch/pytorch.h"),
        str(root / "ops/gr_gate_up_swi_glu/pytorch/pytorch_npu_helper.hpp"),
    ],
    # CalcuOpUtil/AclInterface include torch-npu's bundled ACL headers directly.
    # Resolve our <acl/...> includes to that same version before CANN headers;
    # otherwise CANN 9 and torch-npu can redeclare RI capture types/functions.
    include_dirs=[str(npu / "include/third_party/acl/inc"), str(cann / "include")],
    extra_compile_args=["-O2", "-fvisibility=hidden"],
    libraries=["dl"],
)
setup(
    name="gr_ffn_ops",
    version="0.0.1",
    ext_modules=[extension],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
