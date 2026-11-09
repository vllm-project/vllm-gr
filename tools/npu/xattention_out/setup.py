# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the additive XAttention binding without installing packages."""

import os
from pathlib import Path
from typing import Any, cast

import torch_npu
from build import runtime_versions, validate_sources
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch_npu.utils.cpp_extension import NpuExtension

root = Path(__file__).resolve().parent
source_dir = Path(os.environ["VLLM_GR_XATTENTION_SOURCE_DIR"]).resolve()
validate_sources(source_dir)
runtime_versions()
npu_root = Path(torch_npu.__file__).resolve().parent
npu_lib = npu_root / "lib"

setup(
    name="vllm_gr_xattention_out_ext",
    version="0.0.1",
    ext_modules=[
        NpuExtension(
            name="vllm_gr_xattention_out_ext",
            sources=[str(root / "xattention_out.cpp")],
            include_dirs=[
                str(source_dir / "test" / "python_test"),
                str(npu_root / "include/third_party/acl/inc"),
            ],
            library_dirs=[str(npu_lib)],
            libraries=["dl"],
            extra_compile_args=["-O2", "-fvisibility=hidden"],
            extra_link_args=[f"-Wl,-rpath,{npu_lib}"],
        ),
    ],
    cmdclass=cast(
        Any,
        {"build_ext": BuildExtension.with_options(use_ninja=os.getenv("USE_NINJA") == "1")},
    ),
)
