import os
from setuptools import setup, find_packages, Extension
import pybind11


def get_ext_modules():
    # The C++ extension source is expected to be in vllm_gr/csrc/
    # This is based on the import of `minheap_cpp` in the project.
    csrc_dir = os.path.join("vllm_gr", "csrc")
    minheap_cpp_source = os.path.join(csrc_dir, "minheap.cpp")

    if not os.path.exists(minheap_cpp_source):
        # If there's no C++ source, we assume no extensions are to be built.
        # This will cause a runtime error if minheap_cpp is imported,
        # but it allows installation to proceed if the C++ part is optional
        # or located elsewhere.
        print(
            f"Warning: C++ extension source '{minheap_cpp_source}' not found. "
            "Skipping C++ extension build."
        )
        return []

    return [
        Extension(
            "minheap_cpp",
            [minheap_cpp_source],
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=["-std=c++17", "-O3", "-march=native"],
        )
    ]


setup(
    name="vllm_gr",
    version="0.1.0",
    packages=find_packages(),
    # This builds the extension and installs it into the package/environment
    ext_modules=get_ext_modules(),
    install_requires=[
        "vllm",
        "pybind11",
    ],
    extras_require={
        "test": ["pytest-asyncio"],
    },
    zip_safe=False,
)
