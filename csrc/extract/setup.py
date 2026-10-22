"""
Build the extract_events_cuda extension.

Usage:
    pip install -e .          # editable / in-place build
    python setup.py build_ext --inplace   # alternative

Requirements:
    - CUDA toolkit ≥ 11.x  (for CUB headers bundled with CUDA)
    - PyTorch ≥ 2.0 built against the same CUDA version
    - A C++17-capable compiler (GCC ≥ 9 or Clang ≥ 10)
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# CUB is shipped inside the CUDA toolkit since CUDA 11.
# If you are on an older toolkit, pass the path to a standalone CUB install:
#   extra_include_dirs=["/path/to/cub"]
cub_include = []   # leave empty for CUDA ≥ 11

setup(
    name="extract_events_cuda",
    version="1.0.0",
    description="Fused CUDA kernel for spike event stream compaction",
    ext_modules=[
        CUDAExtension(
            name="extract_events_cuda",
            sources=["extract_events_cuda.cu"],
            include_dirs=cub_include,
            extra_compile_args={
                "cxx":  ["-O3", "-std=c++17"],
                # PyTorch targets the visible GPU by default. Set
                # TORCH_CUDA_ARCH_LIST explicitly for cross-compilation.
                "nvcc": ["-O3", "-std=c++17", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
