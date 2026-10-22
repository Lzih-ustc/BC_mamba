from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

restore_dense_sources = [
    "restore_dense.cpp",
    "restore_dense_fwd_fp16.cu",
    "restore_dense_fwd_bf16.cu",
    "restore_dense_fwd_fp32.cu",
    "restore_dense_bwd_fp16.cu",
    "restore_dense_bwd_bf16.cu",
    "restore_dense_bwd_fp32.cu",
]

setup(
    name="restore_dense_cuda_nodecay",
    ext_modules=[
        CUDAExtension(
            name="restore_dense_cuda_nodecay",
            sources=restore_dense_sources,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)