from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

sources = [
    "selective_scan.cpp",
    "selective_scan_fwd_fp16.cu",
    "selective_scan_fwd_bf16.cu",
    "selective_scan_fwd_fp32.cu",
    # "selective_scan_fwd_event_fp16.cu",
    # "selective_scan_fwd_event_bf16.cu",
    # "selective_scan_fwd_event_fp32.cu",
    # "selective_scan_fwd_event_flat_fp16.cu",
    # "selective_scan_fwd_event_flat_bf16.cu",
    # "selective_scan_fwd_event_flat_fp32.cu",
    # fwd_fp*.cu also explicitly instantiates selective_scan_fwd_flat_ch_cuda.
    "selective_scan_bwd_fp16_real.cu",
    "selective_scan_bwd_bf16_real.cu",
    "selective_scan_bwd_fp32_real.cu",
    # "selective_scan_bwd_event_fp16.cu",
    # "selective_scan_bwd_event_bf16.cu",
    # "selective_scan_bwd_event_fp32.cu",
    # "selective_scan_bwd_event_flat_fp16.cu",
    # "selective_scan_bwd_event_flat_bf16.cu",
    # "selective_scan_bwd_event_flat_fp32.cu",
    # bwd_fp*_real.cu also explicitly instantiates selective_scan_bwd_flat_ch_cuda.
    "selective_scan_bwd_fp16_complex.cu",
    "selective_scan_bwd_bf16_complex.cu",
    "selective_scan_bwd_fp32_complex.cu",
]

setup(
    name="selective_scan_cuda",
    ext_modules=[
        CUDAExtension(
            name="selective_scan_cuda",
            sources=sources,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
