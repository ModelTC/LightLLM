import os
import torch
import sys
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob
import copy
import subprocess

version = "0.0.3"
if "--version-suffix" in sys.argv:
    version_suffix_idx = sys.argv.index("--version-suffix")
    version_suffix = sys.argv[version_suffix_idx + 1]
    sys.argv.remove("--version-suffix")
    sys.argv.pop(version_suffix_idx)
    version += version_suffix

extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": [
        "-O3",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",  # 这个是启用half不是禁用，可能默认是forbidden的。
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        # "--ptxas-options=-v",
        # "--ptxas-options=-O2",
        "-lineinfo",
    ],
}


def find_cuda_include():
    if os.path.isdir("/usr/local/cuda/include"):
        return "/usr/local/cuda/include"
    # 1. Check if CUDA_HOME or CUDA_PATH environment variable is set
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        cuda_include = os.path.join(cuda_home, "include")
        if os.path.isdir(cuda_include):
            return cuda_include

    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        cuda_include = os.path.join(cuda_path, "include")
        if os.path.isdir(cuda_include):
            return cuda_include

    # 2. Check if 'nvcc' is available and find its path
    try:
        nvcc_path = subprocess.check_output(["which", "nvcc"], stderr=subprocess.STDOUT).decode("utf-8").strip()
        if nvcc_path:
            # Typically, nvcc is located under /usr/local/cuda/bin/nvcc, so we can use that to find the include directory
            cuda_dir = os.path.dirname(os.path.dirname(nvcc_path))  # Go two levels up from nvcc's location
            cuda_include = os.path.join(cuda_dir, "include")
            if os.path.isdir(cuda_include):
                return cuda_include
    except subprocess.CalledProcessError:
        pass

    raise EnvironmentError(
        "CUDA not found. Please set the CUDA_HOME environment variable to your CUDA installation directory."
    )


def get_vllm_kernel_extensions():
    major, minor = torch.cuda.get_device_capability()
    # 组合成 sm_xx 格式
    cuda_arch = f"{major}{minor}"
    marlin_archs = ["80", "86", "89", "90"]
    scaled_mm_3x_archs = ["90"]
    library_name = "vllm_total"
    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, library_name)
    vllm_extra_compile_args = copy.deepcopy(extra_compile_args)
    sources = [
        "csrc/cache_kernels.cu",
        "csrc/attention/paged_attention_v1.cu",
        "csrc/attention/paged_attention_v2.cu",
        "csrc/pos_encoding_kernels.cu",
        "csrc/activation_kernels.cu",
        "csrc/layernorm_kernels.cu",
        "csrc/layernorm_quant_kernels.cu",
        "csrc/quantization/gptq/q_gemm.cu",
        "csrc/quantization/compressed_tensors/int8_quant_kernels.cu",
        "csrc/quantization/fp8/common.cu",
        "csrc/cuda_utils_kernels.cu",
        "csrc/prepare_inputs/advance_step.cu",
        "csrc/torch_bindings.cpp",
        "csrc/mamba/mamba_ssm/selective_scan_fwd.cu",
        "csrc/mamba/causal_conv1d/causal_conv1d.cu",
        "csrc/quantization/aqlm/gemm_kernels.cu",
        "csrc/quantization/awq/gemm_kernels.cu",
        "csrc/quantization/gguf/gguf_kernel.cu",
        "csrc/custom_all_reduce.cu",
        "csrc/custom_all_gather.cu",
        "csrc/permute_cols.cu",
        "csrc/quantization/cutlass_w8a8/scaled_mm_entry.cu",
        "csrc/cutlass_extensions/common.cpp",
    ]
    if cuda_arch in marlin_archs:
        sources += [
            "csrc/quantization/fp8/fp8_marlin.cu",
            "csrc/quantization/marlin/dense/marlin_cuda_kernel.cu",
            "csrc/quantization/marlin/sparse/marlin_24_cuda_kernel.cu",
            "csrc/quantization/marlin/qqq/marlin_qqq_gemm_kernel.cu",
            "csrc/quantization/gptq_marlin/gptq_marlin.cu",
            "csrc/quantization/gptq_marlin/gptq_marlin_repack.cu",
            "csrc/quantization/gptq_marlin/awq_marlin_repack.cu",
        ]
    if cuda_arch in scaled_mm_3x_archs:
        sources += ["csrc/quantization/cutlass_w8a8/scaled_mm_c3x.cu"]
        vllm_extra_compile_args["nvcc"].append("-DENABLE_SCALED_MM_C3X=1")
        vllm_extra_compile_args["nvcc"].append("-gencode=arch=compute_90a, code=sm_90a")
    else:
        sources += ["csrc/quantization/cutlass_w8a8/scaled_mm_c2x.cu"]
        vllm_extra_compile_args["nvcc"].append("-DENABLE_SCALED_MM_C2X=1")

    sources = [src.replace("csrc", extensions_dir) for src in sources]

    # replace it with your own environment
    include_dirs = [
        find_cuda_include(),
        os.path.abspath("./cutlass/include"),
    ]
    for root, dirs, files in os.walk(extensions_dir):
        include_dirs.append(os.path.abspath(root))
    return [
        CUDAExtension(
            library_name,
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args=vllm_extra_compile_args,
            extra_link_args=["-lcuda"],
        ),
        CUDAExtension(
            "vllm_moe",
            sources=list(glob.glob("./vllm_total/moe/*.cpp")) + list(glob.glob("./vllm_total/moe/*.cu")),
            include_dirs=[
                os.path.abspath("./vllm_total"),
                os.path.abspath("./vllm_total/core"),
                os.path.abspath("./vllm_total/moe"),
            ],
            extra_compile_args=extra_compile_args,
        ),
    ]


setup(
    name="lightllm_kernel",
    version=version,
    author="wangzaijun",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            "flash_llm_fp6_llm",
            ["./fp6_llm/csrc/fp6_linear.cu", "./fp6_llm/csrc/pybind.cpp"],
            include_dirs=["./fp6_llm/csrc", "./fp6_llm/csrc/include", "./fp6_llm/csrc/utils"],
            # define_macros=[('TURBOMIND_ARCH_SM80', None)],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-std=c++17",
                    "-maxrregcount=255",
                    "--ptxas-options=-v,-warn-lmem-usage,--warn-on-spills",
                    "-gencode=arch=compute_80,code=sm_80",
                ],
            },
        ),
        CUDAExtension(
            "lightllm_constraint_decode_kernel",
            sources=["./constraint_decode_kernel/src/check_dpda_kernel.cu"],
            extra_compile_args={
                "cxx": ["-std=c++17"],
                "nvcc": [
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",  # 这个是启用half不是禁用，可能默认是forbidden的。
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    # "--ptxas-options=-v",
                    # "--ptxas-options=-O2",
                    "-lineinfo",
                ],
            },
        ),
        *get_vllm_kernel_extensions(),
    ],
    cmdclass={"build_ext": BuildExtension},
)
