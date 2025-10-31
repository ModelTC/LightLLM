import os
import torch
from setuptools import Extension
from torch.utils.cpp_extension import CUDAExtension, load

PROGRAM_NAME = "KVCache"
VERSION = "0.0.3"  # 2025.4.15
EXTENSION_BUILD_DIR = "build"
INCLUDE_DIR = "src"
SOURCE_DIR = "src"

class __CompileHelper__:
    def __init__(self) -> None:
        self.BUILD_DIR = EXTENSION_BUILD_DIR
        self.__CUDA_EXTENSION__ = None

        if torch.__version__ < "1.6.0":
            raise RuntimeError(
                f"{PROGRAM_NAME} cannot finish compile; "
                "PyTorch version 1.6 or higher is required."
            )

    def compile(self) -> CUDAExtension:
        """
        Compile a CUDA Extension using this function.

        This function will compile c/c++ plugin Just-In-Time and return a
            CUDAExtension object that can be called from the Python side.
        Users can call the defined extension functions using CUDAExtension.function.

        All .c, .cc files will be compiled using gcc.
        All .cu files will be compiled using nvcc.

        Requires cuda 10.0+ and c++ 13.
        """
        print(
            f"{PROGRAM_NAME}: Performing just-in-time compilation. "
            "This process may take a few minutes."
        )
        
        # check SM version, require SM > 80
        major, _ = torch.cuda.get_device_capability()
        if major is None or major < 8:
            raise RuntimeError(
                f"{PROGRAM_NAME} requires an Nvidia GPU with SM80 or higher architecture to complete compilation. "
                "It seems that your graphics card does not meet this requirement. "
                "Please complete the compilation on a device that meets the necessary specifications."
            )

        # Delete the compilation lock file.
        # This file is used by PyTorch to manage the code compilation process.
        # If this file already exists before the compilation begins,
        # the compilation process may enter an infinite wait state.
        # Therefore, it is necessary to manually delete this file.
        lock_file = os.path.join(
            os.path.dirname(__file__) + "/python", self.BUILD_DIR, "lock"
        )
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
            except Exception:
                raise RuntimeError(f"Can not delete lock file at {lock_file}!")

        # Automatically detect .cpp and .cu source files
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        source_dir = os.path.join(root_dir, SOURCE_DIR)
        sources = []
        file_names = []  # Store file names for printing
        for subdir, _, files in os.walk(source_dir):
            for file in files:
                if file.endswith((".cpp", ".cu")):
                    sources.append(os.path.join(subdir, file))
                    file_names.append(file)

        # Print all detected source file names
        print(f"{PROGRAM_NAME}: Detected source files:")
        for file_name in file_names:
            print(f"  - {file_name}")

        # Compile the extension
        self.__CUDA_EXTENSION__ = load(
            name=PROGRAM_NAME,
            sources=sources,
            extra_include_paths=[
                os.path.join(root_dir, INCLUDE_DIR),
            ],
            build_directory=os.path.join(root_dir + "/python", self.BUILD_DIR),
            with_cuda=True,
            extra_cuda_cflags=["-O3", "-use_fast_math"],
            extra_cflags=["-O3"],
        )
        return self.__CUDA_EXTENSION__

    @property
    def EXTENSION(self) -> Extension:
        if self.__CUDA_EXTENSION__ is None:
            self.__CUDA_EXTENSION__ = self.compile()
        return self.__CUDA_EXTENSION__

CompileHelper = __CompileHelper__()
CompiledExension = CompileHelper.EXTENSION
