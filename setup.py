from setuptools import setup, find_packages

package_data = {"lightllm": ["common/all_kernel_configs/*/*.json", "common/triton_utils/*/*/*/*/*.json"]}
setup(
    name="lightllm",
    version="1.2.0",
    packages=find_packages(exclude=("build", "include", "test", "dist", "docs", "benchmarks", "lightllm.egg-info")),
    author="model toolchain",
    author_email="",
    description="lightllm for inference LLM",
    long_description="",
    long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Linux",
    ],
    python_requires=">=3.10",
    install_requires=[
        "pyzmq",
        "uvloop",
        "transformers",
        "einops",
        "packaging",
        "rpyc",
        "ninja",
        "safetensors",
        "triton",
        "orjson",
        # Keep the final image install on the FFI version tested with TileLang 0.1.9.
        "apache-tvm-ffi==0.1.11",
        "xgrammar>=0.2.8,<0.3",
    ],
    package_data=package_data,
)
