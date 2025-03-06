from setuptools import setup, find_packages

package_data = {"lightllm": ["common/all_kernel_configs/*/*.json"]}
setup(
    name="lightllm",
    version="1.0.1",
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
    python_requires=">=3.9.16",
    install_requires=[
        "pyzmq",
        "uvloop",
        "torch",
        "transformers",
        "einops",
        "packaging",
        "rpyc",
        "ninja",
        "safetensors",
        "triton",
    ],
    package_data=package_data,
)
