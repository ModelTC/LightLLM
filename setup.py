from setuptools import setup, find_packages

package_data = {"lightllm": ["common/all_kernel_configs/*/*.json", "common/triton_utils/*/*/*/*/*.json"]}
setup(
    name="lightllm",
    version="1.1.0",
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
        "transformers",
        "einops",
        "packaging",
        "rpyc",
        "ninja",
        "safetensors",
        "triton",
        "orjson",
    ],
    extras_require={
        # /v1/messages (Anthropic Messages compatibility) uses litellm's
        # adapter for request/response translation. Only install if you
        # plan to serve Anthropic-SDK clients.
        "anthropic": ["litellm>=1.52.0,<1.85"],
    },
    package_data=package_data,
)
