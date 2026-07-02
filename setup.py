from setuptools import setup, find_packages

try:
    from setuptools_rust import Binding, RustExtension
except ImportError:
    Binding = None
    RustExtension = None

package_data = {"lightllm": ["common/all_kernel_configs/*/*.json", "common/triton_utils/*/*/*/*/*.json"]}
rust_extensions = []
if RustExtension is not None and Binding is not None:
    rust_extensions = [
        RustExtension(
            "lightllm.server.httpserver_for_pd_master.pd_selector._pd_tree_rust",
            path="rust/pd_tree/Cargo.toml",
            binding=Binding.PyO3,
        )
    ]
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
    package_data=package_data,
    rust_extensions=rust_extensions,
    zip_safe=False,
)
