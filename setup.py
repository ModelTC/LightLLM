import os
from pathlib import Path

from setuptools import setup, find_namespace_packages

version = (Path(__file__).parent / "VERSION").read_text().strip()
docker_build = os.environ.get("LIGHTLLM_DOCKER_BUILD") == "1"
cuda_profile = os.environ.get("LIGHTLLM_CUDA_PROFILE", "cuda13")
if cuda_profile not in {"cuda13", "cuda128"}:
    raise ValueError(f"Unknown CUDA profile: {cuda_profile}")
requirements_files = ("docker.txt",) if docker_build else ("runtime.txt", f"{cuda_profile}.txt")
if cuda_profile == "cuda128" and not docker_build:
    version += "+cu128"
runtime_requirements = [
    line.strip()
    for filename in requirements_files
    for line in (Path(__file__).parent / "requirements" / filename).read_text().splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]

package_data = {"lightllm": ["common/all_kernel_configs/*/*.json", "common/triton_utils/*/*/*/*/*.json"]}
setup(
    name="lightllm",
    version=version,
    packages=find_namespace_packages(
        include=("lightllm", "lightllm.*"),
        exclude=(
            "*.__pycache__",
            "lightllm.common.all_kernel_configs.*",
            "lightllm.common.triton_utils.autotune_kernel_configs.*",
        ),
    ),
    author="model toolchain",
    author_email="",
    description="lightllm for inference LLM",
    long_description="",
    long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.10,<3.14",
    install_requires=runtime_requirements,
    entry_points={"console_scripts": ["lightllm=lightllm.cli:main"]},
    package_data=package_data,
)
