from setuptools import setup, find_namespace_packages
from pathlib import Path

version = (Path(__file__).parent / "VERSION").read_text().strip()
nightly = ".dev" in version
requirements_file = "nightly.txt" if nightly else "runtime.txt"
runtime_requirements = [
    line.strip()
    for line in (Path(__file__).parent / "requirements" / requirements_file).read_text().splitlines()
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
    python_requires=">=3.12,<3.13" if nightly else ">=3.10,<3.14",
    install_requires=runtime_requirements,
    entry_points={"console_scripts": ["lightllm=lightllm.cli:main"]},
    package_data=package_data,
)
