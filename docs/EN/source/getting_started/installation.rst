.. _installation:

Installation Guide
==================

Lightllm is a pure Python-based inference framework with operators written in Triton.

Environment Requirements
------------------------

* Operating System: Linux
* Python: 3.10–3.13 (3.12 recommended for the native installation below)
* NVIDIA GPU and a driver compatible with CUDA 13.0 for the default native runtime.
  Hardware support also depends on the selected model and kernels.

Native installation with uv
---------------------------

From a checkout of this version:

.. code-block:: console

    $ uv venv --python 3.12
    $ source .venv/bin/activate
    $ uv pip install .
    $ lightllm serve /models/my-model

The package declares its runtime dependencies, including PyTorch, Transformers,
the HTTP server, XGrammar and SGLang kernels. A separate ``requirements.txt`` installation is not
needed. Use ``uv pip install -e .`` for development. The default PyTorch 2.14.1
Linux wheel uses CUDA 13.0; installing this package does not install a GPU driver.
Start with a supported, unquantized model that fits on one GPU.

Use ``lightllm serve --help`` for common options and ``--help-all`` for the full
list. All existing server options and their defaults are preserved, including
``--model_dir``. The original Python module entry point remains available.

To deliver an installable artifact:

.. code-block:: console

    $ uv build --sdist
    $ uv build --wheel
    $ uv pip install dist/lightllm-1.2.0-py3-none-any.whl

Build the wheel directly from the checkout with ``uv build --wheel``. uv's
source archive extractor rejects colons in the bundled kernel configuration
paths. After a release containing these changes is published, users can install
its version with ``uv pip install lightllm==<release-version>``. Older PyPI
releases do not necessarily contain this CLI or dependency metadata.

CUDA 12.8 on Linux x86_64
~~~~~~~~~~~~~~~~~~~~~~~~~

Download and extract the ``Native package`` CI artifact. Install the
``+cu128`` wheel into a fresh environment:

.. code-block:: console

    $ uv venv --python 3.12
    $ source .venv/bin/activate
    $ uv pip install ./lightllm-1.2.0+cu128-py3-none-any.whl

The wheel pins PyTorch 2.11.0 and torchvision 0.26.0 for CUDA 12.8,
Triton 3.6.0, the SGLang 0.4.5 CUDA 12.9 kernel wheel and CuPy for CUDA 12.
The bundled ``lightllm-install.py`` selects ``cuda13`` on drivers 580.0 and
newer. CUDA 12.8 requires explicit selection on all eligible drivers (570.124.06
or newer). GPU generation was verified on driver 580.95.05. Run
``python lightllm-install.py --profile cuda128 --venv .venv-cu128`` to select
CUDA 12.8 explicitly. To build that wheel from source, run
``LIGHTLLM_CUDA_PROFILE=cuda128 uv build --wheel``.

The base installation includes PyTorch/Triton kernels and the SGLang kernels
required by Qwen3.5 linear attention. Optional FlashInfer, standalone FlashMLA,
DeepGEMM, DeepEP, NIXL, LightMem and vLLM quantization kernels still
require their own compatible builds. In particular, AWQ and some FP8 models
need these additional kernels. Follow the corresponding model/deployment guide;
the existing Docker build includes more of these extensions. ``requirements.txt``
remains the legacy image/development environment, not the native package manifest.
The Docker build uses that environment to compile extensions, then installs LightLLM
with Docker-specific dependency metadata and keeps the installed versions.

Vendor runtimes such as MUSA must use their vendor PyTorch and Triton builds.
After preparing that environment with its platform guide, install this checkout
with ``uv pip install --no-deps .`` to preserve those builds. The default native
dependency set targets NVIDIA CUDA, and is not a universal hardware resolver.

Native dependency versions
~~~~~~~~~~~~~~~~~~~~~~~~~~

``requirements/runtime.txt`` defines the shared native dependencies.
``requirements/cuda13.txt`` pins the default CUDA 13 runtime:

.. list-table::
   :header-rows: 1

   * - Component
     - Version
   * - PyTorch / torchvision
     - 2.14.1 / 0.29.1
   * - Triton / SGLang kernel
     - 3.8.0 / 0.4.10
   * - CuPy CUDA 13 / Transformers
     - 14.2.0 / 5.20.2
   * - XGrammar / TVM FFI
     - 0.2.8 / 0.1.12
   * - LiteLLM
     - 1.106.2 (below 1.107)

Upgrade PyTorch, torchvision, Triton and SGLang kernels together: PyTorch constrains
Triton, and SGLang kernel 0.4.10 requires PyTorch 2.14.1. Validate installation,
model startup, text and image requests, and structured output in a fresh environment
after upgrading; successful dependency resolution does not establish GPU kernel compatibility.
Tuning configurations depend on the Triton version and GPU; measure performance
again after upgrading and retune when needed.
TVM FFI 0.1.12 satisfies optional TileLang 0.1.15's ``<0.1.13`` constraint.
LiteLLM 1.106.2 requires ``filelock<4`` and ``huggingface-hub<2``. Other
dependencies use compatible ranges resolved by uv. Python 3.12 can use Librosa 1.0,
Pillow 13 and WebSockets 17; older Python versions select supported releases.
Record the complete delivery
environment with ``uv pip freeze``. The legacy image dependencies are maintained
separately; do not install the root ``requirements.txt`` over this native environment.

Stable and nightly builds
~~~~~~~~~~~~~~~~~~~~~~~~~

Both channels install ``lightllm`` and use the same ``lightllm serve`` command.
Stable builds come from ``vX.Y.Z`` tags matching ``VERSION``. Daily main-branch
builds and manual ``nightly`` builds use the next patch's development version,
such as ``1.2.1.dev1234``, where ``1234`` is the CI run number. Nightly builds
package the latest LightLLM code with the same CUDA 13 and CUDA 12.8 profiles as
stable builds. Upgrade the pinned PyTorch, Triton and SGLang kernels together
and validate the new combination on a GPU.

The ``Native package`` workflow currently uploads wheels, source distributions
and installed-version records as GitHub Actions artifacts. It does not yet
publish to PyPI or a nightly index. Install downloaded wheels with
``uv pip install <wheel-path>``. Once published to an index, select an actual
released version, for example:

.. code-block:: console

    $ uv pip install lightllm==1.2.0
    $ uv pip install lightllm==1.2.1.dev1234

.. _build_from_docker:

Installation via Docker
-----------------------
The easiest way to install Lightllm is using the official image. You can directly pull the official image and run it:

.. code-block:: console

    $ # Pull the official image
    $ docker pull ghcr.io/modeltc/lightllm:main
    $
    $ # Run，The current LightLLM service relies heavily on shared memory.
    $ # Before starting, please make sure that you have allocated enough shared memory
    $ # in your Docker settings; otherwise, the service may fail to start properly.
    $ #
    $ # 1. For text-only services, it is recommended to allocate more than 2GB of shared memory.
    $ # If your system has sufficient RAM, allocating 16GB or more is recommended.
    $ # 2.For multimodal services, it is recommended to allocate 16GB or more of shared memory.
    $ # You can adjust this value according to your specific requirements.
    $ #
    $ # If you do not have enough shared memory available, you can try lowering
    $ # the --running_max_req_size parameter when starting the service.
    $ # This will reduce the number of concurrent requests, but also decrease shared memory usage.
    $ docker run -it --gpus all -p 8080:8080            \
    $   --shm-size 2g -v your_local_path:/data/         \
    $   ghcr.io/modeltc/lightllm:main /bin/bash

You can also manually build the image from source and run it:

.. code-block:: console

    $ # move into lightllm root dir
    $ cd /lightllm
    $ # Manually build the image
    $ docker build -t <image_name> -f ./docker/Dockerfile .
    $
    $ # Run,
    $ docker run -it --gpus all -p 8080:8080            \
    $   --shm-size 2g -v your_local_path:/data/         \
    $   <image_name> /bin/bash

Or you can directly use the script to launch the image and run it with one click:

.. code-block:: console

    $ # View script parameters
    $ python tools/quick_launch_docker.py --help

.. note::
    If you use multiple GPUs, you may need to increase the --shm-size parameter setting above. 

Installation from Source
------------------------

You can also install Lightllm from source:

.. code-block:: console

    $ # (Recommended) Create a new conda environment
    $ conda create -n lightllm python=3.10 -y
    $ conda activate lightllm
    $
    $ # Download the latest Lightllm source code
    $ git clone https://github.com/ModelTC/lightllm.git
    $ cd lightllm
    $
    $ # Install Lightllm dependencies (CUDA 13.0)
    $ pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
    $
    $ # Install Lightllm dependencies (Moore Threads GPU)
    $ ./generate_requirements_musa.sh
    $ pip install -r requirements-musa.txt
    $
    $ # Install Lightllm
    $ pip install --no-deps .
