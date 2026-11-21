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

    $ uv build
    $ uv pip install dist/lightllm-1.2.0-py3-none-any.whl

``uv build`` builds the wheel from the source distribution as well. After a
release containing these changes is published, users can install its version
with ``uv pip install lightllm==<release-version>``. Older PyPI releases do not
necessarily contain this CLI or dependency metadata.

The base installation includes PyTorch/Triton kernels and the SGLang kernels
required by Qwen3.5 linear attention. Optional FlashInfer, standalone FlashMLA,
DeepGEMM, DeepEP, NIXL, LightMem and vLLM quantization kernels still
require their own compatible builds. In particular, AWQ and some FP8 models
need these additional kernels. Follow the corresponding model/deployment guide;
the existing Docker build includes more of these extensions. ``requirements.txt``
remains the legacy image/development environment, not the native package manifest.

Vendor runtimes such as MUSA must use their vendor PyTorch and Triton builds.
After preparing that environment with its platform guide, install this checkout
with ``uv pip install --no-deps .`` to preserve those builds. The default native
dependency set targets NVIDIA CUDA, and is not a universal hardware resolver.

Native dependency versions
~~~~~~~~~~~~~~~~~~~~~~~~~~

``requirements/runtime.txt`` defines the native installation dependencies. Core versions are pinned:

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
such as ``1.2.1.dev1234``, where ``1234`` is the CI run number. Stable builds read
``requirements/runtime.txt``; nightly builds use the fully frozen environment in
``requirements/nightly.txt``.

``requirements/nightly.in`` orders packages by priority: Torch, Triton and CUDA
first, followed by model loading and inference kernels, then serving, protocol
adapters and auxiliary libraries. The updater selects the newest installable
version, including nightly and RC releases, and freezes each selection before
resolving the next package. Lower-priority packages may downgrade to preserve
earlier selections; an unsatisfiable step fails the build. Uninstalled optional
extensions do not constrain the base environment.

Compatible dependency metadata does not guarantee binary compatibility. Builds
also import the native operators. An ABI failure rejects the candidate and
requires a kernel wheel rebuilt against the new Torch; ignoring dependencies
or downgrading the higher-priority core does not count as success.
Maintainers can supply rebuilt kernel wheels through ``UV_FIND_LINKS`` before
updating the dependencies. Keep the initially selected core snapshot during a
build: after rebuilding kernels, use ``UV_CONSTRAINT`` with those core pins or
wheel URLs to resume resolution. Versions uploaded during the build belong to
the next snapshot. Such nightly bundles must include those wheels and
can be installed with ``uv pip install dist/*.whl`` after extraction. Delivering
the LightLLM wheel alone requires its native dependencies to be available on
the corresponding nightly index first.

Maintainers can run ``python tools/update_nightly.py`` with uv and packaging
installed; nightly CI runs it automatically. This profile currently targets
Linux x86_64 and CPython 3.12, using PyPI and the official PyTorch CUDA 13.4
nightly index. The package embeds Torch, Triton and torchvision wheel URLs with
SHA256 hashes, so installation does not require extra index configuration.
Validate each new combination with GPU model and API regressions before delivery.

The ``Native package`` workflow currently uploads wheels, source distributions
and installed-version records as GitHub Actions artifacts. It does not yet
publish to PyPI or a nightly index. Install downloaded wheels with
``uv pip install <wheel-path>``. Once published to an index, select an actual
released version, for example:

.. code-block:: console

    $ uv pip install lightllm==1.2.0
    $ uv pip install lightllm==1.2.1.dev1234

Dependency downgrades are allowed during updates. Wheel installation follows the
frozen dependency set rather than resolving a new combination at deployment time.
CPU package checks do not replace GPU model regressions.

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
